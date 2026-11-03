import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
import time
import uuid
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.telemetry import Timer


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'developer', 'user', 'assistant']
    content: str = Field(max_length=32000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: Literal['auto'] = 'auto'
    messages: list[Message] = Field(min_length=1, max_length=64)
    stream: bool = False
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=2048)


@dataclass(frozen=True)
class Principal:
    tenant: str
    authenticated: bool
    secret: str = ''


def default_principal(settings):
    return Principal('default' if settings.gateway_key else 'anonymous-mock', bool(settings.gateway_key), settings.gateway_key)


class GatewayError(Exception):
    def __init__(self, message, status=502):
        self.message, self.status = message, status
        super().__init__(message)


def tokens(text):
    return max(1, math.ceil(len(text) / 4)) if text else 0


def route(req, settings):
    text = ' '.join(m.content for m in req.messages)
    markers = re.findall(r'\b(?:code|debug|algorithm|proof|derive|reason|analy[sz]e|architecture|refactor|step.by.step)\b|```', text, re.I)
    score = tokens(text) + 100 * len(markers)
    return (settings.deep_model if score >= 180 else settings.fast_model), score


def make_key(req, model, settings, principal=None):
    # Exact content is intentional: whitespace folding would change code and system instructions.
    principal = principal or default_principal(settings)
    material = {'v': 3, 'policy': 'complexity-1', 'scope': 'chat', 'tenant': principal.tenant,
                'request': req.model_dump(exclude={'stream'}), 'model': model,
                'upstream': settings.upstream_url, 'mock': settings.mock,
                'credential': hashlib.sha256((settings.upstream_key + '\0' + principal.secret).encode()).hexdigest()}
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    secret = settings.cache_hmac_secret or principal.secret or 'anonymous-mock-only'
    return hmac.new(secret.encode(), encoded, hashlib.sha256).hexdigest()


def cost(inp, out, ip, op):
    return (inp * ip + out * op) / 1_000_000 if ip is not None and op is not None else None


def chunk(cid, created, model, delta, finish=None, **extra):
    return {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model,
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}], **extra}


class Flight:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.events = []
        self.done = False
        self.error = None
        self.members = set()
        self.owner = None
        self.task = None


class Engine:
    def __init__(self, settings, cache, client, metrics):
        self.settings, self.cache, self.client, self.metrics = settings, cache, client, metrics
        self.flights = {}
        self.flight_lock = asyncio.Lock()

    async def close(self):
        async with self.flight_lock:
            tasks = [flight.task for flight in self.flights.values() if flight.task and not flight.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _produce(self, flight, req, cache, principal):
        gen = self._generate_direct(req, cache=cache, principal=principal, record=False)
        try:
            async for event in gen:
                async with flight.condition:
                    if len(flight.events) >= 16384:
                        raise GatewayError('Upstream emitted too many chunks')
                    flight.events.append(event)
                    flight.condition.notify_all()
        except asyncio.CancelledError:
            raise
        except GatewayError as exc:
            flight.error = exc
        except Exception:
            flight.error = GatewayError('Upstream processing failed')
        finally:
            await gen.aclose()
            async with flight.condition:
                flight.done = True
                flight.condition.notify_all()

    async def generate(self, req, *, baseline=False, cache=None, record=True, principal=None):
        timer = Timer()
        principal = principal or default_principal(self.settings)
        cache = cache if cache is not None else self.cache
        s = self.settings
        eligible = not baseline and (s.mock or (s.enable_upstream_cache and principal.authenticated and bool(s.cache_hmac_secret))) and cache.size > 0 and cache.ttl > 0
        model, score = route(req, s)
        key = make_key(req, model, s, principal)
        # Completed entries replay directly. Only concurrent misses join a flight.
        if not eligible or await cache.get(key) is not None:
            gen = self._generate_direct(req, baseline=baseline, cache=cache, record=record, principal=principal, timer=timer)
            try:
                async for event in gen:
                    yield event
            finally:
                await gen.aclose()
            return

        member = object()
        registry_key = (id(cache), key)
        async with self.flight_lock:
            flight = self.flights.get(registry_key)
            follower = flight is not None
            if flight is None:
                if len(self.flights) >= 256:
                    raise GatewayError('Too many in-flight requests', 503)
                flight = Flight()
                self.flights[registry_key] = flight
            if len(flight.members) >= 128:
                raise GatewayError('Too many requests waiting for this completion', 503)
            flight.members.add(member)
            if flight.task is None:
                flight.task = asyncio.create_task(self._produce(flight, req, cache, principal))
        cid, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())
        index, success = 0, False
        row = {'model': model, 'mode': 'mock' if s.mock else 'upstream', 'cache_hit': False,
               'coalesced': follower, 'error': False, 'complexity_score': score}
        try:
            while True:
                async with flight.condition:
                    await flight.condition.wait_for(lambda: index < len(flight.events) or flight.done)
                    if index < len(flight.events):
                        event = flight.events[index]
                        index += 1
                        if event['type'] == 'final':
                            # First successful subscriber to consume the result owns the shared upstream charge.
                            charge_owner = not getattr(flight, 'billed', False)
                            flight.billed = True
                    else:
                        if flight.error:
                            raise flight.error
                        raise GatewayError('Shared completion ended unexpectedly')
                if event['type'] == 'content':
                    timer.token()
                    yield {'type': 'content', 'text': event['text'], 'id': cid, 'created': created,
                           'model': event['model'], 'cache_hit': False}
                else:
                    base = event['metrics']
                    row.update(base)
                    row['coalesced'] = follower
                    row['cache_hit'] = bool(base['cache_hit'] and not follower)
                    row['shared_cost_attributed'] = bool(charge_owner and not base['cache_hit'])
                    if not charge_owner and not base['cache_hit']:
                        row['estimated_cost_usd'] = 0
                        row['modeled_cost_usd'] = 0
                        row['estimated_cost_saved_usd'] = cost(row['input_tokens'], row['output_tokens'], s.deep_input_price, s.deep_output_price)
                    timings = timer.finish(row['output_tokens'])
                    timings['first_content_observed_ms'] = timings['ttft_ms']
                    timings['first_content_available_ms'] = timings['ttft_ms'] if req.stream else timings['total_latency_ms']
                    if not req.stream:
                        timings['ttft_ms'] = None
                    row.update(timings)
                    success = True
                    yield {'type': 'final', 'id': cid, 'created': created, 'model': event['model'],
                           'content': event['content'], 'finish_reason': event['finish_reason'], 'metrics': row}
                    break
        finally:
            if not success:
                row.update(timer.finish(0), error=True)
            if record:
                self.metrics.add(row, principal.tenant)
            task = None
            async with self.flight_lock:
                flight.members.discard(member)
                if not flight.members:
                    if self.flights.get(registry_key) is flight:
                        del self.flights[registry_key]
                    if flight.task and not flight.task.done():
                        task = flight.task
                        task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    async def _generate_direct(self, req, *, baseline=False, cache=None, record=True, principal=None, timer=None):
        """Yield content pieces, then final metadata. Exceptions after first yield become SSE errors."""
        timer = timer or Timer()
        s = self.settings
        principal = principal or default_principal(s)
        selected, score = route(req, s)
        model = s.deep_model if baseline else selected
        cache = cache if cache is not None else self.cache
        cid, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())
        key = make_key(req, model, s, principal)
        input_tokens = sum(tokens(m.content) for m in req.messages)
        out_tokens, hit, text, success = 0, False, '', False
        finish_reason = 'stop'
        usage_estimated = True
        row = {'model': model, 'mode': 'mock' if s.mock else 'upstream', 'cache_hit': False,
               'coalesced': False, 'error': False, 'complexity_score': score}
        try:
            cache_enabled = not baseline and (s.mock or (s.enable_upstream_cache and principal.authenticated and bool(s.cache_hmac_secret)))
            cached = await cache.get(key) if cache_enabled else None
            if cached is not None:
                hit = True
                text = cached['content']
                out_tokens = cached['output_tokens']
                input_tokens = cached['input_tokens']
                usage_estimated = cached.get('usage_estimated', True)
                finish_reason = cached.get('finish_reason', 'stop')
                if text:
                    timer.token()
                    yield {'type': 'content', 'text': text, 'id': cid, 'created': created, 'model': model, 'cache_hit': True}
            elif s.mock:
                # Explicit deterministic synthetic completion, never presented as a model response.
                last = next((m.content for m in reversed(req.messages) if m.role == 'user'), '')
                answer = ('[MOCK, not an LLM] ' + last)[:req.max_tokens * 4]
                for part in re.findall(r'\S+\s*', answer):
                    timer.token()
                    text += part
                    yield {'type': 'content', 'text': part, 'id': cid, 'created': created, 'model': model, 'cache_hit': False}
                out_tokens = tokens(text)
            else:
                payload = req.model_dump(exclude={'model', 'stream'})
                payload.update(model=model, stream=True, stream_options={'include_usage': True})
                upstream = None
                try:
                    request = self.client.build_request('POST', s.upstream_url + '/chat/completions',
                                                        json=payload, headers={'Authorization': 'Bearer ' + s.upstream_key})
                    deadline = time.monotonic() + s.upstream_timeout
                    upstream = await asyncio.wait_for(self.client.send(request, stream=True), timeout=max(0, deadline - time.monotonic()))
                    if upstream.status_code >= 400:
                        raise GatewayError(f'Upstream returned HTTP {upstream.status_code}', 502)
                    done = False
                    async def frames():
                        lines, size = [], 0
                        async for line in upstream.aiter_lines():
                            if not line:
                                if lines:
                                    yield '\n'.join(lines)
                                    lines, size = [], 0
                            elif line.startswith('data:'):
                                value = line[5:]
                                size += len(value)
                                if size > 1_200_000:
                                    raise GatewayError('Upstream SSE frame exceeds size limit')
                                lines.append(value[1:] if value.startswith(' ') else value)
                        if lines:
                            yield '\n'.join(lines)
                    async def timed_frames():
                        iterator = frames()
                        try:
                            while True:
                                try:
                                    yield await asyncio.wait_for(anext(iterator), timeout=max(0, deadline - time.monotonic()))
                                except StopAsyncIteration:
                                    break
                        finally:
                            await iterator.aclose()
                    async for data in timed_frames():
                        if data == '[DONE]':
                            done = True
                            break
                        try:
                            obj = json.loads(data)
                            if obj.get('error'):
                                raise GatewayError('Upstream returned a stream error')
                            usage = obj.get('usage')
                            if (isinstance(usage, dict) and type(usage.get('prompt_tokens')) is int and
                                    type(usage.get('completion_tokens')) is int and
                                    usage['prompt_tokens'] >= 0 and usage['completion_tokens'] >= 0):
                                input_tokens, out_tokens = usage['prompt_tokens'], usage['completion_tokens']
                                usage_estimated = False
                            choices = obj.get('choices') or []
                            delta = choices[0].get('delta', {}) if choices else {}
                            if delta.get('tool_calls') or delta.get('function_call') or delta.get('refusal') or delta.get('audio'):
                                raise GatewayError('Unsupported upstream response modality')
                            if choices and choices[0].get('finish_reason'):
                                finish_reason = choices[0]['finish_reason']
                            piece = delta.get('content') or ''
                            if not isinstance(piece, str):
                                raise GatewayError('Unsupported upstream content format')
                            if piece:
                                timer.token()
                                text += piece
                                if len(text) > 1_000_000:
                                    raise GatewayError('Upstream response exceeds size limit')
                                yield {'type': 'content', 'text': piece, 'id': cid, 'created': created, 'model': model, 'cache_hit': False}
                        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                            raise GatewayError('Malformed upstream SSE event') from exc
                    if not done:
                        raise GatewayError('Upstream stream ended without [DONE]')
                except (httpx.TimeoutException, TimeoutError) as exc:
                    raise GatewayError('Upstream timed out', 504) from exc
                except httpx.HTTPError as exc:
                    raise GatewayError('Upstream connection failed', 502) from exc
                finally:
                    if upstream is not None:
                        await upstream.aclose()
                if usage_estimated:
                    out_tokens = tokens(text)
            if not hit and cache_enabled and finish_reason == 'stop' and len(text.encode('utf-8')) <= 262144:
                await cache.set(key, {'content': text, 'input_tokens': input_tokens, 'output_tokens': out_tokens,
                                      'usage_estimated': usage_estimated, 'finish_reason': finish_reason})
            baseline_cost = cost(input_tokens, out_tokens, s.deep_input_price, s.deep_output_price)
            actual = 0 if hit or s.mock else (None if usage_estimated else cost(input_tokens, out_tokens,
                s.fast_input_price if model == s.fast_model else s.deep_input_price,
                s.fast_output_price if model == s.fast_model else s.deep_output_price))
            # Savings are a modelled comparison, not actual money saved in mock mode.
            modeled = 0 if hit else cost(input_tokens, out_tokens,
                s.fast_input_price if model == s.fast_model else s.deep_input_price,
                s.fast_output_price if model == s.fast_model else s.deep_output_price)
            timings = timer.finish(out_tokens)
            timings['first_content_observed_ms'] = timings['ttft_ms']
            timings['first_content_available_ms'] = timings['ttft_ms'] if req.stream else timings['total_latency_ms']
            if not req.stream:
                timings['ttft_ms'] = None  # JSON first-content availability is not streamed TTFT.
            row.update(timings, cache_hit=hit, input_tokens=input_tokens, output_tokens=out_tokens,
                       usage_estimated=usage_estimated, estimated_cost_usd=actual,
                       modeled_cost_usd=modeled, estimated_cost_saved_usd=(baseline_cost - modeled) if baseline_cost is not None and modeled is not None else None)
            success = True
            yield {'type': 'final', 'id': cid, 'created': created, 'model': model, 'content': text,
                   'finish_reason': finish_reason, 'metrics': row}
        finally:
            if not success:
                row.update(timer.finish(out_tokens), error=True, cache_hit=hit)
            if record:
                self.metrics.add(row, principal.tenant)
