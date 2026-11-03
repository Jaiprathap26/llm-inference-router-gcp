import asyncio
from dataclasses import replace
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache import Cache
from app.config import Settings
from app.main import create_app
from app.router import ChatRequest, Engine, GatewayError, Message, make_key, route
from app.telemetry import Metrics


def settings(**changes):
    changes.setdefault('cache_hmac_secret', 'test-hmac-secret')
    return replace(Settings.from_env(), **changes)


def payload(text='hello', **kw):
    return {'model': 'auto', 'messages': [{'role': 'user', 'content': text}], **kw}


def event(obj):
    return 'data: ' + json.dumps(obj, ensure_ascii=False) + '\n\n'


def upstream_sse(text='Café', finish='stop'):
    return (event({'choices': [{'delta': {'role': 'assistant'}}]}) +
            event({'choices': [{'delta': {'content': text}, 'finish_reason': None}]}) +
            event({'choices': [{'delta': {}, 'finish_reason': finish}]}) +
            event({'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 2}}) + 'data: [DONE]\n\n')


def test_json_cache_auth_and_validation():
    with TestClient(create_app(settings(gateway_key='secret'))) as client:
        assert client.post('/v1/chat/completions', json=payload()).status_code == 401
        headers = {'Authorization': 'Bearer secret'}
        first = client.post('/v1/chat/completions', json=payload(), headers=headers).json()
        second = client.post('/v1/chat/completions', json=payload(), headers=headers).json()
        assert not first['cache_hit'] and second['cache_hit']
        assert second['id'] != first['id']
        assert second['choices'][0]['message']['content'].startswith('[MOCK')
        assert second['router']['estimated_cost_usd'] == 0
        assert second['router']['ttft_ms'] is None
        assert client.post('/v1/chat/completions', json=payload(tools=[]), headers=headers).status_code == 422
        assert client.post('/v1/chat/completions', json=payload(model='other'), headers=headers).status_code == 422
        metrics = client.get('/v1/metrics', headers=headers).json()
        assert metrics['sample_count'] == 2 and metrics['cache_hits'] == 1
        assert client.get('/v1/metrics').status_code == 401


def test_stream_cache_and_protocol():
    with TestClient(create_app(settings())) as client:
        for hit in [False, True]:
            response = client.post('/v1/chat/completions', json=payload(stream=True))
            assert response.headers['content-type'].startswith('text/event-stream')
            lines = [x[6:] for x in response.text.splitlines() if x.startswith('data: ')]
            assert lines[-1] == '[DONE]'
            chunks = [json.loads(x) for x in lines[:-1]]
            assert chunks[0]['choices'][0]['delta'] == {'role': 'assistant'}
            assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'
            assert chunks[-1]['cache_hit'] is hit
            assert chunks[-1]['router']['ttft_ms'] is not None
            assert len({x['id'] for x in chunks}) == 1


def test_keys_are_exact_and_scoped():
    s = settings(gateway_key='tenant-A')
    a = ChatRequest(messages=[Message(role='user', content='a  b')])
    b = ChatRequest(messages=[Message(role='user', content='a b')])
    assert make_key(a, s.fast_model, s) != make_key(b, s.fast_model, s)
    assert make_key(a, s.fast_model, s) != make_key(a, s.fast_model, replace(s, gateway_key='tenant-B'))
    assert make_key(a, s.fast_model, s) != make_key(a, s.deep_model, s)
    assert make_key(a, s.fast_model, s) != make_key(a.model_copy(update={'temperature': 1}), s.fast_model, s)
    assert route(a, s)[0] == s.fast_model
    assert route(ChatRequest(messages=[Message(role='user', content='analyze code and derive proof')]), s)[0] == s.deep_model


@pytest.mark.asyncio
async def test_ttl_and_lru(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('app.cache.time.monotonic', lambda: clock[0])
    cache = Cache(2, 2)
    await cache.set('a', {'x': 1}); await cache.set('b', {'x': 2})
    assert await cache.get('a') == {'x': 1}
    await cache.set('c', {'x': 3})
    assert await cache.get('b') is None
    clock[0] += 3
    assert await cache.get('a') is None


def test_real_upstream_cache_opt_in_and_usage():
    calls = []
    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, text=upstream_sse(), headers={'content-type': 'text/event-stream'})
    s = settings(upstream_key='upstream', gateway_key='gateway', enable_upstream_cache=True,
                 fast_input_price=1, fast_output_price=2, deep_input_price=4, deep_output_price=8)
    with TestClient(create_app(s, httpx.MockTransport(handler))) as client:
        h = {'Authorization': 'Bearer gateway'}
        one = client.post('/v1/chat/completions', json=payload(), headers=h).json()
        two = client.post('/v1/chat/completions', json=payload(), headers=h).json()
        assert len(calls) == 1 and calls[0]['stream'] is True
        assert one['choices'][0]['message']['content'] == 'Café'
        assert one['usage']['completion_tokens'] == 2 and not one['router']['usage_estimated']
        assert one['router']['estimated_cost_usd'] == pytest.approx(9 / 1_000_000)
        assert two['cache_hit'] and two['router']['estimated_cost_usd'] == 0


def test_real_cache_disabled_without_verified_principal_or_optin():
    for changes in [{'gateway_key': '', 'enable_upstream_cache': True}, {'gateway_key': 'g', 'enable_upstream_cache': False}]:
        calls = []
        def handler(req):
            calls.append(1)
            return httpx.Response(200, text=upstream_sse())
        s = settings(upstream_key='key', **changes)
        with TestClient(create_app(s, httpx.MockTransport(handler))) as client:
            h = {'Authorization': 'Bearer g'} if s.gateway_key else {}
            for _ in range(2):
                assert not client.post('/v1/chat/completions', json=payload(), headers=h).json()['cache_hit']
            assert len(calls) == 2


def test_upstream_http_error_and_truncated_sse():
    def handler(request):
        if json.loads(request.content)['messages'][0]['content'] == 'bad':
            return httpx.Response(503, text='private provider details')
        return httpx.Response(200, text=event({'choices': [{'delta': {'content': 'partial'}}]}))
    with TestClient(create_app(settings(upstream_key='x'), httpx.MockTransport(handler))) as client:
        result = client.post('/v1/chat/completions', json=payload('bad'))
        assert result.status_code == 502 and 'private' not in result.text
        stream = client.post('/v1/chat/completions', json=payload('partial', stream=True))
        assert stream.status_code == 200 and 'partial' in stream.text and 'upstream_error' in stream.text
        assert 'data: [DONE]\n\n' not in stream.text
        metrics = client.get('/v1/metrics').json()
        assert metrics['errors'] == 2


def test_length_finish_reason_not_cached():
    calls = []
    def handler(req):
        calls.append(1)
        return httpx.Response(200, text=upstream_sse(finish='length'))
    s = settings(upstream_key='x', gateway_key='g', enable_upstream_cache=True)
    with TestClient(create_app(s, httpx.MockTransport(handler))) as client:
        for _ in range(2):
            body = client.post('/v1/chat/completions', json=payload(), headers={'Authorization': 'Bearer g'}).json()
            assert body['choices'][0]['finish_reason'] == 'length' and not body['cache_hit']
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_cancellation_closes_upstream():
    closed = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({'choices': [{'delta': {'content': 'hi'}}]}).encode()
            await asyncio.sleep(60)
        async def aclose(self):
            closed.append(True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=Stream()))) as client:
        engine = Engine(settings(upstream_key='x'), Cache(5, 10), client, Metrics())
        gen = engine.generate(ChatRequest(stream=True, messages=[Message(role='user', content='hi')]))
        assert (await anext(gen))['text'] == 'hi'
        await gen.aclose()
        assert closed and engine.metrics.snapshot()['errors'] == 1


def test_benchmark_and_paid_guard():
    s = settings(fast_input_price=1, fast_output_price=2, deep_input_price=5, deep_output_price=10)
    with TestClient(create_app(s)) as client:
        result = client.post('/v1/benchmark', json={'requests': 20, 'concurrency': 5}).json()
        assert result['mode'] == 'mock' and result['repeat_fraction'] == .35
        assert result['baseline']['cache_hit_rate'] == 0
        assert result['optimized']['cache_hit_rate'] == .35
        assert result['baseline']['estimated_cost_usd'] == result['optimized']['estimated_cost_usd'] == 0
        assert result['modeled_cost_reduction_percent'] > 0 and '| Baseline |' in result['markdown']
    with TestClient(create_app(settings(upstream_key='key'))) as client:
        assert client.post('/v1/benchmark', json={'allow_paid_upstream': True}).status_code == 400
