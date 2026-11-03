import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.benchmark import BenchmarkRequest, run_benchmark
from app.cache import Cache
from app.config import Settings
from app.limits import LocalRateLimiter
from app.router import ChatRequest, Engine, GatewayError, Principal, chunk
from app.telemetry import Metrics


def create_app(settings=None, transport=None):
    settings = settings or Settings.from_env()
    parsed = urlsplit(settings.upstream_url)
    if settings.upstream_key and (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Real upstream URL must be HTTPS with a hostname and no embedded credentials, query or fragment')
    if max(settings.chat_rate_per_minute, settings.benchmark_rate_per_minute) > 10000:
        raise ValueError('Rate limits must be at most 10000 per minute')
    if settings.gateway_key and settings.tenant_api_keys:
        raise ValueError('Configure either GATEWAY_API_KEY or TENANT_API_KEYS, not both')
    if settings.upstream_key and settings.enable_upstream_cache and (settings.gateway_key or settings.tenant_api_keys) and not settings.cache_hmac_secret:
        raise ValueError('Real-provider caching requires CACHE_HMAC_SECRET')

    @asynccontextmanager
    async def lifespan(app):
        redis = None
        if settings.redis_url:
            from redis.asyncio import Redis
            redis = Redis.from_url(settings.redis_url, socket_connect_timeout=1, socket_timeout=1)
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.upstream_timeout), transport=transport) as client:
            app.state.engine = Engine(settings, Cache(settings.cache_size, settings.cache_ttl, redis), client, Metrics())
            app.state.limiter = LocalRateLimiter()
            app.state.benchmark_lock = asyncio.Lock()
            try:
                yield
            finally:
                await app.state.engine.close()
                if redis:
                    await redis.aclose()

    app = FastAPI(title='LLM Inference Router', version='1.0.0', lifespan=lifespan)

    async def authorize(request: Request):
        credentials = settings.tenant_api_keys or ((('default', settings.gateway_key),) if settings.gateway_key else ())
        if not credentials:
            return Principal('anonymous-mock', False)
        supplied = request.headers.get('Authorization', '').encode('utf-8')
        principal = None
        for tenant, secret in credentials:
            if hmac.compare_digest(supplied, ('Bearer ' + secret).encode('utf-8')):
                principal = Principal(tenant, True, secret)
        if principal is None:
            raise HTTPException(status_code=401, detail='Invalid gateway bearer token', headers={'WWW-Authenticate': 'Bearer'})
        return principal

    @app.get('/healthz')
    async def health():
        return {'status': 'ok', 'mode': 'mock' if settings.mock else 'upstream'}

    @app.get('/v1/metrics')
    async def metrics(request: Request, principal: Principal = Depends(authorize)):
        return request.app.state.engine.metrics.snapshot(principal.tenant)

    @app.post('/v1/chat/completions')
    async def completions(body: ChatRequest, request: Request, principal: Principal = Depends(authorize)):
        if not request.app.state.limiter.allow(principal.tenant, 'chat', settings.chat_rate_per_minute):
            raise HTTPException(status_code=429, detail='Chat rate limit exceeded', headers={'Retry-After': '60'})
        if sum(len(m.content) for m in body.messages) > 128000:
            raise HTTPException(status_code=413, detail='Combined message content exceeds 128000 characters')
        generator = request.app.state.engine.generate(body, principal=principal)
        # Prime before sending response headers, so connection/status errors remain HTTP errors.
        try:
            first = await anext(generator)
        except GatewayError as exc:
            return JSONResponse(status_code=exc.status, content={'error': {'message': exc.message, 'type': 'upstream_error'}})
        if not body.stream:
            final = first if first['type'] == 'final' else None
            try:
                async for event in generator:
                    if event['type'] == 'final':
                        final = event
            except GatewayError as exc:
                return JSONResponse(status_code=exc.status, content={'error': {'message': exc.message, 'type': 'upstream_error'}})
            if final is None:
                return JSONResponse(status_code=502, content={'error': {'message': 'Incomplete upstream response'}})
            m = final['metrics']
            return {'id': final['id'], 'object': 'chat.completion', 'created': final['created'], 'model': final['model'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': final['content']}, 'finish_reason': final['finish_reason']}],
                    'usage': {'prompt_tokens': m['input_tokens'], 'completion_tokens': m['output_tokens'],
                              'total_tokens': m['input_tokens'] + m['output_tokens']},
                    'cache_hit': m['cache_hit'], 'router': m}

        async def stream():
            def encode(obj):
                return 'data: ' + json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n\n'
            async def send_event(e):
                if e['type'] == 'content':
                    return encode(chunk(e['id'], e['created'], e['model'], {'content': e['text']}, cache_hit=e['cache_hit']))
                return encode(chunk(e['id'], e['created'], e['model'], {}, e['finish_reason'], cache_hit=e['metrics']['cache_hit'], router=e['metrics']))
            try:
                yield encode(chunk(first['id'], first['created'], first['model'], {'role': 'assistant'}))
                yield await send_event(first)
                async for event in generator:
                    yield await send_event(event)
                yield 'data: [DONE]\n\n'
            except GatewayError as exc:
                # After headers, use an error event and close without a success sentinel.
                yield encode({'error': {'message': exc.message, 'type': 'upstream_error'}})
            finally:
                await generator.aclose()
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post('/v1/benchmark')
    async def benchmark(body: BenchmarkRequest, request: Request, principal: Principal = Depends(authorize)):
        if not request.app.state.limiter.allow(principal.tenant, 'benchmark', settings.benchmark_rate_per_minute):
            raise HTTPException(status_code=429, detail='Benchmark rate limit exceeded', headers={'Retry-After': '60'})
        if request.app.state.benchmark_lock.locked():
            raise HTTPException(status_code=429, detail='A benchmark is already running on this instance')
        try:
            async with request.app.state.benchmark_lock:
                return await run_benchmark(request.app.state.engine, body, principal)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GatewayError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.message) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail='Benchmark exceeded its 120-second deadline') from exc

    return app


app = create_app()
