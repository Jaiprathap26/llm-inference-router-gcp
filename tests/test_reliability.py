import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache import Cache
from app.main import create_app
from app.limits import LocalRateLimiter
from app.router import ChatRequest, Engine, GatewayError, Message
from app.telemetry import Metrics, Timer
from test_gateway import settings, payload, event
from test_multitenant_singleflight import configured


def test_timer_formulas_and_unknown_token_boundaries():
    times = iter([1_000_000_000, 1_015_000_000, 1_035_000_000, 1_050_000_000])
    timer = Timer(clock=lambda: next(times))
    timer.token(); timer.token()
    result = timer.finish(4)
    assert result['ttft_ms'] == 15 and result['last_content_ms'] == 35
    assert result['total_latency_ms'] == 50 and result['tokens_per_second'] == 80
    assert result['tpot_ms'] is None


def test_total_timeout_after_first_stream_chunk_no_success_sentinel():
    calls = []
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({'choices': [{'delta': {'content': 'partial'}}]}).encode()
            await asyncio.sleep(5)
    def handler(req):
        calls.append(1)
        return httpx.Response(200, stream=Slow())
    cfg = settings(upstream_key='key', gateway_key='g', enable_upstream_cache=True, upstream_timeout=.05)
    with TestClient(create_app(cfg, httpx.MockTransport(handler))) as client:
        h = {'Authorization': 'Bearer g'}
        for _ in range(2):
            response = client.post('/v1/chat/completions', json=payload(stream=True), headers=h)
            assert response.status_code == 200 and 'partial' in response.text
            assert 'timed out' in response.text and 'data: [DONE]\n\n' not in response.text
        assert len(calls) == 2
        assert client.get('/v1/metrics', headers=h).json()['errors'] == 2


@pytest.mark.asyncio
async def test_failed_shared_stream_does_not_poison_key():
    release = asyncio.Event()
    started = asyncio.Event()
    calls = []
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({'choices': [{'delta': {'content': 'partial'}}]}).encode()
            started.set()
            await release.wait()
            yield b'data: [invalid json]\n\n'
    def handler(req):
        calls.append(1)
        return httpx.Response(200, stream=Broken())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = Cache(10, 10)
        engine = Engine(configured(), cache, client, Metrics())
        req = ChatRequest(stream=True, messages=[Message(role='user', content='x')])
        a, b = engine.generate(req), engine.generate(req)
        await anext(a); await started.wait(); await anext(b)
        release.set()
        async def consume(gen):
            with pytest.raises(GatewayError):
                async for _ in gen: pass
        await asyncio.gather(consume(a), consume(b))
        assert not engine.flights and len(calls) == 1
        c = engine.generate(req)
        await anext(c)
        await c.aclose()
        assert len(calls) == 2


def test_local_rate_limit_and_per_tenant_isolation():
    now = [100.0]
    limiter = LocalRateLimiter(clock=lambda: now[0])
    assert limiter.allow('a', 'chat', 1)
    assert not limiter.allow('a', 'chat', 1)
    assert limiter.allow('b', 'chat', 1)
    now[0] += 60
    assert limiter.allow('a', 'chat', 1)
    cfg = settings(tenant_api_keys=(('a', 'a-key'), ('b', 'b-key')), chat_rate_per_minute=1)
    with TestClient(create_app(cfg)) as client:
        h = {'Authorization': 'Bearer a-key'}
        assert client.post('/v1/chat/completions', json=payload(), headers=h).status_code == 200
        assert client.post('/v1/chat/completions', json=payload(), headers=h).status_code == 429
        assert client.post('/v1/chat/completions', json=payload(), headers={'Authorization': 'Bearer b-key'}).status_code == 200


def test_ambiguous_auth_and_hmac_config_fail_closed():
    with pytest.raises(ValueError):
        create_app(settings(gateway_key='g', tenant_api_keys=(('alpha', 'a'),)))
    with pytest.raises(ValueError):
        create_app(settings(gateway_key='g', upstream_key='u', enable_upstream_cache=True, cache_hmac_secret=''))
    with pytest.raises(ValueError):
        create_app(settings(upstream_key='u', upstream_url='http://example.com/v1'))
