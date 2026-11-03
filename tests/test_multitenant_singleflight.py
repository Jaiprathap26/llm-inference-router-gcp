import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache import Cache
from app.main import create_app
from app.router import ChatRequest, Engine, Message, Principal
from app.telemetry import Metrics
from test_gateway import settings, payload, event, upstream_sse


def test_verified_tenant_cache_and_metrics_isolation():
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, text=upstream_sse())
    cfg = settings(upstream_key='provider', tenant_api_keys=(('alpha', 'alpha-key'), ('beta', 'beta-key')),
                   enable_upstream_cache=True)
    with TestClient(create_app(cfg, httpx.MockTransport(handler))) as client:
        def post(key, spoof):
            return client.post('/v1/chat/completions', json=payload(),
                               headers={'Authorization': 'Bearer ' + key, 'X-Tenant-ID': spoof})
        assert client.post('/v1/chat/completions', json=payload()).status_code == 401
        assert not post('alpha-key', 'beta').json()['cache_hit']
        assert not post('beta-key', 'alpha').json()['cache_hit']
        assert post('alpha-key', 'beta').json()['cache_hit']
        assert len(calls) == 2
        a = client.get('/v1/metrics', headers={'Authorization': 'Bearer alpha-key'}).json()
        b = client.get('/v1/metrics', headers={'Authorization': 'Bearer beta-key'}).json()
        assert a['sample_count'] == 2 and a['cache_hits'] == 1 and a['cache_hit_rate'] == .5
        assert b['sample_count'] == 1 and b['cache_hits'] == 0


class PausedStream(httpx.AsyncByteStream):
    def __init__(self, started, release, closed):
        self.started, self.release, self.closed = started, release, closed
    async def __aiter__(self):
        yield event({'choices': [{'delta': {'content': 'first '}}]}).encode()
        self.started.set()
        await self.release.wait()
        yield event({'choices': [{'delta': {'content': 'second'}, 'finish_reason': 'stop'}]}).encode()
        yield event({'choices': [], 'usage': {'prompt_tokens': 4, 'completion_tokens': 2}}).encode()
        yield b'data: [DONE]\n\n'
    async def aclose(self):
        self.closed.append(True)


def configured():
    return settings(upstream_key='provider', gateway_key='gateway', enable_upstream_cache=True,
                    fast_input_price=1, fast_output_price=2, deep_input_price=5, deep_output_price=10)


async def wait_members(engine, n):
    for _ in range(500):
        if engine.flights and len(next(iter(engine.flights.values())).members) == n:
            return
        await asyncio.sleep(.001)
    raise AssertionError('subscribers did not join')


@pytest.mark.asyncio
async def test_asgi_concurrent_miss_coalesces_and_cost_once():
    started, release, closed, calls = asyncio.Event(), asyncio.Event(), [], []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, stream=PausedStream(started, release, closed))
    app = create_app(configured(), httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            h = {'Authorization': 'Bearer gateway'}
            one = asyncio.create_task(client.post('/v1/chat/completions', json=payload(), headers=h))
            await asyncio.wait_for(started.wait(), 2)
            two = asyncio.create_task(client.post('/v1/chat/completions', json=payload(), headers=h))
            await wait_members(app.state.engine, 2)
            release.set()
            a, b = await asyncio.gather(one, two)
            x, y = a.json(), b.json()
            assert len(calls) == 1 and closed
            assert x['id'] != y['id'] and x['choices'][0]['message']['content'] == y['choices'][0]['message']['content'] == 'first second'
            assert sorted([x['router']['coalesced'], y['router']['coalesced']]) == [False, True]
            assert not x['cache_hit'] and not y['cache_hit']
            assert sum(r['router']['estimated_cost_usd'] for r in (x, y)) == pytest.approx(8 / 1_000_000)
            assert sum(r['router']['shared_cost_attributed'] for r in (x, y)) == 1
            assert (await client.post('/v1/chat/completions', json=payload(), headers=h)).json()['cache_hit']


@pytest.mark.asyncio
async def test_owner_cancellation_keeps_follower_and_attributes_cost():
    started, release, closed, calls = asyncio.Event(), asyncio.Event(), [], []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, stream=PausedStream(started, release, closed))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = Engine(configured(), Cache(5, 10), client, Metrics())
        req = ChatRequest(stream=True, messages=[Message(role='user', content='hello')])
        a, b = engine.generate(req), engine.generate(req)
        assert (await anext(a))['text'] == 'first '
        await started.wait()
        assert (await anext(b))['text'] == 'first '
        await a.aclose()
        assert not closed
        release.set()
        events = [item async for item in b]
        assert events[-1]['metrics']['coalesced'] and events[-1]['metrics']['shared_cost_attributed']
        assert events[-1]['metrics']['estimated_cost_usd'] == pytest.approx(8 / 1_000_000)
        assert len(calls) == 1 and closed and not engine.flights


@pytest.mark.asyncio
async def test_last_subscriber_cancellation_closes_upstream():
    started, release, closed = asyncio.Event(), asyncio.Event(), []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=PausedStream(started, release, closed)))) as client:
        engine = Engine(configured(), Cache(5, 10), client, Metrics())
        req = ChatRequest(stream=True, messages=[Message(role='user', content='hello')])
        a, b = engine.generate(req), engine.generate(req)
        await anext(a); await started.wait(); await anext(b)
        await a.aclose(); assert not closed
        await b.aclose(); assert closed and not engine.flights


@pytest.mark.asyncio
async def test_distinct_tenants_do_not_join_same_flight():
    started, release, closed, calls = asyncio.Event(), asyncio.Event(), [], []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, stream=PausedStream(started, release, closed))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = Engine(configured(), Cache(5, 10), client, Metrics())
        req = ChatRequest(stream=True, messages=[Message(role='user', content='same')])
        a = engine.generate(req, principal=Principal('alpha', True, 'a'))
        b = engine.generate(req, principal=Principal('beta', True, 'b'))
        await anext(a); await anext(b)
        assert len(calls) == 2 and len(engine.flights) == 2
        release.set()
        first = [e async for e in a]; second = [e async for e in b]
        assert not first[-1]['metrics']['coalesced'] and not second[-1]['metrics']['coalesced']
        assert len(closed) == 2


@pytest.mark.asyncio
async def test_asgi_stream_disconnect_cancels_upstream():
    started, release, closed = asyncio.Event(), asyncio.Event(), []
    app = create_app(configured(), httpx.MockTransport(lambda req: httpx.Response(200, stream=PausedStream(started, release, closed))))
    body = json.dumps(payload(stream=True)).encode()
    disconnected = asyncio.Event()
    received = False
    async def receive():
        nonlocal received
        if not received:
            received = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await disconnected.wait()
        return {'type': 'http.disconnect'}
    async def send(message):
        if message['type'] == 'http.response.body' and b'first ' in message.get('body', b''):
            disconnected.set()
    scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'http_version': '1.1',
             'method': 'POST', 'scheme': 'http', 'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
             'query_string': b'', 'headers': [(b'authorization', b'Bearer gateway'), (b'content-type', b'application/json')],
             'client': ('127.0.0.1', 111), 'server': ('test', 80), 'root_path': ''}
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), 3)
        for _ in range(100):
            if closed: break
            await asyncio.sleep(.01)
        assert closed and not app.state.engine.flights


def test_benchmark_default_seed_and_cache_hit_rate():
    with TestClient(create_app(settings())) as client:
        report = client.post('/v1/benchmark', json={}).json()
        assert report['seed'] == 42 and report['baseline']['requests'] == 30
        assert report['repeat_count'] == 10 and report['repeat_fraction'] == pytest.approx(1/3)
        assert report['baseline']['cache_hit_rate'] == 0
        assert report['optimized']['cache_hit_rate'] == pytest.approx(1/3)
        assert report['optimized']['coalesced_rate'] == 0
