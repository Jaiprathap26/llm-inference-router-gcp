import json
import httpx
import pytest
from fastapi.testclient import TestClient
from app.cache import Cache
from app.main import create_app
from test_gateway import settings, payload, event


def test_fragmented_multiline_utf8_sse():
    data = ': heartbeat\n\ndata: {"choices": [\ndata: {"delta": {"content": "café"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    class Fragmented(httpx.AsyncByteStream):
        async def __aiter__(self):
            for byte in data.encode('utf-8'):
                yield bytes([byte])
    transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=Fragmented()))
    with TestClient(create_app(settings(upstream_key='key'), transport)) as client:
        response = client.post('/v1/chat/completions', json=payload())
        assert response.status_code == 200
        assert response.json()['choices'][0]['message']['content'] == 'café'
        assert response.json()['router']['usage_estimated'] is True
        assert response.json()['router']['estimated_cost_usd'] is None


def test_timeout_before_headers_and_invalid_frame():
    def timeout(req):
        raise httpx.ReadTimeout('private')
    with TestClient(create_app(settings(upstream_key='key'), httpx.MockTransport(timeout))) as client:
        response = client.post('/v1/chat/completions', json=payload(stream=True))
        assert response.status_code == 504 and 'private' not in response.text
    with TestClient(create_app(settings(upstream_key='key'), httpx.MockTransport(lambda req: httpx.Response(200, text='data: nope\n\n')))) as client:
        response = client.post('/v1/chat/completions', json=payload(stream=True))
        assert response.status_code == 502


@pytest.mark.asyncio
async def test_redis_failure_fails_open(caplog):
    class Broken:
        async def get(self, key): raise RuntimeError('password should not appear')
        async def set(self, *args, **kwargs): raise RuntimeError('password should not appear')
    cache = Cache(2, 10, Broken())
    assert await cache.get('missing') is None
    await cache.set('key', {'content': 'ok'})
    assert await cache.get('key') == {'content': 'ok'}
    assert 'password should not appear' not in caplog.text
