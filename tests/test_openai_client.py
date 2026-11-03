import httpx
import pytest
from openai import AsyncOpenAI
from app.main import create_app
from test_gateway import settings


@pytest.mark.asyncio
async def test_openai_python_json_and_stream_contract():
    app = create_app(settings(gateway_key='client-key'))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as http:
            sdk = AsyncOpenAI(api_key='client-key', base_url='http://test/v1', http_client=http)
            args = {'model': 'auto', 'messages': [{'role': 'user', 'content': 'sdk contract'}]}
            result = await sdk.chat.completions.create(**args)
            assert result.object == 'chat.completion'
            assert result.choices[0].message.content.startswith('[MOCK')
            stream = await sdk.chat.completions.create(**args, stream=True)
            chunks = [chunk async for chunk in stream]
            assert chunks[0].choices[0].delta.role == 'assistant'
            assert chunks[-1].choices[0].finish_reason == 'stop'
            assert ''.join(c.choices[0].delta.content or '' for c in chunks).startswith('[MOCK')
