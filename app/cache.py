import asyncio
from collections import OrderedDict
import json
import time
import logging

log = logging.getLogger(__name__)


class Cache:
    """Per-process LRU with optional shared Redis, both TTL bounded. Cache errors fail open."""
    def __init__(self, size: int, ttl: int, redis=None):
        self.size, self.ttl, self.redis = size, ttl, redis
        self.data = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key):
        if not self.size or not self.ttl:
            return None
        async with self.lock:
            item = self.data.get(key)
            if item:
                expiry, value = item
                if expiry > time.monotonic():
                    self.data.move_to_end(key)
                    return json.loads(value)
                del self.data[key]
        if self.redis:
            try:
                value = await self.redis.get('llmr:v1:' + key)
                if value:
                    return json.loads(value)
            except Exception:
                log.warning('Redis cache read failed')
        return None

    async def set(self, key, value):
        if not self.size or not self.ttl:
            return
        encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        async with self.lock:
            self.data[key] = (time.monotonic() + self.ttl, encoded)
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)
        if self.redis:
            try:
                await self.redis.set('llmr:v1:' + key, encoded, ex=self.ttl)
            except Exception:
                log.warning('Redis cache write failed')
