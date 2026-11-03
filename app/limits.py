from collections import OrderedDict, deque
from threading import Lock
import time


class LocalRateLimiter:
    """Bounded per-process sliding window. Use a distributed edge limiter for scale-out."""
    def __init__(self, max_scopes=2048, clock=time.monotonic):
        self.lock = Lock()
        self.scopes = OrderedDict()
        self.max_scopes = max_scopes
        self.clock = clock

    def allow(self, tenant, operation, limit):
        if limit == 0:
            return True
        now = self.clock()
        key = (tenant, operation)
        with self.lock:
            values = self.scopes.setdefault(key, deque())
            self.scopes.move_to_end(key)
            while len(self.scopes) > self.max_scopes:
                self.scopes.popitem(last=False)
            while values and values[0] <= now - 60:
                values.popleft()
            if len(values) >= limit:
                return False
            values.append(now)
            return True
