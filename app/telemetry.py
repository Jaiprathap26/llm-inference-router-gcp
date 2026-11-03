from collections import deque
from threading import Lock
import time


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


class Timer:
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter_ns
        self.start = self.clock()
        self.first = None
        self.last = None

    def token(self):
        self.last = self.clock()
        if self.first is None:
            self.first = self.last

    def finish(self, output_tokens):
        end = self.clock()
        total = (end - self.start) / 1e6
        ttft = (self.first - self.start) / 1e6 if self.first is not None else None
        return {'ttft_ms': ttft, 'total_latency_ms': total,
                'last_content_ms': (self.last - self.start) / 1e6 if self.last is not None else None,
                # SSE chunks are not token boundaries, even with an aggregate usage count.
                'tpot_ms': None,
                'tokens_per_second': output_tokens / (total / 1000) if total else None}


class Metrics:
    def __init__(self, limit=10000):
        self.lock = Lock()
        self.rows = deque(maxlen=limit)

    def add(self, row, tenant='anonymous-mock'):
        with self.lock:
            self.rows.append((tenant, dict(row)))

    def snapshot(self, tenant=None):
        with self.lock:
            rows = [row for scope, row in self.rows if tenant is None or scope == tenant]
        ttft = [r['ttft_ms'] for r in rows if r.get('ttft_ms') is not None]
        costs = [r['estimated_cost_usd'] for r in rows if r.get('estimated_cost_usd') is not None]
        savings = [r['estimated_cost_saved_usd'] for r in rows if r.get('estimated_cost_saved_usd') is not None]
        tpots = [r['tpot_ms'] for r in rows if r.get('tpot_ms') is not None]
        throughputs = [r['tokens_per_second'] for r in rows if r.get('tokens_per_second') is not None]
        return {'scope': 'this process; bounded rolling window', 'sample_count': len(rows),
                'cache_hits': sum(bool(r.get('cache_hit')) for r in rows),
                'coalesced_requests': sum(bool(r.get('coalesced')) for r in rows),
                'cache_hit_rate': sum(bool(r.get('cache_hit')) for r in rows) / len(rows) if rows else None,
                'errors': sum(bool(r.get('error')) for r in rows),
                'p50_ttft_ms': percentile(ttft, .50), 'p95_ttft_ms': percentile(ttft, .95),
                'p50_tpot_ms': percentile(tpots, .50),
                'mean_tokens_per_second': sum(throughputs) / len(throughputs) if throughputs else None,
                'estimated_cost_usd': sum(costs) if costs else None,
                'estimated_cost_saved_usd': sum(savings) if savings else None,
                'unknown_cost_samples': sum(r.get('estimated_cost_usd') is None for r in rows),
                'recent': rows[-100:]}
