import asyncio
import random
import time
from pydantic import BaseModel, ConfigDict, Field
from app.cache import Cache
from app.router import ChatRequest, Message
from app.telemetry import percentile


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    requests: int = Field(default=30, ge=4, le=50)
    concurrency: int = Field(default=5, ge=1, le=10)
    allow_paid_upstream: bool = False


async def run_benchmark(engine, spec, principal=None):
    if not engine.settings.mock and (not engine.settings.enable_paid_benchmark or not spec.allow_paid_upstream):
        raise ValueError('Live benchmarks require operator ENABLE_PAID_BENCHMARK=true and allow_paid_upstream=true')
    # Unique requests complete before their repeats; repeat portion is measured, not prewarmed.
    repeated = round(spec.requests / 3)
    unique = spec.requests - repeated
    prompts = [f'Question {i}: explain a simple everyday concept.' if i % 2 == 0 else
               f'Question {i}: analyze this code and derive a step-by-step algorithm.' for i in range(unique)]
    requests = [ChatRequest(stream=True, messages=[Message(role='user', content=p)]) for p in prompts]
    repeated_requests = [requests[i] for i in random.Random(42).sample(range(unique), repeated)]

    async def phase(baseline):
        cache = Cache(max(1, unique), max(1, engine.settings.cache_ttl))
        sem = asyncio.Semaphore(spec.concurrency)
        async def one(req):
            async with sem:
                last = None
                async for event in engine.generate(req, baseline=baseline, cache=cache, record=False, principal=principal):
                    if event['type'] == 'final':
                        last = event['metrics']
                if last is None:
                    raise RuntimeError('Incomplete benchmark request')
                return last
        async def batch(work):
            tasks = [asyncio.create_task(one(req)) for req in work]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        start = time.perf_counter_ns()
        rows = await batch(requests)
        rows += await batch(repeated_requests)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        ttft = [x['ttft_ms'] for x in rows if x['ttft_ms'] is not None]
        return {'requests': len(rows), 'p50_ttft_ms': percentile(ttft, .5), 'p95_ttft_ms': percentile(ttft, .95),
                'tokens_per_second': sum(r['output_tokens'] for r in rows) / elapsed if elapsed else None,
                'cache_hit_rate': sum(r['cache_hit'] for r in rows) / len(rows),
                'coalesced_rate': sum(r.get('coalesced', False) for r in rows) / len(rows),
                'missing_ttft_samples': len(rows) - len(ttft), 'failed_requests': 0,
                'estimated_cost_usd': sum(r['estimated_cost_usd'] for r in rows) if all(r['estimated_cost_usd'] is not None for r in rows) else None,
                'modeled_cost_usd': sum(r['modeled_cost_usd'] for r in rows) if all(r['modeled_cost_usd'] is not None for r in rows) else None, 'elapsed_seconds': elapsed}
    async with asyncio.timeout(120):
        baseline = await phase(True)
        optimized = await phase(False)
    reduction = (1 - optimized['modeled_cost_usd'] / baseline['modeled_cost_usd']) * 100 if baseline['modeled_cost_usd'] and optimized['modeled_cost_usd'] is not None else None
    def fmt(v):
        return 'N/A' if v is None else f'{v:.3f}'
    markdown = '| Scenario | p50 TTFT ms | p95 TTFT ms | tokens/sec | hit rate | modeled cost USD |\n|---|---:|---:|---:|---:|---:|\n'
    for name, item in [('Baseline', baseline), ('Optimized', optimized)]:
        markdown += f"| {name} | {fmt(item['p50_ttft_ms'])} | {fmt(item['p95_ttft_ms'])} | {fmt(item['tokens_per_second'])} | {item['cache_hit_rate']:.1%} | {fmt(item['modeled_cost_usd'])} |\n"
    markdown += f"\nModeled cost reduction: {fmt(reduction)}%" if reduction is not None else '\nModeled cost reduction: N/A (baseline price is zero or unknown).'
    return {'mode': 'mock' if engine.settings.mock else 'upstream', 'seed': 42, 'repeat_count': repeated,
            'repeat_fraction': repeated / spec.requests, 'baseline_model': engine.settings.deep_model,
            'concurrency': spec.concurrency, 'baseline': baseline, 'optimized': optimized,
            'modeled_cost_reduction_percent': reduction, 'markdown': markdown,
            'note': 'Synthetic requests; isolated cold caches; repeats run after unique requests. Mock timings and costs are not provider benchmarks.'}
