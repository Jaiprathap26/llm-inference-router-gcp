# LLM inference router for Cloud Run

An OpenAI-style chat gateway with verified tenant-scoped exact prompt replay and bounded per-key single-flight, deterministic complexity routing, server-observed metrics and an isolated synthetic benchmark. With no `UPSTREAM_API_KEY` it **does not call an LLM**: it returns explicitly labeled deterministic mock text. The mock incurs no upstream token charges; Cloud Run, network and infrastructure may still cost money.

**Security and readiness:** the default unauthenticated mock is for demonstrations only. For production configure distinct high-entropy tenant bearer keys via `TENANT_API_KEYS`, private ingress/IAM, rate limiting, quotas and spend limits. A single `GATEWAY_API_KEY` instead represents one shared trust domain. Real-provider caching requires authenticated identity, `ENABLE_UPSTREAM_CACHE=true`, and a dedicated `CACHE_HMAC_SECRET`; otherwise it is disabled. Tenant identity comes only from verified configured keys, not caller-supplied headers. This is static bearer authentication with one shared upstream account; it is not a complete identity or authorization platform. Distributed rate limiting, hard budget enforcement, cache deletion, live Vertex ADC, and semantic similarity search are not implemented. `GET /healthz` is intentionally unauthenticated. Public benchmark/metrics access is unsafe without protection.

## Quickstart

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

```sh
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
curl -s localhost:8080/v1/benchmark -H 'Content-Type: application/json' \
  -d '{"requests":30,"concurrency":5}'
curl -s localhost:8080/v1/metrics
```

If `GATEWAY_API_KEY` or `TENANT_API_KEYS` is set, add `Authorization: Bearer <assigned key>` to all `/v1` requests. Never put secrets in source control or logs. Authentication happens before cache lookup. The gateway key is distinct from the upstream key. Cloud Run IAM and application bearer auth may require a proxy or a separate `X-Serverless-Authorization` identity token because both otherwise use Authorization.

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `GATEWAY_API_KEY` | empty | One bearer credential for one shared trust domain; cannot be combined with `TENANT_API_KEYS`. |
| `TENANT_API_KEYS` | `{}` | JSON object mapping stable verified tenant IDs to distinct non-empty bearer keys, e.g. `{"tenant-a":"...","tenant-b":"..."}`; supply as a managed secret, not a command-line literal. Empty together with `GATEWAY_API_KEY` disables API auth. |
| `CACHE_HMAC_SECRET` | empty | Dedicated high-entropy secret used for cache-key HMAC; required for real-provider caching with authentication. Rotate to invalidate keys. |
| `UPSTREAM_API_KEY` | empty | If empty: offline mock; otherwise real upstream, no fallback on error. |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Operator-controlled compatible endpoint; `/chat/completions` is appended. |
| `FAST_MODEL`, `DEEP_MODEL` | `gemini-2.5-flash`, `gpt-4.1` | Both must exist at the SAME selected upstream. Defaults are labels in mock mode, not proof that the default OpenAI endpoint serves Gemini. Set models appropriate to your provider. |
| `ENABLE_UPSTREAM_CACHE` | false | Set `true` to opt into replay for real upstream; requires verified tenant or gateway key and `CACHE_HMAC_SECRET`. Sampling results are replayed, not independently generated. |
| `CACHE_TTL_SECONDS`, `CACHE_MAX_ENTRIES` | 300, 512 | Local LRU limits; zero disables cache. Values over 256 KiB are not cached. |
| `REDIS_URL` | empty | Optional shared Redis, TTL applied. Use `rediss://`, authentication, network isolation, retention controls and separate Redis per trust domain. Failures fall back to local/provider, never bypass auth. Local entries are not invalidated across instances. |
| `UPSTREAM_TIMEOUT_SECONDS` | 60 | httpx operation timeout and overall upstream streaming deadline in seconds, excluding initial authentication/cache lookup. |
| `CHAT_RATE_PER_MINUTE`, `BENCHMARK_RATE_PER_MINUTE` | 120, 2 | Bounded per-tenant, per-instance sliding window; 0 disables. At most one benchmark runs per instance. This is not a distributed rate limit. |
| `FAST_INPUT_USD_PER_M`, `FAST_OUTPUT_USD_PER_M`, `DEEP_INPUT_USD_PER_M`, `DEEP_OUTPUT_USD_PER_M` | unset | Operator-supplied USD per million tokens; unknown prices remain unknown, not zero. Maintain independently when provider rates change. |
| `ENABLE_PAID_BENCHMARK` | false | Operator opt-in for live benchmarks; request must also include `allow_paid_upstream: true`. No application-level spend cap is implemented. |
| `PORT` | 8080 | Container listening port. |

A real upstream URL must use HTTPS and cannot contain embedded credentials, a query or a fragment. Real upstream requires OpenAI-style streaming with `stream_options.include_usage` and `[DONE]`. Some providers do not support this exact subset. There is **no native Vertex ADC, token refresh or Google service-account integration**. A Vertex-compatible proxy may be used only if it supplies the compatible streaming protocol and bearer authentication, and supports both configured models; verify compatibility and credentials. Do not point this at an endpoint you do not trust with all prompts and the bearer key.

## API and behavior

`POST /v1/chat/completions` supports `model: "auto"` (default), `messages` with string content and roles system/developer/user/assistant, `stream`, `temperature` (0–2), and `max_tokens` (1–2048). Up to 64 messages and 128,000 combined characters. Unsupported fields and arbitrary explicit model names are rejected rather than silently ignored. No tools, multimodal, n>1, JSON schema, logprobs, stream options, or true semantic similarity cache. Successful responses add `cache_hit` and `router`. JSON includes usage, estimated when the provider does not return it. SSE emits a role chunk, incremental content, a terminal finish chunk containing `router`, and `[DONE]`. A hit replays the complete cached content in one chunk, not original token timing. After response headers, errors are emitted as an SSE `error` object and the stream closes **without** `[DONE]`; clients must treat this as failure. Before headers, safe 502/504 error envelopes are returned. Client cancellation closes the upstream stream.

Canonical keys preserve exact content, case, Unicode, whitespace, order and all accepted parameters except `stream`; they include verified tenant, authorization scope, policy version, resolved model, endpoint, mock mode and credential identity and use HMAC. Never use anonymous mock for sensitive data. The cache retains sensitive response text in memory/Redis even though keys are opaque. It caches complete `stop` responses only, with fresh response IDs on replay; partial, length-limited, filtered or failed outputs are not cached. Concurrent same-tenant, same-key cold misses in one process join a bounded flight; followers have `coalesced: true`, not cache hits. Upstream runs while any subscriber remains and is canceled after the last disconnect. The first subscriber to receive the completed result is attributed the upstream cost; partial canceled provider calls may still be billed but have unknown usage. Limits are 256 flights, 128 subscribers per flight and 16,384 upstream content events. Coalescing is not distributed across instances; duplicate external calls remain possible. There is no purge endpoint; use TTL, restart local instances and delete the isolated Redis namespace according to your retention policy.

Routing estimates prompt length as `ceil(characters/4)` plus 100 points per code/reasoning marker; score >=180 uses the deep model. This is a heuristic, not a tokenizer or quality guarantee. A cheaper model may produce different quality. Inputs are not normalized beyond JSON serialization, deliberately preserving whitespace-sensitive prompts. **<15 ms cache-hit latency is a target to measure on your deployment, not a guarantee.**

## Telemetry and benchmark interpretation

`GET /v1/metrics` is a tenant-filtered, per-process bounded rolling snapshot (10,000 most recent engine invocations across all tenants, last 100 matching observations returned); heavy activity by one tenant can evict another tenant's older samples. It excludes rejected/unauthenticated requests and benchmark runs; aggregate across instances externally. It contains no raw prompt text, but protect it nonetheless. Durations use `perf_counter_ns` from engine entry to first content and engine completion, not full HTTP request/client latency; cached replay is not provider throughput. `ttft_ms` is populated for streaming; JSON exposes response-ready `first_content_available_ms` instead and separately records `first_content_observed_ms`. `tpot_ms` is null: aggregate token counts and SSE chunks do not establish individual token boundaries. `tokens_per_second` is output tokens divided by total engine elapsed time, includes cache lookup and replay; tokens are rough character estimates when `usage_estimated=true`. Empty content has null TTFT. Timings exclude emission of the final response and network transit.

On cache hit, incremental upstream token cost is $0, not infrastructure cost. Mock actual upstream cost is $0 in both arms. Live `estimated_cost_usd` requires provider usage and prices or is null; it is not a bill. `modeled_cost_usd` uses supplied prices even for estimated tokens, and `estimated_cost_saved_usd` is a **hypothetical** deep-tier counterfactual minus modeled cost; it may be negative or null and is not realized savings. Benchmark modeled reduction is null when prices are missing or the baseline is zero. Never report mock hypothetical reduction as real dollars saved.

Benchmark defaults to 30 fixed synthetic requests with seed 42: 20 unique followed by 10 exact repeats (33.3%). It permits 4–50 requests, concurrency 1–10 and has a 120-second overall deadline. Both arms use identical requests, same concurrency, first all unique requests then repeats, isolated cold caches, and streaming-engine timing. Baseline forces the deep model and bypasses cache; optimized routes and caches subject to normal policy. p50/p95 use the rounded nearest index in the sorted sample; throughput is aggregate output tokens divided by scenario wall time. Failures abort the benchmark and cancel remaining tasks; no partial statistics are returned. Coalescing and completed-cache hits are reported separately. There is no quality comparison or external client/multi-instance measurement. Small samples and synthetic/mock timings do not establish production performance. For paid upstream, configure provider-side budgets and quotas before enabling it.

## Container and Cloud Run

```sh
docker build -t llm-router .
docker run --rm -p 8080:8080 -e PORT=8080 llm-router
# From an authorized project/region, after configuring access and secrets:
gcloud run deploy llm-router --source . --region REGION --no-allow-unauthenticated
```

Deploy only with deliberate IAM, ingress, service account, Secret Manager, request timeout, concurrency, max instances, memory and distributed rate/spend limits. Enforce a raw HTTP request-body limit at the ingress: message limits here apply after JSON parsing. Dependencies are top-level pinned, not a hash-locked transitive supply chain; scan and lock them for your environment. Do not paste keys into command history; mount/inject managed secrets. The runtime runs as non-root, listens on `0.0.0.0:$PORT`, and uses one uvicorn process. Cache/metrics are per-instance and vanish on restart; Redis stores shared cached responses if configured. Cloud Run can scale to multiple instances, and infrastructure usage is billable. Validate container and staging deployment, streaming cancellation, provider compatibility, spend and latency in your environment before production.

References: Google Cloud Run container contract https://cloud.google.com/run/docs/container-contract ; OpenAI Python streaming chunk schema https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_chunk.py .
