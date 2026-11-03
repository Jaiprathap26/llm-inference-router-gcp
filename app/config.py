from dataclasses import dataclass
import math
import json
import os


def number(name: str, default: str, cast):
    try:
        value = cast(os.getenv(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError()
        return value
    except ValueError as exc:
        raise ValueError(f'{name} must be a non-negative number') from exc


def price(name):
    return number(name, os.environ[name], float) if name in os.environ else None


def tenant_keys():
    try:
        value = json.loads(os.getenv('TENANT_API_KEYS', '{}'))
        if not isinstance(value, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in value.items()):
            raise ValueError('expected non-empty tenant names and keys')
        if len(value) > 1000 or len(set(value.values())) != len(value):
            raise ValueError('tenant keys must be unique and limited to 1000')
        return tuple(sorted(value.items()))
    except (ValueError, TypeError) as exc:
        raise ValueError('TENANT_API_KEYS must be a JSON object of unique tenant names and non-empty bearer keys') from exc


@dataclass(frozen=True)
class Settings:
    upstream_url: str
    upstream_key: str
    gateway_key: str
    fast_model: str
    deep_model: str
    cache_ttl: int
    cache_size: int
    redis_url: str
    upstream_timeout: float
    fast_input_price: float | None
    fast_output_price: float | None
    deep_input_price: float | None
    deep_output_price: float | None
    enable_upstream_cache: bool
    enable_paid_benchmark: bool
    tenant_api_keys: tuple[tuple[str, str], ...]
    cache_hmac_secret: str
    chat_rate_per_minute: int
    benchmark_rate_per_minute: int

    @classmethod
    def from_env(cls):
        return cls(
            upstream_url=os.getenv('UPSTREAM_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
            upstream_key=os.getenv('UPSTREAM_API_KEY', ''),
            gateway_key=os.getenv('GATEWAY_API_KEY', ''),
            fast_model=os.getenv('FAST_MODEL', 'gemini-2.5-flash'),
            deep_model=os.getenv('DEEP_MODEL', 'gpt-4.1'),
            cache_ttl=number('CACHE_TTL_SECONDS', '300', int),
            cache_size=number('CACHE_MAX_ENTRIES', '512', int),
            redis_url=os.getenv('REDIS_URL', ''),
            upstream_timeout=number('UPSTREAM_TIMEOUT_SECONDS', '60', float),
            fast_input_price=price('FAST_INPUT_USD_PER_M'),
            fast_output_price=price('FAST_OUTPUT_USD_PER_M'),
            deep_input_price=price('DEEP_INPUT_USD_PER_M'),
            deep_output_price=price('DEEP_OUTPUT_USD_PER_M'),
            enable_upstream_cache=os.getenv('ENABLE_UPSTREAM_CACHE', '').lower() == 'true',
            enable_paid_benchmark=os.getenv('ENABLE_PAID_BENCHMARK', '').lower() == 'true',
            tenant_api_keys=tenant_keys(),
            cache_hmac_secret=os.getenv('CACHE_HMAC_SECRET', ''),
            chat_rate_per_minute=number('CHAT_RATE_PER_MINUTE', '120', int),
            benchmark_rate_per_minute=number('BENCHMARK_RATE_PER_MINUTE', '2', int),
        )

    @property
    def mock(self):
        return not self.upstream_key
