from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# LiteLLM community-maintained pricing data (updated with each model release)
_LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main"
    "/model_prices_and_context_window.json"
)
_CACHE_PATH = Path.home() / ".config" / "headroom" / "pricing.json"
_CACHE_TTL = 86_400  # 24 hours in seconds


@dataclass(frozen=True)
class ModelPricing:
    """Cost per million tokens for a Claude model."""

    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float

    def cost_for(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Return total cost in USD for the given token counts."""
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
            + cache_write_tokens * self.cache_write_per_mtok / 1_000_000
            + cache_read_tokens * self.cache_read_per_mtok / 1_000_000
        )


# Fallback prices — used when the cache is absent and the network is unavailable.
# Prices current as of 2026-03. https://www.anthropic.com/pricing
_FALLBACK: dict[str, ModelPricing] = {
    "claude-opus-4-6": ModelPricing(15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(0.80, 4.00, 1.00, 0.08),
    "claude-haiku-4-5": ModelPricing(0.80, 4.00, 1.00, 0.08),
    "claude-opus-4-5": ModelPricing(15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 1.00, 0.08),
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 18.75, 1.50),
}

# Module-level live cache, populated from disk or network at startup.
_live: dict[str, ModelPricing] = {}
_live_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    try:
        data = json.loads(_CACHE_PATH.read_text())
        age = time.time() - data.get("fetched_at", 0)
        return age < _CACHE_TTL
    except Exception:
        return False


def _load_cache() -> dict[str, ModelPricing]:
    """Load pricing from the local disk cache. Returns {} on any error."""
    try:
        data = json.loads(_CACHE_PATH.read_text())
        return _parse_litellm(data.get("models", {}))
    except Exception:
        return {}


def _save_cache(models: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(
            json.dumps({"fetched_at": time.time(), "models": models}, indent=2)
        )
    except Exception as exc:
        log.debug("headroom: could not write pricing cache: %s", exc)


# ---------------------------------------------------------------------------
# LiteLLM parsing
# ---------------------------------------------------------------------------

def _parse_litellm(raw: dict) -> dict[str, ModelPricing]:
    """Extract Claude model pricing from LiteLLM's pricing JSON."""
    result: dict[str, ModelPricing] = {}
    for model_id, info in raw.items():
        if not model_id.startswith("claude"):
            continue
        try:
            # LiteLLM stores costs per *token*, we want per *million tokens*.
            def _mtok(key: str) -> float:
                v = info.get(key)
                return float(v) * 1_000_000 if v is not None else 0.0

            result[model_id] = ModelPricing(
                input_per_mtok=_mtok("input_cost_per_token"),
                output_per_mtok=_mtok("output_cost_per_token"),
                cache_write_per_mtok=_mtok("cache_creation_input_token_cost"),
                cache_read_per_mtok=_mtok("cache_read_input_token_cost"),
            )
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Background refresh
# ---------------------------------------------------------------------------

def _fetch_and_cache() -> dict[str, ModelPricing]:
    """Fetch LiteLLM pricing, update disk cache, return parsed pricing."""
    import httpx  # local import — only needed here

    try:
        resp = httpx.get(_LITELLM_PRICING_URL, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        parsed = _parse_litellm(raw)
        if parsed:
            _save_cache(raw)
            log.debug("headroom: pricing refreshed (%d Claude models)", len(parsed))
        return parsed
    except Exception as exc:
        log.debug("headroom: pricing fetch failed: %s", exc)
        return {}


def _refresh_in_background() -> None:
    """Spawn a daemon thread to refresh pricing without blocking the caller."""
    def _worker():
        fresh = _fetch_and_cache()
        if fresh:
            with _live_lock:
                _live.update(fresh)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Startup: load from cache or kick off a background fetch
# ---------------------------------------------------------------------------

def _initialise() -> None:
    """Called once at import time to populate _live from the best available source."""
    if _cache_is_fresh():
        cached = _load_cache()
        if cached:
            with _live_lock:
                _live.update(cached)
            return
    # Cache is absent or stale — load whatever we have and refresh in the background.
    stale = _load_cache()
    if stale:
        with _live_lock:
            _live.update(stale)
    _refresh_in_background()


_initialise()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pricing(model: str) -> ModelPricing | None:
    """Return the best available pricing for *model*.

    Lookup order: live cache (refreshed daily) → fallback table.
    Returns None if the model is not recognised anywhere.
    """
    with _live_lock:
        live_copy = dict(_live)

    for table in (live_copy, _FALLBACK):
        if model in table:
            return table[model]
        for key, pricing in table.items():
            if model.startswith(key) or key.startswith(model):
                return pricing
    return None


def calculate_turn_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return USD cost for a single API turn. Returns 0.0 if model is unknown."""
    pricing = get_pricing(model)
    if pricing is None:
        return 0.0
    return pricing.cost_for(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )
