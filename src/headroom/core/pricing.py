from __future__ import annotations

from dataclasses import dataclass


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


# Prices current as of 2026-03 — update when Anthropic changes rates.
# https://www.anthropic.com/pricing
PRICING: dict[str, ModelPricing] = {
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


def get_pricing(model: str) -> ModelPricing | None:
    """Return pricing for *model*, or None if unknown."""
    # Exact match first, then prefix match for unversioned aliases.
    if model in PRICING:
        return PRICING[model]
    for key, pricing in PRICING.items():
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
