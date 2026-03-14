from headroom.core.budget import TokenBudget, TokenUsage


def test_headroom_calculation():
    budget = TokenBudget(limit=1000, reserve=100)
    assert budget.usable == 900
    assert budget.headroom(400) == 500
    assert budget.headroom(900) == 0
    assert budget.headroom(950) == 0  # clamped at 0


def test_used_fraction():
    budget = TokenBudget(limit=1000, reserve=0)
    assert budget.used_fraction(0) == 0.0
    assert budget.used_fraction(500) == 0.5
    assert budget.used_fraction(1000) == 1.0
    assert budget.used_fraction(1200) == 1.0  # clamped


def test_status_transitions():
    budget = TokenBudget(limit=1000, warn_at=0.5, act_at=0.7, reserve=0)
    assert budget.status(400) == "ok"
    assert budget.status(500) == "warn"
    assert budget.status(700) == "act"
    assert budget.status(1000) == "overflow"
    assert budget.status(1100) == "overflow"


def test_for_model():
    budget = TokenBudget.for_model("claude-opus-4-6")
    assert budget.limit == 200_000


def test_token_usage_headroom():
    u = TokenUsage(used=50_000, limit=200_000, reserve=1024)
    assert u.headroom == 200_000 - 1024 - 50_000
    assert 0 < u.used_fraction < 1
    assert 0 < u.headroom_pct <= 100
