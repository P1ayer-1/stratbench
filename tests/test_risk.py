"""The veto: every gate, every reason, and the short path out."""

import ast
from decimal import Decimal
from pathlib import Path

from predkit.risk import Limits, RiskEngine

D = Decimal
RESOLVES = 1_000_000_000
NOW = RESOLVES - 300_000          # five minutes out


def engine(**kw):
    return RiskEngine(Limits(**kw))


def order(e, **kw):
    args = dict(market_id="M", side="buy", outcome="yes", price=D("0.48"), size=D("10"),
                resolves_at_ms=RESOLVES, now_ms=NOW)
    args.update(kw)
    return e.check_order(**args)


def test_risk_imports_nothing_from_the_package():
    """The veto cannot depend on what it vetoes."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "predkit" / "risk.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("predkit") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("predkit"), node.module


def test_a_clean_order_is_allowed():
    assert order(engine()).allowed


def test_every_failing_gate_is_reported_not_just_the_first():
    e = engine(max_contracts_per_market=D("5"), max_open_notional=D("1"), max_daily_loss=D("10"))
    e.trip("test")
    e.realized_pnl_today = D("-10")
    veto = order(e, now_ms=RESOLVES - 30_000)         # inside the 60 s window too
    assert not veto
    text = "\n".join(veto.reasons)
    for expected in ("kill switch", "resolves in 30s", "daily loss", "contracts", "open notional"):
        assert expected in text, expected
    assert len(veto.reasons) == 5


def test_no_new_risk_inside_the_resolution_window_or_after():
    e = engine(resolution_window_s=60)
    assert order(e, now_ms=RESOLVES - 61_000).allowed
    assert not order(e, now_ms=RESOLVES - 60_000).allowed
    assert not order(e, now_ms=RESOLVES + 1).allowed


def test_position_limit_counts_yes_equivalents_across_outcomes():
    e = engine(max_contracts_per_market=D("10"))
    e.on_fill("M", "buy", "yes", D("0.5"), D("8"))
    assert not order(e, size=D("3")).allowed                     # 8 + 3 > 10
    assert order(e, outcome="no", size=D("3")).allowed           # 8 - 3 = 5: reduces
    assert order(e, outcome="no", size=D("18")).allowed          # 8 - 18 = -10: at the limit


def test_open_notional_is_dollars_paid():
    e = engine(max_open_notional=D("10"))
    e.on_fill("A", "buy", "yes", D("0.5"), D("10"))             # $5 at risk
    assert order(e, market_id="B", price=D("0.5"), size=D("10")).allowed        # 5 + 5 = 10
    assert not order(e, market_id="B", price=D("0.6"), size=D("10")).allowed    # 5 + 6 > 10


def test_reduce_only_bypasses_the_kill_switch_but_is_verified():
    """The gates stop you OPENING risk; none is a reason to be unable to exit."""
    e = engine()
    e.on_fill("M", "buy", "yes", D("0.5"), D("10"))
    e.trip("something")
    assert order(e, side="sell", size=D("10"), reduce_only=True).allowed
    assert order(e, side="buy", outcome="no", size=D("10"), reduce_only=True).allowed
    flip = order(e, side="sell", size=D("11"), reduce_only=True)
    assert not flip and "flip" in flip.reasons[0]
    lie = order(e, side="buy", size=D("1"), reduce_only=True)
    assert not lie and "does not reduce" in lie.reasons[0]


def test_realised_pnl_is_hand_computed():
    e = engine()
    e.on_fill("M", "buy", "yes", D("0.40"), D("10"))            # paid 4.00
    e.on_fill("M", "sell", "yes", D("0.55"), D("10"), fee=D("0.10"))
    # proceeds 5.50 - cost 4.00 - fee 0.10 = +1.40
    assert e.realized_pnl_today == D("1.40")
    assert e.contracts["M"] == D("0") and e.at_risk["M"] == D("0")


def test_resolution_pays_a_dollar_per_winning_contract():
    e = engine()
    e.on_fill("M", "buy", "no", D("0.30"), D("10"))             # paid 3.00 for NO
    e.on_resolution("M", resolved_yes=False)                    # NO wins: +10.00 - 3.00
    assert e.realized_pnl_today == D("7.00")
    e2 = engine()
    e2.on_fill("M", "buy", "no", D("0.30"), D("10"))
    e2.on_resolution("M", resolved_yes=True)
    assert e2.realized_pnl_today == D("-3.00")


def test_daily_loss_stop_blocks_new_risk_after_the_loss():
    e = engine(max_daily_loss=D("5"))
    e.on_fill("M", "buy", "yes", D("0.60"), D("10"))
    e.on_resolution("M", resolved_yes=False)                    # -6.00
    veto = order(e, market_id="N")
    assert not veto and "daily loss" in veto.reasons[0]
    e.start_new_day("2026-09-13")
    assert order(e, market_id="N").allowed
