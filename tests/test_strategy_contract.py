"""The three verbs every strategy has, asserted against every registered strategy.

The Protocols in `strategies/__init__.py` are load-bearing only if something
checks them. These do, structurally: the strategies do not import the
Protocols, do not inherit from them, and their `plan.py` files have no path
to an order endpoint, which is checked with a grep because that is what
makes it checkable at all.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from predkit.backtest import WindowRow
from predkit.strategies import NAMES, Outcome, Plan, Report, Strategy, load
from predkit.strategies.simple_maker import ExecutionResult as MakerOutcome, MonitorReport as MakerReport, QuotePlan
from predkit.strategies.simple_taker import ExecutionResult as TakerOutcome, MonitorReport as TakerReport, TakePlan

STRATEGIES = Path(__file__).resolve().parents[1] / "predkit" / "strategies"


@pytest.mark.parametrize("name", NAMES)
def test_each_strategy_is_a_strategy(name):
    assert isinstance(load(name), Strategy)


@pytest.mark.parametrize("cls", (QuotePlan, TakePlan))
def test_each_plan_is_a_plan(cls):
    assert isinstance(cls(market_id="M"), Plan)


@pytest.mark.parametrize("cls", (MakerOutcome, TakerOutcome))
def test_each_execution_result_is_an_outcome(cls):
    assert isinstance(cls(market_id="M", dry_run=True), Outcome)


@pytest.mark.parametrize("cls", (MakerReport, TakerReport))
def test_each_monitor_report_is_a_report(cls):
    assert isinstance(cls(market_id="M"), Report)


def test_the_protocols_are_not_inherited_from():
    for cls in (QuotePlan, TakePlan, MakerOutcome, TakerOutcome, MakerReport, TakerReport):
        assert not {b.__name__ for b in cls.__mro__} & {"Plan", "Outcome", "Report", "Strategy"}, cls


@pytest.mark.parametrize("name", NAMES)
def test_planners_have_no_path_to_an_order_endpoint(name):
    source = (STRATEGIES / name / "plan.py").read_text(encoding="utf-8")
    assert "place_order" not in source and "cancel(" not in source, name
    assert "from predkit.strategies import" not in source            # not even the Protocols
    assert "from predkit.venues" not in source


def test_a_plan_reports_every_refusal():
    plan = QuotePlan(market_id="M")
    plan.reasons.extend(["no fair value", "inside the window"])
    assert not plan.ok and len(plan.reasons) == 2


def test_an_outcome_knows_whether_it_happened():
    assert MakerOutcome("M", dry_run=True).dry_run and MakerOutcome("M", dry_run=True).ok
    assert not MakerOutcome("M", dry_run=False, problems=["x"]).ok


def test_a_report_exposes_a_verdict_a_pager_can_key_on():
    assert MakerReport("M", alerts=["foreign order"], critical=True).critical is True


@pytest.mark.parametrize("name", NAMES)
def test_decide_returns_nothing_without_a_signal(name):
    row = WindowRow("kalshi", "M", "default", 0, 1000, "kalshi:M", True, Decimal("0.5"), Decimal("0.48"), None, 2000)
    assert load(name).decide(row) is None
