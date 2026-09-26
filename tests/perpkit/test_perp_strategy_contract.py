"""The three verbs every strategy has, asserted against the one that exists.

`perpkit/strategies/__init__.py` states a lifecycle - plan, execute, monitor -
as Protocols rather than base classes, on the grounds that an interface
extracted from a single example encodes that example's accidents. The risk of
stating a contract nothing is checked against is that it quietly stops being
true, and a second strategy then inherits a description of the first one from
six months ago.

So these assert that carry conforms. Structurally: `CarryPlan` and friends do
not import the Protocols, do not inherit from them, and do not know they
exist. If a rename or a refactor breaks the shape, this is what says so.
"""

from decimal import Decimal

from perpkit.strategies import Outcome, Plan, Report
from perpkit.strategies.carry import (
    Baseline,
    CarryPlan,
    ExecutionResult,
    MonitorReport,
    compare,
)
from perpkit.strategies.carry.monitor import Snapshot


def test_a_carry_plan_is_a_plan():
    assert isinstance(CarryPlan(inst_id="SUI-USDT"), Plan)


def test_an_execution_result_is_an_outcome():
    assert isinstance(
        ExecutionResult(inst_id="SUI-USDT", dry_run=True), Outcome)


def test_a_monitor_report_is_a_report():
    assert isinstance(MonitorReport(inst_id="SUI-USDT"), Report)


def test_the_protocols_are_not_inherited_from():
    """Structural, not nominal. Carry must not know these exist.

    A strategy that had to import the contract to satisfy it would make the
    contract a dependency, and the next strategy would be shaped by carry's
    imports before anyone had decided it should be.
    """
    for cls in (CarryPlan, ExecutionResult, MonitorReport):
        bases = {base.__name__ for base in cls.__mro__}
        assert not bases & {"Plan", "Outcome", "Report"}, cls


def test_a_plan_reports_every_refusal_not_just_the_first():
    """`reasons` is plural in the contract for a reason: fix one gate, re-run,
    find another is a worse loop than being told all of them at once."""
    plan = CarryPlan(inst_id="SUI-USDT")
    plan.reasons.extend(["liquidation too close", "leverage over the limit"])
    assert not plan.ok
    assert len(plan.reasons) == 2


def test_an_outcome_knows_whether_it_actually_happened():
    """`dry_run` lives on the result, not only on the executor, so a caller
    holding a result can never be confused about whether it describes
    something real."""
    rehearsed = ExecutionResult(inst_id="SUI-USDT", dry_run=True)
    real = ExecutionResult(inst_id="SUI-USDT", dry_run=False)
    assert rehearsed.dry_run and not real.dry_run
    assert rehearsed.ok and real.ok          # no problems recorded yet


def test_a_report_exposes_a_verdict_a_scheduler_can_act_on():
    """`critical` has to be readable without parsing prose - that is what a
    pager or a cron wrapper keys on."""
    baseline = Baseline(inst_id="SUI-USDT", opened_at_ms=0, position_id="p1",
                        perp_contracts=Decimal("-100"),
                        perp_entry=Decimal("1"),
                        notional_usd=Decimal("100"))
    # Perp leg gone, spot still held: the unhedged long, and the one state
    # that must always come back critical.
    naked = compare(baseline, Snapshot(at_ms=1, inst_id="SUI-USDT", open=False,
                                       spot_base=Decimal("100")))
    assert naked.critical is True
    assert isinstance(naked, Report)
