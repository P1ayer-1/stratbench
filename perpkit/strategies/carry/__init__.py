"""Delta-neutral funding carry: LONG spot + SHORT perp, collect funding.

TEACHING EXAMPLE, NO KNOWN EDGE. This package shows the plan / execute /
monitor split on a two-leg position with real leg-ordering and unwind logic.
It makes no claim that funding carry earns money after fees, spreads, basis
moves and liquidation risk; `perpkit.analysis.carry_backtest` is how you would
find out, on your own data.

    plan.py     sizes both legs, prices the risk, says whether to do it
    execute.py  puts it on, perp first; takes it off, spot first
    monitor.py  reads both legs back and scores realised against predicted
    broker.py   the REST calls all three are written against

The return source is the funding payment a perpetual pays its shorts, not a
price forecast. The two legs cancel price exposure so that what is left is the
funding, which is why nothing here predicts anything.

Delta-neutral is not risk-neutral, and that is the thing to keep in mind
whichever module you are in: the legs margin separately, so a rally that
leaves the PAIR flat can still liquidate the short - and what survives that is
an unhedged long spot position, the opposite of the trade.
"""

from .broker import BlofinBroker
from .execute import CarryExecutor, ExecutionResult, Step
from .monitor import (
    Baseline,
    MonitorReport,
    Snapshot,
    build_snapshot,
    carry_tag,
    compare,
    fills_totals,
    implied_funding,
    reconstruct_baseline,
)
from .plan import CarryPlan, Market, Wallets, plan_carry, round_down

__all__ = [
    "Baseline",
    "BlofinBroker",
    "CarryExecutor",
    "CarryPlan",
    "ExecutionResult",
    "Market",
    "MonitorReport",
    "Snapshot",
    "Step",
    "Wallets",
    "build_snapshot",
    "carry_tag",
    "compare",
    "fills_totals",
    "implied_funding",
    "plan_carry",
    "reconstruct_baseline",
    "round_down",
]
