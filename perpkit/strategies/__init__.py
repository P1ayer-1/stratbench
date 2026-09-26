"""One strategy per directory, each with the same three verbs.

    plan     what to do, and every reason not to.  SENDS NOTHING.
    execute  put it on, or leave nothing behind trying.  Dry by default.
    monitor  read it back and score it against the plan.  READ ONLY.

The split is not filing. It is a safety property: every sizing and margin question gets answered and reviewed
while the answer is still only text, and the code that can move money is a
separate thing you have to ask for by name.

The contract, verb by verb
--------------------------
**plan** computes orders and cannot send them. There is no code path from a
planner to `placeOrder`, which is checkable with a grep and worth checking
that way. A planner returns its refusals *plural* - every failing gate, not
the first - because the second reason a trade is bad does not stop mattering
once the first is found.

**execute** is dry unless a caller explicitly says otherwise, and reports the
same result shape either way, so the difference between a rehearsal and the
real thing is one flag and nothing else. It owns leg ordering and unwinding:
which leg to send first is a real decision, and whichever fills leaves you
directional until the other does.

**monitor** has no `--confirm` and no path to `placeOrder` at all. It is the
one you want to be able to run at 3am without reading the source first to
check what it might do. It verifies against the EXCHANGE rather than against
the plan, because a plan that agrees with itself has established nothing.

Why there is no Strategy base class
-----------------------------------
There is exactly one strategy here (the carry teaching example). A base class derived from a single
example encodes that example's accidents, and carry has several: two legs, a
funding payment as the entire return source, a hold measured in weeks, and no
prediction anywhere in it. None of those are properties of a directional
strategy, and the next one would spend its life fighting an interface built
before anyone knew what it needed.

So the lifecycle is stated as Protocols, which are satisfied structurally and
inherited from never. `CarryPlan`, `ExecutionResult` and `MonitorReport`
already conform without knowing these exist, which is the point - and
`test_strategy_contract.py` asserts it, so the shape is load-bearing rather
than decorative. When a second strategy arrives and both genuinely want the
same behaviour, that is the moment to extract it, not now.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class Plan(Protocol):
    """What a strategy decided to do, and every reason not to.

    `ok` is the whole verdict. `reasons` is why not, in full - a planner that
    returns only the first failing gate teaches its caller to fix one thing,
    re-run, and find another.
    """

    inst_id: str
    ok: bool
    reasons: List[str]
    warnings: List[str]


@runtime_checkable
class Outcome(Protocol):
    """What happened when a plan was executed, real or rehearsed.

    `dry_run` is on the result rather than only on the executor so that a
    caller reading a result can never be confused about whether it describes
    something that happened.
    """

    inst_id: str
    dry_run: bool
    problems: List[str]

    @property
    def ok(self) -> bool: ...


@runtime_checkable
class Report(Protocol):
    """A position read back from the exchange and scored against its plan.

    `critical` exists so a caller - a scheduler, a pager - can act on the
    verdict without parsing prose.
    """

    inst_id: str
    alerts: List[Any]

    @property
    def critical(self) -> bool: ...


__all__ = ["Outcome", "Plan", "Report"]
