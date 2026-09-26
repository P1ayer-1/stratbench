"""Each planner's gates and arithmetic, hand-computed."""

from decimal import Decimal

from predkit.ledger import OwnLedger, RunLog
from predkit.schema import Book, Contract, Fill, Level, Outcome, Side, utc
from predkit.strategies import Context
from predkit.strategies.simple_maker import MakerBot, plan_quote
from predkit.strategies.simple_maker.execute import cancel_own, execute_quote
from predkit.strategies.simple_maker.monitor import monitor_quote
from predkit.strategies.simple_taker import devig, plan_take

D = Decimal
END = utc(2026, 9, 12, 14, 15)


def kalshi(market="KX-1", tier="default"):
    return Contract("kalshi", market, "q", END, "kalshi:KX", tier)


def poly(tier="crypto_fees_v2"):
    return Contract("polymarket", "0xc", "q", END, "polymarket:uma:0xc", tier, min_size=D("5"),
                    yes_token="Y", no_token="N")


def book(bid="0.47", ask="0.50", bid_size=100, ask_size=100):
    return Book("M", bids=[Level(bid, bid_size)], asks=[Level(ask, ask_size)], ts_ms=0)


def ctx(contract, bk, *, fair=None, seconds_left=200, lag=100.0, position="0", cost="0", reference=None):
    return Context(contract=contract, book=bk, now_ms=contract.resolves_at_ms - seconds_left * 1000,
                   fair=fair, feed_lag_ms=lag, position=D(position), avg_cost=D(cost), reference=reference or {})


# ---- simple_maker ----------------------------------------------------------

def test_maker_refuses_with_every_reason_at_once():
    plan = plan_quote(ctx(poly("unknown"), book(), fair=None, seconds_left=30, lag=400.0))
    text = "\n".join(plan.reasons)
    assert "no fair value" in text and "inside the 60s window" in text and "feed lag 400" in text
    assert "no fee table" in text and "unknown" in text          # the absent tier refuses loudly
    assert len(plan.reasons) == 4 and not plan.intents


def test_maker_bids_one_tick_inside_the_ask_when_fair_is_far_enough():
    # ask 0.50, tick 0.01 -> bid at 0.49; fair 0.52 - 0.49 = 0.03 >= edge 0.02
    plan = plan_quote(ctx(poly(), book(), fair=D("0.52")), size=D("5"))
    assert plan.ok
    (intent,) = plan.intents
    assert (intent.side, intent.outcome, intent.price, intent.post_only) == (Side.BUY, Outcome.YES, D("0.49"), True)


def test_maker_buys_no_when_fair_is_below_the_bid_plus_a_tick():
    # bid 0.47 -> ask at 0.48; 0.48 - fair 0.45 = 0.03 >= 0.02: buy NO at 1 - 0.48 = 0.52
    plan = plan_quote(ctx(poly(), book(), fair=D("0.45")), size=D("5"))
    (intent,) = plan.intents
    assert (intent.outcome, intent.price) == (Outcome.NO, D("0.52"))


def test_maker_will_not_join_a_queue_at_the_touch():
    # spread of one tick: bid 0.49, ask 0.50 -> ask - tick == bid, no bid side
    plan = plan_quote(ctx(poly(), book(bid="0.49", ask="0.50"), fair="0.60"))
    assert plan.ok and not plan.intents
    assert any("join the queue" in w for w in plan.warnings)


def test_maker_stop_exits_at_the_touch_with_a_taker_order():
    # holding 5 YES at 0.49; fair 0.47 <= 0.49 - 0.01
    plan = plan_quote(ctx(poly(), book(), fair=D("0.47"), position="5", cost="0.49"))
    (exit_,) = plan.exits
    assert (exit_.side, exit_.price, exit_.post_only) == (Side.SELL, D("0.47"), False)


def test_maker_decide_mirrors_the_plan_thresholds():
    from predkit.backtest import WindowRow
    row = WindowRow("polymarket", "0xc", "crypto_fees_v2", 0, 1000, "polymarket:uma:0xc", True, D("0.50"), D("0.47"),
                    D("0.52"), 2000)
    decision = MakerBot().decide(row)
    assert (decision.outcome, decision.price, decision.role) == (Outcome.YES, D("0.49"), "maker")


class FakeVenue:
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.reject = False
        self._fills = []
        self._open = []

    def place_order(self, intent):
        if self.reject:
            raise RuntimeError("rejected")
        self.placed.append(intent)
        return f"v{len(self.placed)}"

    def cancel(self, order_id):
        self.cancelled.append(order_id)

    def open_orders(self, market_id=None):
        return self._open

    def fills(self, market_id=None):
        return self._fills


def test_execute_is_dry_unless_told_and_the_result_says_so():
    plan = plan_quote(ctx(poly(), book(), fair=D("0.52")))
    venue, ledger, log = FakeVenue(), OwnLedger(), RunLog(None)
    dry = execute_quote(plan, venue, ledger, log)
    assert dry.dry_run and dry.order_ids[0].startswith("dry-") and venue.placed == []
    real = execute_quote(plan, venue, ledger, log, dry_run=False)
    assert not real.dry_run and real.order_ids == ["v1"] and len(venue.placed) == 1
    assert log.events[-1]["event"] == "placed"


def test_cancel_own_touches_only_the_ledgers_orders():
    plan = plan_quote(ctx(poly(), book(), fair=D("0.52")))
    venue, ledger, log = FakeVenue(), OwnLedger(), RunLog(None)
    execute_quote(plan, venue, ledger, log, dry_run=False)
    venue._open = [{"id": "v1"}, {"id": "somebody-elses"}]
    cancel_own(venue, ledger, log, dry_run=False)
    assert venue.cancelled == ["v1"]


def test_monitor_flags_foreign_orders_and_fills_as_critical():
    venue, ledger = FakeVenue(), OwnLedger()
    venue._open = [{"id": "not-mine"}]
    venue._fills = [Fill("polymarket", "0xc", "not-mine-either", Side.BUY, "0.5", "5", "0", 1, fill_id="f1")]
    report = monitor_quote(venue, ledger, "0xc")
    assert report.critical and len(report.alerts) == 2
    assert ledger.fills == []                       # never adopted


# ---- simple_taker ----------------------------------------------------------

def test_devig_normalises_to_one():
    # 1/1.91 = 0.5236, 1/1.95 = 0.5128; total 1.0364 -> 0.5052 / 0.4948
    yes, no = devig("1.91", "1.95")
    assert yes + no == D("1")
    assert yes.quantize(D("0.0001")) == D("0.5052")


def test_taker_buys_yes_when_fair_beats_the_ask_net_of_fee():
    # fair 0.60, ask 0.50: 0.60 - 0.50 - kalshi taker(0.50, 1)=ceil(0.0175)=0.02 -> 0.08 >= 0.03
    c = kalshi()
    plan = plan_take(ctx(c, book(), seconds_left=7200, reference={"sharp_prob": "0.60"}))
    assert plan.ok and plan.edge_yes == D("0.08")
    (intent,) = plan.intents
    assert (intent.outcome, intent.price, intent.post_only) == (Outcome.YES, D("0.50"), False)


def test_taker_refuses_near_resolution_and_when_already_positioned():
    plan = plan_take(ctx(kalshi(), book(), seconds_left=600, position="3", reference={"sharp_prob": "0.6"}))
    assert not plan.ok and len(plan.reasons) == 2


def test_taker_uses_the_model_over_the_reference_probability_when_both_exist():
    plan = plan_take(ctx(kalshi(), book(), seconds_left=7200,
                         reference={"sharp_prob": "0.30", "model_prob": "0.62"}))
    assert plan.fair == D("0.62")
