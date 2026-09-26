"""Hyperliquid: routing, the rate budget, and what gets read next.

No network. REST takes an injected `opener`, the stream takes a fake socket,
and the tracker takes a fake clock - so every scheduling decision is checked
at exact times instead of raced.

The tests that earn their place are the ones about the budget. Discovery
outruns it by design (~450 new accounts a minute against 450 reads), so a
scheduler that re-reads a market maker on every fill, or never refreshes a
large cross position whose liquidation price is drifting, produces a map that
looks complete and is not.
"""

import asyncio
import json
import os
import urllib.error

import pytest

from perpkit.layout import instrument_dirs
from perpkit.hyperliquid import (
    ExclusiveLock,
    HyperliquidMarketFeed,
    MapWriter,
    PositionTracker,
    WeightBudget,
    fetch_universe,
    post_info,
    summarise,
    validate_coins,
)
from perpkit.liquidation_map import build_map
from perpkit.rawlog import iter_directory

A = "0x" + "a" * 40
B = "0x" + "b" * 40
C = "0x" + "c" * 40


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(*payloads):
    """Returns each payload in turn, then repeats the last. Records requests."""
    requests = []

    def _open(request, timeout=None):
        index = min(len(requests), len(payloads) - 1)
        requests.append(request)
        item = payloads[index]
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)

    _open.requests = requests
    return _open


def chs(*positions, time_ms=1_789_151_132_727):
    """A clearinghouseState response, shaped as measured."""
    return {
        "marginSummary": {"accountValue": "1000"},
        "assetPositions": [{"type": "oneWay", "position": p} for p in positions],
        "time": time_ms,
    }


def pos(coin, szi, liq, value="1000", lev_type="cross"):
    return {"coin": coin, "szi": str(szi), "entryPx": "100",
            "liquidationPx": liq, "positionValue": value,
            "leverage": {"type": lev_type, "value": 10}, "marginUsed": "100"}


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t


def tracker_at(tmp_path, clock, **overrides):
    options = dict(
        data_dir=tmp_path, budget=WeightBudget(1200), clock=clock,
        min_repoll_seconds=30.0, min_refresh_seconds=60.0,
        refresh_rebuild_seconds=0.0, on_log=lambda message: None,
    )
    options.update(overrides)
    coins = options.pop("coins", ["BTC", "ETH"])
    return PositionTracker(coins, **options)


# ---------------------------------------------------------------------------
# REST and the universe
# ---------------------------------------------------------------------------


def test_info_is_a_json_post():
    opener = opener_for({"ok": 1})
    assert post_info({"type": "clearinghouseState", "user": A},
                     opener=opener) == {"ok": 1}
    request = opener.requests[0]
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"type": "clearinghouseState", "user": A}


def test_the_universe_pairs_each_asset_with_its_own_context():
    """The endpoint returns two parallel lists; pairing by index happens once."""
    meta = {"universe": [{"name": "BTC", "maxLeverage": 40},
                         {"name": "kPEPE", "maxLeverage": 10}]}
    contexts = [{"markPx": "77130.0"}, {"markPx": "0.003296"}]
    assets, ctxs = fetch_universe(opener=opener_for([meta, contexts]))
    assert assets["BTC"]["maxLeverage"] == 40
    assert ctxs["kPEPE"]["markPx"] == "0.003296"


UNIVERSE = {"BTC": {"name": "BTC"}, "kPEPE": {"name": "kPEPE"},
            "MATIC": {"name": "MATIC", "isDelisted": True}}


def test_a_wrong_case_name_is_refused_with_the_right_one():
    (problem,) = validate_coins(["KPEPE"], UNIVERSE)
    assert "kPEPE" in problem


def test_every_problem_is_listed_not_just_the_first():
    problems = validate_coins(["NOPE", "MATIC", "BTC"], UNIVERSE)
    assert len(problems) == 2
    assert any("NOPE" in p for p in problems)
    assert any("delisted" in p for p in problems)


def test_names_that_collide_on_a_windows_path_are_refused():
    problems = validate_coins(["BTC", "btc"], UNIVERSE)
    assert any("differ only by case" in p for p in problems)


# ---------------------------------------------------------------------------
# The weight budget
# ---------------------------------------------------------------------------


def test_the_budget_never_grants_more_than_its_rate():
    clock = Clock(0.0)

    async def fake_sleep(seconds):
        clock.t += seconds

    budget = WeightBudget(600, clock=clock, sleep=fake_sleep)

    async def spend_a_minute():
        granted = 0
        while True:
            await budget.acquire(2)
            if clock.t >= 60.0:
                return granted
            granted += 2

    granted = asyncio.run(spend_a_minute())
    assert granted <= 600 + budget.capacity
    assert granted >= 600 - 2, "and it does not throttle below its rate either"


def test_the_largest_request_fits_in_the_bucket():
    """A capacity below 20 would make metaAndAssetCtxs wait forever."""
    assert WeightBudget(60).capacity >= 20


def test_a_penalty_stops_spending_for_a_minute():
    clock = Clock(0.0)
    budget = WeightBudget(1200, clock=clock)
    budget.penalise()
    assert budget.try_take(2) >= 60.0


def test_a_budget_over_the_exchange_limit_is_refused():
    with pytest.raises(ValueError):
        WeightBudget(1201)


# ---------------------------------------------------------------------------
# Ownership of the venue directory
# ---------------------------------------------------------------------------


def test_a_second_recorder_is_refused_by_pid(tmp_path):
    lock = tmp_path / ".recorder.lock"
    first = ExclusiveLock(lock, holder="record_hyperliquid.py").acquire()
    try:
        with pytest.raises(SystemExit) as caught:
            ExclusiveLock(lock, holder="record_hyperliquid.py").acquire()
        assert str(os.getpid()) in str(caught.value)
    finally:
        first.release()


def test_a_stale_lock_is_taken_over(tmp_path):
    lock = tmp_path / ".recorder.lock"
    lock.write_text("999999999", encoding="utf-8")
    held = ExclusiveLock(lock, holder="x").acquire()
    assert int(lock.read_text(encoding="utf-8")) == os.getpid()
    held.release()
    assert not lock.exists()


def test_a_lock_whose_pid_was_reused_by_another_program_is_taken_over(tmp_path, monkeypatch):
    """Guards pid reuse after a reboot: the recorder's old pid (3232) now
    belongs to svchost.exe, a bare `tasklist "PID eq"` test says "alive", and
    the supervisor would refuse every restart while the archive stops. A live
    pid that is not a python interpreter is not our recorder."""
    from perpkit import openinterest
    monkeypatch.setattr(openinterest, "_process_identity",
                        lambda pid: ("svchost.exe", None))
    lock = tmp_path / ".recorder.lock"
    lock.write_text("3232", encoding="utf-8")
    held = ExclusiveLock(lock, holder="record_hyperliquid.py").acquire()
    assert int(lock.read_text(encoding="utf-8")) == os.getpid()
    held.release()


def test_a_lock_held_by_a_python_running_another_script_is_taken_over(tmp_path, monkeypatch):
    """Pid reuse by a different python (e.g. record.py) must not block this
    recorder either: the holder's tokens are checked against the command line."""
    from perpkit import openinterest
    monkeypatch.setattr(
        openinterest, "_process_identity",
        lambda pid: ("python.exe", '"python.exe" -u "bot\\perpkit\\record.py" --instruments BTC-USDT'))
    lock = tmp_path / ".recorder.lock"
    lock.write_text("4242", encoding="utf-8")
    held = ExclusiveLock(lock, holder="record_hyperliquid.py").acquire()
    assert int(lock.read_text(encoding="utf-8")) == os.getpid()
    held.release()


def test_a_lock_held_by_the_same_recorder_is_still_refused(tmp_path, monkeypatch):
    """The fix must not weaken the rule: a python whose command line carries
    every token of the holder IS a second writer and is refused. The holder
    may hold several tokens ("record_liquidations.py --venue binance") that the
    command line separates with quotes."""
    from perpkit import openinterest
    monkeypatch.setattr(
        openinterest, "_process_identity",
        lambda pid: ("python.exe", '"python.exe" -u "bot\\perpkit\\record_liquidations.py" --venue binance'))
    lock = tmp_path / ".recorder.lock"
    lock.write_text("4243", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        ExclusiveLock(lock, holder="record_liquidations.py --venue binance").acquire()
    assert "4243" in str(caught.value)
    # ...and the bybit recorder is a different holder on a different lock,
    # but if it pointed at this lock it would be taken over, not refused.
    held = ExclusiveLock(lock, holder="record_liquidations.py --venue bybit").acquire()
    held.release()


def test_an_unreadable_identity_is_assumed_alive(tmp_path, monkeypatch):
    """When the process table cannot be read, never clobber a live lock."""
    from perpkit import openinterest
    monkeypatch.setattr(openinterest, "_process_identity", lambda pid: ("?", None))
    lock = tmp_path / ".recorder.lock"
    lock.write_text("77", encoding="utf-8")
    with pytest.raises(SystemExit):
        ExclusiveLock(lock, holder="record_hyperliquid.py").acquire()


def test_pid_alive_reads_tasklist_csv_by_pid_column(monkeypatch):
    """`_process_identity` must match the PID column, not any digits in the
    row: a pid of 32 must not be found inside the memory column "3,232 K"."""
    from perpkit import openinterest
    if os.name != "nt":
        pytest.skip("tasklist parsing is Windows-only")
    import subprocess

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "tasklist":
            return Done('"svchost.exe","3232","Services","0","3,232 K"\n')
        return Done("")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert openinterest._process_identity(3232) == ("svchost.exe", None)
    assert openinterest._process_identity(32) is None
    assert openinterest._pid_alive(3232, holder="record_hyperliquid.py") is False
    assert openinterest._pid_alive(32) is False


# ---------------------------------------------------------------------------
# The market feed
# ---------------------------------------------------------------------------


TRADE = {"channel": "trades", "data": [{
    "coin": "BTC", "side": "A", "px": "77123.0", "sz": "0.22",
    "time": 1789151132727, "hash": "0xb1", "tid": 921888350010899,
    "users": [A, B]}]}
BOOK = {"channel": "l2Book", "data": {
    "coin": "ETH", "time": 1789151141638,
    "levels": [[{"px": "2543.2", "sz": "1.5", "n": 3}],
               [{"px": "2543.3", "sz": "2.0", "n": 1}]]}}
CTX = {"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": {
    "funding": "0.0000051293", "openInterest": "36327.72504",
    "oraclePx": "77158.0", "markPx": "77117.5", "midPx": "77129.5"}}}


def feed_at(tmp_path, **overrides):
    options = dict(data_dir=tmp_path, on_log=lambda message: None)
    options.update(overrides)
    return HyperliquidMarketFeed(["BTC", "ETH"], **options)


def archived(tmp_path, coin, channel):
    return [m for _, _, m in iter_directory(
        tmp_path / "hyperliquid" / coin / "raw", channel)]


def test_each_coin_archives_verbatim_to_its_own_directory(tmp_path):
    feed = feed_at(tmp_path)
    for message in (TRADE, BOOK, CTX):
        feed.handle_message(message)
    feed.close()

    assert archived(tmp_path, "BTC", "trades") == [TRADE]
    assert archived(tmp_path, "BTC", "activeAssetCtx") == [CTX]
    assert archived(tmp_path, "ETH", "l2Book") == [BOOK]
    assert archived(tmp_path, "ETH", "trades") == []


def test_a_trade_reports_both_accounts(tmp_path):
    seen = []
    feed = feed_at(tmp_path, on_trade_users=lambda coin, users, ms:
                   seen.append((coin, list(users), ms)))
    feed.handle_message(TRADE)
    feed.close()
    assert seen == [("BTC", [A, B], 1789151132727)]


def test_the_context_carries_mark_and_open_interest(tmp_path):
    feed = feed_at(tmp_path)
    feed.handle_message(CTX)
    feed.close()
    assert feed.mark("BTC") == pytest.approx(77117.5)
    assert feed.open_interest("BTC") == pytest.approx(36327.72504)
    assert feed.mark("ETH") is None


def test_a_stale_mark_is_not_offered_as_current(tmp_path):
    feed = feed_at(tmp_path)
    feed.handle_message(CTX)
    feed.contexts["BTC"]["received"] -= 120
    feed.close()
    assert feed.mark("BTC", max_age_s=60) is None
    assert feed.mark("BTC") is not None


def test_control_messages_and_other_coins_are_not_archived(tmp_path):
    feed = feed_at(tmp_path)
    feed.handle_message({"channel": "pong"})
    feed.handle_message({"channel": "subscriptionResponse", "data": {}})
    feed.handle_message({**CTX, "data": {"coin": "SOL", "ctx": {}}})
    feed.close()
    assert not (tmp_path / "hyperliquid" / "SOL").exists()
    assert feed.messages == 0
    assert feed.unrouted == 1


def test_a_failing_consumer_does_not_cost_the_archive(tmp_path):
    def explode(*args):
        raise RuntimeError("tracker bug")

    feed = feed_at(tmp_path, on_trade_users=explode)
    feed.handle_message(TRADE)
    feed.close()
    assert archived(tmp_path, "BTC", "trades") == [TRADE]
    assert feed.errors == 1


def test_the_venue_is_invisible_to_blofin_tools(tmp_path):
    """Hyperliquid BTC must never be mistaken for an instrument directory."""
    (tmp_path / "hyperliquid" / "BTC" / "raw").mkdir(parents=True)
    (tmp_path / "BTC-USDT" / "raw").mkdir(parents=True)
    assert list(instrument_dirs(tmp_path)) == ["BTC-USDT"]


def test_more_subscriptions_than_the_exchange_allows_is_refused(tmp_path):
    coins = [f"C{i}" for i in range(334)]
    with pytest.raises(ValueError, match="subscriptions"):
        HyperliquidMarketFeed(coins, data_dir=tmp_path, record_raw=False)


class FakeSocket:
    def __init__(self, messages=(), gap=0.0):
        self.queue = [json.dumps(message) for message in messages]
        self.gap = gap
        self.sent = []

    async def recv(self):
        if self.queue:
            if self.gap:
                await asyncio.sleep(self.gap)
            return self.queue.pop(0)
        await asyncio.Event().wait()   # an open socket delivering nothing

    async def send(self, text):
        self.sent.append(json.loads(text))


def test_silence_ends_the_stream(tmp_path):
    feed = feed_at(tmp_path, stall_timeout_s=0.05, ping_seconds=10.0)
    asyncio.run(feed._stream(FakeSocket()))
    feed.close()
    assert feed.stalls == 1


def test_a_trickle_is_not_a_stall(tmp_path):
    """Ten messages 20ms apart under a 100ms deadline are all delivered; the
    stall only comes once they stop."""
    feed = feed_at(tmp_path, stall_timeout_s=0.1, ping_seconds=10.0)
    asyncio.run(feed._stream(FakeSocket([CTX] * 10, gap=0.02)))
    feed.close()
    assert feed.messages == 10
    assert feed.stalls == 1


def test_a_ping_goes_out_before_the_server_would_drop_us(tmp_path):
    feed = feed_at(tmp_path, stall_timeout_s=0.3, ping_seconds=0.05)
    socket = FakeSocket()
    asyncio.run(feed._stream(socket))
    feed.close()
    assert {"method": "ping"} in socket.sent
    assert feed.pings >= 2


# ---------------------------------------------------------------------------
# Positions: filing a reading
# ---------------------------------------------------------------------------


def test_positions_are_filed_by_coin_and_closed_ones_forgotten(tmp_path):
    tracker = tracker_at(tmp_path, Clock())
    tracker.apply(A, chs(pos("BTC", "1.5", "70000", value="115000"),
                         pos("ETH", "-2", "3000", value="5000"),
                         pos("SOL", "10", "80")))
    assert set(tracker.positions["BTC"]) == {A}
    assert tracker.positions["ETH"][A].size == pytest.approx(-2.0)
    assert "SOL" not in tracker.positions
    assert tracker.accounts[A].notional == pytest.approx(120_000)

    tracker.apply(A, chs(pos("ETH", "-2", "3000", value="5000")))
    assert A not in tracker.positions["BTC"]
    assert tracker.accounts[A].notional == pytest.approx(5000)
    tracker.close()


def test_the_archive_keeps_whole_accounts_and_marks_unchanged_reads(tmp_path):
    tracker = tracker_at(tmp_path, Clock())
    state = chs(pos("BTC", "1", "70000"), pos("SOL", "10", "80"))
    tracker.apply(A, state)
    tracker.apply(A, state)
    tracker.apply(A, chs(pos("BTC", "1", "70500"), pos("SOL", "10", "80")))
    tracker.close()

    events = [m for _, _, m in iter_directory(
        tmp_path / "hyperliquid" / "_accounts" / "raw", "clearinghouseState")]
    assert [("data" in e, e.get("unchanged", False)) for e in events] == [
        (True, False), (False, True), (True, False)]
    assert all(event["user"] == A for event in events)
    kept = [p["position"]["coin"] for p in events[0]["data"]["assetPositions"]]
    assert "SOL" in kept, "untracked coins move a cross liquidation price"


def test_a_malformed_response_is_a_failure_not_a_crash(tmp_path):
    tracker = tracker_at(tmp_path, Clock(), opener=opener_for({"error": "x"}))
    tracker.observe_trade("BTC", [A])
    assert asyncio.run(tracker.poll_next()) is True
    assert tracker.failures == 1
    tracker.close()


def test_poll_next_reads_files_and_spends_weight(tmp_path):
    budget = WeightBudget(1200)
    opener = opener_for(chs(pos("BTC", "-0.4004", "209806.4493876494",
                                value="30905")))
    tracker = tracker_at(tmp_path, Clock(), budget=budget, opener=opener)
    tracker.observe_trade("BTC", [A])

    assert asyncio.run(tracker.poll_next()) is True
    assert tracker.positions["BTC"][A].liquidation_px == pytest.approx(
        209806.4493876494)
    assert json.loads(opener.requests[0].data) == {
        "type": "clearinghouseState", "user": A}
    assert budget.spent == 2
    assert asyncio.run(tracker.poll_next()) is False, "nothing else is due"
    tracker.close()


# ---------------------------------------------------------------------------
# Positions: what gets read next
# ---------------------------------------------------------------------------


def test_a_trade_queues_both_accounts_once(tmp_path):
    tracker = tracker_at(tmp_path, Clock())
    tracker.observe_trade("BTC", [A, B])
    tracker.observe_trade("BTC", [A, B])
    assert [tracker.next_user(), tracker.next_user(),
            tracker.next_user()] == [A, B, None]


def test_things_that_are_not_addresses_are_ignored(tmp_path):
    tracker = tracker_at(tmp_path, Clock())
    tracker.observe_trade("BTC", ["", "0x123", None, A])
    assert set(tracker.accounts) == {A}


def test_an_account_that_trades_again_waits_out_the_repoll_interval(tmp_path):
    """A market maker filling every block must not spend the budget on itself."""
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock)
    tracker.observe_trade("BTC", [A])
    assert tracker.next_user() == A
    tracker.apply(A, chs(pos("BTC", "1", "70000")))

    tracker.observe_trade("BTC", [A])
    clock.t = 1029.0
    assert tracker.next_user() is None
    clock.t = 1030.0
    assert tracker.next_user() == A


def test_a_trade_during_a_read_is_not_a_second_concurrent_read(tmp_path):
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock)
    tracker.observe_trade("BTC", [A])
    assert tracker.next_user() == A
    tracker.observe_trade("BTC", [A])
    assert tracker.next_user() is None

    tracker.apply(A, chs())
    clock.t += 30
    assert tracker.next_user() == A, "but it is not forgotten either"


def test_refresh_prefers_larger_positions_at_equal_age(tmp_path):
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock)
    tracker.apply(A, chs(pos("BTC", "1", "70000", value="1000")))
    tracker.apply(B, chs(pos("BTC", "1", "70000", value="4000")))
    clock.t = 1100.0
    assert [tracker.next_user(), tracker.next_user(),
            tracker.next_user()] == [B, A, None]


def test_refresh_prefers_older_readings_at_equal_size(tmp_path):
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock)
    tracker.apply(A, chs(pos("BTC", "1", "70000")))
    clock.t = 1050.0
    tracker.apply(B, chs(pos("BTC", "1", "70000")))
    clock.t = 1200.0
    assert tracker.next_user() == A


def test_a_fresh_reading_is_not_refreshed(tmp_path):
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock)
    tracker.apply(A, chs(pos("BTC", "1", "70000")))
    clock.t = 1059.0
    assert tracker.next_user() is None
    clock.t = 1060.0
    assert tracker.next_user() == A


def test_an_account_without_a_tracked_position_is_never_refreshed(tmp_path):
    """It cannot open one without trading, and a trade re-queues it."""
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock, coins=["BTC"])
    tracker.apply(C, chs(pos("ETH", "5", "3000")))
    clock.t += 1e6
    assert tracker.next_user() is None


def test_one_read_in_four_goes_to_refresh_when_both_queues_have_work(tmp_path):
    clock = Clock(1000.0)
    tracker = tracker_at(tmp_path, clock, refresh_every=4)
    holders = ["0x" + f"{i:040x}" for i in range(1, 11)]
    for user in holders:
        tracker.apply(user, chs(pos("BTC", "1", "70000")))
    clock.t = 2000.0
    traders = ["0x" + f"{i:040x}" for i in range(100, 110)]
    tracker.observe_trade("BTC", traders)

    picks = [tracker.next_user() for _ in range(8)]
    assert sum(pick in holders for pick in picks) == 2
    assert sum(pick in traders for pick in picks) == 6


def test_an_empty_queue_gives_its_turn_to_the_other(tmp_path):
    tracker = tracker_at(tmp_path, Clock(), refresh_every=4)
    traders = ["0x" + f"{i:040x}" for i in range(100, 104)]
    tracker.observe_trade("BTC", traders)
    assert [tracker.next_user() for _ in range(4)] == traders


# ---------------------------------------------------------------------------
# Positions: failure and restart
# ---------------------------------------------------------------------------


def test_a_rate_limit_empties_the_budget_and_requeues_the_account(tmp_path):
    clock = Clock(1000.0)
    budget = WeightBudget(1200)
    limited = urllib.error.HTTPError(
        "https://api.hyperliquid.xyz/info", 429, "Too Many Requests",
        hdrs=None, fp=None)
    tracker = tracker_at(tmp_path, clock, budget=budget,
                         opener=opener_for(limited), retry_seconds=10.0)
    tracker.observe_trade("BTC", [A])

    assert asyncio.run(tracker.poll_next()) is True
    assert (tracker.failures, tracker.rate_limited) == (1, 1)
    assert budget.try_take(2) >= 59.0
    clock.t = 1009.0
    assert tracker.next_user() is None
    clock.t = 1010.0
    assert tracker.next_user() == A


def test_any_other_failure_is_counted_but_leaves_the_budget_alone(tmp_path):
    budget = WeightBudget(1200)
    tracker = tracker_at(tmp_path, Clock(), budget=budget,
                         opener=opener_for(OSError("connection reset")))
    tracker.observe_trade("BTC", [A])
    asyncio.run(tracker.poll_next())
    assert (tracker.failures, tracker.rate_limited) == (1, 0)
    assert budget.try_take(2) == 0.0


def test_the_address_book_survives_a_restart_largest_first(tmp_path):
    clock = Clock(1000.0)
    first = tracker_at(tmp_path, clock)
    first.apply(A, chs(pos("BTC", "1", "70000", value="1000")))
    first.apply(B, chs(pos("BTC", "1", "70000", value="9000")))
    first.apply(C, chs(pos("SOL", "1", "80")))    # holds nothing tracked
    first.close()

    second = tracker_at(tmp_path, clock)
    assert second.restored == 2
    assert second.positions["BTC"] == {}, "old readings are not current ones"
    assert [second.next_user(), second.next_user(),
            second.next_user()] == [B, A, None]
    second.close()


def test_a_restart_before_restored_accounts_are_read_keeps_them(tmp_path):
    """Saving mid-rebuild must not shrink the book to what was re-read so far."""
    clock = Clock(1000.0)
    first = tracker_at(tmp_path, clock)
    first.apply(A, chs(pos("BTC", "1", "70000", value="1000")))
    first.close()
    tracker_at(tmp_path, clock).close()
    assert tracker_at(tmp_path, clock).restored == 1


def test_reads_stranded_by_a_restart_are_requeued(tmp_path):
    tracker = tracker_at(tmp_path, Clock())
    tracker.observe_trade("BTC", [A])
    assert tracker.next_user() == A           # handed out, never finished
    assert tracker.next_user() is None
    assert tracker.recover_in_flight() == 1
    assert tracker.next_user() == A


# ---------------------------------------------------------------------------
# The map on disk and on the console
# ---------------------------------------------------------------------------


def test_the_map_lands_beside_the_coin_one_line_per_snapshot(tmp_path):
    writer = MapWriter(tmp_path)
    result = build_map("BTC", [], mark_px=77130.0,
                       as_of_ms=1_789_151_132_727, open_interest=36327.5)
    writer.write(result)
    writer.write(result)
    writer.close()

    (path,) = (tmp_path / "hyperliquid" / "BTC").glob("liquidation-levels-*")
    assert path.name == "liquidation-levels-2026-09-11.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["mark_px"] == 77130.0


def test_the_summary_is_ascii_and_says_what_it_covers(tmp_path):
    tracker = tracker_at(tmp_path, Clock())
    tracker.apply(A, chs(pos("BTC", "2", "75000", value="154000"),
                         time_ms=1_000_000))
    result = build_map("BTC", tracker.positions["BTC"].values(),
                       mark_px=77130.0, as_of_ms=1_010_000,
                       open_interest=40.0)
    line = summarise(result)
    line.encode("ascii")
    assert "coverage L 5.0%" in line
    assert "down" in line and "up" in line
    tracker.close()


@pytest.mark.parametrize("module", ["record_hyperliquid", "record_liquidations"])
def test_the_recorders_lock_holder_matches_how_they_are_run(tmp_path, monkeypatch, module):
    """The shipped way to run a recorder is `python -m perpkit.<module>`. If
    the holder token did not appear in that command line, a second instance
    would call the live recorder's lock stale, take it over, and two writers
    would share one archive."""
    import importlib

    from perpkit import openinterest

    holder = importlib.import_module("perpkit." + module).LOCK_HOLDER
    monkeypatch.setattr(
        openinterest, "_process_identity",
        lambda pid: ("python.exe", "python.exe -m perpkit." + module + " --check"))
    assert openinterest._pid_alive(os.getpid() + 1, holder=holder)
