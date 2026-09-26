"""The runner refuses live loudly, paper-fills from the tape, and stops on KILL."""

from decimal import Decimal
from pathlib import Path

import pytest

from predkit.runner import PidLock, RunConfig, Runner, refusals
from predkit.schema import Book, Contract, Level, Side, Trade, utc
from predkit.strategies.simple_maker import MakerBot

D = Decimal


def test_live_needs_confirm_residency_and_a_signer_and_says_all_three():
    reasons = refusals(RunConfig("simple_maker", "polymarket", "0xc", live=True, residency="CA"))
    text = "\n".join(reasons)
    assert "--confirm" in text and "91-102" in text and "no signer" in text
    assert len(reasons) == 3


def test_us_residency_is_refused_on_polymarket_but_not_kalshi():
    assert refusals(RunConfig("simple_maker", "polymarket", "0xc", live=True, confirm=True, residency="US"),
                    venue_can_sign=True)
    assert refusals(RunConfig("simple_maker", "kalshi", "K", live=True, confirm=True, residency="US"),
                    venue_can_sign=True) == []


def test_paper_needs_nothing():
    assert refusals(RunConfig("simple_maker", "polymarket", "0xc")) == []


class FakeVenue:
    def __init__(self, contract):
        self._contract = contract
        self.books = [Book(contract.market_id, bids=[Level("0.47", 100)], asks=[Level("0.50", 100)], ts_ms=1)]
        self.trades_out = []

    def market(self, market_id):
        return self._contract

    def book(self, market_id):
        return self.books[-1]

    def trades(self, market_id, limit=20):
        return self.trades_out


def contract():
    return Contract("polymarket", "0xc", "q", utc(2026, 9, 12, 14, 15), "polymarket:uma:0xc", "crypto_fees_v2",
                    min_size=D("5"), yes_token="Y", no_token="N")


async def _no_sleep(_):
    pass


def make_runner(tmp_path, fair=D("0.52")):
    c = contract()
    now = [c.resolves_at_ms / 1000 - 200]
    config = RunConfig("simple_maker", "polymarket", "0xc", data_dir=tmp_path, max_iterations=3)
    venue = FakeVenue(c)
    runner = Runner(config, venue, MakerBot(), fair=lambda book: fair, clock=lambda: now[0], sleep=_no_sleep)
    return runner, venue, now


async def test_paper_step_posts_and_fills_from_the_tape(tmp_path):
    runner, venue, now = make_runner(tmp_path)
    runner.step()
    assert runner.stats()["mode"] == "paper"
    assert "dry-" in next(iter(runner._paper_orders))
    now[0] += 2
    venue.trades_out = [Trade("0xc", "0.49", "5", Side.SELL, int(now[0] * 1000))]
    runner.step()
    assert runner.stats()["paperFills"]["optimistic"] >= 1
    assert runner.risk.contracts.get("0xc", D("0")) > 0 or runner.stats()["paperFills"]["pessimistic"] == 0


async def test_the_veto_is_consulted_before_execute(tmp_path):
    runner, venue, now = make_runner(tmp_path)
    runner.risk.trip("test")
    runner.step()
    assert runner.vetoed == 1 and runner._paper_orders == {}


async def test_kill_file_stops_the_loop_and_cancels_own_orders(tmp_path):
    runner, venue, now = make_runner(tmp_path)
    runner.step()
    assert runner._paper_orders
    runner.kill_file.parent.mkdir(parents=True, exist_ok=True)
    runner.kill_file.write_text("stop")
    await runner.run()
    assert runner.stopped and runner.risk.kill_switch_active
    assert runner.paper.open() == []


def test_pid_lock_refuses_a_second_live_holder_and_takes_over_a_dead_one(tmp_path):
    import os
    (tmp_path / "x.pid").write_text(str(os.getppid()))     # a process that is alive: not ours
    with pytest.raises(SystemExit):
        PidLock(tmp_path / "x.pid").acquire()
    (tmp_path / "x.pid").write_text("999999")            # not a live pid
    PidLock(tmp_path / "x.pid").acquire()
    assert (tmp_path / "x.pid").read_text() == str(os.getpid())
