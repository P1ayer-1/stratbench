"""Run one bot on one market: paper or live, one process, a kill switch.

    python -m predkit.runner --strategy simple_maker --venue polymarket --market <condition_id>
    python -m predkit.runner --strategy simple_taker --venue kalshi --market <ticker> --residency <CC> --live --confirm --key-name kalshi

Modes
-----
paper (default)  production market data; the plan's intents are logged and
                 filled on paper by the replay fill model from the live
                 tape, both bounds. Nothing is sent anywhere.
--live           the executor sends. Refused without --confirm, refused
                 without a residency the geofence accepts, refused without a
                 signer on the venue. All reasons listed at once.

One process per (venue, market, strategy), held by a pid lock in the data
directory; a stale lock from a dead pid is taken over. The kill switch is a
file, `<market_dir>/KILL`: its presence trips the risk engine, cancels this
process's own orders (never anyone else's) and stops the loop. Deleting the
file does not restart anything; that is a decision.

Every iteration: read the book, build the context, `plan`, put each intent
through `risk.check_order`, `execute` what survived, `monitor` on a slower
cadence, log. Per-iteration errors are counted, not raised; the loop runs
under `supervise` anyway.

Keys: `--keystore` and `--key-name` name an entry in `predkit.keys`; the
passphrase comes from `PREDKIT_PASSPHRASE` or a prompt, and the decrypted
secret goes straight into the venue's signer and nowhere else.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from predkit import geofence
from predkit.record import market_dir
from predkit.ledger import RunLog
from predkit.replay import MakerFillModel, MakerOrder
from predkit.risk import Limits, RiskEngine
from predkit.schema import Book, OrderIntent, Outcome, Side
from predkit.strategies import NAMES, Context, load
from predkit.supervise import log, supervise


@dataclass
class RunConfig:
    strategy: str
    venue: str
    market_id: str
    residency: str = ""
    live: bool = False
    confirm: bool = False
    data_dir: Path = Path("data")
    interval_s: float = 1.0
    monitor_every: int = 30
    max_iterations: int = -1
    limits: Limits = field(default_factory=Limits)
    params: Dict[str, Any] = field(default_factory=dict)


def refusals(config: RunConfig, *, venue_can_sign: bool = False) -> List[str]:
    """Every reason not to start, in one list."""
    reasons: List[str] = []
    if config.strategy not in NAMES:
        reasons.append(f"unknown strategy {config.strategy!r}; known: {list(NAMES)}")
    if config.venue not in geofence.KNOWN_VENUES:
        reasons.append(f"unknown venue {config.venue!r}; known: {list(geofence.KNOWN_VENUES)}")
    if config.live:
        if not config.confirm:
            reasons.append("--live without --confirm: live is asked for twice, by name")
        reasons.extend(geofence.refusals(config.venue, config.residency))
        if not venue_can_sign:
            reasons.append("the venue adapter has no signer; nothing can be sent")
    return reasons


class PidLock:
    """Exclusive per (venue, market, strategy). Stale locks are taken over."""

    def __init__(self, path: Path):
        self.path = path
        self.held = False

    @staticmethod
    def _alive(pid: int) -> bool:
        """psutil, not `os.kill(pid, 0)`: on Windows that call TERMINATES
        the target process rather than probing it."""
        import psutil

        return pid > 0 and psutil.pid_exists(pid)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "x") as handle:
                handle.write(str(os.getpid()))
        except FileExistsError:
            try:
                other = int(self.path.read_text().strip() or "0")
            except ValueError:
                other = 0
            if other != os.getpid() and self._alive(other):
                raise SystemExit(f"{self.path} is held by live pid {other}: one process per market "
                                 "per strategy. Stop that one first.")
            self.path.write_text(str(os.getpid()))
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.held = False


class Runner:
    def __init__(self, config: RunConfig, venue: Any, strategy: Any, *,
                 fair: Optional[Callable[[Book], Optional[Decimal]]] = None,
                 reference: Optional[Callable[[], Dict[str, Any]]] = None,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 log_path: Optional[Path] = None):
        self.config = config
        self.venue = venue
        self.strategy = strategy
        self.fair = fair or (lambda book: None)
        self.reference = reference or (lambda: {})
        self.clock = clock
        self.sleep = sleep
        self.risk = RiskEngine(config.limits)
        self.directory = market_dir(config.data_dir, config.venue, config.market_id)
        self.kill_file = self.directory / "KILL"
        self.lock = PidLock(self.directory / f"{config.strategy}.pid")
        self.paper = MakerFillModel()
        self.log = RunLog(log_path)
        self.iterations = 0
        self.errors = 0
        self.vetoed = 0
        self.stopped = False
        self.contract = None
        self._paper_orders: Dict[str, OrderIntent] = {}

    @property
    def dry_run(self) -> bool:
        return not (self.config.live and self.config.confirm)

    def _to_maker_order(self, order_id: str, intent: OrderIntent, now_ms: int, book: Book) -> MakerOrder:
        # Paper fills are simulated on the YES book: a NO buy at q is a YES ask at 1 - q.
        if intent.outcome is Outcome.YES:
            side, price = intent.side, intent.price
        else:
            side = Side.SELL if intent.side is Side.BUY else Side.BUY
            price = Decimal(1) - intent.price
        ttl = getattr(self.strategy, "ttl_ms", 10_000)
        order = MakerOrder(order_id, side, price, intent.size, now_ms, ttl)
        return order

    def step(self) -> None:
        self.iterations += 1
        now_ms = int(self.clock() * 1000)
        if self.kill_file.exists() and not self.risk.kill_switch_active:
            self.risk.trip(f"{self.kill_file} present")
            log(f"KILL: {self.kill_file} present; cancelling own orders and stopping")
            self._cancel_own()
            self.stopped = True
            return
        try:
            if self.contract is None:
                self.contract = self.venue.market(self.config.market_id)
            book = self.venue.book(self.config.market_id)
        except Exception as exc:
            self.errors += 1
            log(f"iteration {self.iterations}: read failed: {type(exc).__name__}: {exc}")
            return
        ledger = getattr(self.strategy, "ledger", None)
        position = ledger.position(self.config.market_id) if ledger else Decimal(0)
        avg_cost = ledger.avg_cost(self.config.market_id) if ledger else Decimal(0)
        context = Context(contract=self.contract, book=book, now_ms=now_ms, fair=self.fair(book),
                          reference=self.reference(), position=position, avg_cost=avg_cost,
                          own_open_orders=len(ledger.open_order_ids(self.config.market_id)) if ledger else 0)
        try:
            plan = self.strategy.plan(context)
        except Exception as exc:
            self.errors += 1
            log(f"iteration {self.iterations}: plan raised: {type(exc).__name__}: {exc}")
            return
        for warning in plan.warnings:
            self.log.write("warning", text=warning)
        if not plan.ok:
            self.log.write("refused", reasons=plan.reasons)
            return
        survivors = []
        for intent in list(plan.intents) + list(getattr(plan, "exits", [])):
            reduce_only = intent in getattr(plan, "exits", [])
            veto = self.risk.check_order(
                market_id=intent.contract.market_id, side=intent.side.value,
                outcome=intent.outcome.value, price=intent.price, size=intent.size,
                resolves_at_ms=intent.contract.resolves_at_ms, now_ms=now_ms, reduce_only=reduce_only)
            if veto:
                survivors.append(intent)
            else:
                self.vetoed += 1
                self.log.write("vetoed", market=intent.contract.market_id, reasons=veto.reasons)
        plan.intents = [i for i in survivors if i not in getattr(plan, "exits", [])]
        if hasattr(plan, "exits"):
            plan.exits = [i for i in survivors if i in plan.exits]
        if not survivors:
            return
        try:
            outcome = self.strategy.execute(plan, self.venue, dry_run=self.dry_run)
        except Exception as exc:
            self.errors += 1
            log(f"iteration {self.iterations}: execute raised: {type(exc).__name__}: {exc}")
            return
        for problem in outcome.problems:
            log(f"execute: {problem}")
        for order_id, intent in zip(outcome.order_ids, survivors):
            self.risk.open_orders[intent.contract.market_id] = \
                self.risk.open_orders.get(intent.contract.market_id, 0) + 1
            if self.dry_run:
                self._paper_orders[order_id] = intent
                self.paper.place(self._to_maker_order(order_id, intent, now_ms, book), book)
        if self.dry_run:
            self._paper_fill(book)
        if self.iterations % self.config.monitor_every == 0 and not self.dry_run:
            report = self.strategy.monitor(self.venue, self.config.market_id)
            for alert in report.alerts:
                log(f"monitor: {alert}")
            if report.critical:
                self.risk.trip("monitor critical: " + "; ".join(report.alerts))

    def _paper_fill(self, book: Book) -> None:
        """Fill paper orders from the live book, both bounds, and book the
        pessimistic bound into risk so exposure is never understated."""
        before = len(self.paper.fills["pessimistic"])
        self.paper.on_book(book)
        try:
            for trade in self.venue.trades(self.config.market_id, limit=20):
                self.paper.on_trade(trade)
        except Exception:
            pass
        for fill in self.paper.fills["pessimistic"][before:]:
            intent = self._paper_orders.get(fill.order_id)
            if intent is None:
                continue
            self.risk.on_fill(intent.contract.market_id, intent.side.value, intent.outcome.value,
                              intent.price, fill.size)
            self.risk.open_orders[intent.contract.market_id] = max(
                0, self.risk.open_orders.get(intent.contract.market_id, 0) - 1)
            self.log.write("paper_fill", order_id=fill.order_id, bound=fill.bound, cause=fill.cause,
                           price=str(fill.price), size=str(fill.size))

    def _cancel_own(self) -> None:
        cancel = getattr(self.strategy, "cancel_all", None)
        if cancel is not None:
            try:
                cancel(self.venue, dry_run=self.dry_run)
            except Exception as exc:
                log(f"cancel on kill failed: {exc}")
        for order_id in list(self._paper_orders):
            self.paper.cancel(order_id)

    async def run(self) -> None:
        self.lock.acquire()
        try:
            while not self.stopped:
                self.step()
                if 0 <= self.config.max_iterations <= self.iterations:
                    break
                await self.sleep(self.config.interval_s)
        finally:
            self.lock.release()

    def stats(self) -> Dict[str, Any]:
        return {"iterations": self.iterations, "errors": self.errors, "vetoed": self.vetoed,
                "paperFills": {k: len(v) for k, v in self.paper.fills.items()},
                "risk": self.risk.status(), "mode": "paper" if self.dry_run else "LIVE"}


def build_venue(name: str, *, keystore: Optional[Path], key_name: Optional[str], passphrase: Optional[str]):
    """The venue adapter, with a signer only if a key was named."""
    if name == "kalshi":
        from predkit.venues.kalshi import Kalshi, RsaPssSigner
        signer = None
        if key_name:
            from predkit.keys import Keystore
            store = Keystore(keystore)
            pem = store.load(key_name, passphrase or "")
            signer = RsaPssSigner(store.meta(key_name).get("key_id", ""), pem)
        return Kalshi(signer=signer)
    if name == "polymarket":
        from predkit.venues.polymarket import CLOB, Polymarket
        clob = None
        if key_name:
            from py_clob_client.client import ClobClient
            from predkit.keys import Keystore
            store = Keystore(keystore)
            private_key = store.load(key_name, passphrase or "").decode()
            clob = ClobClient(CLOB, key=private_key, chain_id=137)
            clob.set_api_creds(clob.create_or_derive_api_creds())
        return Polymarket(clob=clob)
    raise SystemExit(f"unknown venue {name!r}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strategy", required=True, choices=NAMES)
    parser.add_argument("--venue", required=True, choices=geofence.KNOWN_VENUES)
    parser.add_argument("--market", required=True)
    parser.add_argument("--residency", default="", help="ISO 3166-1 alpha-2, declared by the user")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--minutes", type=float, default=0, help="stop after this long; 0 = forever")
    parser.add_argument("--keystore", default=None)
    parser.add_argument("--key-name", default=None)
    parser.add_argument("--max-contracts", type=Decimal, default=Limits.max_contracts_per_market)
    parser.add_argument("--max-notional", type=Decimal, default=Limits.max_open_notional)
    parser.add_argument("--max-daily-loss", type=Decimal, default=Limits.max_daily_loss)
    args = parser.parse_args(argv)

    limits = Limits(max_contracts_per_market=args.max_contracts, max_open_notional=args.max_notional,
                    max_daily_loss=args.max_daily_loss)
    config = RunConfig(args.strategy, args.venue, args.market, residency=args.residency,
                       live=args.live, confirm=args.confirm, data_dir=Path(args.data_dir),
                       interval_s=args.interval, limits=limits,
                       max_iterations=int(args.minutes * 60 / args.interval) if args.minutes else -1)
    passphrase = os.getenv("PREDKIT_PASSPHRASE")
    if args.key_name and not passphrase:
        import getpass
        passphrase = getpass.getpass("keystore passphrase: ")
    venue = build_venue(args.venue, keystore=Path(args.keystore) if args.keystore else None,
                        key_name=args.key_name, passphrase=passphrase)
    can_sign = getattr(venue, "signer", None) is not None or getattr(venue, "clob", None) is not None
    reasons = refusals(config, venue_can_sign=can_sign)
    if reasons:
        raise SystemExit("refusing to start:\n  - " + "\n  - ".join(reasons))
    strategy = load(args.strategy)
    run_dir = market_dir(config.data_dir, config.venue, config.market_id) / "runs"
    stamp = time.strftime("%Y-%m-%dT%H%M%S", time.gmtime())
    mode = "live" if config.live and config.confirm else "paper"
    runner = Runner(config, venue, strategy, log_path=run_dir / f"{stamp}-{args.strategy}-{mode}.jsonl")
    strategy.log = runner.log
    log(f"{mode.upper()} {args.strategy} on {args.venue}:{args.market}; log {runner.log.path}")
    try:
        asyncio.run(supervise("runner", runner.run, max_restarts=0))
    except KeyboardInterrupt:
        runner._cancel_own()
    log(str(runner.stats()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
