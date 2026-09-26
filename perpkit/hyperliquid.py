"""Hyperliquid: its market data, and the liquidation levels it publishes.

Why a second venue, and why this one
------------------------------------
The other recorders here capture BloFin, and that venue cannot show anybody
else's positions. A liquidation map there
is a reconstruction from open interest through an assumed leverage mix, with
no ground truth to score it against.

Hyperliquid's ledger is public. Any account's positions can be read, and each
comes back with the exchange's own `liquidationPx`. So the levels are not
modelled here - they are collected. The hard question moves from "what
leverage do other traders use", which has no answer, to "which accounts
exist", which does:

    every trade on the `trades` channel names BOTH accounts,
    `users: [buyer, seller]`.

Measured on BTC, ETH, SOL and HYPE: 449 distinct accounts in 60
seconds of trades; 58 of 60 sampled held a position and 57 held one in those
four coins; 115 of 121 of those positions were cross margin.

What lands where
----------------
Market data, one websocket, archived verbatim per coin:

    data/hyperliquid/<COIN>/raw/<day>/{l2Book,trades,activeAssetCtx}-HH.jsonl.gz

  * `l2Book` - 20 levels a side, a full snapshot every ~5.3s (measured). Not a
    diff stream, so there is no sequence to lose and nothing to resync.
  * `trades` - every fill, both addresses included.
  * `activeAssetCtx` - mark, oracle, funding, premium and open interest, about
    once a second. Everything BloFin needs two REST pollers for, pushed.

Account state, REST, archived verbatim for every account read:

    data/hyperliquid/_accounts/raw/<day>/clearinghouseState-HH.jsonl.gz

Per account rather than per coin because that is what the response describes:
a cross-margin liquidation price depends on everything else the account
holds, so the whole account is kept, untracked coins included. A read that
changed nothing is written as `{"user", "unchanged": true, "time"}` rather
than skipped - "looked, and it was the same" bounds when a position closed,
and silence cannot.

Capture-or-lose, again
----------------------
`clearinghouseState` returns the present only. Sizes could be partly rebuilt
from fills later, but a past liquidation PRICE depends on the account's margin
and every other position's mark at that instant, and nothing serves it. Same
asymmetry as the order book: cheap now, unrecoverable after.

The rate budget is the real constraint
--------------------------------------
REST is 1,200 weight a minute per IP, and `clearinghouseState` weighs 2
(Hyperliquid docs, "Rate limits and user limits"). At the
default budget of 900 that is 450 reads a minute - against ~450 NEW accounts a
minute on four coins at the start of a run. Discovery outruns reading, so what
gets read next is most of the design:

  1. **An account that just traded.** A position's size can only change
     through a fill, and every fill on a tracked coin names both parties, so
     the trade feed is not only discovery, it is change notification. A
     re-read is held back `min_repoll_seconds` after the last one, so a market
     maker filling every block cannot spend the budget on itself.
  2. **A holder whose reading has aged.** Size is frozen between fills but a
     CROSS liquidation price is not: it moves with the account's other
     positions and with funding. Staleness is scored `age * sqrt(notional)`,
     so large positions refresh often and small ones are never starved. One
     read in `refresh_every` goes here whenever both queues have work.
  3. **An account holding no tracked coin is never refreshed.** It cannot
     open a position without trading, and trading puts it back in (1).

The docs do not state the status code for exceeding the limit. 429 is what
HTTP means by it, and is taken to mean something else on this IP is spending
the same budget: the bucket is emptied for a minute. Every other failure is
counted separately, so an unexpected code shows up in the counts rather than
hiding inside a retry.

One websocket, not one per coin
-------------------------------
BloFin gets a connection per instrument so a desync on one cannot touch
another. That reason does not carry over - `l2Book` is a snapshot, there is
nothing to desync - and Hyperliquid caps an IP at 10 websocket connections,
which a per-coin layout would exhaust at ten coins. The silence watchdog still
applies: `activeAssetCtx` arrives about once a second per coin, so 30 seconds
of nothing is a dead feed, not a quiet one. The server also drops a connection
it has not sent anything to for 60 seconds, and the documented `{"method":
"ping"}` goes out on a timer regardless.

Kept apart from BloFin's data
-----------------------------
`data/hyperliquid/BTC/` and `data/BTC-USDT/` hold similar-looking events about
different markets. `perpkit/layout.py` only recognises `BASE-QUOTE`
directories directly under `data/`, and `hyperliquid` is not one, so no BloFin
tool picks these up by accident. Combining venues is something to do on
purpose, with the venue in a column.

No SDK
------
The info API is public and unsigned. `urllib` for REST, and `websockets`
imported inside `run()`, keep this module importable and testable with no
network - the same rule `openinterest.py` follows.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

from .liquidation_map import LiquidationMap, TrackedPosition, positions_from_state
from .openinterest import _lock_owner, _pid_alive
from .rawlog import RawEventLog

INFO_URL = "https://api.hyperliquid.xyz/info"
WS_URL = "wss://api.hyperliquid.xyz/ws"

VENUE_DIR = "hyperliquid"
# No Hyperliquid coin name contains an underscore, so this cannot collide with
# a coin's directory beside it.
ACCOUNTS_DIR = "_accounts"
ADDRESS_BOOK = "addresses.json"

MARKET_CHANNELS = ("l2Book", "trades", "activeAssetCtx")
POSITIONS_CHANNEL = "clearinghouseState"

# Hyperliquid docs, "Rate limits and user limits".
WEIGHT_LIMIT_PER_MINUTE = 1200
WEIGHT_CLEARINGHOUSE_STATE = 2
WEIGHT_META_AND_ASSET_CTXS = 20
MAX_WS_SUBSCRIPTIONS = 1000

# 75% of the limit. The remainder is headroom for a `--check`, another tool,
# or anything else on this IP, because the limit is per IP and not per process.
DEFAULT_WEIGHT_PER_MINUTE = 900

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_STALL_TIMEOUT_S = 30.0
# Half the server's 60s idle cutoff.
DEFAULT_PING_SECONDS = 30.0

USER_AGENT = "hyperliquid-recorder/1.0"
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}")


def venue_dir(data_dir: Path) -> Path:
    return Path(data_dir) / VENUE_DIR


def coin_dir(data_dir: Path, coin: str) -> Path:
    return venue_dir(data_dir) / coin


def accounts_dir(data_dir: Path) -> Path:
    return venue_dir(data_dir) / ACCOUNTS_DIR


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


def post_info(
    body: Dict[str, Any],
    *,
    url: str = INFO_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., Any]] = None,
) -> Any:
    """One request to the public info endpoint. Raises on transport failure."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    fetch = opener or urllib.request.urlopen
    with fetch(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_universe(
    *,
    url: str = INFO_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Every perp's metadata and live context, by coin name.

    `metaAndAssetCtxs` returns the two as parallel lists; they are paired by
    index here so nothing downstream has to trust that alignment again.
    """
    payload = post_info({"type": "metaAndAssetCtxs"}, url=url,
                        timeout=timeout, opener=opener)
    if not (isinstance(payload, list) and len(payload) == 2
            and isinstance(payload[0], dict)):
        raise ValueError("unexpected metaAndAssetCtxs payload")
    universe = payload[0].get("universe") or []
    contexts = payload[1] if isinstance(payload[1], list) else []
    assets: Dict[str, Dict[str, Any]] = {}
    ctxs: Dict[str, Dict[str, Any]] = {}
    for index, asset in enumerate(universe):
        name = (asset or {}).get("name")
        if not name:
            continue
        assets[name] = asset
        if index < len(contexts) and isinstance(contexts[index], dict):
            ctxs[name] = contexts[index]
    return assets, ctxs


def validate_coins(coins: Sequence[str],
                   assets: Dict[str, Dict[str, Any]]) -> List[str]:
    """Every reason these coins cannot be recorded. Plural, not the first.

    Names are case-sensitive (`kPEPE`), and Windows paths are not, so two
    requested names differing only by case would share a directory - the
    exact merge `record.py` exists to prevent, arriving by filesystem instead.
    """
    problems: List[str] = []
    by_fold: Dict[str, List[str]] = {}
    for name in assets:
        by_fold.setdefault(name.casefold(), []).append(name)

    seen: Dict[str, str] = {}
    for coin in coins:
        folded = coin.casefold()
        if folded in seen and seen[folded] != coin:
            problems.append(
                f"{seen[folded]} and {coin} differ only by case, and would "
                f"share one directory on Windows.")
        seen.setdefault(folded, coin)

        asset = assets.get(coin)
        if asset is None:
            near = [name for name in by_fold.get(folded, []) if name != coin]
            hint = (f" - did you mean {', '.join(near)}? Names are "
                    f"case-sensitive." if near else "")
            problems.append(f"{coin} is not a Hyperliquid perp{hint}")
            continue
        if asset.get("isDelisted"):
            problems.append(f"{coin} is delisted: no book and no trades to "
                            f"record.")
    return problems


class WeightBudget:
    """A token bucket over Hyperliquid's per-IP request weight.

    The capacity is a few seconds of budget rather than a whole minute, so a
    restart cannot open with a burst the exchange sees as the whole minute's
    allowance spent at once.
    """

    def __init__(
        self,
        per_minute: float = DEFAULT_WEIGHT_PER_MINUTE,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ):
        if not 0 < per_minute <= WEIGHT_LIMIT_PER_MINUTE:
            raise ValueError(
                f"weight per minute must be in (0, {WEIGHT_LIMIT_PER_MINUTE}]"
                f", got {per_minute}")
        self.per_minute = float(per_minute)
        self.capacity = max(float(WEIGHT_META_AND_ASSET_CTXS),
                            self.per_minute / 12)
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated = clock()
        self.spent = 0
        self.penalties = 0

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(
            self.capacity,
            self._tokens + (now - self._updated) * self.per_minute / 60.0)
        self._updated = now

    def try_take(self, weight: float) -> float:
        """Take `weight` if available and return 0; else return seconds to wait."""
        self._refill()
        # A tolerance, because the refill is float arithmetic. Without it a
        # deficit of 1e-15 tokens asks for a sleep too short to move the
        # clock, and the caller spins forever without ever being granted.
        if self._tokens >= weight - 1e-9:
            self._tokens -= weight
            self.spent += weight
            return 0.0
        return max((weight - self._tokens) * 60.0 / self.per_minute, 1e-3)

    async def acquire(self, weight: float) -> None:
        while True:
            wait = self.try_take(weight)
            if wait <= 0:
                return
            await self._sleep(wait)

    def penalise(self, seconds: float = 60.0) -> None:
        """Stop spending for `seconds`: our count and the exchange's disagree."""
        self._refill()
        self._tokens = min(self._tokens, 0.0) - seconds * self.per_minute / 60.0
        self.penalties += 1


class ExclusiveLock:
    """One owner per path, the same pid-file rule the snapshot pollers use.

    A second recorder would archive every event twice. (Two writers on one
    hourly gzip file can destroy it outright - 0 of 40 records survived in
    one test, see `openinterest.py` - which is why `RawEventLog` creates its
    files exclusively.) This process writes a file per coin per
    channel plus the accounts archive, so the whole venue directory has
    exactly one owner.
    """

    def __init__(self, path: Path, *, holder: str):
        self.path = Path(path)
        self.holder = holder
        self.held = False

    def acquire(self) -> "ExclusiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner = _lock_owner(self.path)
        if owner is not None and _pid_alive(owner, holder=self.holder):
            raise SystemExit(
                f"{self.holder} is already running as pid {owner} "
                f"({self.path}).\nA second one would archive every event "
                f"twice. Stop that process first.")
        if owner is not None:
            self.path.unlink(missing_ok=True)   # stale: the owner is gone
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(
                f"{self.path} exists but names no process. If no "
                f"{self.holder} is running, delete it and start again.")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.held = True
        return self

    def release(self) -> None:
        if self.held and _lock_owner(self.path) == os.getpid():
            self.path.unlink(missing_ok=True)
        self.held = False


# ---------------------------------------------------------------------------
# The market feed
# ---------------------------------------------------------------------------


def _coin_of(channel: Any, data: Any) -> Optional[str]:
    if channel == "trades":
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0].get("coin")
        return None
    if isinstance(data, dict):
        return data.get("coin")
    return None


class HyperliquidMarketFeed:
    """Book, trades and asset context for a set of coins, over one socket.

    Archives first and parses second, like `MicrostructureFeed`: a bug in a
    consumer of the trade users can never cost the event that exposed it.
    """

    def __init__(
        self,
        coins: Sequence[str],
        *,
        data_dir: Path,
        record_raw: bool = True,
        channels: Sequence[str] = MARKET_CHANNELS,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        ping_seconds: float = DEFAULT_PING_SECONDS,
        url: str = WS_URL,
        on_trade_users: Optional[Callable[[str, Sequence[str], int], Any]] = None,
        on_log: Callable[[str], Any] = print,
        connect: Optional[Callable[..., Any]] = None,
    ):
        self.coins = list(dict.fromkeys(coins))
        self.channels = tuple(channels)
        unknown = set(self.channels) - set(MARKET_CHANNELS)
        if unknown:
            raise ValueError(f"unknown channel(s): {', '.join(sorted(unknown))}")
        if len(self.coins) * len(self.channels) > MAX_WS_SUBSCRIPTIONS:
            raise ValueError(
                f"{len(self.coins)} coins x {len(self.channels)} channels "
                f"exceeds Hyperliquid's {MAX_WS_SUBSCRIPTIONS} subscriptions "
                f"per IP")
        self.data_root = Path(data_dir)
        self.stall_timeout_s = stall_timeout_s
        self.ping_seconds = ping_seconds
        self.url = url
        self.on_trade_users = on_trade_users
        self._log = on_log
        self._connect = connect

        self.logs: Dict[str, RawEventLog] = {
            coin: RawEventLog(coin_dir(self.data_root, coin),
                              enabled=record_raw, channels=set(self.channels))
            for coin in self.coins
        }
        # coin -> latest parsed activeAssetCtx, with its local receive time.
        self.contexts: Dict[str, Dict[str, Any]] = {}

        self.connected = False
        self.messages = 0
        self.control_messages = 0
        self.unrouted = 0
        self.errors = 0
        self.reconnects = 0
        self.stalls = 0
        self.pings = 0
        self.last_error: Optional[str] = None
        self._last_message = time.monotonic()

    # ---- messages --------------------------------------------------------

    def handle_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            self.unrouted += 1
            return
        channel = message.get("channel")
        data = message.get("data")
        if channel in ("pong", "subscriptionResponse"):
            self.control_messages += 1
            return
        if channel == "error":
            self.errors += 1
            self.last_error = str(data)[:300]
            self._log(f"[hyperliquid] server error: {self.last_error}")
            return

        coin = _coin_of(channel, data)
        log = self.logs.get(coin) if channel in self.channels else None
        if log is None:
            self.unrouted += 1
            return

        self.messages += 1
        log.write(channel, message)   # archive first, parse second

        try:
            if channel == "trades":
                self._on_trades(data)
            elif channel == "activeAssetCtx":
                self._on_context(coin, data)
        except Exception as exc:  # noqa: BLE001 - the archive already has it
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._log(f"[hyperliquid] consumer error on {channel}: "
                      f"{self.last_error}")

    def _on_trades(self, data: Any) -> None:
        if self.on_trade_users is None:
            return
        for trade in data:
            if not isinstance(trade, dict):
                continue
            users = trade.get("users")
            if users:
                self.on_trade_users(trade.get("coin"), users,
                                    int(trade.get("time") or 0))

    def _on_context(self, coin: str, data: Dict[str, Any]) -> None:
        ctx = data.get("ctx") or {}
        self.contexts[coin] = {
            "markPx": _to_float(ctx.get("markPx")),
            "oraclePx": _to_float(ctx.get("oraclePx")),
            "openInterest": _to_float(ctx.get("openInterest")),
            "funding": _to_float(ctx.get("funding")),
            "received": time.time(),
        }

    def _context(self, coin: str, max_age_s: Optional[float]) -> Optional[Dict[str, Any]]:
        ctx = self.contexts.get(coin)
        if ctx is None:
            return None
        if max_age_s is not None and time.time() - ctx["received"] > max_age_s:
            return None
        return ctx

    def mark(self, coin: str, *, max_age_s: Optional[float] = None) -> Optional[float]:
        ctx = self._context(coin, max_age_s)
        return None if ctx is None else ctx["markPx"]

    def open_interest(self, coin: str, *,
                      max_age_s: Optional[float] = None) -> Optional[float]:
        ctx = self._context(coin, max_age_s)
        return None if ctx is None else ctx["openInterest"]

    # ---- the loop --------------------------------------------------------

    async def run(self) -> None:
        """Connect, subscribe, stream; reconnect forever. Run under `supervise`."""
        connect = self._connect
        if connect is None:
            import websockets  # here, so the module imports without it

            connect = websockets.connect
        backoff = 1.0
        while True:
            try:
                async with connect(self.url, max_size=8 * 2 ** 20) as ws:
                    for coin in self.coins:
                        for channel in self.channels:
                            await ws.send(json.dumps({
                                "method": "subscribe",
                                "subscription": {"type": channel, "coin": coin},
                            }))
                    self.connected = True
                    backoff = 1.0
                    self._log(f"[hyperliquid] connected: {len(self.coins)} "
                              f"coin(s) x [{', '.join(self.channels)}]")
                    await self._stream(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect is the answer
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._log(f"[hyperliquid] feed error: {self.last_error} - "
                          f"reconnecting in {backoff:g}s")
            finally:
                self.connected = False
                self.reconnects += 1
                self.flush()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _stream(self, ws: Any) -> None:
        """Pump messages until the socket closes or goes silent.

        Same failure `ingest._stream` exists for: an open socket that stops
        delivering raises nothing, so `supervise` would call it healthy while
        the archive writes nothing. The deadline is per message, never per
        stream. Unlike there, a timeout here may only mean a ping is due, so
        the read is resumed afterwards - which `websockets` documents as safe:
        a cancelled `recv()` loses no message.
        """
        self._last_message = time.monotonic()
        next_ping = self._last_message + self.ping_seconds
        while True:
            now = time.monotonic()
            silent_until = self._last_message + self.stall_timeout_s
            if now >= silent_until:
                self.stalls += 1
                self._log(f"[hyperliquid] feed stalled: nothing in "
                          f"{self.stall_timeout_s:g}s (socket still open) - "
                          f"reconnecting.")
                return
            if now >= next_ping:
                await ws.send(json.dumps({"method": "ping"}))
                self.pings += 1
                next_ping = now + self.ping_seconds
            timeout = max(0.0, min(silent_until, next_ping) - now)
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            self._last_message = time.monotonic()
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                self.unrouted += 1
                continue
            self.handle_message(message)

    # ---- lifecycle -------------------------------------------------------

    def flush(self) -> None:
        for log in self.logs.values():
            log.flush()

    def close(self) -> None:
        for log in self.logs.values():
            log.close()

    def summary_line(self) -> str:
        return (f"[market] {'connected' if self.connected else 'DISCONNECTED'}"
                f"; {self.messages:,} messages, {self.reconnects} reconnect(s),"
                f" {self.stalls} stall(s), {self.errors} error(s)")


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@dataclass
class AccountEntry:
    last_polled: Optional[float] = None
    notional: float = 0.0            # |value| across tracked coins, at last read
    fingerprint: Optional[int] = None
    queued: bool = False
    in_flight: bool = False
    redirty: bool = False            # traded while being read


def _fingerprint(state: Dict[str, Any]) -> int:
    """What would make a new read worth archiving in full.

    Size, entry, liquidation price and leverage, across ALL of the account's
    positions: opening an untracked coin moves a cross liquidation price, and
    an isolated top-up moves that position's own.
    """
    rows = []
    for item in state.get("assetPositions") or []:
        position = (item or {}).get("position") or {}
        leverage = position.get("leverage") or {}
        rows.append(tuple(str(value) for value in (
            position.get("coin"), position.get("szi"), position.get("entryPx"),
            position.get("liquidationPx"), leverage.get("type"),
            leverage.get("value"), leverage.get("rawUsd"),
        )))
    return hash(tuple(sorted(rows)))


class PositionTracker:
    """Discovers accounts from trades and keeps their positions current.

    `positions[coin][user]` is the latest reading of every tracked position,
    which is all `liquidation_map.build_map` needs. See the module docstring
    for the order in which accounts are read and why.
    """

    def __init__(
        self,
        coins: Sequence[str],
        *,
        data_dir: Path,
        budget: WeightBudget,
        url: str = INFO_URL,
        opener: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        archive: bool = True,
        persist: bool = True,
        workers: int = 3,
        min_repoll_seconds: float = 30.0,
        min_refresh_seconds: float = 60.0,
        refresh_every: int = 4,
        refresh_rebuild_seconds: float = 5.0,
        retry_seconds: float = 10.0,
        idle_seconds: float = 0.25,
        persist_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        on_log: Callable[[str], Any] = print,
    ):
        self.coins: Set[str] = set(coins)
        self.root = accounts_dir(data_dir)
        self.budget = budget
        self.url = url
        self.opener = opener
        self.timeout_seconds = timeout_seconds
        self.persist = persist
        self.workers = workers
        self.min_repoll_seconds = min_repoll_seconds
        self.min_refresh_seconds = min_refresh_seconds
        self.refresh_every = max(1, refresh_every)
        self.refresh_rebuild_seconds = refresh_rebuild_seconds
        self.retry_seconds = retry_seconds
        self.idle_seconds = idle_seconds
        self.persist_seconds = persist_seconds
        self._clock = clock
        self._log = on_log

        self.log = RawEventLog(self.root, enabled=archive,
                               channels={POSITIONS_CHANNEL})

        self.positions: Dict[str, Dict[str, TrackedPosition]] = {
            coin: {} for coin in self.coins}
        self.accounts: Dict[str, AccountEntry] = {}
        self._holders: Set[str] = set()

        # (eligible_at, seq, user). seq keeps insertion order among equals,
        # which is what makes a restored address book come back largest first.
        self._dirty: List[Tuple[float, int, str]] = []
        self._seq = itertools.count()
        self._refresh_order: Deque[str] = deque()
        self._refresh_built = -math.inf
        self._slot = 0

        self.discovered = 0
        self.polls = 0
        self.changed = 0
        self.unchanged = 0
        self.failures = 0
        self.rate_limited = 0
        self._last_failure_log = -math.inf

        self.restored = self._load_addresses() if persist else 0

    # ---- what to read next -----------------------------------------------

    def observe_trade(self, coin: Any, users: Sequence[Any], trade_ms: int = 0) -> None:
        """Both parties to a fill: their size may have just changed."""
        now = self._clock()
        for user in users:
            if not isinstance(user, str) or not ADDRESS.fullmatch(user):
                continue
            entry = self.accounts.get(user)
            if entry is None:
                entry = self.accounts[user] = AccountEntry()
                self.discovered += 1
            self._mark_dirty(user, entry, now)

    def _mark_dirty(self, user: str, entry: AccountEntry, now: float,
                    delay: float = 0.0) -> None:
        if entry.in_flight:
            entry.redirty = True
            return
        if entry.queued:
            return
        eligible = now + delay
        if entry.last_polled is not None:
            eligible = max(eligible, entry.last_polled + self.min_repoll_seconds)
        entry.queued = True
        heapq.heappush(self._dirty, (eligible, next(self._seq), user))

    def _refresh_eligible(self, entry: Optional[AccountEntry], now: float) -> bool:
        return (entry is not None and entry.notional > 0 and not entry.queued
                and not entry.in_flight
                and (entry.last_polled is None
                     or now - entry.last_polled >= self.min_refresh_seconds))

    def _peek_refresh(self, now: float) -> Optional[str]:
        # Rebuilt on a timer, not whenever empty: an empty candidate list
        # would otherwise rescan every holder on every single read.
        if now - self._refresh_built >= self.refresh_rebuild_seconds:
            scored = []
            for user in self._holders:
                entry = self.accounts[user]
                if not self._refresh_eligible(entry, now):
                    continue
                age = (math.inf if entry.last_polled is None
                       else now - entry.last_polled)
                scored.append((-age * math.sqrt(entry.notional), user))
            scored.sort()
            self._refresh_order = deque(user for _, user in scored)
            self._refresh_built = now
        while self._refresh_order:
            user = self._refresh_order[0]
            if self._refresh_eligible(self.accounts.get(user), now):
                return user
            self._refresh_order.popleft()
        return None

    def next_user(self, now: Optional[float] = None) -> Optional[str]:
        """The account to read next, marked in flight; None if nothing is due."""
        now = self._clock() if now is None else now
        while True:
            dirty_ready = bool(self._dirty) and self._dirty[0][0] <= now
            refresh_user = self._peek_refresh(now)
            if not dirty_ready and refresh_user is None:
                return None

            self._slot = (self._slot + 1) % self.refresh_every
            wants_refresh = self._slot == 0
            if dirty_ready and (not wants_refresh or refresh_user is None):
                _, _, user = heapq.heappop(self._dirty)
                entry = self.accounts[user]
                entry.queued = False
                if entry.in_flight:
                    entry.redirty = True
                    continue
            else:
                user = refresh_user
                self._refresh_order.popleft()
                entry = self.accounts[user]
            entry.in_flight = True
            return user

    # ---- reading ---------------------------------------------------------

    def fetch_state(self, user: str) -> Dict[str, Any]:
        state = post_info({"type": POSITIONS_CHANNEL, "user": user},
                          url=self.url, timeout=self.timeout_seconds,
                          opener=self.opener)
        if not isinstance(state, dict) or "assetPositions" not in state:
            raise ValueError(f"unexpected {POSITIONS_CHANNEL} response: "
                             f"{str(state)[:200]}")
        return state

    def apply(self, user: str, state: Dict[str, Any],
              now: Optional[float] = None) -> None:
        """File one account's reading: positions by coin, and the archive."""
        now = self._clock() if now is None else now
        entry = self.accounts.get(user)
        if entry is None:
            entry = self.accounts[user] = AccountEntry()
        entry.in_flight = False
        entry.last_polled = now
        self.polls += 1

        observed_ms = int(_to_float(state.get("time")) or now * 1000)
        held = {position.coin: position for position in positions_from_state(
            user, state, observed_ms=observed_ms, coins=self.coins)}
        for coin, book in self.positions.items():
            if coin in held:
                book[user] = held[coin]
            else:
                book.pop(user, None)
        entry.notional = sum(position.position_value for position in held.values())
        if entry.notional > 0:
            self._holders.add(user)
        else:
            self._holders.discard(user)

        fingerprint = _fingerprint(state)
        if fingerprint == entry.fingerprint:
            self.unchanged += 1
            self.log.write(POSITIONS_CHANNEL, {
                "user": user, "unchanged": True, "time": state.get("time")})
        else:
            entry.fingerprint = fingerprint
            self.changed += 1
            self.log.write(POSITIONS_CHANNEL, {"user": user, "data": state})

        if entry.redirty:
            entry.redirty = False
            self._mark_dirty(user, entry, now)

    def _on_failure(self, user: str, exc: BaseException) -> None:
        self.failures += 1
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
            self.rate_limited += 1
            self.budget.penalise()
        entry = self.accounts.get(user)
        if entry is not None:
            entry.in_flight = False
            entry.redirty = False
            self._mark_dirty(user, entry, self._clock(), delay=self.retry_seconds)
        now = time.monotonic()
        if now - self._last_failure_log >= 30.0:
            self._last_failure_log = now
            self._log(f"[positions] read failed: {type(exc).__name__}: {exc} "
                      f"({self.failures:,} failed so far, "
                      f"{self.rate_limited:,} rate-limited)")

    async def poll_next(self) -> bool:
        """Read one account if one is due. Returns False if none was.

        Never raises on a failed read: a tracker that dies on a dropped
        connection is one that silently stops tracking.
        """
        user = self.next_user()
        if user is None:
            return False
        try:
            await self.budget.acquire(WEIGHT_CLEARINGHOUSE_STATE)
            state = await asyncio.to_thread(self.fetch_state, user)
        except asyncio.CancelledError:
            self._recover(user)
            raise
        except Exception as exc:  # noqa: BLE001
            self._on_failure(user, exc)
            return True
        try:
            self.apply(user, state)
        except Exception as exc:  # noqa: BLE001
            self._on_failure(user, exc)
        return True

    def _recover(self, user: str) -> None:
        entry = self.accounts.get(user)
        if entry is not None and entry.in_flight:
            entry.in_flight = False
            entry.redirty = False
            self._mark_dirty(user, entry, self._clock())

    def recover_in_flight(self) -> int:
        """Requeue reads a previous `run()` left half-done. Returns how many."""
        stranded = [user for user, entry in self.accounts.items()
                    if entry.in_flight]
        for user in stranded:
            self._recover(user)
        return len(stranded)

    async def _worker(self) -> None:
        while True:
            try:
                busy = await self.poll_next()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never end the run
                self._log(f"[positions] worker error: {type(exc).__name__}: {exc}")
                busy = False
            if not busy:
                await asyncio.sleep(self.idle_seconds)

    async def _persist_loop(self) -> None:
        while True:
            await asyncio.sleep(self.persist_seconds)
            try:
                self.save_addresses()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[positions] address book not saved: {exc}")

    async def run(self) -> None:
        """Read accounts forever. Run under `supervise`."""
        self.recover_in_flight()
        await asyncio.gather(self._persist_loop(),
                             *(self._worker() for _ in range(self.workers)))

    # ---- the address book ------------------------------------------------

    def save_addresses(self) -> None:
        """Accounts holding a tracked coin, so a restart is not a cold start.

        Written atomically: a torn address book would cost every account
        discovered so far, which at the measured rate is hours of reads.
        """
        if not self.persist:
            return
        path = self.root / ADDRESS_BOOK
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_ms": int(self._clock() * 1000),
            "coins": sorted(self.coins),
            "holders": {user: round(self.accounts[user].notional, 2)
                        for user in sorted(self._holders)},
        }
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)

    def _load_addresses(self) -> int:
        path = self.root / ADDRESS_BOOK
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return 0
        except (OSError, ValueError) as exc:
            self._log(f"[positions] address book unreadable ({exc}) - "
                      f"discovering from zero.")
            return 0
        holders = payload.get("holders") if isinstance(payload, dict) else None
        ranked = []
        for user, notional in (holders or {}).items():
            value = _to_float(notional)
            if isinstance(user, str) and ADDRESS.fullmatch(user) and value:
                ranked.append((value, user))
        ranked.sort(reverse=True)
        now = self._clock()
        for notional, user in ranked:
            # Queued, largest first. The positions themselves are NOT
            # restored: a reading hours old is exactly the stale level the
            # map must not present as current, so each is read again.
            entry = self.accounts[user] = AccountEntry(notional=notional)
            self._holders.add(user)
            self._mark_dirty(user, entry, now)
        return len(ranked)

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        try:
            self.save_addresses()
        finally:
            self.log.close()

    def summary_line(self) -> str:
        return (f"[positions] {len(self.accounts):,} accounts known, "
                f"{len(self._holders):,} holding a tracked coin; "
                f"{self.polls:,} reads ({self.changed:,} changed, "
                f"{self.unchanged:,} unchanged); {len(self._dirty):,} queued; "
                f"{self.failures:,} failed ({self.rate_limited:,} rate-limited)")


# ---------------------------------------------------------------------------
# The derived map, on disk and on the console
# ---------------------------------------------------------------------------


class MapWriter:
    """One JSON line per map, per coin per UTC day, beside the coin's raw/.

    Derived, and regenerable from the two archives. Plain JSONL rather than
    gzip because it is small - one line a minute - and meant to be read while
    the recorder is still writing it.
    """

    def __init__(self, data_dir: Path, *, enabled: bool = True):
        self.data_root = Path(data_dir)
        self.enabled = enabled
        self._handles: Dict[str, Any] = {}
        self._days: Dict[str, str] = {}
        self.lines_written = 0

    def path_for(self, coin: str, day: str) -> Path:
        return coin_dir(self.data_root, coin) / f"liquidation-levels-{day}.jsonl"

    def write(self, result: LiquidationMap) -> None:
        if not self.enabled:
            return
        day = datetime.fromtimestamp(result.as_of_ms / 1000,
                                     tz=timezone.utc).strftime("%Y-%m-%d")
        coin = result.coin
        if self._days.get(coin) != day:
            old = self._handles.pop(coin, None)
            if old is not None:
                old.close()
            path = self.path_for(coin, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[coin] = open(path, "a", encoding="utf-8")
            self._days[coin] = day
        handle = self._handles[coin]
        handle.write(json.dumps(result.to_dict(), separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        self.lines_written += 1

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - shutdown path
                pass
        self._handles.clear()
        self._days.clear()


def _usd(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


def summarise(result: LiquidationMap) -> str:
    """One console line per coin. ASCII only: a Windows console is cp1252."""
    coverage = result.coverage()

    def share(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    def unpriced(side) -> str:
        if not side.size:
            return "n/a"
        return f"{side.no_liquidation_size / side.size * 100:.0f}%"

    bands = "  ".join(
        f"{pct * 100:g}%: {_usd(result.longs.within[pct])} down / "
        f"{_usd(result.shorts.within.get(pct, 0.0))} up"
        for pct in sorted(result.longs.within))
    age = ("n/a" if result.weighted_age_s is None
           else f"{result.weighted_age_s:.0f}s")
    return (f"{result.coin:<6} mark {result.mark_px:,.6g}  "
            f"accounts {result.accounts:,}  coverage L "
            f"{share(coverage['long'])} S {share(coverage['short'])}  | "
            f"{bands}  | no liq px L {unpriced(result.longs)} S "
            f"{unpriced(result.shorts)}  age {age}")
