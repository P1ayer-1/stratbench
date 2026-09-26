"""Record Hyperliquid, and track where its positions are liquidated.

    python -m perpkit.record_hyperliquid
    python -m perpkit.record_hyperliquid --coins BTC,ETH,SOL,HYPE
    python -m perpkit.record_hyperliquid --coins BTC --no-positions
    python -m perpkit.record_hyperliquid --check --coins BTC,kPEPE

A separate venue gets a separate process. Nothing is shared with `record.py`:
different directories, a different lock, a different exchange's rate limit.
Starting or stopping this touches no BloFin run.

What lands where
----------------
    data/hyperliquid/<COIN>/raw/               l2Book, trades, activeAssetCtx   irreplaceable
    data/hyperliquid/_accounts/raw/            clearinghouseState per account   irreplaceable
    data/hyperliquid/_accounts/addresses.json  accounts holding a tracked coin  restart state
    data/hyperliquid/<COIN>/liquidation-levels-<day>.jsonl  one map a minute   derived

Reading the map
---------------
Each interval, one console line per coin:

    BTC    mark 77,130  accounts 812  coverage L 18.2% S 21.0%  | 1%: $4.1M down / $3.2M up ...

`1%: $4.1M down` is the notional of tracked LONGS whose own exchange-reported
liquidation price lies within a 1% fall; `up` is shorts within a 1% rise.

Coverage is tracked size against open interest, and it starts near zero:
accounts are only known once they trade, so the map fills in over hours. Read
every number through the coverage printed beside it, never on its own. A
restart does not reset it - accounts holding a tracked coin are saved and
re-read first, largest first.

See `perpkit/hyperliquid.py` for what is read in what order and why, and
`perpkit/liquidation_map.py` for what a map does and does not claim.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from perpkit.config import (
    DATA_DIR,
    FEED_STALL_TIMEOUT_S,
    HYPERLIQUID_BUCKET_PCT,
    HYPERLIQUID_COINS,
    HYPERLIQUID_MAP_SECONDS,
    HYPERLIQUID_WEIGHT_PER_MINUTE,
    RECORD_RAW,
)
from perpkit.supervise import log, supervise
from perpkit.hyperliquid import (
    MARKET_CHANNELS,
    POSITIONS_CHANNEL,
    WEIGHT_CLEARINGHOUSE_STATE,
    WEIGHT_LIMIT_PER_MINUTE,
    ExclusiveLock,
    HyperliquidMarketFeed,
    MapWriter,
    PositionTracker,
    WeightBudget,
    accounts_dir,
    coin_dir,
    fetch_universe,
    summarise,
    validate_coins,
    venue_dir,
)
from perpkit.liquidation_map import build_map

# The token a live recorder's command line must contain for its lock to count
# as held. Matches `python -m perpkit.record_hyperliquid` and a script path
# ending in record_hyperliquid.py alike.
LOCK_HOLDER = "record_hyperliquid"

# A map drawn against a mark this old describes a market that has moved on.
MARK_MAX_AGE_S = 60.0


def _number(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _usd(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value / 1e3:.0f}k"


def report(coins: List[str], assets: Dict[str, Dict[str, Any]],
           contexts: Dict[str, Dict[str, Any]], *, weight_per_minute: int,
           positions: bool) -> None:
    print(f"\n{len(coins)} Hyperliquid perp(s), live:")
    print(f"  {'coin':<10}{'max lev':>8}{'mark':>14}{'open interest':>16}"
          f"{'24h volume':>13}")
    for coin in coins:
        ctx = contexts.get(coin, {})
        mark = _number(ctx.get("markPx"))
        interest = _number(ctx.get("openInterest"))
        print(f"  {coin:<10}{assets[coin].get('maxLeverage', '?'):>8}"
              f"{mark if mark is not None else float('nan'):>14,.6g}"
              f"{_usd(interest * mark if interest and mark else None):>16}"
              f"{_usd(_number(ctx.get('dayNtlVlm'))):>13}")

    if not positions:
        print("\n--no-positions: market data only, so no liquidation map.")
        return
    reads = weight_per_minute / WEIGHT_CLEARINGHOUSE_STATE
    print(f"\nAccount reads: {weight_per_minute} of {WEIGHT_LIMIT_PER_MINUTE} "
          f"weight/min = {reads:.0f} reads/min.")
    print("Measured: ~450 new accounts/min across BTC, ETH, SOL and "
          "HYPE at the\nstart of a run. Discovery outruns reading at first, "
          "so coverage builds over hours.")


async def map_loop(feed: HyperliquidMarketFeed, tracker: PositionTracker,
                   writer: MapWriter, coins: List[str], *, every: float,
                   bucket_pct: float) -> None:
    while True:
        await asyncio.sleep(every)
        now_ms = int(time.time() * 1000)
        for coin in coins:
            mark = feed.mark(coin, max_age_s=MARK_MAX_AGE_S)
            if mark is None or mark <= 0:
                log(f"{coin:<6} no mark price in the last "
                    f"{MARK_MAX_AGE_S:.0f}s - map skipped rather than drawn "
                    f"against a stale one.")
                continue
            result = build_map(
                coin, list(tracker.positions[coin].values()),
                mark_px=mark, as_of_ms=now_ms,
                open_interest=feed.open_interest(coin, max_age_s=MARK_MAX_AGE_S),
                bucket_pct=bucket_pct,
            )
            writer.write(result)
            log(summarise(result))
        log(tracker.summary_line())
        log(feed.summary_line())


async def run(coins: List[str], data_dir: Path, *, weight_per_minute: int,
              positions: bool, map_seconds: float, bucket_pct: float) -> None:
    tracker: Optional[PositionTracker] = None
    writer: Optional[MapWriter] = None
    if positions:
        tracker = PositionTracker(coins, data_dir=data_dir,
                                  budget=WeightBudget(weight_per_minute),
                                  archive=RECORD_RAW, on_log=log)
        writer = MapWriter(data_dir)
    feed = HyperliquidMarketFeed(
        coins, data_dir=data_dir, record_raw=RECORD_RAW,
        stall_timeout_s=FEED_STALL_TIMEOUT_S,
        on_trade_users=tracker.observe_trade if tracker else None,
        on_log=log,
    )

    archive = "" if RECORD_RAW else "  ARCHIVE OFF (PERPKIT_RECORD_RAW)"
    log(f"Recording {len(coins)} Hyperliquid coin(s) -> {venue_dir(data_dir)}")
    for coin in coins:
        log(f"  {coin:<10} -> {coin_dir(data_dir, coin)}  "
            f"[{', '.join(MARKET_CHANNELS)}]{archive}")
    if tracker is not None:
        log(f"  accounts   -> {accounts_dir(data_dir)}  "
            f"[{POSITIONS_CHANNEL}]{archive}")
        if tracker.restored:
            log(f"  restored {tracker.restored:,} account(s) holding a tracked "
                f"coin; re-reading them largest first.")
        log(f"  a liquidation map per coin every {map_seconds:g}s, the first "
            f"after one interval.")

    tasks = [supervise("hyperliquid:market", feed.run)]
    if tracker is not None and writer is not None:
        tasks.append(supervise("hyperliquid:positions", tracker.run))
        tasks.append(supervise(
            "hyperliquid:maps",
            lambda: map_loop(feed, tracker, writer, coins, every=map_seconds,
                             bucket_pct=bucket_pct)))
    try:
        await asyncio.gather(*tasks)
    finally:
        closers = [("market", feed.close)]
        if tracker is not None:
            closers.append(("positions", tracker.close))
        if writer is not None:
            closers.append(("maps", writer.close))
        for name, closer in closers:
            try:
                closer()
            except Exception as exc:  # pragma: no cover - shutdown path
                log(f"  {name}: close failed: {exc}")
        log("Hyperliquid writers flushed and closed.")


def parse_coins(text: str) -> List[str]:
    coins = [part.strip() for part in text.split(",") if part.strip()]
    if not coins:
        raise SystemExit("No coins given.")
    duplicates = sorted({coin for coin in coins if coins.count(coin) > 1})
    if duplicates:
        raise SystemExit(f"Coin(s) listed more than once: {', '.join(duplicates)}")
    return coins


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--coins", default=HYPERLIQUID_COINS,
        help="Comma-separated Hyperliquid perp names, case-sensitive "
             "(default: HYPERLIQUID_COINS).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--weight-per-minute", type=int, default=HYPERLIQUID_WEIGHT_PER_MINUTE,
        help=f"REST weight to spend reading accounts, of the "
             f"{WEIGHT_LIMIT_PER_MINUTE} per IP the exchange allows.")
    parser.add_argument(
        "--no-positions", action="store_true",
        help="Market data only: no account reads and no liquidation map.")
    parser.add_argument("--map-seconds", type=float,
                        default=HYPERLIQUID_MAP_SECONDS)
    parser.add_argument("--bucket-pct", type=float,
                        default=HYPERLIQUID_BUCKET_PCT,
                        help="Map band width as a fraction of mark.")
    parser.add_argument(
        "--check", action="store_true",
        help="Validate the coins against the live universe, print what "
             "recording them involves, and exit without recording.")
    args = parser.parse_args(argv)

    coins = parse_coins(args.coins)
    if not 0 < args.weight_per_minute <= WEIGHT_LIMIT_PER_MINUTE:
        raise SystemExit(f"--weight-per-minute must be in 1..."
                         f"{WEIGHT_LIMIT_PER_MINUTE}, the exchange's per-IP limit.")
    if args.map_seconds <= 0 or args.bucket_pct <= 0:
        raise SystemExit("--map-seconds and --bucket-pct must be positive.")
    positions = not args.no_positions
    if not RECORD_RAW and not positions:
        raise SystemExit("PERPKIT_RECORD_RAW is false and --no-positions leaves "
                         "nothing else to produce.\nNothing to do.")

    try:
        assets, contexts = fetch_universe()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not read Hyperliquid's universe "
                         f"({type(exc).__name__}: {exc}).\nNothing is started "
                         f"against coins that cannot be checked.")
    problems = validate_coins(coins, assets)
    if problems:
        raise SystemExit("Refusing to start:\n"
                         + "\n".join(f"  - {problem}" for problem in problems))

    report(coins, assets, contexts, weight_per_minute=args.weight_per_minute,
           positions=positions)
    if args.check:
        return 0

    lock = ExclusiveLock(venue_dir(args.data_dir) / ".recorder.lock",
                         holder=LOCK_HOLDER).acquire()
    try:
        asyncio.run(run(coins, args.data_dir,
                        weight_per_minute=args.weight_per_minute,
                        positions=positions, map_seconds=args.map_seconds,
                        bucket_pct=args.bucket_pct))
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
