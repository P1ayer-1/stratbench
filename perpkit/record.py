"""Record several instruments at once, headless. No chart, no HTTP, no ports.

    python -m perpkit.record --instruments BTC-USDT,ADA-USDT,PUMP-USDT
    python -m perpkit.record --instruments ADA-USDT --data-dir data
    python -m perpkit.record --list-cost --instruments A,B,C   # disk only, no run

Collecting a dataset usually means several symbols at once - different
research questions need different instruments, and each needs continuity.
Running one process per symbol means one set of connections and logs per
symbol to keep straight. Instead this runs N `MicrostructureFeed`s as N
supervised asyncio tasks in one process, with no servers at all.

Each feed opens its own websocket connection and writes to its own
`data/<INST-ID>/` directory. Nothing is shared between them except the
process, so one instrument desyncing, stalling or reconnecting cannot touch
another's data.

Every feed runs under `perpkit.supervise.supervise`, so no exception in one
loop can end the run: a collection run that dies at 03:00 costs the hours
nobody was awake for, and the raw archive is the half that cannot be
re-collected.

Cost
----
Roughly **140 MB/day per instrument** (~100 MB raw + ~40 MB features at the 1s
sample interval). That is measured on a live recorder, not estimated.
Six instruments is therefore ~840 MB/day and ~25 GB/month. `--list-cost`
prints the arithmetic for a chosen set and exits without connecting, which is
worth doing before leaving anything running for a week.

Memory ceiling
--------------
A leaking recorder can grow to tens of GB and exhaust the machine's commit
charge. One known cause is SDK clients that survive `close()` and keep
queueing a whole feed nobody reads (guarded against by
`perpkit.ingest.close_ws_client`). A healthy process is 40-170 MB. As a
backstop against the next leak, the process logs its memory every
`--memory-log-s` and, once it passes `--max-memory-mb` (default 2048), flushes
every writer and exits with code 3. Your process supervisor (e.g. a scheduled
task or systemd unit) restarts it, and the new process writes `.r001` files
beside the open hour's. The number checked is the
larger of RSS and private bytes. On Windows private bytes is the commit charge
that actually ran out, and RSS understates it once pages are trimmed to the
page file.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import psutil


from perpkit.config import (
    BOOK_DEPTH,
    DATA_DIR,
    FEATURE_SAMPLE_INTERVAL_MS,
    FEED_STALL_TIMEOUT_S,
    INST_ID,
    LABEL_HORIZONS,
    LABEL_THRESHOLD_BPS,
    RECORD_FEATURES,
    RECORD_RAW,
    TAPE_WINDOW_SECONDS,
    USE_DEMO,
)
from perpkit.supervise import log, supervise
from perpkit.ingest import MicrostructureFeed
from perpkit.openinterest import OpenInterestPoller
from perpkit.recorder import LabelConfig

# Measured on a live recorder. Raw dominates and scales with message rate,
# so a busier instrument costs more than a quiet one; this is the BTC-USDT
# figure and therefore an upper-ish bound for the thinner symbols.
MB_PER_DAY_PER_INSTRUMENT = 140

# See "Memory ceiling" above. 2 GB is over ten times the largest healthy
# process (~170 MB) and far below what a leak can reach.
MAX_MEMORY_MB = 2048
MEMORY_CHECK_S = 30.0
MEMORY_LOG_S = 900.0
EXIT_MEMORY_CEILING = 3


def process_memory_mb() -> Tuple[float, float]:
    """(RSS, private bytes) of this process in MB. Private is 0 where the OS has none."""
    info = psutil.Process().memory_info()
    return info.rss / 2**20, getattr(info, "private", 0) / 2**20


async def memory_ceiling(
    ceiling_mb: float,
    *,
    check_s: float = MEMORY_CHECK_S,
    log_every_s: float = MEMORY_LOG_S,
    read: Optional[Callable[[], Tuple[float, float]]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Log memory every `log_every_s`; return the MB used once it passes the ceiling."""
    read = read or process_memory_mb
    last_log: Optional[float] = None
    while True:
        try:
            rss, private = read()
        except Exception as exc:   # a guard must never end the recording
            log(f"memory: could not read ({exc}); still recording")
            await asyncio.sleep(check_s)
            continue
        used = max(rss, private)
        now = clock()
        if last_log is None or now - last_log >= log_every_s:
            log(f"memory: rss {rss:.0f} MB, private {private:.0f} MB "
                f"(ceiling {ceiling_mb:.0f} MB)")
            last_log = now
        if used > ceiling_mb:
            return used
        await asyncio.sleep(check_s)


def build_feed(inst_id: str, data_dir: Path) -> MicrostructureFeed:
    """One feed, configured from `perpkit.config`."""
    return MicrostructureFeed(
        inst_id,
        use_demo=USE_DEMO,
        book_depth=BOOK_DEPTH,
        data_dir=data_dir,
        record=RECORD_FEATURES,
        record_raw=RECORD_RAW,
        sample_interval_ms=FEATURE_SAMPLE_INTERVAL_MS,
        tape_window_seconds=TAPE_WINDOW_SECONDS,
        stall_timeout_s=FEED_STALL_TIMEOUT_S,
        label_config=LabelConfig(
            horizons_seconds=LABEL_HORIZONS,
            threshold_bps=LABEL_THRESHOLD_BPS,
        ),
    )


def report_cost(instruments: List[str], data_dir: Path) -> None:
    daily = len(instruments) * MB_PER_DAY_PER_INSTRUMENT
    print(f"\n{len(instruments)} instrument(s) -> {data_dir}")
    for inst_id in instruments:
        print(f"  {inst_id:<18} {data_dir / inst_id}")
    print(f"\nMeasured cost, at ~{MB_PER_DAY_PER_INSTRUMENT} MB/day each:")
    print(f"  per day      {daily / 1000:.2f} GB")
    print(f"  per week     {daily * 7 / 1000:.2f} GB")
    print(f"  per 30 days  {daily * 30 / 1000:.2f} GB")
    print("\nThe raw archive is most of that, and it is the half that cannot "
          "be re-collected.\nFeature CSVs are regenerable from it with "
          "perpkit/analysis/replay.py, so if disk gets\ntight, delete those first and "
          "never the archive.")


async def run(
    instruments: List[str],
    data_dir: Path,
    *,
    max_memory_mb: float = MAX_MEMORY_MB,
    memory_log_s: float = MEMORY_LOG_S,
    memory_check_s: float = MEMORY_CHECK_S,
) -> int:
    """Record until cancelled. Returns an exit code: 3 when the memory ceiling tripped."""
    feeds = [(inst_id, build_feed(inst_id, data_dir)) for inst_id in instruments]
    # Open interest, one HTTP request per poll for the whole set. It is the
    # only public input to estimating where OTHER traders are liquidated, and
    # BloFin serves a snapshot with no history endpoint - so it is collected
    # live or not at all. `record_oi.py` runs the same poller standalone, for
    # adding it to a run that is already going without restarting this one.
    #
    # None when the raw archive is off: a poller that polled and discarded
    # would spend requests to write nothing.
    oi_poller = (
        OpenInterestPoller(instruments, data_dir=data_dir)
        if RECORD_RAW else None
    )

    log(f"Recording {len(feeds)} instrument(s), headless. No chart, no ports.")
    for inst_id, feed in feeds:
        writers = []
        if feed.recorder.enabled:
            writers.append("features")
        if feed.raw_log.enabled:
            writers.append("raw")
        log(f"  {inst_id:<18} -> {feed.instrument_dir}  "
            f"[{', '.join(writers) if writers else 'NOTHING ENABLED'}]")
    if oi_poller is not None:
        log(f"  open interest      one poll every "
            f"{oi_poller.poll_seconds:.0f}s for all {len(instruments)} "
            f"[raw]")
    else:
        log("  open interest      DISABLED (PERPKIT_RECORD_RAW)")

    if not any(feed.recorder.enabled or feed.raw_log.enabled
               for _, feed in feeds):
        raise SystemExit(
            "Both writers are disabled (PERPKIT_RECORD_FEATURES and "
            "PERPKIT_RECORD_RAW), so this\nwould connect and discard "
            "everything. Nothing to do."
        )

    log(f"Nothing reaches the feature CSVs for the first "
        f"{max(LABEL_HORIZONS) / 60:.0f} minutes - a row cannot be written "
        f"until its\n  forward window has closed. The raw archive starts "
        f"immediately.")

    if max_memory_mb > 0:
        log(f"  memory ceiling     {max_memory_mb:.0f} MB, then flush and exit "
            f"{EXIT_MEMORY_CEILING} for the supervisor to restart")

    recorders = guard = None
    try:
        # Supervised, so a failure in one instrument's loop restarts that loop
        # instead of ending the process and every other instrument with it.
        tasks = [supervise(f"feed:{inst_id}", feed.run)
                 for inst_id, feed in feeds]
        if oi_poller is not None:
            tasks.append(
                supervise("open-interest", lambda: oi_poller.run(on_log=log))
            )
        recorders = asyncio.gather(*tasks)
        if max_memory_mb <= 0:
            await recorders
            return 0
        guard = asyncio.ensure_future(memory_ceiling(
            max_memory_mb, check_s=memory_check_s, log_every_s=memory_log_s))
        await asyncio.wait({recorders, guard},
                           return_when=asyncio.FIRST_COMPLETED)
        if not guard.done():
            await recorders   # only reached if it ended; re-raises its error
            return 0
        log(f"memory {guard.result():.0f} MB is past the {max_memory_mb:.0f} MB "
            f"ceiling - flushing and exiting so the supervisor restarts this "
            f"process.")
        return EXIT_MEMORY_CEILING
    finally:
        for task in (guard, recorders):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if oi_poller is not None:
            try:
                oi_poller.close()
            except Exception as exc:  # pragma: no cover - shutdown path
                log(f"  open-interest: close failed: {exc}")
        # Flush every writer, even on Ctrl-C. The pending-row buffer is lost
        # either way - those rows have no closed forward window yet - but what
        # is already labelled belongs on disk.
        for inst_id, feed in feeds:
            try:
                feed.close()
            except Exception as exc:  # pragma: no cover - shutdown path
                log(f"  {inst_id}: close failed: {exc}")
        log("Recorders flushed and closed.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--instruments", default=INST_ID,
        help="Comma-separated instrument ids (default: BLOFIN_INST_ID).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--list-cost", action="store_true",
        help="Print the disk arithmetic for this set and exit without "
             "connecting.")
    parser.add_argument(
        "--max-memory-mb", type=float, default=MAX_MEMORY_MB,
        help="Flush and exit with code 3 once RSS or private bytes pass this "
             "(default %(default)g; 0 disables).")
    parser.add_argument(
        "--memory-log-s", type=float, default=MEMORY_LOG_S,
        help="Seconds between memory log lines (default %(default)g).")
    args = parser.parse_args(argv)

    instruments = [part.strip() for part in args.instruments.split(",")
                   if part.strip()]
    if not instruments:
        raise SystemExit("No instruments given.")

    duplicates = {name for name in instruments if instruments.count(name) > 1}
    if duplicates:
        # Two feeds on one symbol would interleave rows into one file from two
        # independent books, which is corruption rather than more data.
        raise SystemExit(
            f"Instrument(s) listed more than once: {', '.join(sorted(duplicates))}"
        )

    if args.list_cost:
        report_cost(instruments, args.data_dir)
        return 0

    report_cost(instruments, args.data_dir)
    try:
        return asyncio.run(run(instruments, args.data_dir,
                               max_memory_mb=args.max_memory_mb,
                               memory_log_s=args.memory_log_s))
    except KeyboardInterrupt:
        log("Interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
