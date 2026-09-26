"""Record open interest alongside an already-running recorder.

    python -m perpkit.record_oi --instruments BTC-USDT,ADA-USDT,PUMP-USDT
    python -m perpkit.record_oi --instruments BTC-USDT --once
    python -m perpkit.record_oi --match-running

**This does not require restarting `record.py`.** It is a separate process
writing a separate channel: `raw/<day>/open-interest-<HH>.jsonl.gz`, never the
`books-` or `trades-` files a live recorder holds open. Start it against a run
that is already three days in and nothing about that run changes.

That property is the reason it is a standalone entrypoint rather than another
task inside `record.py`. Adding it there would mean stopping a recorder to
start collecting a series - and the argument for collecting OI at all is that
the hours you do not have are gone, which a restart makes worse before it
makes better.

**`record.py` now starts a poller of its own, so do not run both.** They would
archive every minute twice. An earlier version was worse: both appended to
the same `open-interest-<HH>.jsonl.gz`, and two gzip writers on one file
produce a stream that does not decode - measured at 0 of 40 records recovered.
Files are now created exclusively so that cannot recur, but each poller still
takes an exclusive per-instrument lock and the second is refused with the
owning pid. This entrypoint is for instruments the running recorder is not
covering, or for a recorder that does not run a poller of its own.

What it collects and why
------------------------
Open interest is the one publicly available input to estimating where OTHER
traders are liquidated. `perpkit/risk.py` models our own liquidation price
exactly; nothing models anyone else's, because none of `books`, `trades` or
`funding-rate` carries a fact about someone else's position. OI does, and
BloFin serves only a snapshot - no history endpoint - so the series has to be
accumulated live or it does not exist. See `perpkit/openinterest.py` for the
measured behaviour of the endpoint and for what is deliberately NOT built yet
(the cluster model itself).

Cost: one HTTP request per poll for ALL instruments, and roughly 300 KB/day
per instrument before compression. Against ~140 MB/day/instrument for the
book and trade feed, this is free.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


from perpkit.config import DATA_DIR, INST_ID, RECORD_RAW
from perpkit.supervise import log, supervise
from perpkit.openinterest import (
    DEFAULT_POLL_SECONDS,
    OpenInterestPoller,
    SnapshotPoller,
    PRODUCTION_BASE_URL,
)


def running_recorder_instruments() -> Optional[List[str]]:
    """The instrument list of a `record.py` already running on this machine.

    Typing the list twice is how the two processes end up recording different
    symbol sets, which produces an OI series with holes exactly where the book
    data is densest. Reading it off the running process removes that failure
    mode entirely. Windows-only (WMIC/CIM); returns None anywhere else, or if
    no recorder is running, and the caller falls back to --instruments.
    """
    if sys.platform != "win32":
        return None
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
             "| Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in (completed.stdout or "").splitlines():
        if "record.py" in line and "record_oi.py" not in line:
            match = re.search(r"--instruments\s+(\S+)", line)
            if match:
                return [part.strip() for part in match.group(1).split(",")
                        if part.strip()]
    return None


# What each channel's cadence actually means, so the banner cannot claim the
# wrong thing about whichever one is running.
CADENCE_NOTE = {
    "open-interest":
        "  One request covers every instrument; rows are written only when "
        "the exchange's\n  minute-stamped value actually changes, so this is "
        "one row per instrument per minute.",
    "mark-price":
        "  One request covers every instrument. This series publishes "
        "CONTINUOUSLY rather\n  than on a boundary, so the poll interval IS "
        "the sample rate - and it carries both\n  mark and index price, whose "
        "difference is the basis nothing else here observes.",
}


async def run(poller: SnapshotPoller) -> None:
    channel = poller.CHANNEL
    log(f"Polling {channel} every "
        f"{poller.poll_seconds:.0f}s for {len(poller.instruments)} "
        "instrument(s).")
    note = CADENCE_NOTE.get(channel)
    if note:
        log(note)
    for inst_id in poller.instruments:
        log(f"  {inst_id:<18} -> {poller.data_root / inst_id / 'raw'}")
    log("Different channel from books/trades, so a live record.py's feed "
        "files are untouched.")
    log(f"  A second {channel} poller on the same instrument is refused: two "
        f"of them archive\n  every row twice. A "
        f"poller on a DIFFERENT channel is fine.")

    try:
        # Supervised for the same reason every other loop here is: a dropped
        # HTTP connection must not end an overnight collection run.
        await supervise(channel, lambda: poller.run(on_log=log))
    finally:
        poller.close()
        log(f"{channel} poller closed. {poller.rows_written} row(s) "
            f"written, {poller.failures} failed poll(s).")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--instruments", default=None,
        help="Comma-separated instrument ids (default: BLOFIN_INST_ID, or the "
             "running recorder's list with --match-running).")
    parser.add_argument(
        "--match-running", action="store_true",
        help="Take the instrument list from a record.py already running on "
             "this machine, so the two cannot drift apart.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--channel", default="open-interest",
        choices=["open-interest", "mark-price"],
        help="Which market snapshot to poll. mark-price carries BOTH mark "
             "and index price, and has no websocket channel at all - REST is "
             "the only route to it. Safe to run alongside an open-interest "
             "poller on the same instrument: the exclusive lock is named "
             "after the channel.")
    parser.add_argument("--poll-seconds", type=float, default=None,
                        help="Default: 20s for open-interest (it publishes "
                             "once a minute), 10s for mark-price (it "
                             "publishes continuously, so this is the "
                             "sample rate).")
    parser.add_argument("--base-url", default=PRODUCTION_BASE_URL)
    parser.add_argument(
        "--once", action="store_true",
        help="One poll, then exit. Use this to prove the endpoint and the "
             "write path work before leaving it running.")
    args = parser.parse_args(argv)

    instruments: List[str] = []
    if args.match_running:
        found = running_recorder_instruments()
        if found:
            log(f"Matched running recorder: {len(found)} instrument(s).")
            instruments = found
        else:
            log("No running record.py found - falling back to --instruments.")
    if not instruments:
        instruments = [part.strip()
                       for part in (args.instruments or INST_ID).split(",")
                       if part.strip()]
    if not instruments:
        raise SystemExit("No instruments given.")

    duplicates = {name for name in instruments if instruments.count(name) > 1}
    if duplicates:
        raise SystemExit(
            f"Instrument(s) listed more than once: {', '.join(sorted(duplicates))}"
        )

    if not RECORD_RAW:
        raise SystemExit(
            "PERPKIT_RECORD_RAW is false, so this would poll and discard every "
            "response.\nNothing to do."
        )

    if args.channel == "mark-price":
        from perpkit.markprice import DEFAULT_POLL_SECONDS as MARK_POLL_SECONDS
        from perpkit.markprice import MarkPricePoller

        factory, default_poll = MarkPricePoller, MARK_POLL_SECONDS
    else:
        factory, default_poll = OpenInterestPoller, DEFAULT_POLL_SECONDS

    poller = factory(
        instruments,
        data_dir=args.data_dir,
        base_url=args.base_url,
        poll_seconds=(default_poll if args.poll_seconds is None
                      else args.poll_seconds),
    )
    log(f"Polling {args.channel} for {len(instruments)} instrument(s) "
        f"every {poller.poll_seconds:g}s.")

    if args.once:
        written = poller.poll_once(on_log=log)
        poller.close()
        log(f"One poll: {written} row(s) written to {args.data_dir}.")
        if written == 0 and poller.failures:
            return 1
        return 0

    try:
        asyncio.run(run(poller))
    except KeyboardInterrupt:
        log("Interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
