"""How far the raw archive's receive clock ran behind BloFin's, and where.

    python -m perpkit.analysis.recorder_lag --date 2026-01-01
    python -m perpkit.analysis.recorder_lag --date 2026-01-01 --instruments SUI-USDT,BTC-USDT --hours
    python -m perpkit.analysis.recorder_lag --since 2026-01-01 --workers 2

Read-only, so it is safe beside a live recorder - but it parses every book
update, so it competes with that recorder for CPU. `--workers` defaults to 2.

Why this exists
---------------
`perpkit/rawlog.py` stamps every message with `t`, when this process wrote
it. BloFin stamps its own: `data.ts` on `books`, `ts` on each `trades` row,
`ts` on `mark-price`. In bursts `t - ts` on a book has reached ~53 s, and any
tool that joins the archive to another clock on `t` is wrong by that much in
exactly the volatile hours.

What it measures
----------------
Per instrument, day, UTC hour of `t`, and channel: the distribution of
`t - ts` and the share over `--late-ms`. Each socket message's lag is split at
the newest stamp the socket had delivered by then (the running maximum, as in
`replay.on_exchange_clock`):

  venue   `freshest - ts`. One websocket per instrument carries books, trades
          and funding, and TCP plus the SDK's queue are first in, first out,
          so arrival order is the venue's send order. A book arriving after a
          trade stamped 15 s later was already 15 s late when BloFin sent it.
          Nothing in this process can make that number; it is a lower bound
          on the venue's own delay.
  after   `t - freshest`. Everything after the venue sent its newest message:
          the link, this machine, this process's queue - and a venue behind
          on every channel at once, which no single socket can tell apart.

Then, per hour, whether the instruments were late together. Every instrument
has its own socket, but all of them share one event loop in `perpkit.record`;
a backlog there delays all of them at once, while a slow book at the venue
need not. For each second of receive time: which instruments delivered a
stamped message, and which of those were over `--late-ms` (on `after`, where
the event loop would show). Reported: seconds with anyone late, the median
share of active instruments late in those seconds, and the lift - P(another
instrument late | one is) over P(an instrument late). A lift near 1 is
independent; near 1/P, fully shared.

Throughput is the last check. A process that cannot keep up writes at its
ceiling while the backlog lasts; a stalled link writes nothing and then a
burst. So: messages written per second by the whole recording (all
instruments), in late seconds and the rest of the hour.

`mark-price` is recorded by a different process (`perpkit.record_oi`), polled
over REST, and is reported beside the socket channels as the control: the
same machine and link, a different event loop.

What it has found
-----------------
Over several days of 15 instruments and ~19M book updates: the median lag is
70-100 ms every day; the tail is the problem and it clusters in busy hours
(US morning). In the worst hours over 40% of an instrument's book updates
arrived more than a second late, p90 ~44 s, worst single message ~56 s. The
cause was NOT the recorder: one event loop at ~10% of a core kept some
instruments' sockets under a second while others sat ~50 s behind for
minutes; the `venue` split was ~0, so books and trades were equally late;
and when a lagging socket died ~43 s behind, only one message was left in the
process to write. The delay was queued upstream - the venue's gateway or the
path to it - and the REST control never exceeded 400 ms. So a bad hour here
is not a reason to restructure the recorder; it is a reason to read the
archive on the exchange clock.
"""

from __future__ import annotations

import argparse
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


from perpkit.layout import instrument_dirs
from perpkit.analysis.replay import SOCKET_CHANNELS, channel_of, exchange_ms, merged_events
from perpkit.rawlog import iter_events

from perpkit.config import DATA_DIR
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
# Channels reported, in print order. `books5` is folded into `books`.
REPORTED = ("books", "trades", "mark-price")
CONTROL = "mark-price"


@dataclass
class ChannelLags:
    """Stamped messages of one channel, in arrival order."""

    received: np.ndarray   # t, ms
    stamp: np.ndarray      # the venue's ts, ms
    freshest: np.ndarray   # newest ts the socket had delivered, this one included

    @property
    def lag(self) -> np.ndarray:
        return self.received - self.stamp

    @property
    def venue(self) -> np.ndarray:
        return self.freshest - self.stamp

    @property
    def after(self) -> np.ndarray:
        return self.received - self.freshest


@dataclass
class HourStats:
    n: int
    p50: float
    p90: float
    p99: float
    max: int
    min: int
    late_share: float
    venue_p90: float
    after_p90: float


@dataclass
class DayScan:
    """Everything one instrument-day contributes; small enough to pickle back."""

    instrument: str
    day: str
    hours: Dict[str, Dict[int, HourStats]] = field(default_factory=dict)
    # Per second of the day, over books + trades: the largest `after` and total
    # lag among messages written that second (NaN: none written), and how many
    # socket messages of any channel were written.
    after_max: Optional[np.ndarray] = None
    lag_max: Optional[np.ndarray] = None
    written: Optional[np.ndarray] = None


def _percentiles(values: np.ndarray) -> Tuple[float, float, float]:
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return float(p50), float(p90), float(p99)


def socket_lags(events: Iterable[Tuple[int, int, dict]]) -> Tuple[Dict[str, ChannelLags], List[int]]:
    """Split one socket's arrival-ordered events into per-channel lags.

    Returns the stamped channels, and the receive time of every socket message
    (stamped or not), for the throughput count.
    """
    columns: Dict[str, Tuple[array, array, array]] = {}
    written: List[int] = []
    freshest: Optional[int] = None
    for received, _, message in events:
        written.append(received)
        stamp = exchange_ms(message)
        if stamp is None:
            continue
        if freshest is None or stamp > freshest:
            freshest = stamp
        channel = (message.get("arg") or {}).get("channel")
        if channel == "books5":
            channel = "books"
        cols = columns.setdefault(channel, (array("q"), array("q"), array("q")))
        cols[0].append(received)
        cols[1].append(stamp)
        cols[2].append(freshest)
    return {
        channel: ChannelLags(*(np.frombuffer(col, dtype=np.int64) for col in cols))
        for channel, cols in columns.items()
    }, written


def polled_lags(events: Iterable[Tuple[int, int, dict]]) -> ChannelLags:
    """A polled channel's lags. Each poll stands alone: nothing to be behind."""
    received, stamps = array("q"), array("q")
    for t, _, message in events:
        stamp = exchange_ms(message)
        if stamp is not None:
            received.append(t)
            stamps.append(stamp)
    stamp = np.frombuffer(stamps, dtype=np.int64)
    return ChannelLags(np.frombuffer(received, dtype=np.int64), stamp, stamp)


def hourly_stats(lags: ChannelLags, day_start_ms: int, late_ms: int) -> Dict[int, HourStats]:
    """Distribution of `t - ts` per UTC hour of receive time."""
    out: Dict[int, HourStats] = {}
    if not len(lags.received):
        return out
    hour = (lags.received - day_start_ms) // HOUR_MS
    lag, venue, after = lags.lag, lags.venue, lags.after
    for h in np.unique(hour):
        mask = hour == h
        values = lag[mask]
        p50, p90, p99 = _percentiles(values)
        out[int(h)] = HourStats(
            n=int(mask.sum()), p50=p50, p90=p90, p99=p99,
            max=int(values.max()), min=int(values.min()),
            late_share=float(np.mean(values > late_ms)),
            venue_p90=float(np.percentile(venue[mask], 90)),
            after_p90=float(np.percentile(after[mask], 90)),
        )
    return out


def per_second_max(received: np.ndarray, values: np.ndarray, day_start_ms: int) -> np.ndarray:
    """Largest value among messages written in each second of the day; NaN where none."""
    out = np.full(DAY_MS // 1000, np.nan)
    second = (received - day_start_ms) // 1000
    keep = (second >= 0) & (second < len(out))
    np.fmax.at(out, second[keep], values[keep].astype(float))
    return out


def day_start(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def scan_day(raw_dir: Path, day: str, late_ms: int = 1000) -> DayScan:
    instrument = raw_dir.parent.name
    start = day_start(day)
    scan = DayScan(instrument=instrument, day=day)
    files = sorted((raw_dir / day).glob("*.jsonl.gz"))
    socket_files = [path for path in files if channel_of(path) in SOCKET_CHANNELS]
    if socket_files:
        lags, written = socket_lags(merged_events(raw_dir, day, channels=SOCKET_CHANNELS, quiet=True))
    else:
        lags, written = {}, []
    control_files = [path for path in files if channel_of(path) == CONTROL]
    if control_files:
        lags[CONTROL] = polled_lags(event for path in control_files
                                    for event in iter_events(path, warn=False))

    for channel in REPORTED:
        if channel in lags:
            scan.hours[channel] = hourly_stats(lags[channel], start, late_ms)

    stamped = [lags[channel] for channel in ("books", "trades") if channel in lags]
    if stamped:
        received = np.concatenate([item.received for item in stamped])
        scan.after_max = per_second_max(received, np.concatenate([item.after for item in stamped]), start)
        scan.lag_max = per_second_max(received, np.concatenate([item.lag for item in stamped]), start)
    counts = np.zeros(DAY_MS // 1000, dtype=np.int32)
    if written:
        seconds = (np.asarray(written, dtype=np.int64) - start) // 1000
        seconds = seconds[(seconds >= 0) & (seconds < len(counts))]
        np.add.at(counts, seconds, 1)
    scan.written = counts
    return scan


@dataclass
class Together:
    late_seconds: int
    median_share: float
    p_late: float
    p_late_given_other: float

    @property
    def lift(self) -> float:
        return self.p_late_given_other / self.p_late if self.p_late > 0 else float("nan")


def together(matrix: np.ndarray, late_ms: int) -> Together:
    """Were instruments late in the same seconds?

    `matrix` is instruments x seconds of per-second lag, NaN where an
    instrument wrote nothing that second. Only seconds with at least two
    active instruments count.
    """
    active = ~np.isnan(matrix)
    late = np.where(active, matrix > late_ms, False)
    n_active = active.sum(axis=0)
    n_late = late.sum(axis=0)
    usable = n_active >= 2
    any_late = usable & (n_late > 0)
    shares = n_late[any_late] / n_active[any_late]
    p_late = n_late[usable].sum() / n_active[usable].sum() if usable.any() else 0.0
    # Over every late (instrument, second): the share of the OTHER active
    # instruments late too.
    pairs = (late[:, usable] * (n_late[usable] - 1)).sum()
    others = (late[:, usable] * (n_active[usable] - 1)).sum()
    return Together(
        late_seconds=int(any_late.sum()),
        median_share=float(np.median(shares)) if len(shares) else float("nan"),
        p_late=float(p_late),
        p_late_given_other=float(pairs / others) if others else float("nan"),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_ms(value: float) -> str:
    if value >= 10_000:
        return f"{value / 1000:.0f}s"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}"


def _row(label: str, stats: HourStats) -> str:
    return (f"  {label:<22} {stats.n:>8,} {_fmt_ms(stats.p50):>6} {_fmt_ms(stats.p90):>6} "
            f"{_fmt_ms(stats.p99):>6} {_fmt_ms(stats.max):>6} {stats.min:>5} "
            f"{stats.late_share:>6.1%} {_fmt_ms(stats.venue_p90):>6} {_fmt_ms(stats.after_p90):>6}")


def combine(hours: Dict[int, HourStats]) -> HourStats:
    """A day from its hours: counts and extremes exactly, quantiles n-weighted.

    Quantiles do not combine exactly; the weighted mean of hourly quantiles
    is labelled as such where printed.
    """
    items = list(hours.values())
    n = sum(item.n for item in items)
    w = np.array([item.n for item in items], dtype=float) / n

    def mean(name: str) -> float:
        return float(np.dot(w, [getattr(item, name) for item in items]))

    return HourStats(n=n, p50=mean("p50"), p90=mean("p90"), p99=mean("p99"),
                     max=max(item.max for item in items),
                     min=min(item.min for item in items),
                     late_share=mean("late_share"), venue_p90=mean("venue_p90"),
                     after_p90=mean("after_p90"))


def report(scans: Sequence[DayScan], *, late_ms: int, hours: bool) -> None:
    days = sorted({scan.day for scan in scans})
    header = (f"  {'':<22} {'n':>8} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6} {'min':>5} "
              f"{'>' + _fmt_ms(late_ms):>6} {'venue':>6} {'after':>6}")
    print(f"t - ts in ms (s where marked). Day rows average the hourly quantiles, "
          f"weighted by n;\nmax, min and the late share are exact. venue/after: "
          f"p90 of the split described in the docstring.\n{CONTROL} is the "
          f"control: another process, polling REST, on this machine.")
    for day in days:
        print(f"\n=== {day}")
        print(header)
        for scan in sorted((s for s in scans if s.day == day), key=lambda s: s.instrument):
            for channel in REPORTED:
                by_hour = scan.hours.get(channel)
                if not by_hour:
                    continue
                label = f"{scan.instrument} {channel}"
                print(_row(label, combine(by_hour)))
                if hours:
                    for h, stats in sorted(by_hour.items()):
                        print(_row(f"    {h:02d}:00", stats))

        day_scans = [s for s in scans if s.day == day and s.after_max is not None]
        if len(day_scans) < 2:
            continue
        names = [s.instrument for s in sorted(day_scans, key=lambda s: s.instrument)]
        ordered = sorted(day_scans, key=lambda s: s.instrument)
        print(f"\n  Share of book updates over {_fmt_ms(late_ms)} late, by UTC hour "
              f"(columns: {', '.join(names)})")
        width = 5
        print("  hour " + "".join(f"{name.split('-')[0][:width - 1]:>{width}}" for name in names))
        for h in range(24):
            cells = []
            for scan in ordered:
                stats = scan.hours.get("books", {}).get(h)
                cells.append(f"{'':>{width}}" if stats is None else
                             f"{stats.late_share * 100:>{width}.0f}")
            if any(cell.strip() for cell in cells):
                print(f"  {h:02d}   " + "".join(cells))

        after = np.vstack([s.after_max for s in ordered])
        lag = np.vstack([s.lag_max for s in ordered])
        written = np.vstack([s.written for s in ordered]).sum(axis=0)
        print(f"\n  Late together? Per UTC hour with any late second, over books + trades.")
        print(f"  {'hour':<5} {'late s':>7} {'share':>6} {'P(late)':>8} {'P(|other)':>9} {'lift':>6}"
              f"   {'total lag':>9} {'lift':>6}   written/s: late p50 max | rest p50 max")
        for h in range(24):
            window = slice(h * 3600, (h + 1) * 3600)
            on_after = together(after[:, window], late_ms)
            on_lag = together(lag[:, window], late_ms)
            if on_lag.late_seconds == 0 and on_after.late_seconds == 0:
                continue
            late_now = (np.nan_to_num(after[:, window], nan=0.0) > late_ms).any(axis=0)
            hour_written = written[window]
            spans = []
            for chosen in (late_now, ~late_now):
                values = hour_written[chosen]
                spans.append(f"{np.median(values):>4.0f} {values.max():>4d}" if len(values) else f"{'-':>4} {'-':>4}")
            print(f"  {h:02d}    {on_after.late_seconds:>7,} {on_after.median_share:>6.0%} "
                  f"{on_after.p_late:>8.1%} {on_after.p_late_given_other:>9.1%} {on_after.lift:>6.1f}"
                  f"   {on_lag.late_seconds:>9,} {on_lag.lift:>6.1f}"
                  f"   {spans[0]} | {spans[1]}")


def _scan_job(job: Tuple[Path, str, int]) -> DayScan:
    raw_dir, day, late_ms = job
    return scan_day(raw_dir, day, late_ms)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--instruments", default=None,
                        help="comma-separated; default every instrument directory")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, or several comma-separated")
    parser.add_argument("--since", default=None, help="every archived day from YYYY-MM-DD on")
    parser.add_argument("--late-ms", type=int, default=1000)
    parser.add_argument("--hours", action="store_true", help="print every hour, not just days")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel instrument-days; each competes with a live recorder for CPU")
    args = parser.parse_args(argv)

    found = instrument_dirs(args.data_dir)
    if args.instruments:
        wanted = [name.strip() for name in args.instruments.split(",") if name.strip()]
        missing = [name for name in wanted if name not in found]
        if missing:
            raise SystemExit(f"No instrument directory for {', '.join(missing)} under "
                             f"{args.data_dir}.\nAvailable: {', '.join(found) or 'none'}")
        found = {name: found[name] for name in wanted}

    jobs = []
    for name, path in found.items():
        raw = path / "raw"
        if not raw.is_dir():
            continue
        for day_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
            day = day_dir.name
            if args.date and day not in args.date.split(","):
                continue
            if args.since and day < args.since:
                continue
            jobs.append((raw, day, args.late_ms))
    if not jobs:
        raise SystemExit(f"No archived days match under {args.data_dir} "
                         f"(--date {args.date}, --since {args.since}).")

    print(f"Scanning {len(jobs)} instrument-day(s) with {args.workers} worker(s)...",
          file=sys.stderr)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            scans = list(pool.map(_scan_job, jobs))
    else:
        scans = [_scan_job(job) for job in jobs]
    report(scans, late_ms=args.late_ms, hours=args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
