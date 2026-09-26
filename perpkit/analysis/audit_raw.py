"""Which archived hours the old raw reader cut short, and what is recoverable.

    python -m perpkit.analysis.audit_raw
    python -m perpkit.analysis.audit_raw --data-dir data --verify

Read-only. Files are opened for reading and nothing is written, so it is safe
to run against a live recorder.

Why this exists
---------------
An earlier `RawEventLog` opened each hourly file with
`gzip.open(path, "at")`, so a restart inside an hour appended a second gzip
member to the first run's file. If that first run was hard-killed - a power
cut, `os._exit`, a `timeout` kill - its member had no trailer, and
`iter_events` read through the tear into the next member's header, raised,
and stopped. Reproduced: five records flushed and killed, five appended by a
clean restart, **0 of 10** read back - the five from before the crash
included.

Every reader of the archive goes through `iter_events` (`replay.py` and
anything built on `replay.merged_events`), so anything computed from an hour
like that silently used less than was on disk. The reader now resumes at
the next real gzip header; this script says which hours that changes, and by
how many records.

How
---
One structural pass decodes every file without parsing JSON and notes what
kind of damage it has. A file whose only damage is an unterminated LAST member
reads identically under both readers - the old one already tolerated that,
because it is what the current hour looks like the whole time it is recorded.
So only files with damage before the end are read by both readers and
counted.

`--verify` counts EVERY file with both readers instead. That is a check on the
new reader rather than on the data: wherever the old reader did not stop
early, the two must agree record for record. It parses the whole archive
twice, so expect minutes rather than seconds.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


from perpkit.rawlog import iter_events, iter_lines

# A file with an unterminated tail modified this recently is most likely the
# hour a recorder is writing right now, not a crash.
LIVE_SECONDS = 15 * 60


def legacy_iter_events(path: Path):
    """`iter_events` exactly as it was before the torn-member fix, kept for comparison."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = iter(handle)
        while True:
            try:
                line = next(lines)
            except StopIteration:
                return
            except (EOFError, zlib.error, OSError):
                return

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield record.get("t", 0), record.get("n", 0), record.get("m", {})


@dataclass
class FileReport:
    path: Path
    size: int
    modified: float
    damage: List[Tuple[str, int, int]] = field(default_factory=list)
    old: Optional[int] = None
    new: Optional[int] = None
    old_failure: str = ""
    old_is_prefix: bool = True

    @property
    def kinds(self) -> List[str]:
        return [kind for kind, _, _ in self.damage]

    @property
    def damaged_before_end(self) -> bool:
        return any(kind != "tail" for kind in self.kinds)

    @property
    def live(self) -> bool:
        return "tail" in self.kinds and time.time() - self.modified < LIVE_SECONDS


def keys(records) -> Tuple[List[Tuple[int, int]], str]:
    """(t, n) of every record a reader yields, and how it failed if it did.

    The legacy reader can raise on lines the new one skips (it calls `.get`
    on whatever JSON parsed), so a failure is recorded rather than allowed to
    end the audit.
    """
    found: List[Tuple[int, int]] = []
    try:
        for received, sequence, _ in records:
            found.append((received, sequence))
    except Exception as exc:  # noqa: BLE001 - reporting it is the point
        return found, f"{type(exc).__name__}: {exc}"
    return found, ""


def audit_file(path: Path, *, verify: bool) -> FileReport:
    stat = path.stat()
    report = FileReport(path=path, size=stat.st_size, modified=stat.st_mtime)

    def note(kind: str, start: int, end: int) -> None:
        report.damage.append((kind, start, end))

    for _ in iter_lines(path, on_damage=note):
        pass

    if verify or report.damaged_before_end:
        old, report.old_failure = keys(legacy_iter_events(path))
        new, _ = keys(iter_events(path, warn=False))
        report.old, report.new = len(old), len(new)
        report.old_is_prefix = new[:len(old)] == old
    return report


def describe(report: FileReport) -> str:
    counts = {}
    for kind in report.kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return ", ".join(f"{n} {kind}" if n > 1 else kind for kind, n in counts.items())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--verify", action="store_true",
                        help="count every file with both readers, not just damaged ones")
    args = parser.parse_args(argv)

    root = args.data_dir
    files = sorted(root.rglob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"No *.jsonl.gz under {root.resolve()}. Pass --data-dir "
                         f"pointing at the directory perpkit.record writes into.")

    started = time.time()
    reports: List[FileReport] = []
    for index, path in enumerate(files, start=1):
        reports.append(audit_file(path, verify=args.verify))
        if index % 250 == 0:
            print(f"  {index}/{len(files)} files, {time.time() - started:.0f}s",
                  file=sys.stderr)

    total_mb = sum(r.size for r in reports) / 1e6
    damaged = [r for r in reports if r.damaged_before_end]
    tail_only = [r for r in reports if r.damage and not r.damaged_before_end]
    live = [r for r in tail_only if r.live]
    clean = [r for r in reports if not r.damage]

    print(f"\nScanned {len(reports)} file(s), {total_mb:,.0f} MB, "
          f"in {time.time() - started:.0f}s under {root.resolve()}\n")
    print(f"  clean, every member terminated            {len(clean):6d}")
    print(f"  unterminated last member only             {len(tail_only):6d}"
          f"   ({len(live)} modified in the last {LIVE_SECONDS // 60} min: "
          f"likely still recording)")
    print(f"  torn or corrupt member before the end     {len(damaged):6d}"
          f"   <- the old reader stops early here")

    short = [r for r in damaged if r.new > r.old]
    if damaged:
        width = max(len(str(r.path.relative_to(root))) for r in damaged)
        print(f"\n  {'file':<{width}}  {'old':>9}  {'new':>9}  {'recovered':>9}  damage")
        for r in damaged:
            flags = []
            if r.old_failure:
                flags.append(f"old reader raised {r.old_failure}")
            if not r.old_is_prefix:
                flags.append("old reader yielded records the new one does not")
            print(f"  {str(r.path.relative_to(root)):<{width}}  {r.old:>9,}  "
                  f"{r.new:>9,}  {r.new - r.old:>9,}  {describe(r)}"
                  + ("  [" + "; ".join(flags) + "]" if flags else ""))
        print(f"\n  {len(short)} file(s) where the old reader returned fewer records; "
              f"{sum(r.new - r.old for r in short):,} record(s) recovered in total.")
    else:
        print("\n  No file has damage before its end, so no hour read short under "
              "the old reader.")

    if args.verify:
        disagree = [r for r in reports if not r.damaged_before_end
                    and (r.old != r.new or not r.old_is_prefix)]
        print(f"\n  --verify: readers disagree on {len(disagree)} file(s) the old "
              f"reader did not stop early on (must be 0).")
        for r in disagree:
            print(f"    {r.path.relative_to(root)}: old {r.old:,}, new {r.new:,}"
                  f"{' [old reader raised ' + r.old_failure + ']' if r.old_failure else ''}")
        return 1 if disagree else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
