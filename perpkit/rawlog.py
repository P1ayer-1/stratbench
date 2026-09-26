"""Archives raw websocket messages, exactly as received.

Why this matters more than the feature CSV
------------------------------------------
The feature recorder saves *derived* values. If in three weeks you want a
feature that doesn't exist yet — book slope, depth at 50 levels, queue
position, OFI over a different window, imbalance weighted by distance from mid
— you cannot compute it from `features-*.csv`. The information was thrown away
at write time.

The raw log keeps the source events, so any future feature can be recomputed
over all the history you've collected, and `perpkit/analysis/replay.py` can regenerate
a feature file from scratch. That is the difference between a dataset that
appreciates and one that is frozen the day you designed the schema.

It is also what makes an honest backtest possible: replaying the actual book
updates reproduces the exact sequence the bot saw, including the desyncs.

Format
------
Gzipped JSON Lines, one file per channel per hour per process:

    data/BTC-USDT/raw/2026-09-06/books-14.jsonl.gz
    data/BTC-USDT/raw/2026-09-06/books-14.r001.jsonl.gz   (a restart that hour)

Each line is a compact object (short keys, because they repeat millions of
times):

    {"t": 1757183000123, "n": 48213, "m": {...the exact exchange message...}}

  t = local receive time in ms — when THIS process wrote the message, not
      when the venue sent it. Kept alongside the exchange's own timestamp so
      you can measure feed latency after the fact, which you cannot
      reconstruct later if you don't store it now. Measured on a live
      recorder: p50 86 ms, and up to 56 s in bursts, upstream of this
      process. So `t` orders events, and
      `perpkit.analysis.replay.merged_events(clock="exchange")` places them
      in time; `perpkit/analysis/recorder_lag.py` reports the difference.
  n = a monotonic counter across ALL channels in this process.
  m = the message verbatim, unparsed and unmodified.

`n` exists because millisecond timestamps are not a fine enough clock to
recover arrival order. Book updates arrive ~10/s and trades arrive in bursts,
so a book update and a trade routinely share a millisecond. Merging the
per-channel files on `t` alone would let replay apply them in a different
order than they actually arrived, and a trade applied before rather than after
a book update produces different features. Replay must sort on `(t, n)`.

JSONL + gzip rather than a database because the write pattern is append-only,
the read pattern is a full sequential scan, and a power cut mid-write costs
what had not been flushed instead of a corrupt table - provided nothing is
ever appended after the tear, which is the next section.

A crash must cost seconds, not the hour
---------------------------------------
An earlier version opened each hourly file with `gzip.open(path, "at")`, so
a restart inside an hour appended a second gzip member to the first run's
file. Harmless after a clean exit. After a hard kill - a power cut, or
`os._exit`, or a `timeout` kill - the first member has no
trailer, and a gzip reader cannot tell where it ends: it decodes straight on
into the next member's bytes. Reproduced with real processes: five
records flushed and the process killed, five more appended by a clean
restart, and the reader returned **0 of 10**. It raised on the second
member's header and discarded the output of the read that raised, which held
the first run's five records too.

Both halves changed.

**The writer never appends.** Each file is created exclusively (`"xt"`): if
`books-14.jsonl.gz` exists, this process writes `books-14.r001.jsonl.gz`, then
`r002`, and so on. One file holds one process's output, so a tear can only be
at the end of a file, which the reader has always tolerated. The name keeps
the `<channel>-<HH>` prefix readers glob on and sorts after the hour's first
file, up to `r999` - a thousand restarts in one hour is a crash loop, not a
recording.

Checking whether the existing file ended cleanly, and appending only if so,
was rejected on cost: that check is a full decode, measured at 0.78s for a
39.3 MB `books` hour (IOST-USDT, 184 MB decoded), on the event loop, for
every channel of every instrument at every restart - and it would still race
a second live writer. Exclusive create is one syscall and cannot race. It also
means two writers on one channel (see `openinterest.py`) now write two files
rather than shredding one; the pid locks still refuse that, because the
archive would hold every row twice.

**The reader recovers every member.** Hours written the old way are still on
disk, so `iter_lines` decodes member by member, and a member that does not
terminate is cut off at the next real gzip header rather than decoded into
it. That "rather than" is the whole design. Measured over 30 kills
between flushes (1,000 to 30,000 unflushed trades), a decoder that only
resynchronises when zlib raises carried on through the next member as if it
continued the torn deflate block: in 13 of 30 it never raised, silently
swallowing the member a restart appended, and in 2 it completed the torn last
line into a well-formed record with the next sequence number and the wrong
contents - which no ordering check could catch. So a torn member is only ever
fed the bytes in front of the next header.

A header found by searching has to be real. Its three magic bytes also occur
by chance in compressed data, about once per 16 MB, so a candidate must carry
only the header flag Python's gzip writer sets (FNAME) - by the header layout,
1 chance match in 16 would otherwise pass zlib's own checks with FEXTRA set,
whose length field can swallow any trial decode - and must decode cleanly
from a fresh decoder up to the next candidate.
Compressed data read as a new stream fails within a few bytes.

`perpkit/analysis/audit_raw.py` says which archived hours the old reader cut short.
"""

from __future__ import annotations

import gzip
import itertools
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, TextIO, Tuple


class RawEventLog:
    """Hourly-rotated, gzipped archive of raw feed messages.

    Writes are buffered and flushed on a line/time threshold, so the asyncio
    event loop is not doing a gzip syscall on every book update. Every file it
    writes is one it created; see the module docstring for why it never
    appends to an existing one.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: bool = True,
        channels: Optional[set] = None,
        flush_lines: int = 200,
        flush_seconds: float = 5.0,
        compress_level: int = 6,
    ):
        self.root = Path(data_dir) / "raw"
        self.enabled = enabled
        # `tickers` is deliberately excluded by default: it is redundant with
        # the book's top level and would roughly double the file count for no
        # extra information.
        self.channels = channels or {"books", "books5", "trades", "funding-rate"}
        self.flush_lines = flush_lines
        self.flush_seconds = flush_seconds
        self.compress_level = compress_level

        self._handles: Dict[str, TextIO] = {}
        self._slots: Dict[str, str] = {}
        self._pending: Dict[str, int] = {}
        self._last_flush = time.time()
        # Monotonic across every channel, so replay can recover exact arrival
        # order even when many messages share a millisecond.
        self._sequence = 0

        self.lines_written = 0
        self.bytes_estimate = 0

    # ---- writing ---------------------------------------------------------

    @staticmethod
    def _slot(now: float) -> tuple:
        stamp = datetime.fromtimestamp(now, tz=timezone.utc)
        return stamp.strftime("%Y-%m-%d"), stamp.strftime("%H")

    def _handle_for(self, channel: str, now: float) -> Optional[TextIO]:
        day, hour = self._slot(now)
        slot = f"{day}-{hour}"

        if self._slots.get(channel) == slot:
            return self._handles.get(channel)

        # Hour rolled over (or first write): close the old file cleanly first.
        old = self._handles.pop(channel, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        handle = self._create(directory, channel, hour)
        self._handles[channel] = handle
        self._slots[channel] = slot
        self._pending[channel] = 0
        return handle

    def _create(self, directory: Path, channel: str, hour: str) -> TextIO:
        """Create this hour's file, or the next free restart suffix beside it.

        `x` fails if the file exists, atomically, so neither a restart after a
        crash nor a second live writer can ever add to a file it did not
        create. Zero-padded so the names sort in the order they were made.
        """
        for restart in itertools.count():
            suffix = f".r{restart:03d}" if restart else ""
            path = directory / f"{channel}-{hour}{suffix}.jsonl.gz"
            try:
                return gzip.open(path, "xt", compresslevel=self.compress_level,
                                 encoding="utf-8")
            except FileExistsError:
                continue

    def write(self, channel: str, message: Dict[str, Any]) -> None:
        if not self.enabled or channel not in self.channels:
            return

        now = time.time()
        handle = self._handle_for(channel, now)
        if handle is None:
            return

        self._sequence += 1
        line = json.dumps(
            {"t": int(now * 1000), "n": self._sequence, "m": message},
            separators=(",", ":"),   # no spaces; this is written millions of times
            default=str,
        )
        handle.write(line)
        handle.write("\n")

        self.lines_written += 1
        self.bytes_estimate += len(line) + 1
        self._pending[channel] = self._pending.get(channel, 0) + 1

        if (
            self._pending[channel] >= self.flush_lines
            or now - self._last_flush >= self.flush_seconds
        ):
            self.flush()

    def flush(self) -> None:
        for channel, handle in self._handles.items():
            try:
                handle.flush()
            except Exception:
                pass
            self._pending[channel] = 0
        self._last_flush = time.time()

    def close(self) -> None:
        self.flush()
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._slots.clear()

    # ---- reporting -------------------------------------------------------

    def disk_bytes(self) -> int:
        """Actual compressed bytes on disk so far."""
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*.jsonl.gz"))

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "linesWritten": self.lines_written,
            "uncompressedBytes": self.bytes_estimate,
            "diskBytes": self.disk_bytes(),
        }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_GZIP_MAGIC = b"\x1f\x8b\x08"   # ID1, ID2, CM=deflate: how every member begins
_FNAME = 0x08                   # the only header flag Python's gzip writer sets
_READ_CHUNK = 1 << 16           # compressed bytes decoded per step
_PROBE_BYTES = 1 << 12          # how far a candidate header is trial-decoded

DamageCallback = Callable[[str, int, int], None]

_DAMAGE_WARNINGS = {
    "tail": "  {name}: stopped at an incomplete tail - still being written, "
            "or truncated.",
    "torn": "  {name}: the gzip member at byte {start:,} stops without a "
            "trailer at byte {end:,} (a writer killed before a restart "
            "appended to the file); resumed at the next member.",
    "corrupt": "  {name}: undecodable data at byte {end:,} in the gzip member "
               "starting at byte {start:,}; skipped to the next member, if any.",
    "junk": "  {name}: skipped bytes {start:,}-{end:,}, which are not gzip.",
}


class _Window:
    """A forward-only view of a file, addressed by absolute byte offset.

    Holds one decode step plus a probe's worth of look-ahead, never the whole
    file: `replay.py` opens every file of a run at once through
    `heapq.merge`, so reading whole files would hold the archive in RAM.
    Measured merging one IOST-USDT day (96 files, 505 MB
    compressed): 27 MB held, 46 MB peak - about half a megabyte per open file,
    against 8 MB peak for the `gzip` reader this replaced. Same speed.
    """

    def __init__(self, handle):
        self._handle = handle
        self._data = bytearray()
        self._base = 0
        self._exhausted = False

    @property
    def end(self) -> int:
        return self._base + len(self._data)

    def fill(self, until: int) -> None:
        """Hold the file up to `until`, or to its end if it is shorter."""
        while self.end < until and not self._exhausted:
            block = self._handle.read(max(until - self.end, _READ_CHUNK))
            if block:
                self._data += block
            else:
                self._exhausted = True

    def release(self, before: int) -> None:
        drop = before - self._base
        if drop > 0:
            del self._data[:drop]
            self._base = before

    def slice(self, start: int, end: int) -> bytearray:
        return self._data[start - self._base:end - self._base]

    def find_magic(self, start: int, end: int) -> int:
        """First gzip magic lying wholly inside [start, end), or -1."""
        found = self._data.find(_GZIP_MAGIC, start - self._base, end - self._base)
        return -1 if found < 0 else found + self._base


def _inflate(decoder, data) -> Tuple[bytes, Any, Optional[int]]:
    """Decompress `data`, keeping the output in front of a zlib error.

    zlib raises without returning anything from the call that failed, so one
    call over a whole chunk would throw away every good record in that chunk
    ahead of the fault. On failure the decoder is rewound and the chunk
    replayed in smaller pieces until the failing byte is isolated.

    Returns (output, decoder, offset of the failing byte or None). After a
    failure the decoder returned is the one positioned just before it.
    """
    checkpoint = decoder.copy()
    try:
        return decoder.decompress(data), decoder, None
    except zlib.error:
        if len(data) <= 1:
            return b"", checkpoint, 0
    decoder, output = checkpoint, []
    step = max(1, len(data) // 16)
    for start in range(0, len(data), step):
        piece, decoder, failed = _inflate(decoder, data[start:start + step])
        output.append(piece)
        if failed is not None:
            return b"".join(output), decoder, start + failed
    return b"".join(output), decoder, None


def _is_member_start(data: _Window, at: int) -> bool:
    """Does a gzip member really begin at `at`, or is the magic a chance match?

    A real member written here has no header flag but FNAME, and decodes
    cleanly from a fresh decoder - at least up to the next candidate, since a
    member torn by a kill fails no earlier than where the member after it
    begins. Compressed data read as the start of a stream fails within a few
    bytes.
    """
    data.fill(at + _PROBE_BYTES)
    head = data.slice(at, at + 4)
    if len(head) == 4 and head[3] & ~_FNAME:
        return False
    end = min(data.end, at + _PROBE_BYTES)
    _, _, failed = _inflate(zlib.decompressobj(31), data.slice(at, end))
    if failed is None:
        return True
    following = data.find_magic(at + 1, end)
    return following >= 0 and at + failed >= following


def _next_member(data: _Window, start: int) -> Tuple[Optional[int], bool]:
    """The first real member at or after `start`, reading forward.

    Returns (offset, or None at end of file; whether any byte skipped on the
    way was something other than zero padding).
    """
    position, nonzero = start, False
    while True:
        data.fill(position + _READ_CHUNK + _PROBE_BYTES)
        if position >= data.end:
            return None, nonzero
        end = min(data.end, position + _READ_CHUNK)
        # A magic straddling `end` starts in this step, so look slightly past.
        search_end = min(data.end, end + len(_GZIP_MAGIC) - 1)
        candidate = data.find_magic(position, search_end)
        while candidate >= 0 and not _is_member_start(data, candidate):
            candidate = data.find_magic(candidate + 1, search_end)
        stop = end if candidate < 0 else candidate
        nonzero = nonzero or bool(data.slice(position, stop).strip(b"\0"))
        if candidate >= 0:
            return candidate, nonzero
        position = end
        data.release(position)


def _member_lines(data: _Window, start: int, report: DamageCallback):
    """Yield the lines of the member at `start`; return where to carry on
    (the next member's offset, or None at end of file)."""
    decoder = zlib.decompressobj(31)   # 31: gzip header expected, trailer verified
    position = start
    examined = start + 1               # candidate headers before this are ruled out
    pending = b""
    while True:
        data.fill(position + _READ_CHUNK + _PROBE_BYTES)
        end = min(data.end, position + _READ_CHUNK)
        if position >= end:
            if pending:
                yield pending
            report("tail", start, data.end)
            return None

        # A real header inside this step is where the member must stop being
        # fed, terminated or not. Decoding past it is what invents records.
        search_end = min(data.end, end + len(_GZIP_MAGIC) - 1)
        header = data.find_magic(max(examined, position), search_end)
        while header >= 0 and not _is_member_start(data, header):
            header = data.find_magic(header + 1, search_end)
        stop = end if header < 0 else header
        examined = stop

        output, decoder, failed = _inflate(decoder, data.slice(position, stop))
        lines = (pending + output).split(b"\n")
        pending = lines.pop()
        yield from lines

        if failed is not None:
            if pending:
                yield pending
            report("corrupt", start, position + failed)
            resume, _ = _next_member(data, position + failed + 1)
            return resume
        if decoder.eof:
            if pending:
                yield pending
            return stop - len(decoder.unused_data)
        position = stop
        data.release(position)
        if header >= 0:
            if pending:
                yield pending   # a torn line; the JSON parse rejects it
            report("torn", start, header)
            return header


def _ignore_damage(kind: str, start: int, end: int) -> None:
    pass


def iter_lines(path: Path, *, on_damage: Optional[DamageCallback] = None) -> Iterator[bytes]:
    """Every line of a raw log file, as bytes, from every gzip member in it.

    `on_damage(kind, start, end)` is called for everything that is not a
    cleanly terminated member, with offsets into the compressed file:

      "tail"     the last member has no trailer; it runs from `start` to the
                 end of the file. The current hour looks like this while it is
                 recorded, and a past hour does when its writer was killed.
      "torn"     a member stops without a trailer at `end`, where another one
                 begins: a hard kill, then a restart that appended.
      "corrupt"  a member fails to decode at byte `end` - not a tear at a
                 member boundary, so the bytes themselves are damaged.
      "junk"     bytes `start` to `end` are not gzip, and not zero padding.

    The final line of a member is yielded even without its newline; whether
    it is a whole record is for the JSON parse to decide.
    """
    report = on_damage or _ignore_damage
    with open(path, "rb") as handle:
        data = _Window(handle)
        position: Optional[int] = 0
        while position is not None:
            data.release(position)
            data.fill(position + len(_GZIP_MAGIC))
            if position >= data.end:
                return
            if data.slice(position, position + len(_GZIP_MAGIC)) == _GZIP_MAGIC:
                position = yield from _member_lines(data, position, report)
                continue
            resume, nonzero = _next_member(data, position)
            if nonzero:
                report("junk", position, data.end if resume is None else resume)
            position = resume


def iter_events(path: Path, *, warn: bool = True,
                on_damage: Optional[DamageCallback] = None):
    """Read back one raw log file, yielding (receive_ms, sequence, message).

    Sort merged streams on the (receive_ms, sequence) pair — see the module
    docstring for why the timestamp alone is not sufficient.

    Yields every complete record in the file, from every gzip member, and
    tolerates the damage the archive actually takes:

    **A truncated JSON line**, from the process being killed mid-write. One
    lost message, not a lost file.

    **An unterminated last member**, which is what the CURRENT hour's file
    looks like the entire time it is being appended to. This is the normal
    case, not the exceptional one: analysis runs while recording continues.
    Without this, every tool reading the archive dies on `zlib.error` the
    moment a recorder is live — which is precisely when you want to look.

    **A torn member with more after it**, from a hard kill followed by a
    restart that appended to the same file, as an earlier version of the
    writer did. Its records up to the tear are kept, and reading carries on
    from the next member rather than stopping - see the module docstring for
    what stopping cost.

    None of it is silent: with `warn`, one stderr line per damaged member
    naming the file, because a file still being written and a genuinely
    damaged archive must not look alike. `on_damage` receives the same events
    instead (kinds are listed on `iter_lines`).
    """
    path = Path(path)
    if on_damage is None and warn:
        def on_damage(kind: str, start: int, end: int) -> None:
            print(_DAMAGE_WARNINGS[kind].format(name=path.name, start=start, end=end),
                  file=sys.stderr)

    for line in iter_lines(path, on_damage=on_damage):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a truncated line, or bytes that are not UTF-8 at all
        if not isinstance(record, dict):
            continue
        yield record.get("t", 0), record.get("n", 0), record.get("m", {})


def iter_directory(root: Path, channel: str):
    """Yield (receive_ms, sequence, message) for one channel across all
    archived hours, in chronological order.

    A restart's `<channel>-<HH>.rNNN.jsonl.gz` sorts after that hour's first
    file, so name order is still time order.
    """
    for path in sorted(Path(root).rglob(f"{channel}-*.jsonl.gz")):
        yield from iter_events(path)
