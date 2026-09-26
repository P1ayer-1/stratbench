"""The raw archive: every venue message verbatim, and the reader that gets
it all back.

Why the archive matters most
----------------------------
Most prediction-market backtests run on price history. Few record the
book. Derived data (a rebuilt book, a fill simulation, a labelled window) is
regenerable from this archive; the archive is not regenerable from anything.
Any future feature, any future fill model, is computable over every hour
recorded since day one only if the source events were kept. So: archive
first, parse second. If parsing throws, the event is already on disk.

Format
------
Gzipped JSON Lines, one file per channel per hour per process, under the
market's own directory (see `record.market_dir`):

    data/kalshi/KXBTC15M-26SEP1218/raw/2026-09-12/book-14.jsonl.gz
    data/kalshi/KXBTC15M-26SEP1218/raw/2026-09-12/book-14.r001.jsonl.gz  (a restart)

Each line is `{"t": receive_ms, "n": counter, "m": message}`: `t` is local
receive time, kept beside the venue's own timestamp so feed lag can be
measured after the fact; `n` is monotonic across every channel in the
process, because book updates and trades routinely share a millisecond and
replay must recover arrival order by sorting on `(t, n)`; `m` is the message
untouched. Channels recorded by other processes (a reference feed) have an
unrelated `n`, so join them on `t` only.

The two rules, with the measurements that made them
---------------------------------------------------
**The writer never appends.** Each file is created exclusively (`"xt"`): if
`book-14.jsonl.gz` exists, this process writes `book-14.r001.jsonl.gz`, then
`r002`, up to `r999`. Appending a second gzip member after a hard kill left
the first member without a trailer, and a reader decoded straight through
it into the next member's bytes: reproduced with real processes, five
records flushed, killed, five appended by a clean restart, and the reader
returned 0 of 10. Exclusive create is one syscall,
cannot race a second writer, and means two writers on one channel produce
two files rather than shredding one. Checking whether an existing file
ended cleanly was rejected on cost: a full decode, 0.78 s for a 39 MB hour.

**The reader recovers every member.** Files written by an older appending
writer, or by a writer killed mid-block, are read member by member, and a
member that does not terminate is cut off at the next REAL gzip header
rather than decoded into it. Measured over 30 mid-block kills: a decoder
that only resynchronised on a zlib error swallowed the appended member in
13 of 30 and in 2 completed the torn last line into a well-formed record
with the wrong contents, which no ordering check could catch. A header
found by searching must carry only the FNAME flag Python's writer sets and
must decode cleanly from a fresh decoder up to the next candidate; chance
matches of the three magic bytes occur about once per 16 MB of compressed
data and fail within a few bytes.

The current hour is an unterminated stream for as long as it is recorded,
and analysis runs while recording continues, so that case is the normal one
and is reported as "tail", never raised.
"""

from __future__ import annotations

import gzip
import itertools
import json
import queue
import sys
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, TextIO, Tuple


class _Flush:
    def __init__(self) -> None:
        self.done = threading.Event()


class _Close:
    pass


class RawEventLog:
    """Hourly-rotated, gzipped archive of raw feed messages.

    The event loop only stamps and enqueues. Measured over 8.5 h
    of recording eight Polymarket subscriptions: encoding and gzipping every
    message on the loop let the socket reader fall behind on busy windows
    (200 messages/s), the venue closed the socket 17 times with "slow
    consumer: send buffer full", and each reconnect lost 1.5 to 14 s of
    book updates. So with `threaded=True` (the default) `write` assigns
    `t` and `n` on the caller's thread, which is what keeps arrival order,
    and hands the message to one writer thread that does the JSON, the
    gzip and the file rotation. `flush` and `close` are synchronous: they
    queue a marker and wait for the thread to pass it, so a reader that
    follows a flush sees everything written before it.

    `channels=None` archives every channel: never subscribe to less than
    you might later need, because unrecorded data is gone.
    """

    def __init__(self, data_dir: Path, *, enabled: bool = True,
                 channels: Optional[set] = None, flush_lines: int = 200,
                 flush_seconds: float = 5.0, compress_level: int = 6,
                 clock: Callable[[], float] = time.time, threaded: bool = True):
        self.root = Path(data_dir) / "raw"
        self.enabled = enabled
        self.channels = channels
        self.flush_lines = flush_lines
        self.flush_seconds = flush_seconds
        self.compress_level = compress_level
        self._clock = clock
        self.threaded = threaded

        self._handles: Dict[str, TextIO] = {}
        self._slots: Dict[str, str] = {}
        self._pending: Dict[str, int] = {}
        self._last_flush = clock()
        self._sequence = 0

        self.lines_written = 0          # accepted by write()
        self.lines_on_disk = 0          # written by the thread (== lines_written when not threaded)
        self.bytes_estimate = 0
        self.write_failures = 0
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    # ---- writing ---------------------------------------------------------

    @staticmethod
    def _slot(now: float) -> Tuple[str, str]:
        stamp = datetime.fromtimestamp(now, tz=timezone.utc)
        return stamp.strftime("%Y-%m-%d"), stamp.strftime("%H")

    def _handle_for(self, channel: str, now: float) -> TextIO:
        day, hour = self._slot(now)
        slot = f"{day}-{hour}"
        if self._slots.get(channel) == slot:
            return self._handles[channel]
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

        `x` fails if the file exists, atomically, so neither a restart after
        a crash nor a second live writer can ever add to a file it did not
        create. Zero-padded so the names sort in the order they were made.
        """
        for restart in itertools.count():
            suffix = f".r{restart:03d}" if restart else ""
            path = directory / f"{channel}-{hour}{suffix}.jsonl.gz"
            try:
                return gzip.open(path, "xt", compresslevel=self.compress_level,
                                 encoding="utf-8")
            except FileExistsError:
                if restart >= 999:
                    raise RuntimeError(f"{path.parent}: a thousand restarts in one hour "
                                       "is a crash loop, not a recording")
                continue
        raise AssertionError("unreachable")

    def write(self, channel: str, message: Any) -> None:
        """Stamp, sequence, and either write or enqueue. Never blocks on I/O
        when threaded: the queue is unbounded, and a backlog shows up in
        `stats()["queued"]` rather than in the venue's send buffer.

        `message` is a JSON-serialisable object, or a `str` holding one JSON
        value verbatim from the wire (a websocket text frame). Strings are
        the fast path: no dict is ever built, so the GC has nothing to
        track and the loop does nothing per frame but this call."""
        if not self.enabled or (self.channels is not None and channel not in self.channels):
            return
        if self._closed:
            raise RuntimeError("write after close")
        now = self._clock()
        self._sequence += 1
        self.lines_written += 1
        item = (channel, now, self._sequence, message)
        if not self.threaded:
            self._write_item(item)
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="rawlog-writer", daemon=True)
            self._thread.start()
        self._queue.put(item)

    def _write_item(self, item: Tuple[str, float, int, Any]) -> None:
        channel, now, sequence, message = item
        try:
            handle = self._handle_for(channel, now)
            if isinstance(message, str):
                # A frame exactly as the venue sent it, already JSON: embedded
                # verbatim, never parsed or re-serialised. The loop that
                # received it did no work but a tag and a queue put.
                line = f'{{"t":{int(now * 1000)},"n":{sequence},"m":{message}}}'
            else:
                line = json.dumps({"t": int(now * 1000), "n": sequence, "m": message},
                                  separators=(",", ":"), default=str)
            handle.write(line)
            handle.write("\n")
        except Exception as exc:
            self.write_failures += 1
            if self.write_failures <= 5:
                print(f"  rawlog: write failed ({type(exc).__name__}: {exc}); message lost",
                      file=sys.stderr)
            return
        self.lines_on_disk += 1
        self.bytes_estimate += len(line) + 1
        self._pending[channel] = self._pending.get(channel, 0) + 1
        if (self._pending[channel] >= self.flush_lines
                or now - self._last_flush >= self.flush_seconds):
            self._flush_handles()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if isinstance(item, _Close):
                self._close_handles()
                return
            if isinstance(item, _Flush):
                self._flush_handles()
                item.done.set()
                continue
            self._write_item(item)

    def _flush_handles(self) -> None:
        for channel, handle in self._handles.items():
            try:
                handle.flush()
            except Exception:
                pass
            self._pending[channel] = 0
        self._last_flush = self._clock()

    def _close_handles(self) -> None:
        self._flush_handles()
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._slots.clear()

    def flush(self) -> None:
        """Everything accepted so far is on disk when this returns."""
        if not self.threaded or self._thread is None or not self._thread.is_alive():
            self._flush_handles()
            return
        marker = _Flush()
        self._queue.put(marker)
        marker.done.wait(timeout=60.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.threaded and self._thread is not None and self._thread.is_alive():
            self._queue.put(_Close())
            self._thread.join(timeout=60.0)
            return
        self._close_handles()

    def disk_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*.jsonl.gz"))

    def stats(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "linesWritten": self.lines_written,
                "linesOnDisk": self.lines_on_disk, "queued": self._queue.qsize(),
                "writeFailures": self.write_failures,
                "uncompressedBytes": self.bytes_estimate, "diskBytes": self.disk_bytes()}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_GZIP_MAGIC = b"\x1f\x8b\x08"   # ID1, ID2, CM=deflate: how every member begins
_FNAME = 0x08                   # the only header flag Python's gzip writer sets
_READ_CHUNK = 1 << 16
_PROBE_BYTES = 1 << 12

DamageCallback = Callable[[str, int, int], None]

_DAMAGE_WARNINGS = {
    "tail": "  {name}: stopped at an incomplete tail - still being written, or truncated.",
    "torn": "  {name}: the gzip member at byte {start:,} stops without a trailer at byte "
            "{end:,} (a writer killed before a restart appended); resumed at the next member.",
    "corrupt": "  {name}: undecodable data at byte {end:,} in the gzip member starting at "
               "byte {start:,}; skipped to the next member, if any.",
    "junk": "  {name}: skipped bytes {start:,}-{end:,}, which are not gzip.",
}


class _Window:
    """A forward-only view of a file, addressed by absolute byte offset. Holds
    one decode step plus a probe of look-ahead, never the whole file, so a
    replay that opens every file of a day at once stays in tens of MB."""

    def __init__(self, handle):
        self._handle = handle
        self._data = bytearray()
        self._base = 0
        self._exhausted = False

    @property
    def end(self) -> int:
        return self._base + len(self._data)

    def fill(self, until: int) -> None:
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
        found = self._data.find(_GZIP_MAGIC, start - self._base, end - self._base)
        return -1 if found < 0 else found + self._base


def _inflate(decoder, data) -> Tuple[bytes, Any, Optional[int]]:
    """Decompress, keeping the output in front of a zlib error. zlib returns
    nothing from the call that failed, so on failure the decoder is rewound
    and the chunk replayed in smaller pieces until the failing byte is
    isolated. Returns (output, decoder, failing offset or None)."""
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
    position, nonzero = start, False
    while True:
        data.fill(position + _READ_CHUNK + _PROBE_BYTES)
        if position >= data.end:
            return None, nonzero
        end = min(data.end, position + _READ_CHUNK)
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
    decoder = zlib.decompressobj(31)
    position = start
    examined = start + 1
    pending = b""
    while True:
        data.fill(position + _READ_CHUNK + _PROBE_BYTES)
        end = min(data.end, position + _READ_CHUNK)
        if position >= end:
            if pending:
                yield pending
            report("tail", start, data.end)
            return None
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
                yield pending
            report("torn", start, header)
            return header


def _ignore_damage(kind: str, start: int, end: int) -> None:
    pass


def iter_lines(path: Path, *, on_damage: Optional[DamageCallback] = None) -> Iterator[bytes]:
    """Every line of a raw log file, as bytes, from every gzip member in it.
    `on_damage(kind, start, end)` receives "tail", "torn", "corrupt" or
    "junk" with compressed-file offsets (see the module docstring)."""
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


def _frame_bytes(line: bytes) -> bytes:
    """The message's own bytes out of one archived line: everything after
    the writer's `"m":`, which follows `t` and `n` (two numbers) in every
    line this module writes, up to the record's closing brace. A verbatim
    venue frame comes back byte for byte as the socket delivered it. A line
    in any other shape comes back whole, stamps included, so it can only
    ever equal itself."""
    start = line.find(b'"m":')
    return line[start + 4:-1] if start >= 0 and line.endswith(b"}") else line


def iter_events(path: Path, *, warn: bool = True,
                on_damage: Optional[DamageCallback] = None, frames: bool = False):
    """Yield (receive_ms, sequence, message) for one file, tolerating a
    truncated last line, an unterminated last member (the current hour) and
    torn members with more after them. None of it is silent: with `warn`,
    one stderr line per damaged member names the file, because a file still
    being written and a damaged archive must not look alike.

    With `frames`, each tuple carries a fourth item: the message's bytes as
    archived (`_frame_bytes`). Two copies of one venue frame are equal
    there; re-serialising the parsed message would not be the venue's
    frame, and costs a dump per line."""
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
            continue
        if not isinstance(record, dict):
            continue
        if frames:
            yield record.get("t", 0), record.get("n", 0), record.get("m", {}), _frame_bytes(line)
        else:
            yield record.get("t", 0), record.get("n", 0), record.get("m", {})


def iter_directory(root: Path, channel: str, *, warn: bool = True, frames: bool = False):
    """(receive_ms, sequence, message) for one channel across every archived
    hour, in name order, which is time order: a restart's `.rNNN` file sorts
    after the hour's first. `frames` adds the archived bytes (`iter_events`)."""
    for path in sorted(Path(root).rglob(f"{channel}-*.jsonl.gz")):
        yield from iter_events(path, warn=warn, frames=frames)


def channels_in(root: Path) -> Tuple[str, ...]:
    names = set()
    for path in Path(root).rglob("*.jsonl.gz"):
        names.add(path.name.split("-")[0])
    return tuple(sorted(names))


__all__ = ["RawEventLog", "channels_in", "iter_directory", "iter_events", "iter_lines"]
