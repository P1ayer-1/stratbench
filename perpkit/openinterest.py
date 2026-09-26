"""Records BloFin's open interest, the input a liquidation-cluster model needs.

Why this exists
---------------
`ingest.py` subscribes to `books`, `trades` and `funding-rate`. None of those
say anything about *other people's* positions, so nothing in this repo can
estimate where the rest of the market gets liquidated - only where we would
(`risk.py`). Open interest is the missing ingredient: OI added at a price,
bucketed by an assumed leverage mix and rolled forward through the same
maintenance-margin formula `risk.py` already implements, is what a liquidation
heatmap actually is.

That model is not built here. This module only captures the series it would
need, for one reason:

    GET /api/v1/market/open-interest  returns a SNAPSHOT. There is no history
    endpoint, on this venue or any mirror of it.

So OI is exactly like the order book: capture-or-lose. Every hour the recorder
runs without it is an hour for which no leverage map can ever be built, no
matter how good the model written later is. That asymmetry - cheap now,
impossible later - is the whole argument for recording it before the thing
that consumes it exists.

What the endpoint actually does (measured 2026-09-09, not assumed)
------------------------------------------------------------------
  * It updates **once per minute**, stamped on the exact minute boundary.
  * The new value appears **~15s after** that boundary. Polling at :00 gets
    you the previous minute.
  * Called with **no instId it returns every instrument** - 473 of them in one
    response. Recording 15 symbols therefore costs one HTTP request, not 15.

Hence: poll every 20s, write only when an instrument's `ts` changes. That
lands one row per instrument per minute, tolerates a failed request or two,
and never writes the same minute twice.

Cost is negligible against the feed itself - roughly 300 KB/day/instrument
before compression, against ~140 MB/day for book and trades.

Why a separate process
----------------------
`RawEventLog` writes one file per channel per hour, so this writes
`open-interest-14.jsonl.gz` beside the running recorder's `books-14.jsonl.gz`
and cannot touch it. That is what makes `record_oi.py` startable against a
recorder that is already hours into a run: no restart, no lost continuity, no
shared file handle. The archive is the half that cannot be re-collected, so
"start it without stopping anything" is a requirement, not a convenience.

That safety holds against the BOOK and TRADE files. It does NOT hold against
another open-interest poller. In an earlier version two of them on one instrument
appended interleaved gzip members to the same hourly file, and the result did
not decode - measured, 20 rows each from two writers produced a file yielding
**0 of 40** records. `RawEventLog` now creates every file exclusively, so a
second writer gets its own (`open-interest-14.r001.jsonl.gz`) and nothing is
destroyed. What remains is every minute archived twice, which the `ts` dedupe
cannot prevent, being per-process state.

So each poller still takes an exclusive per-instrument lock and a second one
is refused. `record.py` starts a poller of its own, which makes `record_oi.py`
useful only against a run that predates that change.

Two consequences of that separation, both of which matter when reading the
archive back:

  * `n`, the monotonic counter in each raw line, is **per process**. This
    process starts its own at 1, so `(t, n)` does not order OI against book
    events. Merge OI on the timestamp alone (an as-of join), never on `n`.
  * `perpkit/analysis/replay.py` deliberately ignores this channel. Nothing computes
    an OI feature yet, and replay's job is to reproduce exactly what the
    feature engine saw. When an OI feature is added, add the channel there in
    the same change.

No SDK import
-------------
This uses `urllib` rather than the BloFin SDK's `Client`, so the `perpkit`
package stays importable and testable with no SDK, no credentials and no
network - the rule stated in `perpkit/__init__.py`. The endpoint is public and
unsigned, so the SDK would have contributed nothing but a dependency.

Production, always
------------------
Public market data is read from the production host regardless of
`BLOFIN_USE_DEMO`. A dataset that describes the demo environment's book rather
than the real market's is worse than no dataset, because it looks the same.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .rawlog import RawEventLog

CHANNEL = "open-interest"
PRODUCTION_BASE_URL = "https://openapi.blofin.com"
OPEN_INTEREST_PATH = "/api/v1/market/open-interest"

# The endpoint publishes on the minute, ~15s late. 20s spacing means three
# chances to catch each minute's value, so one timed-out request costs nothing.
DEFAULT_POLL_SECONDS = 20.0
DEFAULT_TIMEOUT_SECONDS = 15.0

# Measured, not defensive: the endpoint returns 403 to urllib's default
# `Python-urllib/3.x` User-Agent and 200 to literally any other value. Sending
# nothing here means every poll fails forever while looking like a network
# problem, so this header is load-bearing.
USER_AGENT = "blofin-recorder/1.0"


def fetch_snapshot(
    *,
    path: str,
    base_url: str = PRODUCTION_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    """Every instrument's current open interest, in one request.

    Returns the raw rows exactly as the exchange sends them - `instId`,
    `openInterest` (contracts), `openInterestCurrency` (base units) and `ts`.
    They are archived unparsed, so no interpretation happens here.

    Raises on transport failure; the caller decides whether that is fatal
    (it is not - see `poll_once`).
    """
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    fetch = opener or urllib.request.urlopen
    with fetch(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("unexpected payload type " + type(payload).__name__)
    if str(payload.get("code")) != "0":
        raise ValueError(
            "API error {0}: {1}".format(payload.get("code"), payload.get("msg"))
        )

    rows = payload.get("data")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def as_message(row: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a REST row in the websocket envelope the rest of the archive uses.

    Every other file under `raw/` holds messages shaped
    `{"arg": {"channel", "instId"}, "data": [...]}`. Matching that shape means
    a reader does not need to know this channel arrived over HTTP rather than
    a socket, and `iter_directory(root, "open-interest")` behaves exactly like
    it does for `trades`.
    """
    return {
        "arg": {"channel": CHANNEL, "instId": row.get("instId")},
        "data": [row],
    }


def _lock_owner(path: Path) -> Optional[int]:
    """The pid recorded in a lock file, or None if there is no usable one."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


_PYTHON_IMAGES = {"python.exe", "pythonw.exe", "python", "python3"}


def _process_identity(pid: int) -> Optional[Tuple[str, Optional[str]]]:
    """(image name, command line or None) of a live process, or None if no
    process has that pid. Split out so the lock logic is testable without a
    process table; tests monkeypatch this."""
    if os.name == "nt":
        import subprocess
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return ("?", None)   # cannot tell; treated as alive, never clobbered
        import csv
        image = None
        for row in csv.reader((completed.stdout or "").splitlines()):
            if len(row) >= 2 and row[1].strip() == str(pid):
                image = row[0].strip()
                break
        if image is None:
            return None
        cmdline: Optional[str] = None
        try:
            probe = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, timeout=20,
            )
            cmdline = (probe.stdout or "").strip() or None
        except (OSError, subprocess.SubprocessError):
            cmdline = None
        return (image, cmdline)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return ("?", None)
    image, cmdline = "?", None
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip() or None
        image = Path(f"/proc/{pid}/comm").read_text().strip() or "?"
    except OSError:
        pass
    return (image, cmdline)


def _pid_alive(pid: int, holder: Optional[str] = None) -> bool:
    """Is that process still running, and is it one of OURS?

    A lock left by a crashed recorder must not block collection forever - the
    whole argument for this module is that missed hours cannot be recovered.
    Pids are reused: after a reboot, Windows can hand a dead recorder's pid
    to an unrelated system process (an svchost.exe, say), and a bare
    "does this pid exist" test (`tasklist "PID eq"`) then calls a stale lock
    live indefinitely while the supervisor refuses every restart. So a
    pid only counts as alive when its image is a python interpreter, and, if
    `holder` is given and the command line is readable, when every token of
    `holder` (e.g. "record_hyperliquid.py") appears in that command line.
    A pid whose identity cannot be read at all is assumed alive rather than
    clobbered.
    """
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True   # our own lock, whatever this process is called
    identity = _process_identity(pid)
    if identity is None:
        return False
    image, cmdline = identity
    if image == "?":
        return True
    if image.lower() not in _PYTHON_IMAGES:
        return False   # pid reused by something else after a reboot
    if holder and cmdline:
        return all(token in cmdline for token in holder.split())
    return True


class SnapshotPoller:
    """Polls one batch market endpoint and archives whatever moved.

    Subclasses set `CHANNEL` and `PATH`; everything else - the per-instrument
    logs, the `ts` dedupe, the exclusive lock, the never-raise polling loop -
    is the same problem whatever the endpoint returns. The lock is named after
    the CHANNEL, so two pollers on different channels coexist on the same
    instrument and two on the same channel still cannot.

    One poller serves any number of instruments, because one request covers
    the whole venue. Each instrument gets its own `RawEventLog` under
    `data/<INST-ID>/raw/`, so the per-instrument isolation that `record.py`
    exists to enforce holds here too: no shared file, ever.
    """

    CHANNEL: str = ""
    PATH: str = ""

    def __init__(
        self,
        instruments: Sequence[str],
        *,
        data_dir: Optional[Path] = None,
        base_url: str = PRODUCTION_BASE_URL,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        enabled: bool = True,
        opener: Optional[Callable[..., Any]] = None,
    ):
        self.instruments = list(dict.fromkeys(instruments))
        self.data_root = Path(data_dir or Path("data"))
        self.base_url = base_url
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.opener = opener

        # channels= is explicit rather than relying on RawEventLog's default
        # set, so this cannot start writing some other channel if that default
        # is ever changed for the websocket feed's benefit.
        self.logs: Dict[str, RawEventLog] = {
            inst_id: RawEventLog(
                self.data_root / inst_id,
                enabled=enabled,
                channels={self.CHANNEL},
                # One row a minute: buffering 200 lines would hold the newest
                # three hours of a cheap series in memory for no gain.
                flush_lines=1,
            )
            for inst_id in self.instruments
        }

        # instId -> last archived exchange ts. This is the dedupe key, and the
        # reason polling faster than the publish rate costs nothing.
        self._last_ts: Dict[str, str] = {}

        # Two pollers on one instrument used to DESTROY that hour's file.
        # Measured, not theorised: two RawEventLogs appending 20 rows each to
        # the same `open-interest-HH.jsonl.gz` produced a file from which 0 of
        # 40 records could be read back. Now each RawEventLog
        # creates its file exclusively, so the second gets a file of its own
        # and both survive - with every row archived twice, once per file.
        #
        # The per-instrument `ts` dedupe does NOT prevent that - it is
        # per-process state, and the two processes never see each other's.
        #
        # This is reachable: `record.py` starts a poller of its own, and
        # `record_oi.py` exists to be run against a recorder that is already
        # going. So ownership is claimed explicitly, and a second poller is
        # refused rather than allowed to quietly double the archive.
        self._locks: List[Path] = []
        if enabled:
            self._claim(inst_ids=self.instruments)

        self.polls = 0
        self.failures = 0
        self.rows_written = 0
        self.started_at = time.time()
        self.last_success_at: Optional[float] = None
        self._warned_missing: set = set()

    # ---- exclusive ownership ---------------------------------------------

    def _claim(self, inst_ids: Sequence[str]) -> None:
        """Take an exclusive lock on each instrument's open-interest channel.

        `O_CREAT | O_EXCL` is atomic, so two processes racing here cannot both
        win. A lock whose recorded pid is no longer alive is stale - a crash
        or a power cut - and is taken over
        rather than left to block collection forever.
        """
        claimed: List[Path] = []
        try:
            for inst_id in inst_ids:
                path = (self.data_root / inst_id / "raw"
                        / f".{self.CHANNEL}.lock")
                path.parent.mkdir(parents=True, exist_ok=True)
                owner = _lock_owner(path)
                if owner is not None and _pid_alive(owner):
                    raise SystemExit(
                        f"{self.CHANNEL} for {inst_id} is already being "
                        f"recorded by pid {owner}.\n"
                        f"Two pollers on one instrument corrupt the hour's "
                        f"archive beyond recovery -\nmeasured, not theorised. "
                        f"Stop that process, or drop {inst_id} from this one.\n"
                        f"(record.py already runs a poller; record_oi.py is "
                        f"only for a run that predates it.)"
                    )
                if owner is not None:
                    path.unlink(missing_ok=True)   # stale: the owner is gone
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                claimed.append(path)
        except BaseException:
            for path in claimed:
                path.unlink(missing_ok=True)
            raise
        self._locks = claimed

    def _release(self) -> None:
        for path in self._locks:
            # Only ever remove a lock this process owns; a stale-takeover race
            # could otherwise delete the winner's claim.
            if _lock_owner(path) == os.getpid():
                path.unlink(missing_ok=True)
        self._locks = []

    # ---- what a subclass supplies ----------------------------------------

    def fetch(self) -> List[Dict[str, Any]]:
        """Every instrument's current value for this channel, in one request."""
        return fetch_snapshot(
            path=self.PATH,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            opener=self.opener,
        )

    def as_message(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap a REST row in the websocket envelope the archive uses."""
        return {
            "arg": {"channel": self.CHANNEL, "instId": row.get("instId")},
            "data": [row],
        }

    # ---- polling ---------------------------------------------------------

    def poll_once(self, *, on_log: Optional[Callable[[str], Any]] = None) -> int:
        """One request; archive whatever moved. Returns rows written.

        Never raises on a transport or API failure. A poller that dies on a
        dropped connection is a poller that silently stops recording, and the
        hours it misses cannot be fetched later.
        """
        self.polls += 1
        try:
            rows = self.fetch()
        except Exception as exc:
            self.failures += 1
            if on_log:
                on_log("[{0}] poll failed: {1}: {2}".format(
                    self.CHANNEL, type(exc).__name__, exc))
            return 0

        self.last_success_at = time.time()
        written = self._archive(rows, on_log=on_log)
        self.rows_written += written
        return written

    def _archive(
        self,
        rows: Iterable[Dict[str, Any]],
        on_log: Optional[Callable[[str], Any]] = None,
    ) -> int:
        wanted = {row.get("instId"): row for row in rows
                  if row.get("instId") in self.logs}

        missing = set(self.logs) - set(wanted)
        for inst_id in sorted(missing - self._warned_missing):
            # Said once per instrument, not once per poll: a delisted symbol
            # would otherwise fill the log with the same line every 20s.
            self._warned_missing.add(inst_id)
            if on_log:
                on_log("[{0}] {1} absent from the response - delisted, "
                       "or a typo in the instrument list.".format(
                           self.CHANNEL, inst_id))

        written = 0
        for inst_id, row in wanted.items():
            ts = str(row.get("ts", ""))
            if ts and ts == self._last_ts.get(inst_id):
                continue  # same minute, already archived
            self._last_ts[inst_id] = ts
            self.logs[inst_id].write(self.CHANNEL, self.as_message(row))
            written += 1
        return written

    async def run(self, *, on_log: Optional[Callable[[str], Any]] = None) -> None:
        """Poll forever. Intended to run under `perpkit.supervise.supervise`."""
        while True:
            # to_thread because `urllib` is blocking and this may share an
            # event loop with feeds that must not stall for a slow response.
            await asyncio.to_thread(self.poll_once, on_log=on_log)
            await asyncio.sleep(self.poll_seconds)

    # ---- lifecycle -------------------------------------------------------

    def flush(self) -> None:
        for log in self.logs.values():
            log.flush()

    def close(self) -> None:
        for log in self.logs.values():
            log.close()
        self._release()

    def status(self) -> Dict[str, Any]:
        return {
            "instruments": len(self.instruments),
            "polls": self.polls,
            "failures": self.failures,
            "rowsWritten": self.rows_written,
            "uptimeSeconds": time.time() - self.started_at,
            "lastSuccessAgeSeconds": (
                None if self.last_success_at is None
                else time.time() - self.last_success_at
            ),
        }


class OpenInterestPoller(SnapshotPoller):
    """Open interest, polled every 20s and written once a minute.

    The endpoint publishes on the minute and ~15s late, so 20s spacing gives
    three chances to catch each minute's value and the `ts` dedupe throws the
    duplicates away. One timed-out request therefore costs nothing.
    """

    CHANNEL = CHANNEL
    PATH = OPEN_INTEREST_PATH


def fetch_open_interest(
    *,
    base_url: str = PRODUCTION_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    """Every instrument's current open interest, in one request."""
    return fetch_snapshot(path=OPEN_INTEREST_PATH, base_url=base_url,
                          timeout=timeout, opener=opener)
