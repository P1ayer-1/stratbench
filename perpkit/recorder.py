"""Writes feature rows to disk with forward-looking labels attached.

Why this module exists
----------------------
None of the ML in the plan — LightGBM, the regime model, anything — can be
built without a dataset of (features at time t, what happened after t). This
records exactly that, so that after a few days of running you have something
real to train on.

The one thing this file must not get wrong: **lookahead bias**. A row is only
written once enough real time has passed to observe its label, and the label
is computed strictly from mid-prices timestamped *after* the feature snapshot.
Rows still inside their horizon sit in a pending buffer and are never written.
Get this wrong and you train a model with a 90% hit rate that loses money on
every live trade.

Output: one CSV per UTC day in the data directory. CSV rather than parquet by
default because it has zero dependencies, appends safely, and survives the
process being killed mid-write — all of which matter more than file size for a
recorder you leave running. `to_parquet.py`-style conversion is trivial later.
"""

from __future__ import annotations

import csv
import os
import time
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .features import FeatureSnapshot, feature_columns


@dataclass
class LabelConfig:
    """Forward horizons (seconds) and the move size that counts as a signal.

    `threshold_bps` should be set relative to your actual round-trip cost.
    Labelling a 1bps move as "up" when a round trip costs 10bps trains the
    model to find moves it cannot profitably capture.

    The defaults are minutes, not seconds, and that is deliberate: the
    measured standard deviation of a 30-second BTC move is 2.41bps against a
    1.2-10bps round trip, so second-scale horizons are unprofitable by
    arithmetic before any model is involved. See `perpkit/config.py` for the
    full calculation. Both values are overridden from config in the live path.
    """

    horizons_seconds: Tuple[float, ...] = (300.0, 900.0, 1800.0)
    threshold_bps: float = 10.0


class FeatureRecorder:
    """Buffers feature snapshots, labels them once their future is known,
    and appends completed rows to a daily CSV."""

    def __init__(
        self,
        data_dir: Path,
        *,
        label_config: Optional[LabelConfig] = None,
        sample_interval_ms: int = 250,
        flush_every: int = 50,
        enabled: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.labels = label_config or LabelConfig()
        self.sample_interval_ms = sample_interval_ms
        self.flush_every = flush_every
        self.enabled = enabled

        self.max_horizon_ms = int(max(self.labels.horizons_seconds) * 1000)

        # Snapshots waiting for their forward window to close.
        self._pending: Deque[FeatureSnapshot] = deque()
        # Mid prices used to resolve labels, as two parallel ordered lists
        # rather than a deque of tuples. Must outlive the longest horizon.
        #
        # Lists, because label lookup is a `bisect` over `_mid_ts` and a deque
        # cannot be bisected in better than O(n). That distinction did not
        # matter at a 30s horizon; at 1800s the buffer holds ~30x more
        # samples AND every lookup targets its far end, so a linear scan made
        # importing a single day take hours. Same reasoning as `_Series` in
        # features.py.
        self._mid_ts: List[int] = []
        self._mid_values: List[float] = []

        self._last_sample_ms = 0
        self._latest_ts = 0
        self._rows_since_flush = 0
        self._handle = None
        self._writer: Optional[csv.DictWriter] = None
        self._open_date: Optional[str] = None
        self._open_path: Optional[Path] = None

        self.rows_written = 0
        self.rows_dropped = 0

    # ---- column layout ---------------------------------------------------

    def _columns(self) -> List[str]:
        columns = feature_columns()
        for horizon in self.labels.horizons_seconds:
            tag = self._tag(horizon)
            columns.append(f"fwd_ret_bps_{tag}")
            columns.append(f"label_{tag}")
        return columns

    @staticmethod
    def _tag(horizon: float) -> str:
        return f"{horizon:g}s".replace(".", "p")

    # ---- ingestion -------------------------------------------------------

    def observe(self, snapshot: FeatureSnapshot) -> None:
        """Feed every computed snapshot in. Downsampling happens here, not at
        the call site, so the feature engine keeps full resolution for its own
        rolling calculations while the file stays a manageable size."""
        if not self.enabled or not snapshot.is_valid or snapshot.mid is None:
            return

        self._latest_ts = max(self._latest_ts, snapshot.ts)
        # Guard the bisect's precondition: out-of-order arrivals would corrupt
        # every subsequent lookup silently. Book events are sequenced, so this
        # should never fire, but "should never" is not a guarantee to bet a
        # training set on.
        if self._mid_ts and snapshot.ts < self._mid_ts[-1]:
            self.rows_dropped += 1
            return
        self._mid_ts.append(snapshot.ts)
        self._mid_values.append(snapshot.mid)
        self._trim_mid_history()

        if snapshot.ts - self._last_sample_ms >= self.sample_interval_ms:
            self._last_sample_ms = snapshot.ts
            self._pending.append(snapshot)

        self._drain()

    # Only compact once there is a worthwhile amount to drop; deleting from
    # the front of a list is O(n), so doing it per event would reintroduce the
    # cost this structure exists to avoid.
    _COMPACT_THRESHOLD = 4096

    def _trim_mid_history(self) -> None:
        # Keep a generous margin beyond the longest horizon so label lookups
        # never fall off the front of the buffer.
        cutoff = self._latest_ts - (self.max_horizon_ms * 3)
        index = bisect_left(self._mid_ts, cutoff)
        if index < self._COMPACT_THRESHOLD:
            return
        del self._mid_ts[:index]
        del self._mid_values[:index]

    def _drain(self) -> None:
        """Write every pending row whose full forward window has elapsed."""
        while self._pending:
            snapshot = self._pending[0]
            if self._latest_ts - snapshot.ts < self.max_horizon_ms:
                break  # future not observed yet — the whole point
            self._pending.popleft()
            row = self._label(snapshot)
            if row is not None:
                self._write(row)
            else:
                self.rows_dropped += 1

    def _mid_at_or_after(self, target_ts: int) -> Optional[float]:
        """First observed mid at or after `target_ts`.

        Strictly forward-looking. If there is no sample at or after the target
        (a feed gap), we return None and the row is dropped rather than
        labelled from a price that predates the horizon.
        """
        index = bisect_left(self._mid_ts, target_ts)
        if index >= len(self._mid_ts):
            return None
        return self._mid_values[index]

    def _label(self, snapshot: FeatureSnapshot) -> Optional[Dict[str, object]]:
        if snapshot.mid is None or snapshot.mid <= 0:
            return None

        row: Dict[str, object] = dict(snapshot.to_dict())
        for horizon in self.labels.horizons_seconds:
            future_mid = self._mid_at_or_after(snapshot.ts + int(horizon * 1000))
            if future_mid is None or future_mid <= 0:
                return None
            forward_bps = (future_mid / snapshot.mid - 1.0) * 10_000.0
            tag = self._tag(horizon)
            row[f"fwd_ret_bps_{tag}"] = round(forward_bps, 4)
            if forward_bps >= self.labels.threshold_bps:
                row[f"label_{tag}"] = 1
            elif forward_bps <= -self.labels.threshold_bps:
                row[f"label_{tag}"] = -1
            else:
                row[f"label_{tag}"] = 0
        return row

    # ---- output ----------------------------------------------------------

    @staticmethod
    def _existing_header(path: Path) -> Optional[List[str]]:
        """First line of an existing CSV, split into column names."""
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                first = handle.readline()
        except OSError:
            return None
        if not first.strip():
            return None
        return next(csv.reader([first]), None)

    def _resolve_path(self, today: str, columns: List[str]) -> Path:
        """Today's file, unless its header describes a different schema.

        The recorder appends, and it writes a header only when starting a new
        file. So a run whose label horizons changed since the last run would
        otherwise append rows in the NEW column order underneath the OLD
        header - every value silently filed under the wrong name, in a dataset
        whose entire purpose is to be trained on. Changing PERPKIT_LABEL_HORIZONS
        and restarting is a completely ordinary thing to do, so this rolls to a
        suffixed file instead of corrupting the existing one.
        """
        path = self.data_dir / f"features-{today}.csv"
        for suffix in range(2, 1000):
            header = self._existing_header(path)
            if header is None or header == columns:
                return path
            path = self.data_dir / f"features-{today}-{suffix}.csv"
        raise RuntimeError(f"too many schema-mismatched files for {today}")

    def _ensure_file(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._handle is not None and self._open_date == today:
            return

        self.close()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        columns = self._columns()
        path = self._resolve_path(today, columns)
        is_new = not path.exists() or path.stat().st_size == 0

        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=columns, extrasaction="ignore"
        )
        if is_new:
            self._writer.writeheader()
        self._open_date = today
        self._open_path = path

    def _write(self, row: Dict[str, object]) -> None:
        self._ensure_file()
        assert self._writer is not None and self._handle is not None
        self._writer.writerow(row)
        self.rows_written += 1
        self._rows_since_flush += 1
        # Flush periodically so an unclean shutdown loses seconds of data,
        # not hours of it.
        if self._rows_since_flush >= self.flush_every:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except Exception:
                pass
            self._handle.close()
        self._handle = None
        self._writer = None
        self._open_date = None
        self._open_path = None

    def stats(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "rowsWritten": self.rows_written,
            "rowsDropped": self.rows_dropped,
            "pending": len(self._pending),
            "maxHorizonSeconds": self.max_horizon_ms / 1000.0,
            "file": self._open_path.name if self._open_path else None,
        }
