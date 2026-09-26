"""Turn finished windows into backtest rows, labelled by the venue.

    python -m predkit.label --data-dir data
    python -m predkit.label --data-dir data --offsets 240,120,60,30

For every window directory under `data/<venue>/` whose contract has
resolved (plus a grace period), the venue is asked for its result once and
the answer is cached in `result.json` beside `contract.json`. Then the
archive is replayed to rebuild the YES book, and at each `offset` seconds
before resolution the best bid and ask are sampled into a `WindowRow`
whose label is the cached result. Rows go to `data/rows/<venue>.jsonl`.

`backtest.label_row` enforces the two rules: a row is written only after
resolution, and only from the contract's own resolution source. This
module never touches Binance or any spot price; `signal` is left empty for
strategies to fill from their own reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from predkit.backtest import WindowRow, label_row, write_rows
from predkit.record_series import read_contract
from predkit.replay import rebuild
from predkit.schema import Contract
from predkit.supervise import log

DEFAULT_OFFSETS = (240, 120, 60, 30)


def window_dirs(data_dir: Path, venue: str) -> List[Path]:
    root = Path(data_dir) / venue
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "contract.json").exists())


def cached_result(directory: Path) -> Optional[bool]:
    path = directory / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("resolved_yes")


def fetch_result(directory: Path, contract: Contract, venue: Any, *, now_s: float) -> Optional[bool]:
    """The venue's result, cached once known. None while unresolved."""
    known = cached_result(directory)
    if known is not None:
        return known
    resolved = venue.resolution(contract.market_id)
    if resolved is None:
        return None
    (directory / "result.json").write_text(json.dumps(
        {"resolved_yes": resolved, "source": contract.resolution_source, "fetched_at_ms": int(now_s * 1000)}),
        encoding="utf-8")
    return resolved


def touch_at_offsets(directory: Path, contract: Contract, offsets: Tuple[int, ...]) -> Dict[int, Tuple[Any, Any, int]]:
    """(best bid, best ask, sample time ms) from the last book seen at or
    before `resolves_at - offset`, for each offset."""
    targets = sorted(((contract.resolves_at_ms - off * 1000), off) for off in offsets)
    out: Dict[int, Tuple[Any, Any, int]] = {}
    last_touch = None                      # a copy: the book state is mutated by the next event
    index = 0
    for t, (kind, event) in rebuild(directory, contract):
        if kind != "book":
            continue
        while index < len(targets) and t > targets[index][0]:
            if last_touch is not None:
                out[targets[index][1]] = last_touch
            index += 1
        if index >= len(targets):
            break
        bid, ask = event.best_bid, event.best_ask
        if bid is not None and ask is not None:
            last_touch = (bid, ask, event.ts_ms or t)
    # Offsets later than the last archived book take that last book.
    while index < len(targets):
        if last_touch is not None:
            out[targets[index][1]] = last_touch
        index += 1
    return out


def rows_for(directory: Path, contract: Contract, resolved_yes: bool, offsets: Tuple[int, ...],
             *, now_s: float, window_s: int) -> List[WindowRow]:
    rows = []
    for offset, (bid, ask, at_ms) in touch_at_offsets(directory, contract, offsets).items():
        if bid >= ask:
            continue                         # a crossed sample is a bad sample
        rows.append(label_row(
            contract, resolved_yes=resolved_yes, resolution_source=contract.resolution_source,
            now_ms=int(now_s * 1000), opens_at_ms=contract.resolves_at_ms - offset * 1000,
            entry_price=ask, entry_bid=bid, signal=None,
            extra={"offset_s": str(offset), "window_s": str(window_s), "sampled_at_ms": str(at_ms)}))
    return rows


def window_seconds(contract: Contract) -> int:
    slug = str(contract.extra.get("slug", ""))
    if "-5m-" in slug:
        return 300
    if "-15m-" in slug or contract.venue == "kalshi":
        return 900
    return 0


def label_venue(data_dir: Path, venue_name: str, venue: Any, *, offsets: Tuple[int, ...] = DEFAULT_OFFSETS,
                grace_s: float = 180.0, now_s: Optional[float] = None) -> Tuple[int, int, int]:
    """Returns (windows resolved, rows written, windows still open)."""
    now_s = time.time() if now_s is None else now_s
    out_path = Path(data_dir) / "rows" / f"{venue_name}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["market_id"], row["extra"].get("offset_s")))
    resolved = pending = written = 0
    for directory in window_dirs(data_dir, venue_name):
        contract = read_contract(directory)
        if contract.resolves_at_ms + grace_s * 1000 > now_s * 1000:
            pending += 1
            continue
        if all((contract.market_id, str(off)) in done for off in offsets):
            resolved += 1
            continue
        try:
            result = fetch_result(directory, contract, venue, now_s=now_s)
        except Exception as exc:
            log(f"[{venue_name}] {contract.market_id}: result lookup failed: {type(exc).__name__}: {exc}")
            pending += 1
            continue
        if result is None:
            pending += 1
            continue
        resolved += 1
        rows = [r for r in rows_for(directory, contract, result, offsets, now_s=now_s,
                                    window_s=window_seconds(contract))
                if (r.market_id, r.extra["offset_s"]) not in done]
        written += write_rows(out_path, rows)
    return resolved, written, pending


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    parser.add_argument("--venue", choices=("polymarket", "kalshi"), default=None)
    args = parser.parse_args(argv)
    offsets = tuple(int(o) for o in args.offsets.split(",") if o.strip())
    from predkit.venues.kalshi import Kalshi
    from predkit.venues.polymarket import Polymarket
    venues = {"polymarket": Polymarket(), "kalshi": Kalshi()}
    for name, venue in venues.items():
        if args.venue and name != args.venue:
            continue
        resolved, written, pending = label_venue(Path(args.data_dir), name, venue, offsets=offsets)
        log(f"{name}: {resolved} windows resolved, {written} rows written, {pending} pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
