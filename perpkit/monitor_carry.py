"""Read an open carry back and score it against the plan. Sends nothing.

    python -m perpkit.monitor_carry --instrument SUI-USDT
    python -m perpkit.monitor_carry --instrument SUI-USDT --json

TEACHING EXAMPLE, NO KNOWN EDGE (see `run_carry.py`).

There is no `--confirm` here and no code path to `placeOrder`, deliberately:
this is the tool you want to be able to run at 3am without reading the source
first to check what it might do.

The baseline
------------
A carry opened by a run that persisted nothing has its prediction (fill
prices, sizes, the funding median, the timestamp) only in a terminal. Rather
than declare it unmeasurable, the baseline is RECONSTRUCTED from the exchange
on first run:
the position carries its entry and creation time, the fills carry their fees,
and the executor tagged both legs with the same client-order id, which is what
makes the spot side findable at all.

It is then written to `data/<INST>/carry/baseline.json` and never rewritten,
so the forecast the position is scored against stops moving the moment it is
first recorded. Snapshots append to `snapshots.jsonl` beside it. A 30-day hold
outlives every process that touches it, so the file is the memory.

The prediction it scores against
--------------------------------
`--funding-per-day-bps` pins the exact number the plan printed. Without it the
median is recomputed from the same funding history the planner used, which
reproduces the plan's method but not necessarily its number, since the history
has moved on. Either way it is frozen into the baseline on first run and the
report says which it used.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional


from perpkit.layout import carry_dir
from perpkit.config import DATA_DIR
from perpkit.guardrails import read_host
from perpkit.keys_env import blofin_credentials
from perpkit.plan_carry import (
    MEASURED_FEE_BUFFER_BPS,
    decimal_of,
    funding_per_day_bps,
)
from perpkit.supervise import log
from perpkit.strategies.carry import (
    Baseline,
    build_snapshot,
    compare,
    reconstruct_baseline,
)
from perpkit.margin_tiers import maintenance_margin_rate
from perpkit.risk import RiskLimits, Side, liquidation_price

BASELINE_FILE = "baseline.json"
SNAPSHOT_FILE = "snapshots.jsonl"


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        # Plain notation, never scientific. `str()` on a Decimal that has been
        # through a division renders exact zero as "0E+39", which round-trips
        # correctly and reads like a bug in a file whose whole purpose is
        # being opened months later and believed.
        return format(value, "f")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def load_baseline(path: Path) -> Optional[Baseline]:
    """The frozen baseline, or None if this instrument has never been scored."""
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {}
    for key, value in raw.items():
        if key in ("inst_id", "position_id", "tag"):
            fields[key] = value
        elif key == "opened_at_ms":
            fields[key] = int(value)
        elif value is None:
            fields[key] = None
        else:
            fields[key] = Decimal(str(value))
    return Baseline(**fields)


def save_baseline(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(asdict(baseline)), indent=2),
                    encoding="utf-8")


def append_snapshot(path: Path, snapshot, report) -> None:
    """One line per reading, raw inputs included.

    The derived funding figure is stored beside the three numbers it was
    derived from. If the accounting identity behind it ever turns out to be
    wrong, a history that kept only the conclusion could not be re-scored.
    """
    row = _jsonable(asdict(snapshot))
    row["funding_booked_usd"] = _jsonable(snapshot.funding_booked_usd)
    row["days_elapsed"] = _jsonable(report.days_elapsed)
    row["realised_funding_per_day_bps"] = (
        _jsonable(report.realised_funding_per_day_bps)
        if report.realised_funding_per_day_bps is not None else None)
    row["net_delta_usd"] = _jsonable(report.net_delta_usd)
    row["total_pnl_usd"] = _jsonable(report.total_pnl_usd)
    row["alerts"] = [alert.code for alert in report.alerts]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


def read_position(client, inst_id: str) -> Optional[Dict[str, Any]]:
    payload = client.get("/api/v1/account/positions",
                         params={"instId": inst_id}, sign=True)
    for row in payload.get("data") or []:
        if row.get("instId") == inst_id and decimal_of(row.get("positions")):
            return row
    return None


def read_spot_balance(client, currency: str) -> Decimal:
    payload = client.get("/api/v1/asset/balances",
                         params={"accountType": "spot"}, sign=True)
    for row in payload.get("data") or []:
        if row.get("currency") == currency:
            return decimal_of(row.get("available"))
    return Decimal(0)


def read_perp_orders(client, inst_id: str) -> List[Dict[str, Any]]:
    payload = client.get("/api/v1/trade/orders-history",
                         params={"instId": inst_id, "limit": "100"}, sign=True)
    return list(payload.get("data") or [])


def read_perp_fills(client, inst_id: str) -> List[Dict[str, Any]]:
    payload = client.get("/api/v1/trade/fills-history",
                         params={"instId": inst_id, "limit": "100"}, sign=True)
    return list(payload.get("data") or [])


def read_spot_orders(client, inst_id: str) -> List[Dict[str, Any]]:
    # `instType` is required here and optional on the futures equivalent.
    payload = client.get("/api/v1/spot/trade/orders-history",
                         params={"instType": "SPOT", "instId": inst_id,
                                 "limit": "100"}, sign=True)
    return list(payload.get("data") or [])


def read_spot_mark(public_client, inst_id: str) -> Decimal:
    from perpkit.analysis.blofin_spot import fetch_spot_tickers, top_of_book

    book = top_of_book(fetch_spot_tickers(public_client).get(inst_id))
    if not book:
        return Decimal(0)
    return (decimal_of(book[0]) + decimal_of(book[1])) / 2


def read_funding_rates(api, inst_id: str, pages: int) -> List[Dict[str, Any]]:
    from perpkit.analysis.funding_carry import fetch_funding_history

    return list(fetch_funding_history(api, inst_id, pages=pages))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ")


def _usd(value: Optional[Decimal]) -> str:
    return "n/a" if value is None else f"${value:+,.4f}"


def report(baseline: Baseline, snapshot, result, *, environment: str,
           model_liquidation: Optional[Decimal]) -> None:
    base = baseline.inst_id.split("-")[0]
    print("\n" + "=" * 78)
    print(f"CARRY MONITOR  {baseline.inst_id}  ({environment})")
    print("=" * 78)

    print("\n  POSITION")
    print(f"    opened               {_stamp(baseline.opened_at_ms)}  "
          f"({result.days_elapsed:.2f} days ago)")
    if not snapshot.open:
        print("    perp                 GONE - no open position")
    else:
        print(f"    perp                 {snapshot.perp_contracts} contracts "
              f"@ {baseline.perp_entry}")
    print(f"    spot                 {snapshot.spot_base} {base} "
          f"@ {baseline.spot_entry}")
    print(f"    notional             ${baseline.notional_usd:,.2f}")

    print("\n  DELTA")
    share = (abs(result.net_delta_usd) / baseline.notional_usd
             if baseline.notional_usd else Decimal(0))
    print(f"    net                  {result.net_delta_base:+} {base} "
          f"= ${result.net_delta_usd:+.2f}  ({share:.3%} of notional)")

    if snapshot.open:
        print("\n  RISK ON THE SHORT LEG")
        print(f"    mark                 {snapshot.perp_mark}")
        if result.liquidation_distance is not None:
            print(f"    liquidation          {snapshot.liquidation}  "
                  f"({result.liquidation_distance:.1%} away)")
        if model_liquidation is not None and snapshot.liquidation:
            error = ((model_liquidation - snapshot.liquidation)
                     / snapshot.liquidation * Decimal("10000"))
            print(f"    our model says       {model_liquidation:.6f}  "
                  f"({error:+.2f} bps out)")
        if snapshot.actual_mmr is not None:
            print(f"    MMR charged          {snapshot.actual_mmr:.5f}")

    print("\n  FUNDING")
    print(f"    settlements          {result.funding_periods}")
    print(f"    booked               {_usd(result.funding_booked_usd)}"
          f"   (realizedPnl + fees - fillPnl)")
    print(f"    implied by rates     {_usd(result.funding_implied_usd)}"
          f"   (the control, priced at today's notional)")
    if result.realised_funding_per_day_bps is not None:
        print(f"    realised             "
              f"{result.realised_funding_per_day_bps:+.2f} bps/day")
    print(f"    predicted            "
          f"{baseline.planned_funding_per_day_bps:.2f} bps/day")

    print("\n  SCOREBOARD")
    print(f"    round trip           {result.round_trip_bps:.2f} bps   "
          f"(fees actually charged x2; entry spreads are not recoverable)")
    if result.planned_breakeven_days is not None:
        print(f"    break-even predicted {result.planned_breakeven_days:.1f} days")
    if result.realised_breakeven_days is not None:
        print(f"    break-even realised  {result.realised_breakeven_days:.1f} days")
    else:
        print("    break-even realised  n/a - funding has not paid yet")
    if result.projected_net_bps is not None:
        print(f"    projected {baseline.hold_days}d        "
              f"{result.projected_net_bps:+.1f} bps = "
              f"${result.projected_net_usd:+.2f}   (on the REALISED rate)")
    else:
        print(f"    projected {baseline.hold_days}d        n/a")

    print("\n  MARK TO MARKET")
    print(f"    perp                 {_usd(result.perp_pnl_usd)}"
          f"   (unrealised + realised, fees included)")
    print(f"    spot                 {_usd(result.spot_pnl_usd)}"
          f"   (held x mark, less what was paid)")
    print(f"    total                {_usd(result.total_pnl_usd)}")

    print("\n" + "=" * 78)
    if result.critical:
        print("ACT NOW")
    elif any(alert.level == "warn" for alert in result.alerts):
        print("HOLDING, WITH SOMETHING TO SAY")
    else:
        print("HOLDING TO PLAN")
    print("=" * 78)
    for alert in result.alerts:
        print(f"  [{alert.level:<8}] {alert.message}")
    if not result.alerts:
        print("  Nothing to report: both legs on, delta flat, funding on "
              "forecast.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--hold-days", type=Decimal, default=Decimal("30"),
                        help="The hold the plan predicted over. Frozen into "
                             "the baseline on first run.")
    parser.add_argument("--funding-per-day-bps", type=Decimal, default=None,
                        help="Pin the prediction to the number the plan "
                             "printed. Without it the median is recomputed.")
    parser.add_argument("--funding-pages", type=int, default=6)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable report on stdout instead.")
    args = parser.parse_args(argv)

    base_url, environment = read_host(production=args.production)
    api_key, secret, passphrase = blofin_credentials()

    from blofin.client import Client
    from blofin.rest_market import MarketAPI

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    public = Client(baseUrl=base_url)
    api = MarketAPI(public)

    inst_id = args.instrument
    base_currency = inst_id.split("-")[0]
    # Not created yet: an instrument with no carry on it should not acquire a
    # directory just because someone asked whether it had one.
    directory = carry_dir(DATA_DIR, inst_id)
    baseline_path = directory / BASELINE_FILE
    snapshot_path = directory / SNAPSHOT_FILE

    if not args.json:
        log(f"{environment}: reading {inst_id} back...")

    position = read_position(client, inst_id)
    baseline = load_baseline(baseline_path)

    if baseline is None:
        if position is None:
            raise SystemExit(
                f"No open {inst_id} position on {environment} and no baseline "
                f"at\n{baseline_path}.\nThere is nothing to monitor: a "
                f"baseline can only be reconstructed while the position that "
                f"produced it still exists.")
        planned = args.funding_per_day_bps
        if planned is None:
            planned = funding_per_day_bps(api, inst_id, args.funding_pages)
            if not args.json:
                log(f"  no --funding-per-day-bps given; recomputed the median "
                    f"at {planned:.2f} bps/day")
        baseline = reconstruct_baseline(
            position,
            read_perp_orders(client, inst_id),
            read_spot_orders(client, inst_id),
            planned_funding_per_day_bps=planned,
            hold_days=args.hold_days,
        )
        if baseline is None:
            raise SystemExit(
                f"Could not reconstruct a baseline for {inst_id}: the "
                f"position is missing an entry price, size or creation time.")
        if baseline.tag is None:
            raise SystemExit(
                f"The open {inst_id} position was not opened by "
                f"run_carry.py - its orders carry no `carry<tag>` client id, "
                f"so the spot leg hedging it cannot be identified.\nA guessed "
                f"pairing would invent a hedge that may not exist.")
        save_baseline(baseline_path, baseline)
        if not args.json:
            log(f"  baseline reconstructed and frozen at {baseline_path}")

    snapshot = build_snapshot(
        at_ms=int(time.time() * 1000),
        inst_id=inst_id,
        position=position,
        spot_base=read_spot_balance(client, base_currency),
        spot_mark=read_spot_mark(public, inst_id),
        fills=read_perp_fills(client, inst_id),
        funding_rates=read_funding_rates(api, inst_id, args.funding_pages),
        baseline=baseline,
    )
    result = compare(baseline, snapshot, limits=RiskLimits())
    append_snapshot(snapshot_path, snapshot, result)

    # The model, run again on the position that exists, against the exchange's
    # own number. Free to compute and it re-validates the liquidation formula
    # on every reading rather than once at open.
    model_liquidation = None
    if position:
        leverage = decimal_of(position.get("leverage"), "0")
        mmr = maintenance_margin_rate(client, inst_id,
                                      abs(baseline.perp_contracts))
        if mmr is not None and leverage > 0:
            model_liquidation = liquidation_price(
                entry_price=baseline.perp_entry, leverage=leverage,
                side=Side.SHORT, maintenance_margin_rate=mmr,
                fee_buffer_bps=MEASURED_FEE_BUFFER_BPS)

    if args.json:
        payload = _jsonable(asdict(result))
        payload["alerts"] = [asdict(alert) for alert in result.alerts]
        payload["model_liquidation"] = (str(model_liquidation)
                                        if model_liquidation else None)
        print(json.dumps(payload, indent=2))
    else:
        report(baseline, snapshot, result, environment=environment,
               model_liquidation=model_liquidation)

    return 1 if result.critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
