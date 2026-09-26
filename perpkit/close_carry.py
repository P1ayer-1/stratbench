"""Take an open carry off, and write down what it actually earned.

    python -m perpkit.close_carry --instrument SUI-USDT
    python -m perpkit.close_carry --instrument SUI-USDT --confirm

TEACHING EXAMPLE, NO KNOWN EDGE (see `run_carry.py`).

Dry unless `--confirm`, demo unless `--production`, and `--production
--confirm` is refused outright, exactly like `run_carry.py`: perpkit never
opens a production carry, so it has none to close. Close a production
position on the exchange yourself.

A separate entrypoint, not `run_carry.py --close`
-------------------------------------------------
If closing were a flag, then typing the open command and forgetting the flag
would OPEN A SECOND CARRY - the most expensive typo available in this
project. One verb per entrypoint removes that failure mode instead of
documenting it.

The last moment funding is observable
-------------------------------------
Funding is derived from `realizedPnl` on the position (see `monitor.py`), and
`realizedPnl` ceases to exist the moment the position does. There is no
account-bills endpoint to recover it from afterwards.

So the final snapshot is taken IMMEDIATELY BEFORE the closing orders are sent,
and that is the number that goes into the record. Reading it afterwards would
return nothing, and a 30-day test that cannot state what it earned has not
concluded - it has only stopped.

The exit fees are read after, because they do not exist until then. The two
halves are stitched together in `closed.json` beside the baseline, so the
result outlives the terminal that produced it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional


from perpkit.layout import carry_dir
from perpkit.config import DATA_DIR
from perpkit.monitor_carry import (
    BASELINE_FILE,
    SNAPSHOT_FILE,
    _jsonable,
    append_snapshot,
    load_baseline,
    read_funding_rates,
    read_perp_fills,
    read_position,
    read_spot_balance,
    read_spot_mark,
)
from perpkit.guardrails import order_host
from perpkit.keys_env import blofin_credentials
from perpkit.plan_carry import decimal_of
from perpkit.supervise import log
from perpkit.strategies.carry import (
    BlofinBroker,
    CarryExecutor,
    build_snapshot,
    compare,
    fills_totals,
    round_down,
)

CLOSED_FILE = "closed.json"


def _num(value: Decimal) -> str:
    """Plain notation, no trailing zeros, no exponent."""
    return format(value.normalize(), "f")


def spot_lot_size(public_client, inst_id: str) -> Decimal:
    """The venue's lot size for the spot leg, so dust is measured not guessed."""
    from perpkit.analysis.blofin_spot import fetch_spot_instruments

    row = fetch_spot_instruments(public_client).get(inst_id) or {}
    return decimal_of(row.get("lotSize"), "0.00000001")


def _stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ")


def report_intent(inst_id: str, *, contracts: Decimal, sellable: Decimal,
                  dust: Decimal, base: str, mark: Decimal,
                  confirm: bool) -> None:
    print("\n" + "=" * 78)
    mode = "WILL BE SENT" if confirm else "DRY RUN - NOTHING SENT"
    print(f"CLOSE PLAN  {inst_id}  ({mode})")
    print("=" * 78)
    print("\n  ORDERS, IN THIS ORDER")
    if sellable > 0:
        print(f"    1. SELL spot   {_num(sellable)} {base}  (~${sellable * mark:.2f})")
    else:
        print(f"    1. SELL spot   nothing - no {base} balance")
    if contracts != 0:
        side = "BUY " if contracts < 0 else "SELL"
        print(f"    2. {side} perp   {_num(abs(contracts))} contracts, reduce_only")
    else:
        print("    2. perp        nothing - no open position")
    print("\n    Spot first on purpose: it is the leg that can be refused, and")
    print("    a refusal there leaves the carry fully hedged and costs nothing.")
    print("    What follows is a reduce_only close, which is the most reliable")
    print("    order this system can send.")
    if dust > 0:
        print(f"\n  DUST\n    {dust} {base} (~${dust * mark:.4f}) cannot be sold: "
              f"below one lot.")


def report_scorecard(baseline, result, *, funding_usd: Decimal,
                     settlements: int, fees_usd: Decimal,
                     days: Decimal, closed_at_ms: int) -> Dict[str, Any]:
    """What the carry actually earned, against what it was sold as."""
    notional = baseline.notional_usd
    net_usd = funding_usd - fees_usd
    net_bps = (net_usd / notional * Decimal("10000")) if notional else Decimal(0)
    predicted_bps = (baseline.planned_funding_per_day_bps * baseline.hold_days
                     - baseline.round_trip_bps)

    print("\n" + "=" * 78)
    print(f"REALISED  {baseline.inst_id}")
    print("=" * 78)
    print(f"  held                 {_stamp(baseline.opened_at_ms)} -> "
          f"{_stamp(closed_at_ms)}  ({days:.2f} days)")
    print(f"  notional             ${notional:,.2f}")
    print(f"\n  funding collected    ${funding_usd:+,.4f}   "
          f"over {settlements} settlement(s)")
    print(f"  fees paid            ${-fees_usd:+,.4f}   "
          f"all four legs, as charged")
    print(f"  {'-' * 50}")
    print(f"  net                  ${net_usd:+,.4f}  = {net_bps:+.1f} bps")
    print(f"\n  predicted            {predicted_bps:+.1f} bps  "
          f"({baseline.planned_funding_per_day_bps:.2f} bps/day x "
          f"{baseline.hold_days}d - {baseline.round_trip_bps:.1f} round trip)")
    print(f"  difference           {net_bps - predicted_bps:+.1f} bps")

    return {
        "inst_id": baseline.inst_id,
        "opened_at_ms": baseline.opened_at_ms,
        "closed_at_ms": closed_at_ms,
        "days_held": days,
        "notional_usd": notional,
        "funding_usd": funding_usd,
        "funding_settlements": settlements,
        "fees_usd": fees_usd,
        "net_usd": net_usd,
        "net_bps": net_bps,
        "predicted_bps": predicted_bps,
        "spot_sold": result.spot_sold,
        "perp_closed": result.perp_closed,
        "dust_base": result.dust_base,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--confirm", action="store_true",
                        help="Actually send the orders. Without this it is a "
                             "rehearsal that sizes from real state.")
    parser.add_argument("--production", action="store_true",
                        help="Rehearse against a production account (read "
                             "only). Refused together with --confirm.")
    parser.add_argument("--funding-pages", type=int, default=6)
    args = parser.parse_args(argv)

    # First, before any key, client or network: --production --confirm is
    # refused outright.
    base_url, environment = order_host(production=args.production,
                                       confirm=args.confirm)
    api_key, secret, passphrase = blofin_credentials()

    from blofin.client import Client
    from blofin.rest_market import MarketAPI
    from blofin.rest_trading import TradingAPI

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    public = Client(baseUrl=base_url)
    api = MarketAPI(public)

    inst_id = args.instrument
    base_currency = inst_id.split("-")[0]
    directory = carry_dir(DATA_DIR, inst_id)
    baseline = load_baseline(directory / BASELINE_FILE)

    log(f"{environment}: reading {inst_id} back before closing...")
    position = read_position(client, inst_id)
    balance = read_spot_balance(client, base_currency)
    lot = spot_lot_size(public, inst_id)
    contracts = decimal_of(position.get("positions")) if position else Decimal(0)
    # The executor's own rounding, not a second copy of it: a rehearsal
    # that rounds differently is not a rehearsal of anything.
    sellable = round_down(balance, lot) if lot > 0 else balance
    mark = read_spot_mark(public, inst_id)

    if contracts == 0 and sellable <= 0:
        print(f"\nNothing open on {inst_id}: no perp position and no "
              f"{base_currency} balance.")
        return 0

    report_intent(inst_id, contracts=contracts, sellable=sellable,
                  dust=balance - sellable, base=base_currency, mark=mark,
                  confirm=args.confirm)

    # The last moment funding is readable. `realizedPnl` goes with the
    # position, and there is no bills endpoint to recover it from.
    final = None
    if baseline is not None:
        final = build_snapshot(
            at_ms=int(time.time() * 1000), inst_id=inst_id, position=position,
            spot_base=balance, spot_mark=mark,
            fills=read_perp_fills(client, inst_id),
            funding_rates=read_funding_rates(api, inst_id, args.funding_pages),
            baseline=baseline)
        report = compare(baseline, final)
        append_snapshot(directory / SNAPSHOT_FILE, final, report)
        print(f"\n  funding to date      ${final.funding_booked_usd:+.4f} "
              f"over {final.funding_periods} settlement(s)  <- captured now, "
              f"because\n                       it is unreadable once the "
              f"position is gone")
    else:
        print(f"\n  No baseline at {directory / BASELINE_FILE}, so this can "
              f"close the position\n  but cannot score it. Run monitor_carry.py "
              f"first if you want the record.")

    if not args.confirm:
        print("\n" + "=" * 78)
        print("REHEARSAL COMPLETE")
        print("=" * 78)
        print("  Nothing was sent. Re-run with --confirm to close.")
        return 0

    log("")
    log("--confirm given: these orders WILL be sent.")
    executor = CarryExecutor(BlofinBroker(client, TradingAPI(client)),
                             dry_run=False, on_log=log)
    result = executor.close(inst_id, base_currency=base_currency,
                            spot_lot_size=lot)

    print("\n" + "=" * 78)
    print(f"CLOSE  {inst_id}")
    print("=" * 78)
    for step in result.steps:
        marker = "ok " if step.ok else "FAIL"
        print(f"  [{marker}] {step.name:<6} {step.detail}")
        if step.error:
            print(f"         {step.error}")

    if not result.ok:
        print("\n" + "=" * 78)
        print("PROBLEMS")
        print("=" * 78)
        for problem in result.problems:
            print(f"  - {problem}")
        return 1

    print("\n  Both legs gone, verified against the exchange.")

    if baseline is not None and final is not None:
        fees, _ = fills_totals(read_perp_fills(client, inst_id),
                               baseline.position_id)
        spot_fees_usd = baseline.spot_fee_base * baseline.spot_entry
        # The exit spot fee is charged in quote on a sell, so it does not show
        # up in the base balance the way the entry fee did. Read from the
        # order rather than inferred.
        exit_spot_fee = _exit_spot_fee(client, inst_id, baseline.tag)
        closed_at = int(time.time() * 1000)
        days = Decimal(closed_at - baseline.opened_at_ms) / Decimal("86400000")
        record = report_scorecard(
            baseline, result,
            funding_usd=final.funding_booked_usd,
            settlements=final.funding_periods,
            fees_usd=fees + spot_fees_usd + exit_spot_fee,
            days=days, closed_at_ms=closed_at)
        path = directory / CLOSED_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(record), indent=2),
                        encoding="utf-8")
        print(f"\n  Written to {path}")

    return 0


def _exit_spot_fee(client, inst_id: str, tag: Optional[str]) -> Decimal:
    """The closing spot sell's fee, in USD.

    A sell is charged in the quote currency, unlike the buy that opened the
    position - which was charged in base and is why the hedge came up 0.254
    SUI short. Read rather than assumed, because that asymmetry is exactly the
    kind of thing that is wrong in the direction nobody checks.
    """
    payload = client.get("/api/v1/spot/trade/orders-history",
                         params={"instType": "SPOT", "instId": inst_id,
                                 "limit": "20"}, sign=True)
    total = Decimal(0)
    for row in payload.get("data") or []:
        if str(row.get("side", "")).lower() != "sell":
            continue
        total += decimal_of(row.get("fee"))
    return total


if __name__ == "__main__":
    raise SystemExit(main())
