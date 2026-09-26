"""Open a planned carry on the DEMO account. Dry run unless --confirm.

TEACHING EXAMPLE, NO KNOWN EDGE. Funding carry is shown because it
exercises every part of an order path (two legs, leg ordering, unwinds,
verification against the exchange) with no prediction in it. Nothing here
claims it earns money after fees, borrow of margin, basis moves and
liquidation risk.

    python -m perpkit.run_carry --instrument SUI-USDT --notional 200
    python -m perpkit.run_carry --instrument SUI-USDT --notional 200 --confirm

Without `--confirm` this rehearses: it builds the plan against live prices,
walks every step the open would take, and prints them without sending
anything. With `--confirm` it sends them, to BloFin's demo host.

`--production --confirm` is refused outright (see `perpkit/guardrails.py`).
`--production` alone rehearses against a production account read-only.
Credentials come from BLOFIN_API_KEY / BLOFIN_API_SECRET /
BLOFIN_API_PASSPHRASE in the environment and are never logged.

The order, and the failure it is chosen for
-------------------------------------------
Perp leg first, spot leg second. Between the two the position is directional,
so the choice is about which exposure to be holding if the second leg fails.
A failed spot leg leaves a SHORT perp - closeable instantly, on the deeper
book, with a `reduce_only` order - and that is what gets unwound automatically.
A failed perp leg leaves nothing on at all, which is the cheapest outcome
available. The reverse order would strand you long spot.

What it checks that a plan cannot
---------------------------------
The plan assumes a maintenance margin rate. MMR is tiered and
instrument-specific - 0.500% measured on SOL-USDT, 0.300% on BTC-USDT - so it
is a guess until a position exists. After both legs are on, this reads the
position back and compares the exchange's own liquidation price to the planned
one. Closer than planned is reported as a problem, immediately.

Margin mode is checked and never changed: on BloFin it is an account-wide
setting, so flipping it for one carry would re-margin every other open
position.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from typing import List, Optional

from perpkit.config import SPOT_TAKER_FEE_BPS, TAKER_FEE_BPS, VIP_TIER
from perpkit.guardrails import order_host
from perpkit.keys_env import blofin_credentials
from perpkit.plan_carry import (
    MEASURED_FEE_BUFFER_BPS,
    MEASURED_MMR,
    build_market,
    funding_per_day_bps,
    read_wallets,
    report as report_plan,
)
from perpkit.supervise import log
from perpkit.strategies.carry import (
    BlofinBroker,
    CarryExecutor,
    plan_carry,
)
from perpkit.margin_tiers import maintenance_margin_rate
from perpkit.risk import RiskLimits


def report_execution(result, *, hold_days: Decimal) -> None:
    print("\n" + "=" * 78)
    mode = "DRY RUN - NOTHING SENT" if result.dry_run else "LIVE"
    print(f"EXECUTION  {result.inst_id}  ({mode})")
    print("=" * 78)

    for step in result.steps:
        marker = "ok " if step.ok else "FAIL"
        sent = "" if step.sent else "  (not sent)"
        print(f"  [{marker}] {step.name:<10} {step.detail}{sent}")
        if step.error:
            print(f"         {step.error}")

    if result.spot_acquired is not None:
        print("\n  HEDGE SIZE, MEASURED FROM THE BALANCE")
        print(f"    spot acquired        {result.spot_acquired} "
              f"{result.inst_id.split('-')[0]}")
        print("    Checked against the balance, not the order response: "
              "`targetCurrency`\n    decides whether a market order's size is "
              "base or quote, and the wrong one\n    fills cleanly at the "
              "wrong size.")

    if result.actual_liquidation is not None:
        print("\n  VERIFIED AGAINST THE EXCHANGE")
        print(f"    planned liquidation  {result.planned_liquidation}")
        print(f"    actual liquidation   {result.actual_liquidation}")
        if result.actual_mmr is not None:
            print(f"    MMR actually charged {result.actual_mmr:.5f}  "
                  f"(plan assumed {MEASURED_MMR})")

    print("\n" + "=" * 78)
    if result.dry_run:
        print("REHEARSAL COMPLETE")
        print("=" * 78)
        print("  Nothing was sent. Re-run with --confirm to place these "
              "orders on the demo\n  account.")
    elif result.opened and result.ok:
        print("CARRY IS ON")
        print("=" * 78)
        print(f"  Both legs filled and the exchange's liquidation price "
              f"matches the plan.\n  Hold it and let funding accrue; the point "
              f"of the test is comparing realised\n  against the "
              f"{hold_days}-day prediction, which needs days, not minutes.")
    elif result.unwound:
        print("UNWOUND - NOTHING IS OPEN")
        print("=" * 78)
        for problem in result.problems:
            print(f"  - {problem}")
    else:
        print("PROBLEMS")
        print("=" * 78)
        for problem in result.problems:
            print(f"  - {problem}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--notional", type=Decimal, default=Decimal("200"),
                        help="Target size in USD. Start small.")
    parser.add_argument("--leverage", type=Decimal, default=Decimal("3"))
    parser.add_argument("--hold-days", type=Decimal, default=Decimal("30"))
    parser.add_argument("--funding-pages", type=int, default=6)
    parser.add_argument("--confirm", action="store_true",
                        help="Actually send the orders. Without this it is a "
                             "rehearsal.")
    parser.add_argument("--production", action="store_true",
                        help="Rehearse against a production account (read "
                             "only). Refused together with --confirm.")
    args = parser.parse_args(argv)

    # First, before any key, client or network: --production --confirm is
    # refused outright.
    base_url, environment = order_host(production=args.production,
                                       confirm=args.confirm)

    if SPOT_TAKER_FEE_BPS is None:
        raise SystemExit(
            f"No spot fee schedule for VIP {VIP_TIER} in perpkit.fees.SPOT_VIP_TIERS.")

    api_key, secret, passphrase = blofin_credentials()

    from blofin.client import Client
    from blofin.rest_market import MarketAPI
    from blofin.rest_trading import TradingAPI

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    public = Client()
    api = MarketAPI(public)

    log(f"{environment}: building the plan from live prices...")
    market = build_market(public, api, args.instrument)
    if market is None:
        return 1
    wallets = read_wallets(client)
    daily = funding_per_day_bps(api, args.instrument, args.funding_pages)

    # MMR from the exchange's own tier table, on the SAME host as the
    # account: demo and production publish different schedules (SUI
    # isolated tier 1 is 0.0050 on production and 0.0065 on demo), and
    # the wrong one makes liquidation look further away than it is.
    wanted_contracts = (args.notional / market.perp_mid
                        / market.contract_value)
    mmr = maintenance_margin_rate(client, args.instrument, wanted_contracts)
    if mmr is None:
        raise SystemExit(
            f"Could not read the margin tier for {args.instrument} on "
            f"{environment}. A liquidation price from a guessed MMR is a "
            "number that looks measured and is not.")
    log(f"  maintenance margin rate {mmr} (tier for ~{wanted_contracts:.0f} "
        f"contracts on {environment})")

    plan = plan_carry(
        market=market, wallets=wallets,
        target_notional_usd=args.notional, leverage=args.leverage,
        funding_per_day_bps=daily, hold_days=args.hold_days,
        spot_taker_bps=Decimal(str(SPOT_TAKER_FEE_BPS)),
        perp_taker_bps=Decimal(str(TAKER_FEE_BPS)),
        limits=RiskLimits(),
        maintenance_margin_rate=mmr,
        fee_buffer_bps=MEASURED_FEE_BUFFER_BPS,
    )
    report_plan(plan, market, wallets, hold_days=args.hold_days,
                environment=environment)
    if not plan.ok:
        return 1

    if args.confirm:
        log("")
        log("--confirm given: these orders WILL be sent.")

    executor = CarryExecutor(
        BlofinBroker(client, TradingAPI(client)),
        dry_run=not args.confirm,
        on_log=log,
    )
    result = executor.open(plan, base_currency=args.instrument.split("-")[0])
    report_execution(result, hold_days=args.hold_days)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
