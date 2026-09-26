"""Print exactly what a carry would do. Sends nothing.

    python -m perpkit.plan_carry --instrument SUI-USDT --notional 2000
    python -m perpkit.plan_carry --instrument XRP-USDT --notional 1000 --leverage 3
    python -m perpkit.plan_carry --instrument CRV-USDT --notional 500 --production

TEACHING EXAMPLE, NO KNOWN EDGE (see `run_carry.py`).

It fetches live prices, the
exchange's own size rules, the account's wallet balances and the instrument's
recent funding, then prints the orders it *would* place, the capital they
need, where the short leg would be liquidated, and what the position is
expected to earn.

It places no orders. There is no code path from here to `placeOrder`, and that
is the point: every sizing, margin and rounding question gets settled while
the answer is still text on a screen.

Read the REFUSALS section first. A plan that comes back `ok` has passed the
same `RiskLimits` the eventual executor will enforce - leverage cap, notional
cap, and the liquidation buffer - and a plan that does not tells you which
limit and by how much.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from typing import List, Optional


from perpkit.config import (
    MAX_LEVERAGE,
    SPOT_TAKER_FEE_BPS,
    TAKER_FEE_BPS,
    VIP_TIER,
)
from perpkit.strategies.carry import Market, Wallets, plan_carry
from perpkit.margin_tiers import maintenance_margin_rate
from perpkit.risk import RiskLimits

from perpkit.guardrails import read_host
from perpkit.keys_env import blofin_credentials

# Measured against a small real position by comparing `risk.liquidation_price`
# with BloFin's own reported liquidation price: the MMR read out as exactly
# 0.005 at that notional, and the residual implied a ~6 bps fee term. MMR is
# tiered by size; `margin_tiers.maintenance_margin_rate` reads the real tier.
MEASURED_MMR = Decimal("0.005")
MEASURED_FEE_BUFFER_BPS = Decimal("6")


def decimal_of(value, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal(default)


def funding_per_day_bps(api, inst_id: str, pages: int) -> Decimal:
    """Median funding over the deepest history the endpoint will give.

    Median, not mean: funding is fat-tailed, and a mean quietly imports a few
    extreme settlements into the headline number.
    """
    from perpkit.analysis.funding_carry import PERIODS_PER_DAY, fetch_funding_history
    from perpkit.analysis.funding_carry import funding_stats

    history = fetch_funding_history(api, inst_id, pages=pages)
    stats = funding_stats(history)
    if not stats:
        return Decimal(0)
    return decimal_of(stats["median"]) * PERIODS_PER_DAY


def build_market(client, api, inst_id: str) -> Optional[Market]:
    from perpkit.analysis.blofin_spot import (
        fetch_spot_instruments,
        fetch_spot_tickers,
        top_of_book,
    )

    spot_instruments = fetch_spot_instruments(client)
    spot_tickers = fetch_spot_tickers(client)
    if inst_id not in spot_instruments:
        print(f"  {inst_id} has no spot market, so it cannot be hedged this way.")
        return None

    perps = {row["instId"]: row for row in (api.getInstruments().get("data") or [])}
    perp_tickers = {row["instId"]: row for row in (api.getTickers().get("data") or [])}
    if inst_id not in perps:
        print(f"  {inst_id} has no perpetual.")
        return None

    spot_book = top_of_book(spot_tickers.get(inst_id))
    perp_book = top_of_book(perp_tickers.get(inst_id))
    if not spot_book or not perp_book:
        print(f"  {inst_id} has no usable quote on one of the legs.")
        return None

    perp = perps[inst_id]
    spot = spot_instruments[inst_id]
    return Market(
        inst_id=inst_id,
        spot_bid=decimal_of(spot_book[0]), spot_ask=decimal_of(spot_book[1]),
        perp_bid=decimal_of(perp_book[0]), perp_ask=decimal_of(perp_book[1]),
        contract_value=decimal_of(perp.get("contractValue"), "1"),
        perp_lot_size=decimal_of(perp.get("lotSize"), "1"),
        perp_min_size=decimal_of(perp.get("minSize"), "1"),
        spot_lot_size=decimal_of(spot.get("lotSize"), "0.00000001"),
        spot_min_size=decimal_of(spot.get("minSize"), "0"),
    )


def read_wallets(client) -> Wallets:
    """USDT available in each wallet. A carry needs both funded."""
    def available(account_type: str) -> Decimal:
        payload = client.get("/api/v1/asset/balances",
                             params={"accountType": account_type}, sign=True)
        for row in payload.get("data") or []:
            if row.get("currency") == "USDT":
                return decimal_of(row.get("available"))
        return Decimal(0)

    return Wallets(spot_usdt=available("spot"),
                   futures_usdt=available("futures"))


def report(plan, market: Market, wallets: Wallets, *, hold_days: Decimal,
           environment: str) -> None:
    print("\n" + "=" * 78)
    print(f"CARRY PLAN  {plan.inst_id}  ({environment}, VIP {VIP_TIER}) "
          f"- DRY RUN, NOTHING SENT")
    print("=" * 78)

    print("\n  ORDERS IT WOULD PLACE")
    print(f"    1. BUY  spot   {plan.spot_base} {plan.inst_id.split('-')[0]} "
          f"@ ~{market.spot_ask}   (${plan.spot_cost_usd:,.2f})")
    print(f"    2. SELL perp   {plan.perp_contracts} contracts "
          f"@ ~{market.perp_bid}   isolated, {plan.leverage}x")
    print(f"       = {plan.perp_base} base, notional ${plan.notional_usd:,.2f}")
    if plan.residual_base:
        print(f"    residual delta {plan.residual_base} base "
              f"(${plan.residual_usd:,.2f}) - unhedged")

    print("\n  CAPITAL")
    print(f"    spot leg cost        ${plan.spot_cost_usd:>12,.2f}   "
          f"(wallet has ${wallets.spot_usdt:,.2f})")
    print(f"    perp isolated margin ${plan.perp_margin_usd:>12,.2f}   "
          f"(wallet has ${wallets.futures_usdt:,.2f})")
    print(f"    total               ${plan.total_capital_usd:>12,.2f}")
    if plan.spot_transfer_usd:
        print(f"    -> transfer ${plan.spot_transfer_usd:,.2f} futures -> spot")
    if plan.futures_transfer_usd:
        print(f"    -> transfer ${plan.futures_transfer_usd:,.2f} spot -> futures")

    print("\n  RISK ON THE SHORT LEG")
    if plan.liquidation_price is not None:
        print(f"    liquidation at       {plan.liquidation_price:>12,.4f}   "
              f"({plan.liquidation_distance:.1%} away)")
        print(f"    perp mark            {market.perp_mid:>12,.4f}")
    print("    Delta-neutral is not risk-neutral: the legs margin separately, "
          "so a rally\n    that leaves the PAIR flat can still liquidate the "
          "short. What survives that\n    is an unhedged long spot position - "
          "the opposite of this trade.")

    print("\n  ECONOMICS")
    print(f"    spot spread          {market.spot_spread_bps:>12.2f} bps")
    print(f"    perp spread          {market.perp_spread_bps:>12.2f} bps")
    print(f"    round trip, 4 legs   {plan.round_trip_bps:>12.2f} bps")
    print(f"    funding per day      {plan.funding_per_day_bps:>12.2f} bps")
    if plan.breakeven_days is not None:
        print(f"    break-even           {plan.breakeven_days:>12.1f} days")
    print(f"    expected over {hold_days:>3} days {plan.expected_net_bps:>+11.1f} bps "
          f"= ${plan.expected_net_usd:,.2f}")

    if plan.warnings:
        print("\n  WARNINGS")
        for warning in plan.warnings:
            print(f"    - {warning}")

    print("\n" + "=" * 78)
    if plan.ok:
        print("PLAN OK - and still not an instruction to trade")
        print("=" * 78)
        print("  It clears every RiskLimits gate the executor will enforce. "
              "What it does NOT\n  cover: which leg to send first (whichever "
              "fills leaves you directional until\n  the other does), what to "
              "do if the second leg fails, and whether funding\n  behaves "
              "like its median for the next "
              f"{hold_days} days. `carry_backtest.py` is the\n  honest "
              "distribution behind that last one.")
    else:
        print("REFUSED")
        print("=" * 78)
        for reason in plan.reasons:
            print(f"  - {reason}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--notional", type=Decimal, default=Decimal("1000"),
                        help="Target position size in USD.")
    parser.add_argument("--leverage", type=Decimal, default=Decimal("3"),
                        help=f"On the perp leg. Capped at {MAX_LEVERAGE}.")
    parser.add_argument("--hold-days", type=Decimal, default=Decimal("30"))
    parser.add_argument("--funding-pages", type=int, default=6,
                        help="Funding history depth; 6 is about 200 days.")
    parser.add_argument("--production", action="store_true",
                        help="Read the live account instead of demo. Still "
                             "places nothing.")
    args = parser.parse_args(argv)

    if SPOT_TAKER_FEE_BPS is None:
        raise SystemExit(
            f"No spot fee schedule for VIP {VIP_TIER} in perpkit.fees.SPOT_VIP_TIERS.")

    base_url, environment = read_host(production=args.production)
    api_key, secret, passphrase = blofin_credentials()

    from blofin.client import Client
    from blofin.rest_market import MarketAPI

    client = Client(apiKey=api_key, apiSecret=secret, passphrase=passphrase,
                    baseUrl=base_url)
    # Market data is public and identical on both hosts; the account is not.
    public = Client()
    api = MarketAPI(public)

    print(f"Reading {environment} account and live market...")
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
    print(f"  maintenance margin rate {mmr} (tier for ~{wanted_contracts:.0f} "
        f"contracts on {environment})")

    plan = plan_carry(
        market=market,
        wallets=wallets,
        target_notional_usd=args.notional,
        leverage=args.leverage,
        funding_per_day_bps=daily,
        hold_days=args.hold_days,
        spot_taker_bps=Decimal(str(SPOT_TAKER_FEE_BPS)),
        perp_taker_bps=Decimal(str(TAKER_FEE_BPS)),
        limits=RiskLimits(),
        maintenance_margin_rate=mmr,
        fee_buffer_bps=MEASURED_FEE_BUFFER_BPS,
    )
    report(plan, market, wallets, hold_days=args.hold_days,
           environment=environment)
    return 0 if plan.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
