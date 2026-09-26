"""The one rule every perpkit order path goes through before it builds a client.

    dry        unless --confirm     (a rehearsal prints every step, sends nothing)
    demo       unless --production  (BloFin's demo-trading host)
    refused    --production --confirm, always

The last line is deliberate and has no override flag. perpkit ships no
production order path: the strategies here are teaching examples with no
known edge, and a toolkit whose tests run on fakes has never proven it sends
the orders it means to send with real money. If you decide to trade live,
that is a decision to make outside this toolkit, deliberately, after the
same code has run end to end on demo. The read-only tools (`plan_carry`,
`monitor_carry`) may read a production account with --production; they
cannot send anything.
"""

from __future__ import annotations

from typing import Tuple

DEMO_BASE_URL = "https://demo-trading-openapi.blofin.com"
PRODUCTION_BASE_URL = "https://openapi.blofin.com"

REFUSAL = (
    "--production with --confirm would send real orders with real money, and "
    "perpkit refuses that\ncombination outright. Rehearse without --confirm, "
    "or run on demo (drop --production).\nSee 'Guardrails' in the README.")


def order_host(*, production: bool, confirm: bool) -> Tuple[str, str]:
    """(base_url, environment name) for an order path, or SystemExit."""
    if production and confirm:
        raise SystemExit(REFUSAL)
    if production:
        return PRODUCTION_BASE_URL, "production"
    return DEMO_BASE_URL, "demo"


def read_host(*, production: bool) -> Tuple[str, str]:
    """(base_url, environment name) for a read-only tool."""
    if production:
        return PRODUCTION_BASE_URL, "production"
    return DEMO_BASE_URL, "demo"
