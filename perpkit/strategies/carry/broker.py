"""The `Broker` protocol, against BloFin's REST API.

Thin on purpose: every method is one call and no logic, so the ordering,
unwinding and verification stay in `execute.py` where the tests can reach them
without a network.

It lives here rather than in an entrypoint because two entrypoints now need
it - `run_carry.py` opens and `close_carry.py` closes - and a broker that
lives in one of them would make the other import a CLI to get at a REST
client.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return Decimal(default)


class BlofinBroker:
    """Every exchange call an open or a close needs, and no decisions."""

    def __init__(self, client, trading_api):
        self.client = client
        self.trading = trading_api

    def transfer(self, *, currency: str, amount: Decimal,
                 from_account: str, to_account: str) -> Dict[str, Any]:
        return self.trading.transfer(
            currency=currency, amount=str(amount),
            fromAccount=from_account, toAccount=to_account)

    def margin_mode(self) -> str:
        payload = self.client.get("/api/v1/account/margin-mode", params={},
                                  sign=True)
        return str((payload.get("data") or {}).get("marginMode", "unknown"))

    def set_leverage(self, inst_id: str, leverage: Decimal) -> Dict[str, Any]:
        return self.trading.setLeverage(
            instId=inst_id, leverage=str(int(leverage)), marginMode="isolated")

    def place_perp(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str,
                   reduce_only: bool = False) -> Dict[str, Any]:
        return self.trading.placeOrder(
            instId=inst_id, marginMode="isolated", positionSide="net",
            side=side, orderType="market", size=str(size),
            reduceOnly="true" if reduce_only else "false",
            clientOrderId=client_order_id)

    def place_spot(self, *, inst_id: str, side: str, size: Decimal,
                   client_order_id: str) -> Dict[str, Any]:
        # `targetCurrency` is OPTIONAL in BloFin's schema and decides whether
        # `size` on a market order means base units or quote. Leaving it to a
        # default would make a hedge of 254 SUI or of 254 USDT-worth of SUI
        # depending on a value not written down here - a 3x mis-hedge that
        # fills cleanly and looks correct. Always explicit.
        #
        # `base_currency` is right in BOTH directions: buying, it is the hedge
        # size; selling, it is the balance being liquidated.
        return self.client.post("/api/v1/spot/trade/order", {
            "instType": "SPOT",
            "instId": inst_id,
            "side": side,
            "orderType": "market",
            "targetCurrency": "base_currency",
            "size": str(size),
            "clientOrderId": client_order_id,
        })

    def perp_position(self, inst_id: str) -> Optional[Dict[str, Any]]:
        payload = self.client.get("/api/v1/account/positions",
                                  params={"instId": inst_id}, sign=True)
        for row in payload.get("data") or []:
            if row.get("instId") == inst_id:
                return row
        return None

    def spot_balance(self, currency: str) -> Decimal:
        payload = self.client.get("/api/v1/asset/balances",
                                  params={"accountType": "spot"}, sign=True)
        for row in payload.get("data") or []:
            if row.get("currency") == currency:
                return _decimal(row.get("available"))
        return Decimal(0)
