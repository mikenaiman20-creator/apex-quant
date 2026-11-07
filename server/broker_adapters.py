from __future__ import annotations

from typing import Any, Protocol


class BaseBrokerAdapter(Protocol):
    def submit_protected_order(
        self, asset_class: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit an order only when the adapter supports protected execution."""


class AlpacaPaperAdapter:
    def __init__(self, client: Any):
        self.client = client

    def submit_protected_order(
        self, asset_class: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        if asset_class != "stock":
            raise NotImplementedError(
                f"Alpaca protected-order routing is not enabled for {asset_class}."
            )
        return self.client.submit_bracket(payload)
