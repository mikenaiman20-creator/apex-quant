"""Read-only broker account reconciliation for Alpaca PAPER and LIVE feeds."""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from typing import Any


ALPACA_ENDPOINTS = {
    "PAPER": "https://paper-api.alpaca.markets",
    "LIVE": "https://api.alpaca.markets",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class ReconciliationError(RuntimeError):
    """Raised when a broker snapshot cannot be read or reconciled."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def reconcile_snapshot(
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    *,
    mode: str,
    target_symbols: list[str] | tuple[str, ...] = (),
    managed_symbols: list[str] | tuple[str, ...] | None = None,
    tolerance_usd: float = 5.0,
    tolerance_pct: float = 0.001,
) -> dict[str, Any]:
    """Compare broker balances with broker positions and optional runner-owned exposure."""
    result: dict[str, Any] = {
        "mode": mode,
        "approved": False,
        "reason": "",
        "position_count": 0,
        "equity": None,
        "cash": None,
        "position_market_value": None,
        "calculated_equity": None,
        "equity_delta": None,
        "tolerance": None,
        "unmanaged_symbols": [],
    }
    if mode not in ALPACA_ENDPOINTS:
        result["reason"] = "unsupported broker mode"
        return result
    if not isinstance(account, dict) or not isinstance(positions, list):
        result["reason"] = "broker account or positions response has an invalid shape"
        return result
    if account.get("trading_blocked") or account.get("account_blocked"):
        result["reason"] = "broker reports the account is blocked"
        return result
    try:
        equity = _number(account.get("equity"), "equity")
        cash = _number(account.get("cash"), "cash")
        if equity <= 0:
            raise ValueError("equity must be greater than zero.")
        if tolerance_usd < 0 or not math.isfinite(tolerance_usd):
            raise ValueError("tolerance_usd must be finite and non-negative.")
        if tolerance_pct < 0 or not math.isfinite(tolerance_pct):
            raise ValueError("tolerance_pct must be finite and non-negative.")

        seen_symbols: set[str] = set()
        market_value = 0.0
        for position in positions:
            if not isinstance(position, dict):
                raise ValueError("positions contain a non-object record.")
            symbol = position.get("symbol")
            if not isinstance(symbol, str) or not symbol or symbol in seen_symbols:
                raise ValueError("positions contain a missing or duplicate symbol.")
            seen_symbols.add(symbol)
            quantity = _number(position.get("qty"), f"{symbol} quantity")
            if quantity == 0:
                raise ValueError(f"{symbol} position has zero quantity.")
            market_value += _number(position.get("market_value"), f"{symbol} market value")

        calculated_equity = cash + market_value
        delta = equity - calculated_equity
        tolerance = max(tolerance_usd, abs(equity) * tolerance_pct)
        target_set = set(target_symbols)
        unmanaged = []
        if managed_symbols is not None:
            managed_set = set(managed_symbols)
            unmanaged = sorted(
                symbol for symbol in seen_symbols & target_set if symbol not in managed_set
            )

        result.update({
            "position_count": len(positions),
            "equity": equity,
            "cash": cash,
            "position_market_value": market_value,
            "calculated_equity": calculated_equity,
            "equity_delta": delta,
            "tolerance": tolerance,
            "unmanaged_symbols": unmanaged,
        })
        if abs(delta) > tolerance:
            result["reason"] = "equity does not reconcile to cash plus position market value"
        elif unmanaged:
            result["reason"] = "broker has target-symbol exposure absent from runner state"
        else:
            result["approved"] = True
            result["reason"] = "broker balances and positions reconciled"
    except ValueError as exc:
        result["reason"] = str(exc)
    return result


class AlpacaReadOnlyClient:
    """Minimal GET-only Alpaca client; it deliberately exposes no trading methods."""

    def __init__(
        self,
        mode: str,
        key: str,
        secret: str,
        *,
        timeout: float = 15.0,
        opener: Any | None = None,
    ) -> None:
        normalized_mode = mode.upper()
        if normalized_mode not in ALPACA_ENDPOINTS:
            raise ValueError("mode must be PAPER or LIVE.")
        if not key or not secret:
            raise ValueError("Alpaca API credentials are required.")
        self.base_url = ALPACA_ENDPOINTS[normalized_mode]
        self.mode = normalized_mode
        self.headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        }
        self.timeout = timeout
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    def _get_json(self, path: str) -> Any:
        request = urllib.request.Request(
            self.base_url + path, headers=self.headers, method="GET",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise ReconciliationError(
                f"Alpaca {self.mode} read failed with HTTP {exc.code}."
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ReconciliationError(
                f"Unable to read the Alpaca {self.mode} account feed."
            ) from exc

    def snapshot(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        account = self._get_json("/v2/account")
        positions = self._get_json("/v2/positions")
        if not isinstance(account, dict) or not isinstance(positions, list):
            raise ReconciliationError("Alpaca returned an invalid account/positions response.")
        return account, positions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read and reconcile the configured Alpaca broker account."
    )
    args = parser.parse_args(argv)
    del args
    try:
        from config.settings import BrokerName, load_settings
        from server.risk_engine import RiskGateClient, RiskGateError

        settings = load_settings(require_credentials=False)
        if settings.broker is not BrokerName.ALPACA:
            raise ReconciliationError("Broker reconciliation currently supports Alpaca only.")
        client = AlpacaReadOnlyClient(
            settings.mode.value, settings.alpaca_key, settings.alpaca_secret,
        )
        account, positions = client.snapshot()
        decision = RiskGateClient().reconcile({
            "mode": settings.mode.value,
            "account": account,
            "positions": positions,
            "target_symbols": list(settings.watchlist),
            "source": "scheduled_read_only_reconciliation",
        })
        print(json.dumps(decision, sort_keys=True))
        return 0 if decision.get("approved") is True else 1
    except (ReconciliationError, RuntimeError, ValueError) as exc:
        print(f"Broker reconciliation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
