"""Publish sanitized PAPER account state for the read-only dashboard."""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values

from server.io_utils import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
PAPER_API_URL = "https://paper-api.alpaca.markets/v2"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
logger = logging.getLogger("apex.paper_snapshot")

ACCOUNT_FIELDS = (
    "status", "currency", "equity", "last_equity", "cash", "buying_power",
    "portfolio_value", "trading_blocked", "account_blocked",
)
POSITION_FIELDS = (
    "symbol", "qty", "side", "market_value", "cost_basis", "unrealized_pl",
    "unrealized_plpc", "current_price", "avg_entry_price",
)
ORDER_FIELDS = (
    "submitted_at", "symbol", "side", "qty", "filled_qty", "type", "status",
    "filled_at",
)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class PaperApiClient:
    def __init__(self, key: str, secret: str):
        self._opener = build_opener(_NoRedirectHandler())
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }

    def get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        url = PAPER_API_URL + path
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers=self._headers, method="GET")
        try:
            with self._opener.open(request, timeout=15) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"Alpaca PAPER API returned HTTP {exc.code} for {path}."
            ) from None
        except URLError as exc:
            raise RuntimeError(
                f"Alpaca PAPER API connection failed for {path}: {exc.reason}"
            ) from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise RuntimeError(f"Alpaca PAPER API response for {path} was too large.")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Alpaca PAPER API returned invalid JSON for {path}."
            ) from exc


def _records(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Alpaca PAPER API returned an invalid {label} response.")
    return value


def _selected_fields(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in fields:
        if key not in item:
            continue
        value = item[key]
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Alpaca PAPER API returned an invalid {key} value.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Alpaca PAPER API returned a non-finite {key} value.")
        selected[key] = value
    return selected


def _write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    atomic_write_json(path, snapshot)


def refresh_snapshot(env_path: Path, output_path: Path) -> dict[str, Any]:
    values = dotenv_values(env_path, encoding="utf-8", interpolate=False)
    if values.get("TRADING_MODE", "").upper() != "PAPER":
        raise ValueError("Paper snapshot service requires TRADING_MODE=PAPER.")
    if values.get("BROKER", "ALPACA").upper() != "ALPACA":
        raise ValueError("Paper snapshot service requires BROKER=ALPACA.")
    key = values.get("ALPACA_API_KEY")
    secret = values.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("Paper snapshot service requires Alpaca API credentials.")

    client = PaperApiClient(key, secret)
    account = client.get_json("/account")
    if not isinstance(account, dict):
        raise ValueError("Alpaca PAPER API returned an invalid account response.")
    positions = _records(client.get_json("/positions"), "positions")
    orders = _records(
        client.get_json(
            "/orders",
            {"status": "all", "limit": "20", "direction": "desc"},
        ),
        "orders",
    )
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "PAPER",
        "paper_api_status": "connected",
        "account": _selected_fields(account, ACCOUNT_FIELDS),
        "positions": [_selected_fields(item, POSITION_FIELDS) for item in positions],
        "orders": [_selected_fields(item, ORDER_FIELDS) for item in orders],
    }
    _write_snapshot(output_path, snapshot)
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs" / "paper_dashboard_snapshot.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        snapshot = refresh_snapshot(args.env_file, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Paper dashboard snapshot refresh failed: %s", exc)
        return 1
    logger.info(
        "Published PAPER dashboard snapshot: positions=%d orders=%d",
        len(snapshot["positions"]),
        len(snapshot["orders"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
