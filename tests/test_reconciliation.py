import json

import pytest

from server.reconciliation import (
    ALPACA_ENDPOINTS,
    AlpacaReadOnlyClient,
    reconcile_snapshot,
)


def test_reconcile_snapshot_matches_cash_plus_position_market_value():
    report = reconcile_snapshot(
        {"equity": "102000", "cash": "98000"},
        [{"symbol": "SPY", "qty": "20", "market_value": "4000"}],
        mode="PAPER",
        target_symbols=["SPY"],
        managed_symbols=["SPY"],
    )
    assert report["approved"] is True
    assert report["calculated_equity"] == 102000
    assert report["equity_delta"] == 0


def test_reconcile_snapshot_rejects_balance_mismatch_and_unmanaged_target():
    mismatch = reconcile_snapshot(
        {"equity": "110000", "cash": "98000"},
        [{"symbol": "SPY", "qty": "20", "market_value": "4000"}],
        mode="PAPER",
    )
    assert mismatch["approved"] is False
    assert "does not reconcile" in mismatch["reason"]

    unmanaged = reconcile_snapshot(
        {"equity": "102000", "cash": "98000"},
        [{"symbol": "SPY", "qty": "20", "market_value": "4000"}],
        mode="PAPER",
        target_symbols=["SPY"],
        managed_symbols=[],
    )
    assert unmanaged["approved"] is False
    assert unmanaged["unmanaged_symbols"] == ["SPY"]


def test_reconcile_snapshot_fails_closed_on_invalid_or_blocked_feed():
    assert not reconcile_snapshot(
        {"equity": "NaN", "cash": "1000"}, [], mode="PAPER",
    )["approved"]
    assert not reconcile_snapshot(
        {"equity": "1000", "cash": "1000", "trading_blocked": True},
        [], mode="PAPER",
    )["approved"]
    assert not reconcile_snapshot(
        {"equity": "1000", "cash": "1000"}, [], mode="UNKNOWN",
    )["approved"]


def test_read_only_client_selects_live_or_paper_and_uses_get(monkeypatch):
    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(self.value).encode()

    responses = iter([
        {"equity": "100000", "cash": "100000"},
        [],
    ])

    class FakeOpener:
        def open(self, request, timeout):
            calls.append((request.full_url, request.get_method(), timeout))
            return Response(next(responses))

    client = AlpacaReadOnlyClient("LIVE", "key", "secret", opener=FakeOpener())
    account, positions = client.snapshot()
    assert client.base_url == ALPACA_ENDPOINTS["LIVE"]
    assert account["equity"] == "100000"
    assert positions == []
    assert [method for _, method, _ in calls] == ["GET", "GET"]
    assert all(url.startswith(ALPACA_ENDPOINTS["LIVE"]) for url, _, _ in calls)
    assert not hasattr(client, "submit_order")
    with pytest.raises(ValueError, match="PAPER or LIVE"):
        AlpacaReadOnlyClient("UNKNOWN", "key", "secret")
