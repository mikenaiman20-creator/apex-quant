from datetime import datetime, timedelta, timezone

from server.universe import DynamicUniverseSelector, _read_constituents


class FakeClient:
    def assets(self):
        return [
            {"symbol": symbol, "status": "active", "tradable": True, "class": "us_equity",
             "exchange": "NASDAQ", "name": f"{symbol} Corporation"}
            for symbol in ("AAA", "BBB")
        ]

    def most_actives(self):
        return [{"symbol": "AAA"}, {"symbol": "BBB"}]

    def bars(self, symbols, timeframe, start, end, limit):
        if timeframe == "1Day":
            rows = [
                {
                    "t": (end - timedelta(days=30 - index)).isoformat(),
                    "o": 100,
                    "h": 102,
                    "l": 98,
                    "c": 101,
                    "v": 500_000,
                }
                for index in range(20)
            ]
            return {symbol: rows for symbol in symbols}
        return {
            symbol: [{"t": start.isoformat(), "v": 2_000, "o": 100, "h": 101, "l": 99, "c": 100}]
            for symbol in symbols
        }

    def latest_quotes(self, symbols):
        return {
            symbol: {"bp": 99.95, "ap": 100.05,
                     "t": datetime(2026, 10, 28, 13, 15, tzinfo=timezone.utc).isoformat()}
            for symbol in symbols
        }


def test_dynamic_selector_applies_liquidity_volatility_premarket_and_spread_filters():
    selector = DynamicUniverseSelector(FakeClient(), top_n=1)
    selected = selector.select(datetime(2026, 10, 28, 13, 15, tzinfo=timezone.utc))
    assert len(selected) == 1
    assert selected[0].symbol in {"AAA", "BBB"}
    assert selected[0].average_dollar_volume >= 20_000_000
    assert 0.005 <= selected[0].atr_pct <= 0.08
    assert selected[0].relative_premarket_volume >= 0.0005
    assert selected[0].spread_pct <= 0.0025


def test_active_asset_fallback_rejects_non_equity_and_nontradable_assets():
    assets = [
        {"symbol": "AAPL", "status": "active", "tradable": True, "class": "us_equity",
         "exchange": "NASDAQ"},
        {"symbol": "TQQQ", "status": "active", "tradable": True, "class": "us_equity",
         "exchange": "NASDAQ", "name": "ProShares UltraPro QQQ"},
        {"symbol": "BAD", "status": "inactive", "tradable": True, "class": "us_equity",
         "exchange": "NASDAQ"},
        {"symbol": "XYZ", "status": "active", "tradable": False, "class": "us_equity",
         "exchange": "NYSE"},
        {"symbol": "LONGER", "status": "active", "tradable": True, "class": "us_equity",
         "exchange": "NYSE"},
    ]
    assert DynamicUniverseSelector._active_symbols(assets) == ["AAPL"]


def test_constituent_file_supports_symbol_column_not_in_first_position(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text("Name,Ticker\nApple,AAPL\nNvidia,NVDA\n", encoding="utf-8")
    assert _read_constituents(path) == ["AAPL", "NVDA"]
