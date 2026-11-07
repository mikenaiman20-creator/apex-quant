import logging

import pytest

import backtest.cli as cli


def test_run_defaults_to_local_data_and_target_symbols(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["csv_dir"] = args.csv_dir
        seen["symbols"] = args.symbols
        return 0

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    assert cli.main(["run"]) == 0
    assert seen == {"csv_dir": "data", "symbols": ["SPY", "QQQ", "NVDA"]}


def test_run_rejects_conflicting_data_sources():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--synthetic", "--csv-dir", "data"])
    assert exc_info.value.code == 2


def test_run_reports_missing_csv_as_logged_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        cli, "load_dir",
        lambda *_args: (_ for _ in ()).throw(FileNotFoundError("data/SPY.csv")),
    )
    with caplog.at_level(logging.ERROR, logger="apex.backtest.cli"):
        assert cli.main(["run", "--csv-dir", "data", "--symbols", "SPY"]) == 1
    assert "Backtest command 'run' failed." in caplog.text
    assert "FileNotFoundError: data/SPY.csv" in caplog.text


def test_run_rejects_csv_with_no_valid_regular_session_bars(monkeypatch, caplog):
    monkeypatch.setattr(cli, "load_dir", lambda *_args: {"SPY": []})
    with caplog.at_level(logging.ERROR, logger="apex.backtest.cli"):
        assert cli.main(["run", "--csv-dir", "data", "--symbols", "SPY"]) == 1
    assert "No valid regular-session bars loaded for SPY" in caplog.text


def test_synthetic_run_defaults_remain_available(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["synthetic"] = args.synthetic
        seen["days"] = args.days
        return 0

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    assert cli.main(["run", "--synthetic"]) == 0
    assert seen == {"synthetic": True, "days": 160}
