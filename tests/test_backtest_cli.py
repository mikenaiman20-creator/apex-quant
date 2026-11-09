import logging
from datetime import datetime, timezone
from types import SimpleNamespace

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


def test_every_cli_run_reports_cpcv_dsr_and_cumulative_trials(monkeypatch, capsys):
    import server.evolution as evolution

    seen = {}
    results = {
        (0.65, 1.25, 2.0, 2.5): [],
        (0.75, 1.75, 2.5, 3.5): [],
    }
    monkeypatch.setattr(cli, "synthetic_bars", lambda **_kwargs: [])
    monkeypatch.setattr(cli, "run_parameter_grid", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(evolution, "register_trials", lambda *_args, **_kwargs: 42)
    monkeypatch.setattr(
        cli, "evaluate_cpcv",
        lambda *_args, **kwargs: seen.update(n_trials=kwargs["n_trials"]) or {
            "method": "CPCV-CSCV", "valid_splits": 20, "splits": 20,
            "groups": 6, "test_groups": 3, "embargo_days": 1,
            "pbo": 0.1, "pbo_threshold": 0.2, "pbo_passed": True,
            "trials_in_run": 2, "n_trials": 42, "oos_observations": 100,
            "passed": True,
            "dsr": {"adjusted_sharpe": 0.97, "passed": True},
        },
    )
    monkeypatch.setattr(cli, "walk_forward", lambda *_args, **_kwargs: ([], [], results))

    assert cli.main([
        "run", "--synthetic", "--symbols", "SPY", "--walk-forward-only",
    ]) == 0
    output = capsys.readouterr().out
    assert "PBO=0.100" in output
    assert "DSR_probability=0.970" in output
    assert "trials_in_run=2  cumulative_trials=42" in output
    assert seen["n_trials"] == 42


def test_orb_cli_uses_independent_signal_and_trial_grid(monkeypatch):
    import server.evolution as evolution

    from backtest.strategies import ORB_PARAMETER_GRID, opening_range_breakout

    seen = {}
    monkeypatch.setattr(cli, "synthetic_bars", lambda **_kwargs: [])
    monkeypatch.setattr(
        cli,
        "run_parameter_grid",
        lambda *_args, **kwargs: seen.update(
            grid=kwargs["grid"], signal_fn=kwargs["signal_fn"],
        ) or {(2.0, 2.5, 0.0025, 0.1): []},
    )
    monkeypatch.setattr(evolution, "register_trials", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        cli,
        "evaluate_cpcv",
        lambda *_args, **kwargs: seen.update(
            cpcv_signal=kwargs["signal_fn"],
        ) or {
            "method": "CPCV-CSCV", "valid_splits": 0, "splits": 0,
            "groups": 6, "test_groups": 3, "embargo_days": 1,
            "pbo": None, "pbo_threshold": 0.2, "pbo_passed": False,
            "trials_in_run": 1, "n_trials": 1, "oos_observations": 0,
            "passed": False, "dsr": {"adjusted_sharpe": None, "passed": False},
        },
    )
    monkeypatch.setattr(
        cli, "walk_forward",
        lambda *_args, **kwargs: seen.update(
            walk_forward_signal=kwargs["signal_fn"],
            walk_forward_grid=kwargs["grid"],
        ) or ([], [], {}),
    )

    assert cli.main([
        "run", "--synthetic", "--symbols", "SPY", "--strategy", "orb",
        "--walk-forward-only",
    ]) == 0
    assert seen == {
        "grid": ORB_PARAMETER_GRID,
        "signal_fn": opening_range_breakout,
        "cpcv_signal": opening_range_breakout,
        "walk_forward_signal": opening_range_breakout,
        "walk_forward_grid": ORB_PARAMETER_GRID,
    }


def test_zscore_cli_runs_cpcv_walk_forward_and_reports_zero_budget(monkeypatch, capsys):
    import server.evolution as evolution

    from backtest.strategies import ZSCORE_PARAMETER_GRID, short_term_zscore_mean_reversion

    seen = {}
    monkeypatch.setattr(cli, "synthetic_bars", lambda **_kwargs: [])

    def parameter_grid(data, config, **kwargs):
        seen.update(
            threshold=config.zscore_entry_threshold,
            spread_bps=config.spread_bps,
            grid=kwargs["grid"],
            signal=kwargs["signal_fn"],
            workers=kwargs["workers"],
        )
        return {(2.0,): [], (2.5,): []}

    monkeypatch.setattr(cli, "run_parameter_grid", parameter_grid)
    monkeypatch.setattr(evolution, "register_trials", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        cli,
        "evaluate_cpcv",
        lambda *_args, **kwargs: seen.update(cpcv_signal=kwargs["signal_fn"]) or {
            "method": "CPCV-CSCV", "valid_splits": 20, "splits": 20,
            "groups": 6, "test_groups": 3, "embargo_days": 1,
            "pbo": 0.4, "pbo_threshold": 0.2, "pbo_passed": False,
            "trials_in_run": 2, "n_trials": 100, "oos_observations": 100,
            "passed": False,
            "dsr": {"adjusted_sharpe": 0.0, "passed": False},
        },
    )
    monkeypatch.setattr(
        cli, "walk_forward",
        lambda *_args, **kwargs: seen.update(
            wf_signal=kwargs["signal_fn"], wf_grid=kwargs["grid"],
        ) or ([], [], {}),
    )

    assert cli.main([
        "run", "--synthetic", "--symbols", "SPY", "--strategy", "zscore",
        "--walk-forward-only",
    ]) == 0
    output = capsys.readouterr().out
    assert seen == {
        "threshold": 2.0,
        "spread_bps": 2.0,
        "grid": ZSCORE_PARAMETER_GRID,
        "signal": short_term_zscore_mean_reversion,
        "workers": 1,
        "cpcv_signal": short_term_zscore_mean_reversion,
        "wf_signal": short_term_zscore_mean_reversion,
        "wf_grid": ZSCORE_PARAMETER_GRID,
    }
    assert "research_risk_budget=$0.00" in output
    assert "execution_risk=$0.00" in output
    assert "live and paper order routing remain disabled" in output


def test_pooled_null_report_uses_combined_capital_and_equity_curve(monkeypatch):
    seen = {}

    def capture_report(title, trades, curve, equity):
        seen.update(title=title, trades=trades, curve=curve, equity=equity)

    monkeypatch.setattr(cli, "_report", capture_report)
    trades = [
        SimpleNamespace(exit_ts=datetime(2026, 1, 2, 15, tzinfo=timezone.utc), pnl_net=-50),
        SimpleNamespace(exit_ts=datetime(2026, 1, 5, 15, tzinfo=timezone.utc), pnl_net=25),
    ]

    cli._report_pooled_null_model(trades, initial_equity=5000)

    assert seen["title"].endswith("3 seeds pooled)")
    assert seen["equity"] == 15000
    assert seen["curve"] == [
        (datetime(2026, 1, 2).date(), 14950),
        (datetime(2026, 1, 5).date(), 14975),
    ]
    assert seen["trades"] == trades
