from types import SimpleNamespace

from backtest.engine import BTConfig
from backtest.walkforward import _quality, apply_params


def test_optimizer_applies_supported_signal_and_risk_parameters():
    config = apply_params(BTConfig(), {
        "min_signal_score": 0.75,
        "atr_stop_mult": 1.25,
        "min_reward_risk": 2.5,
        "chandelier_atr_mult": 2.5,
    })
    assert config.min_signal_score == 0.75
    assert config.risk.atr_stop_mult == 1.25
    assert config.risk.min_reward_risk == 2.5
    assert config.risk.chandelier_atr_mult == 2.5


def test_optimizer_quality_rejects_non_positive_expectancy():
    profitable = [
        SimpleNamespace(r_multiple=1.0, pnl_net=100.0),
        SimpleNamespace(r_multiple=-0.2, pnl_net=-20.0),
        SimpleNamespace(r_multiple=0.5, pnl_net=50.0),
    ]
    losing = [
        SimpleNamespace(r_multiple=-1.0, pnl_net=-100.0),
        SimpleNamespace(r_multiple=0.2, pnl_net=20.0),
        SimpleNamespace(r_multiple=-0.5, pnl_net=-50.0),
    ]
    assert _quality(profitable) > float("-inf")
    assert _quality(losing) == float("-inf")
