"""Known-positive calibration: a planted, modest edge must pass the real gate.

The panel has a cross-sectional 5-day reversal signal with an asset-level
information coefficient near 0.15, Student-t(5) innovations, mild AR(1)
autocorrelation, and a calibrated risk budget that makes the after-cost
portfolio Sharpe 0.50 by construction.  The production CPCV/PBO/DSR code must
accept it.

The 15-year sample is deliberate: at the brief's 3 years (756 observations) a
Sharpe of 0.50 produces DSR 0.806, below the unchanged 0.95 gate.  See
``DECISIONS.md`` ("Gate Calibration Baseline").
"""
from __future__ import annotations

import pytest

from tests.calibration.synthetic import DEFAULT_SEED, build_seed, evaluate_gate


def test_known_positive_passes_cpcv_pbo_dsr_and_cost_gate(monkeypatch):
    seed_panels = build_seed(
        DEFAULT_SEED,
        (0.15,),
        compute_realized_ic=True,
    )
    result = evaluate_gate(seed_panels, 0.15, monkeypatch)

    # The planted asset-level signal is a 5-day reversal with IC near 0.15.
    assert result.realized_ic == pytest.approx(0.15, abs=0.03)

    # The calibrated portfolio clears the unchanged statistical gates.
    assert result.pbo < 0.20
    assert result.dsr_probability >= 0.95
    assert result.net_sharpe > 0.40
    assert result.net_sharpe == pytest.approx(0.50, abs=0.02)
    assert result.passed is True
    assert result.observations >= 3_000
