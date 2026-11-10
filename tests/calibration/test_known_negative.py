"""Known-negative calibration: pure noise must be rejected.

Same generator, same 500-symbol panel, same 15-year sample, same calibrated
risk budget as the positive test, but the planted information coefficient is
zero.  The gate must not pass it: PBO >= 0.20, DSR < 0.95, or net Sharpe <= 0.
"""
from __future__ import annotations

import pytest

from tests.calibration.synthetic import DEFAULT_SEED, build_seed, evaluate_gate


def test_known_negative_is_rejected_by_the_gate(monkeypatch):
    seed_panels = build_seed(
        DEFAULT_SEED,
        (0.0,),
        compute_realized_ic=True,
    )
    result = evaluate_gate(seed_panels, 0.0, monkeypatch)

    # No planted edge: the asset-level IC is statistically indistinguishable
    # from zero.
    assert abs(result.realized_ic) < 0.03

    # At least one unchanged production gate must reject the null.
    assert (
        result.pbo >= 0.20
        or result.dsr_probability < 0.95
        or result.net_sharpe <= 0.0
    )
    assert result.passed is False
