# Decisions

Architectural decision records for Project Apex Quant.  Each entry records the
date, the decision, the alternatives considered, and the consequences.

## Gate Calibration Baseline

- **Date:** 2026-11-10
- **Status:** Accepted
- **Phase:** 1 (gate calibration)

### Decision

The CPCV + PBO + DSR gate is calibrated in both directions with a deterministic
synthetic panel:

- `tests/calibration/test_known_positive.py` must accept a planted edge:
  PBO < 0.20, DSR >= 0.95, and after-cost net Sharpe > 0.40.
- `tests/calibration/test_known_negative.py` must reject pure noise:
  PBO >= 0.20, DSR < 0.95, or net Sharpe <= 0.
- `tests/calibration/test_boundary.py` sweeps IC
  {0.00, 0.05, 0.10, 0.15, 0.20} over 20 fixed seeds, asserts the pass rate is
  non-decreasing, and writes `artifacts/calibration_report.md`.

The PBO < 0.20 and DSR >= 0.95 thresholds are unchanged.  The production gate
code in `backtest/cpcv.py` and `backtest/metrics.py` is unchanged.

### Context

The Phase 1 brief asked for 500 symbols, 3 years of daily bars, a planted
after-cost Sharpe of 0.50, and DSR >= 0.95.  Those requirements are not jointly
satisfiable.  Using the production DSR formula:

- 3 years (756 observations) at Sharpe 0.50 gives DSR 0.8065.
- Approximately 2,773 observations (~11 years) are required for DSR 0.95 at
  Sharpe 0.50.
- 15 years (3,780 observations) at Sharpe 0.50 gives DSR 0.9735.

The gate is therefore not miscalibrated; the 3-year sample simply has too
little power to distinguish a Sharpe-0.50 edge from the trial-adjusted null.
The calibration uses 15 years for margin rather than weakening the threshold or
inflating the planted Sharpe to ~0.95.

A second inconsistency: a fully invested 500-symbol long/short portfolio with
an asset-level IC of 0.15 has a natural Sharpe far above 0.50 (the prototype
measured ~43).  The generator therefore plants the asset-level IC and the
portfolio-level Sharpe independently, exactly as the existing
`generate_synthetic_calibration_signal` does.  A calibrated zero-alpha overlay
sets the portfolio's after-cost Sharpe to 0.50 while the cross-sectional signal
keeps its IC near 0.15.  The overlay is a diversified basket, not a single
fat-tailed series, so the portfolio return's finite-sample kurtosis stays near
normal while the asset-level innovations remain Student-t(5).

### Test design

- 500 symbols, 15 years, daily observations (3,780 return observations).
- Signal: negative rolling 5-day sum of past Student-t(5) noise, standardized
  cross-sectionally each day.
- Noise: Student-t(5), standardized, with AR(1) autocorrelation 0.02.
- Portfolio: long the top signal quintile, short the bottom quintile,
  dollar-neutral, equal weight.
- Costs: production `BTConfig` parameters (1 bp slippage, 2 bps quoted spread,
  0.3 bps sell fee) applied to daily turnover, giving ~2.15 bps per side.
- Gate: the real `evaluate_cpcv` with 6 CPCV groups, a 1-day embargo, and two
  candidate lookbacks (5-day true, 10-day decoy).
- Seeds: 20261110 for the positive/negative tests; 20261110–20261129 for the
  boundary sweep.  No unseeded randomness.

The intraday fill engine is bypassed because the strategy is a daily
cross-sectional portfolio; the test injects pre-costed daily trade records into
the real CPCV/PBO/DSR gate.  This is documented in the test module docstring.

### Alternatives considered

- **Keep 3 years and lower DSR to ~0.80.** Rejected: violates the non-negotiable
  DSR >= 0.95 constraint and would make the gate accept weaker evidence.
- **Raise the planted Sharpe to ~0.95.** Rejected: the brief asks for a modest,
  realistic edge of ~0.50; a 0.95 Sharpe would test a different hypothesis.
- **Use 30,000 observations (the existing smoke test).** Rejected for the
  calibration suite: 119 years of daily data is not a realistic research sample
  and hides the sample-size problem instead of documenting it.
- **Use `n_trials` equal to the two PBO candidates.** Rejected for Phase 1: at
  15 years and Sharpe 0.50, the two-trial DSR penalty pushes the probability
  below 0.95.  Phase 2 owns the multi-trial penalty; Phase 1 isolates the
  single-hypothesis DSR threshold, matching the existing CI smoke test.  The
  positive test still uses two candidates so PBO is exercised.
- **Build a full cross-sectional backtest engine.** Rejected for Phase 1: the
  gate code is the object under test; a new engine would expand scope and add
  untested machinery.  The production `evaluate_cpcv` path is used directly.

### Consequences

- `pytest tests/calibration/ -v` takes roughly 7–8 minutes on the ARM64 CI
  runner; the dedicated `calibration` CI job has a 30-minute timeout.
- The main `verify` job excludes `tests/calibration/` and the dedicated job runs
  it as a blocking check, so the suite is not run twice.
- Any future change that makes the gate accept noise or reject a known-positive
  fails `tests/calibration/` and blocks the merge.
- The boundary report is committed at `artifacts/calibration_report.md`; a
  regenerated report that changes the pass-rate table is visible in review.
