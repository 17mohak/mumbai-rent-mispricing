"""Tests for the DiD causal module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentlens.data.generate import generate_listings
from rentlens.geo.transit import enrich
from rentlens.causal.diff_in_diff import (
    _assign_treatment,
    create_panel,
    run_did,
    parallel_trends_test,
    treatment_effect,
    group_means,
    TRUE_LOG_EFFECT,
    COMMON_DRIFT,
    TREATMENT_DIST_M,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "cities" / "mumbai.yaml"
TRANSIT = Path(__file__).resolve().parents[1] / "data" / "reference" / "transit_mumbai.csv"


@pytest.fixture(scope="module")
def base_df() -> pd.DataFrame:
    raw = generate_listings(CONFIG, n_total=600, seed=5)
    return enrich(raw, TRANSIT)


@pytest.fixture(scope="module")
def panel(base_df) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return create_panel(base_df, rng)


def test_treatment_assignment_uses_distance(base_df):
    df = _assign_treatment(base_df[base_df["locality"] == "Powai"])
    assert "treated" in df.columns
    assert "dist_event_m" in df.columns
    assert df["dist_event_m"].gt(0).all()


def test_treatment_flag_consistent_with_distance(base_df):
    df = _assign_treatment(base_df[base_df["locality"] == "Powai"])
    assert (df[df["treated"] == 1]["dist_event_m"] <= TREATMENT_DIST_M).all()
    assert (df[df["treated"] == 0]["dist_event_m"] > TREATMENT_DIST_M).all()


def test_panel_has_three_periods(panel):
    assert set(panel["period"].unique()) == {-1, 0, 1}


def test_panel_each_listing_appears_three_times(panel):
    counts = panel.groupby("listing_id")["period"].count()
    assert (counts == 3).all()


def test_panel_powai_only(panel):
    assert (panel["locality"] == "Powai").all()


def test_treated_and_control_both_present(panel):
    t0 = panel[panel["period"] == 0]
    assert t0["treated"].sum() > 10, "Need at least 10 treated listings"
    assert t0["treated"].eq(0).sum() > 10, "Need at least 10 control listings"


def test_common_drift_applied_equally(panel):
    """Control group should drift by exactly COMMON_DRIFT per period (plus noise)."""
    ctrl = panel[panel["treated"] == 0]
    mean_by_period = ctrl.groupby("period")["log_rent_panel"].mean()
    drift_0_to_1 = mean_by_period[1] - mean_by_period[0]
    drift_minus1_to_0 = mean_by_period[0] - mean_by_period[-1]
    # Both should be close to COMMON_DRIFT (within 2× noise level)
    assert abs(drift_0_to_1 - COMMON_DRIFT) < 0.03
    assert abs(drift_minus1_to_0 - COMMON_DRIFT) < 0.03


def test_did_results_object(panel):
    results = run_did(panel)
    assert results.rsquared > 0.5, "DiD model should explain most panel variation"
    assert "did_post" in results.params


def test_parallel_trends_passes(panel):
    results = run_did(panel)
    pt = parallel_trends_test(results)
    assert pt["pass"], (
        f"Pre-trend should not be significant; got p={pt['pval']:.3f}, "
        f"coef={pt['coef']:+.4f}"
    )


def test_treatment_effect_recovers_planted(panel):
    results = run_did(panel)
    te = treatment_effect(results)
    assert te["recovered"], (
        f"DiD should recover planted {(np.exp(TRUE_LOG_EFFECT)-1)*100:.0f}% effect; "
        f"estimated {te['pct_effect']:+.1f}% (planted {te['planted_pct']:+.1f}%)"
    )


def test_treatment_effect_is_positive_and_significant(panel):
    results = run_did(panel)
    te = treatment_effect(results)
    assert te["coef"] > 0, "Treatment effect should be positive (metro uplift)"
    assert te["pval"] < 0.05, f"Treatment effect should be significant; p={te['pval']:.3f}"


def test_ci_contains_true_effect(panel):
    results = run_did(panel)
    te = treatment_effect(results)
    assert te["ci_lo"] <= TRUE_LOG_EFFECT <= te["ci_hi"], (
        f"95% CI [{te['ci_lo']:.4f}, {te['ci_hi']:.4f}] should contain "
        f"true effect {TRUE_LOG_EFFECT:.4f}"
    )


def test_group_means_shape(panel):
    gm = group_means(panel)
    assert gm.shape == (2, 3), f"Expected (2 groups × 3 periods); got {gm.shape}"
