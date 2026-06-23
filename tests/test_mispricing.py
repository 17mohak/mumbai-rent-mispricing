"""Tests for mispricing detection and arbitrage ranking."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentlens.data.generate import generate_listings, PLANTED_BIAS
from rentlens.geo.transit import enrich
from rentlens.model.mispricing import (
    fit_cross_market,
    add_cross_market_residuals,
    locality_mispricing,
    verify_planted_signal_gt,
    build_arbitrage_list,
)
from rentlens.model.uncertainty import fit_full as fit_quantile, predict_intervals

CONFIG = Path(__file__).resolve().parents[1] / "config" / "cities" / "mumbai.yaml"
TRANSIT = Path(__file__).resolve().parents[1] / "data" / "reference" / "transit_mumbai.csv"


@pytest.fixture(scope="module")
def full_df() -> pd.DataFrame:
    raw = generate_listings(CONFIG, n_total=1400, seed=42)
    geo = enrich(raw, TRANSIT)
    q10, q50, q90 = fit_quantile(geo)
    ivs = predict_intervals(geo, q10, q50, q90)
    return geo.merge(ivs[["listing_id", "fair_rent_pred", "interval_lower",
                            "interval_upper", "outside_interval"]],
                     on="listing_id")


@pytest.fixture(scope="module")
def scored_df(full_df) -> pd.DataFrame:
    cm_model = fit_cross_market(full_df)
    return add_cross_market_residuals(full_df, cm_model)


def test_cross_market_model_r2(full_df):
    model = fit_cross_market(full_df)
    assert model.rsquared > 0.50, f"Cross-market OLS R² too low: {model.rsquared:.3f}"


def test_residual_columns_present(scored_df):
    for col in ["fundamental_fair_rent", "residual_cm", "residual_cm_pct"]:
        assert col in scored_df.columns


def test_residual_sum_to_near_zero_for_neutral_localities(scored_df):
    neutral = scored_df[scored_df["locality"].isin(["Andheri East", "Goregaon", "Chembur"])]
    median_resid = neutral["residual_cm_pct"].median()
    assert abs(median_resid) < 10, f"Neutral localities show large residual: {median_resid:.1f}%"


def test_powai_residual_is_positive(scored_df):
    powai = scored_df[scored_df["locality"] == "Powai"]
    assert powai["residual_cm_pct"].median() > 5, "Powai should be detectably overpriced"


def test_mulund_residual_is_negative(scored_df):
    mulund = scored_df[scored_df["locality"] == "Mulund"]
    assert mulund["residual_cm_pct"].median() < -5, "Mulund should be detectably underpriced"


def test_planted_signal_recovered(full_df, scored_df):
    # Merge GT column back for the ground-truth check
    df_gt = scored_df.copy()
    if "_fair_rent_gt" not in df_gt.columns:
        gt_col = full_df[["listing_id", "_fair_rent_gt"]]
        df_gt = df_gt.merge(gt_col, on="listing_id", how="left")
    checks = verify_planted_signal_gt(df_gt)
    for loc, res in checks.items():
        assert res["pass"], (
            f"Planted signal not recovered for {loc}: "
            f"expected {res['expected_pct']:+.1f}%, "
            f"observed {res['observed_pct']:+.2f}%"
        )


def test_locality_mispricing_table_shape(scored_df):
    tbl = locality_mispricing(scored_df)
    assert len(tbl) == scored_df["locality"].nunique()
    assert "residual_pct" in tbl.columns
    assert "signal" in tbl.columns


def test_powai_signal_is_overpriced(scored_df):
    tbl = locality_mispricing(scored_df)
    assert tbl.loc["Powai", "signal"] == "OVERPRICED"


def test_mulund_signal_is_underpriced(scored_df):
    tbl = locality_mispricing(scored_df)
    assert tbl.loc["Mulund", "signal"] == "UNDERPRICED"


def test_arbitrage_list_non_empty(scored_df):
    arb = build_arbitrage_list(scored_df)
    assert len(arb) > 0, "Arbitrage list should not be empty"


def test_arbitrage_all_underpriced(scored_df):
    arb = build_arbitrage_list(scored_df)
    assert (arb["residual_cm_pct"] < 0).all(), "All arb candidates must be underpriced vs fundamentals"


def test_arbitrage_all_near_future_transit(scored_df):
    arb = build_arbitrage_list(scored_df, max_future_dist_m=2_500)
    near = (
        (scored_df.loc[scored_df["listing_id"].isin(arb["listing_id"]),
                       "dist_nearest_under_construction_m"] <= 2_500) |
        (scored_df.loc[scored_df["listing_id"].isin(arb["listing_id"]),
                       "dist_nearest_planned_m"] <= 2_500)
    )
    assert near.all(), "All arbitrage candidates must be within 2,500m of future transit"


def test_arbitrage_ranked_by_arb_score(scored_df):
    arb = build_arbitrage_list(scored_df)
    assert arb["arb_score"].is_monotonic_increasing, "Arbitrage list must be sorted by arb_score"
