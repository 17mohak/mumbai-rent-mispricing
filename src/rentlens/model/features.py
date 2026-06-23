"""Shared feature-engineering utilities for all RentLens models."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── feature groups ──────────────────────────────────────────────────────────
NUMERIC = [
    "log_carpet",           # log(carpet_area_sqft)
    "bhk",
    "bathrooms",
    "floor",
    "building_age_years",
    "amenities_count",
    "dist_op_100m",         # dist_nearest_operational_m  / 100
    "dist_uc_100m",         # dist_nearest_under_construction_m / 100
    "dist_pl_100m",         # dist_nearest_planned_m / 100
]
CAT = ["furnishing", "property_type"]
LOCALITY_COL = "locality"
TARGET = "log_rent"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_carpet"] = np.log(out["carpet_area_sqft"])
    out["log_rent"] = np.log(out["monthly_rent"])
    out["dist_op_100m"] = out["dist_nearest_operational_m"] / 100
    out["dist_uc_100m"] = out["dist_nearest_under_construction_m"] / 100
    out["dist_pl_100m"] = out["dist_nearest_planned_m"] / 100
    return out


MIN_FEATURE_COVERAGE = 0.5  # require >=50% non-null to keep a numeric feature


def _available_numeric(d: pd.DataFrame) -> list[str]:
    """NUMERIC columns that carry enough signal to model.

    Two real-data failure modes, both observed in practice rather than
    hypothetical:
      - A distance-to-status feature (e.g. dist_pl_100m) is all-NaN whenever
        the transit table has zero stations of that status — real transit
        tables (unlike the synthetic one) aren't guaranteed to have
        operational/UC/planned stations in every bucket.
      - building_age_years is ~94% NaN on real MagicBricks data, because most
        listings report move-in availability ("Immediately"), not
        construction age, and that's not something to guess at.

    Either breaks statsmodels OLS outright (any NaN in exog raises) and a
    mostly-missing column carries negligible signal for LightGBM anyway, so
    columns below MIN_FEATURE_COVERAGE non-null are dropped rather than
    imputed.
    """
    return [c for c in NUMERIC if d[c].notna().mean() >= MIN_FEATURE_COVERAGE]


def ols_Xy(
    df: pd.DataFrame,
    include_locality: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_with_const, y) for statsmodels OLS."""
    import statsmodels.api as sm

    d = add_derived(df)
    numeric_cols = _available_numeric(d)
    # OLS can't tolerate *any* row-level NaN (unlike LightGBM, which handles
    # missing values natively) — a column can clear the coverage threshold
    # above and still have a few scattered gaps (e.g. real `floor` data at
    # 98% complete). Drop just those rows for the OLS fit; lgbm_Xy below
    # keeps them.
    d = d.dropna(subset=numeric_cols)
    y = d[TARGET]

    cat_cols = (CAT + [LOCALITY_COL]) if include_locality else CAT
    dummies = pd.get_dummies(d[cat_cols], drop_first=True, dtype=float)
    X = pd.concat([d[numeric_cols], dummies], axis=1)
    return sm.add_constant(X, has_constant="add"), y


def lgbm_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, categorical_feature_names) for LightGBM."""
    d = add_derived(df)
    y = d[TARGET]
    numeric_cols = _available_numeric(d)
    cat_cols = CAT + [LOCALITY_COL]
    for c in cat_cols:
        d[c] = d[c].astype("category")
    return d[numeric_cols + cat_cols], y, cat_cols


def cv_folds(df: pd.DataFrame) -> list[tuple[pd.Index, pd.Index, str]]:
    """Yield (train_idx, test_idx, held_out_locality) for spatial LOO-CV."""
    localities = df[LOCALITY_COL].unique()
    for loc in localities:
        train_idx = df.index[df[LOCALITY_COL] != loc]
        test_idx = df.index[df[LOCALITY_COL] == loc]
        yield train_idx, test_idx, loc


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE_Rs": mae, "MAPE_pct": mape, "RMSE_Rs": rmse, "R2": r2}
