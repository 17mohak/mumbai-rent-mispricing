"""Smoke tests for the model layer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentlens.data.generate import generate_listings
from rentlens.geo.transit import enrich
from rentlens.model.features import add_derived, lgbm_Xy, ols_Xy, cv_folds, TARGET

CONFIG = Path(__file__).resolve().parents[1] / "config" / "cities" / "mumbai.yaml"
TRANSIT = Path(__file__).resolve().parents[1] / "data" / "reference" / "transit_mumbai.csv"


@pytest.fixture(scope="module")
def geo_df() -> pd.DataFrame:
    raw = generate_listings(CONFIG, n_total=350, seed=7)
    return enrich(raw, TRANSIT)


def test_add_derived_columns(geo_df):
    d = add_derived(geo_df)
    assert "log_carpet" in d.columns
    assert "log_rent" in d.columns
    assert "dist_op_100m" in d.columns
    assert (d["log_rent"] > 0).all()


def test_ols_Xy_shape(geo_df):
    X, y = ols_Xy(geo_df, include_locality=True)
    assert len(X) == len(geo_df)
    assert len(y) == len(geo_df)
    assert "const" in X.columns
    assert "log_carpet" in X.columns


def test_ols_Xy_no_nans(geo_df):
    X, y = ols_Xy(geo_df, include_locality=True)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_lgbm_Xy_shape(geo_df):
    X, y, cats = lgbm_Xy(geo_df)
    assert len(X) == len(geo_df)
    assert set(cats).issubset(set(X.columns))
    assert not X.isnull().any().any()


def test_cv_folds_covers_all_localities(geo_df):
    localities_seen = set()
    for _, test_idx, loc in cv_folds(geo_df):
        localities_seen.add(loc)
        # holdout rows should only contain the held-out locality
        assert (geo_df.loc[test_idx, "locality"] == loc).all()
    assert localities_seen == set(geo_df["locality"].unique())


def test_cv_folds_no_overlap(geo_df):
    for train_idx, test_idx, _ in cv_folds(geo_df):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_hedonic_ols_fits(geo_df):
    import statsmodels.api as sm
    from rentlens.model.hedonic import fit_full
    model = fit_full(geo_df)
    assert model.rsquared > 0.70, f"OLS R² too low: {model.rsquared:.3f}"


def test_lgbm_fits_and_predicts(geo_df):
    from rentlens.model.gbm import _fit_lgbm
    X, y, cats = lgbm_Xy(geo_df)
    model = _fit_lgbm(X, y, cats)
    preds = model.predict(X)
    assert len(preds) == len(geo_df)
    residuals = np.abs(np.exp(preds) - np.exp(y.values))
    assert residuals.mean() < 20_000, "In-sample MAE suspiciously high"


def test_quantile_q10_lt_q90(geo_df):
    from rentlens.model.uncertainty import _fit_quantile
    from rentlens.model.features import lgbm_Xy

    X, y, cats = lgbm_Xy(geo_df)
    q10 = _fit_quantile(X, y, cats, 0.10)
    q90 = _fit_quantile(X, y, cats, 0.90)
    lo = q10.predict(X)
    hi = q90.predict(X)
    assert (hi > lo).mean() > 0.95, "q90 should be > q10 for most rows"
