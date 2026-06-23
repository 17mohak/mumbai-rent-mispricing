"""
Interpretable log-rent OLS baseline (hedonic pricing model).

Spatial cross-validation: leave-one-locality-out, WITHOUT locality dummies
in the CV design matrix — this is the strict generalization test (can the
pure hedonic + transit signal price a held-out area?).  The final full model
uses locality dummies as fixed effects.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rentlens.model.features import (
    add_derived, cv_folds, ols_Xy, regression_metrics, TARGET,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _pct(coef: float) -> float:
    """Log-point coefficient → percentage change."""
    return (np.exp(coef) - 1) * 100


def print_elasticities(results: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    p = results.params
    pv = results.pvalues

    def sig(name: str) -> str:
        v = pv.get(name, 1.0)
        return "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""

    print("\n  Elasticities (OLS on log-rent):")
    print(f"  {'Feature':<38}  Effect on rent      p-val")
    print("  " + "─" * 65)

    print(f"  {'carpet area  (+1% area)':<38}  {p.get('log_carpet', 0):+.2f}%  elasticity{sig('log_carpet')}")
    print(f"  {'bhk  (+1 room)':<38}  {_pct(p.get('bhk', 0)):+.1f}%{sig('bhk')}")
    print(f"  {'bathrooms  (+1 bath)':<38}  {_pct(p.get('bathrooms', 0)):+.1f}%{sig('bathrooms')}")
    print(f"  {'floor  (+1 floor up)':<38}  {_pct(p.get('floor', 0)):+.2f}%{sig('floor')}")
    print(f"  {'building age  (+10 years older)':<38}  {_pct(p.get('building_age_years', 0) * 10):+.1f}%{sig('building_age_years')}")
    print(f"  {'amenities  (+1 amenity)':<38}  {_pct(p.get('amenities_count', 0)):+.1f}%{sig('amenities_count')}")

    # Detect reference (dropped) category for each categorical group
    def _ref(prefix: str, universe: set[str]) -> str:
        present = {c.removeprefix(prefix) for c in p.index if c.startswith(prefix)}
        refs = universe - present
        return next(iter(refs), "ref")

    furn_ref = _ref("furnishing_", {"unfurnished", "semi", "furnished"})
    ptype_ref = _ref("property_type_", {"apartment", "independent"})

    print("  " + "─" * 65)
    for fc in sorted(c for c in p.index if c.startswith("furnishing_")):
        label = fc.removeprefix("furnishing_") + f" (vs {furn_ref})"
        print(f"  {label:<38}  {_pct(p[fc]):+.1f}%{sig(fc)}")

    for fc in sorted(c for c in p.index if c.startswith("property_type_")):
        label = fc.removeprefix("property_type_") + f" (vs {ptype_ref})"
        print(f"  {label:<38}  {_pct(p[fc]):+.1f}%{sig(fc)}")

    print("  " + "─" * 65 + "  ← transit signals")
    print(f"  {'operational metro  (+100m further)':<38}  {_pct(p.get('dist_op_100m', 0)):+.2f}%{sig('dist_op_100m')}")
    print(f"  {'UC metro           (+100m further)':<38}  {_pct(p.get('dist_uc_100m', 0)):+.2f}%{sig('dist_uc_100m')}")
    print(f"  {'planned metro      (+100m further)':<38}  {_pct(p.get('dist_pl_100m', 0)):+.2f}%{sig('dist_pl_100m')}")

    locality_cols = [c for c in p.index if c.startswith("locality_")]
    if locality_cols:
        print("  " + "─" * 65 + "  ← locality FE (vs baseline)")
        for lc in sorted(locality_cols):
            label = lc.replace("locality_", "")
            print(f"  {label:<38}  {_pct(p[lc]):+.1f}%{sig(lc)}")

    print(f"\n  Model R² = {results.rsquared:.4f}   Adj-R² = {results.rsquared_adj:.4f}"
          f"   N = {int(results.nobs):,}")
    print("  *** p<0.001  ** p<0.01  * p<0.05")
    print("\n  Note: transit elasticities are small because locality fixed effects")
    print("  absorb the between-locality gradient (e.g. Powai is both far from")
    print("  operational metro AND expensive — the locality dummy captures this).")
    print("  Within-locality transit gradients are estimated from fine-grained jitter.")


def spatial_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-locality-out CV without locality dummies (strict spatial test)."""
    rows = []
    d = add_derived(df)

    for train_idx, test_idx, loc in cv_folds(d):
        train = d.loc[train_idx]
        test = d.loc[test_idx]

        X_train, y_train = ols_Xy(train, include_locality=False)
        X_test, _        = ols_Xy(test,  include_locality=False)

        # Align columns (const always present; no locality dummies here)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        model = sm.OLS(y_train, X_train).fit()
        log_pred = model.predict(X_test)
        pred_rent = np.exp(log_pred)
        # ols_Xy drops rows with NaN in any selected numeric feature (real
        # `floor` data has scattered gaps) — use X_test's surviving index,
        # not the original test_idx, so true/pred stay aligned.
        true_rent = np.exp(d.loc[X_test.index, "log_rent"])

        m = regression_metrics(true_rent.values, pred_rent.values)
        m["locality"] = loc
        m["n"] = len(test_idx)
        rows.append(m)

    cv_df = pd.DataFrame(rows).set_index("locality")
    # Weighted overall
    weights = cv_df["n"] / cv_df["n"].sum()
    overall = {col: (cv_df[col] * weights).sum() for col in ["MAE_Rs", "MAPE_pct", "RMSE_Rs", "R2"]}
    overall["locality"] = "OVERALL (wtd)"
    overall["n"] = cv_df["n"].sum()
    cv_df = pd.concat([cv_df, pd.DataFrame([overall]).set_index("locality")])
    return cv_df


def fit_full(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS on full dataset with locality fixed effects."""
    X, y = ols_Xy(df, include_locality=True)
    return sm.OLS(y, X).fit()


def run(df: pd.DataFrame) -> tuple:
    """Fit, print diagnostics, run spatial CV; return (full_model, cv_metrics)."""
    print("\n--- Hedonic OLS (interpretable log-rent baseline) ---")
    model = fit_full(df)
    print_elasticities(model)

    print("\n  Spatial CV (leave-one-locality-out, no locality dummies):")
    cv = spatial_cv(df)
    print(
        cv[["MAE_Rs", "MAPE_pct", "RMSE_Rs", "R2", "n"]]
        .rename(columns={"MAE_Rs": "MAE(Rs)", "MAPE_pct": "MAPE%",
                         "RMSE_Rs": "RMSE(Rs)", "R2": "R²"})
        .round({"MAE(Rs)": 0, "MAPE%": 1, "RMSE(Rs)": 0, "R²": 3})
        .to_string()
    )
    return model, cv


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    df = pd.read_parquet(root / "data" / "processed" / "listings_geo.parquet")
    run(df)
