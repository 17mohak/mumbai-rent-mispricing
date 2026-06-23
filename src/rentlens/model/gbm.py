"""
LightGBM gradient-boosted model + SHAP feature importance.

Spatial CV: locality is a categorical feature, so the model CAN use
locality information — this is a tighter CV (tests within-locality
generalisation) compared to the OLS holdout which strips locality entirely.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from rentlens.model.features import (
    add_derived, cv_folds, lgbm_Xy, regression_metrics, TARGET,
)

warnings.filterwarnings("ignore")

LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)


def _fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series, cat_cols: list[str]) -> LGBMRegressor:
    model = LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    return model


def spatial_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = add_derived(df)

    for train_idx, test_idx, loc in cv_folds(d):
        X_train, y_train, cat_cols = lgbm_Xy(d.loc[train_idx])
        X_test,  y_test,  _        = lgbm_Xy(d.loc[test_idx])

        model = _fit_lgbm(X_train, y_train, cat_cols)
        log_pred = model.predict(X_test)
        m = regression_metrics(np.exp(y_test.values), np.exp(log_pred))
        m["locality"] = loc
        m["n"] = len(test_idx)
        rows.append(m)

    cv_df = pd.DataFrame(rows).set_index("locality")
    weights = cv_df["n"] / cv_df["n"].sum()
    overall = {col: (cv_df[col] * weights).sum() for col in ["MAE_Rs", "MAPE_pct", "RMSE_Rs", "R2"]}
    overall["locality"] = "OVERALL (wtd)"
    overall["n"] = cv_df["n"].sum()
    return pd.concat([cv_df, pd.DataFrame([overall]).set_index("locality")])


def shap_analysis(model: LGBMRegressor, X: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # mean |SHAP| per feature
    importance = pd.DataFrame(
        {"feature": X.columns, "mean_abs_shap": np.abs(shap_vals).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    # bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance.head(12)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2563EB")
    ax.set_xlabel("Mean |SHAP value| (log-rent units)")
    ax.set_title("LightGBM — SHAP Feature Importance (RentLens)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return importance


def run(df: pd.DataFrame, output_dir: Path) -> tuple:
    d = add_derived(df)
    X_full, y_full, cat_cols = lgbm_Xy(d)

    print("\n--- LightGBM ---")
    print("  Spatial CV (leave-one-locality-out, locality as categorical feature):")
    cv = spatial_cv(df)
    print(
        cv[["MAE_Rs", "MAPE_pct", "RMSE_Rs", "R2", "n"]]
        .rename(columns={"MAE_Rs": "MAE(Rs)", "MAPE_pct": "MAPE%",
                         "RMSE_Rs": "RMSE(Rs)", "R2": "R²"})
        .round({"MAE(Rs)": 0, "MAPE%": 1, "RMSE(Rs)": 0, "R²": 3})
        .to_string()
    )

    print("\n  Fitting final model on all data...")
    model = _fit_lgbm(X_full, y_full, cat_cols)

    shap_out = output_dir / "shap_importance.png"
    importance = shap_analysis(model, X_full, shap_out)
    print(f"\n  Top SHAP features:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["mean_abs_shap"] / importance["mean_abs_shap"].max() * 20)
        print(f"    {row['feature']:<30}  {row['mean_abs_shap']:.4f}  {bar}")
    print(f"\n  SHAP plot saved → {shap_out}")

    return model, cv, importance


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    df = pd.read_parquet(root / "data" / "processed" / "listings_geo.parquet")
    run(df, root / "outputs")
