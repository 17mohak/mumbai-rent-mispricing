"""
Difference-in-Differences: Metro Line 6 Opening Treatment Effect.

══════════════════════════════════════════════════════════════════
METHODOLOGY DEMONSTRATION ON SYNTHETIC DATA
══════════════════════════════════════════════════════════════════

Design
──────
Event     : Powai Lake Metro (Line 6) opening — set to 2026-03-31
Treatment : Listings within 1,500 m of Powai Lake Metro station
Control   : Remaining Powai listings (same locality, > 1,500 m)
            Using within-locality comparison to avoid the collinearity
            between locality identity and treatment status that would
            arise in a cross-locality spec with locality fixed effects.

Time periods:
  t = −1  pre-pre  (simulated: 2 quarters before opening)
  t =  0  pre      (reference: 1 quarter before opening)
  t = +1  post     (simulated: 1 quarter after opening)

Outcome : log(monthly_rent) in each period

Data-generating process for the synthetic panel:
  log_rent_it = log(rent_i0)
              + COMMON_DRIFT × (t + 1)           ← same for both groups
              + TRUE_LOG_EFFECT × treated_i × 1[t=1]  ← treatment effect
              + ε_it   where ε ~ N(0, 0.025²)

True planted effect: exp(TRUE_LOG_EFFECT) − 1 = 8 % rent uplift.
The DiD estimator should recover ≈ 8 % (modulo small-sample noise).

Parallel-Trends Assumption
───────────────────────────
Required: treated and control listings would have followed the same rent
trajectory in the absence of treatment. We satisfy this BY CONSTRUCTION
in the synthetic panel (identical common drift for both groups in t=−1
and t=0). We test it formally: the coefficient on (treated × pre-pre)
should be statistically indistinguishable from zero.

Threats to Validity (on real data)
───────────────────────────────────
1. Parallel trends: metro opening may be anticipated — rents near the
   station may rise BEFORE opening, violating the assumption.
2. Selection bias: buyers who value transit may systematically choose
   flats near the planned station (pre-sorting), biasing upward.
3. SUTVA / spillovers: nearby listings outside the 1,500 m radius may
   also benefit, contaminating the control group.
4. Confounding infrastructure: other improvements (road widening, malls)
   may coincide with the metro opening.
5. Attrition: high-demand listings near the station may be harder to
   observe in the pre-period if they are rented quickly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from rentlens.geo.transit import haversine_m
from rentlens.model.features import add_derived

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────

EVENT_STATION_NAME  = "Powai Lake Metro"
EVENT_STATION_LAT   = 19.1182
EVENT_STATION_LON   = 72.9058
TREATMENT_DIST_M    = 1_500      # 1.5 km walking catchment

TRUE_LOG_EFFECT     = np.log(1.08)   # 8 % uplift — the planted effect
COMMON_DRIFT        = np.log(1.02)   # 2 % market drift per period (same for both groups)
PANEL_NOISE_SIGMA   = 0.025          # idiosyncratic period-to-period noise


# ── panel construction ───────────────────────────────────────────────────────

def _assign_treatment(df: pd.DataFrame) -> pd.DataFrame:
    dists = haversine_m(
        df["latitude"].values, df["longitude"].values,
        EVENT_STATION_LAT, EVENT_STATION_LON,
    )
    out = df.copy()
    out["dist_event_m"] = dists
    out["treated"] = (dists <= TREATMENT_DIST_M).astype(int)
    return out


def create_panel(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a synthetic 3-period panel (t ∈ {−1, 0, +1}).

    Only Powai listings are used so the comparison is purely within-locality.
    Listing characteristics are fixed across periods; only log_rent varies.
    """
    powai = df[df["locality"] == "Powai"].copy()
    powai = _assign_treatment(powai)
    powai = add_derived(powai)

    frames = []
    for t in [-1, 0, 1]:
        d = powai.copy()
        d["period"] = t
        d["post"]   = int(t == 1)

        # Cumulative drift: 0 at t=−1, 1× at t=0, 2× at t=+1
        drift     = COMMON_DRIFT * (t + 1)
        treatment = TRUE_LOG_EFFECT * d["treated"] * d["post"]
        noise     = rng.normal(0.0, PANEL_NOISE_SIGMA, len(d))

        d["log_rent_panel"] = d["log_rent"] + drift + treatment + noise
        frames.append(d)

    panel = pd.concat(frames, ignore_index=True)
    return panel


# ── DiD regressions ──────────────────────────────────────────────────────────

def _build_X(panel: pd.DataFrame, periods: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Design matrix for the event-study specification."""
    sub = panel[panel["period"].isin(periods)].copy()

    # Period dummies (reference = t = 0)
    sub["t_minus1"] = (sub["period"] == -1).astype(float)
    sub["t_plus1"]  = (sub["period"] ==  1).astype(float)

    # Interaction terms (treated × period; reference = treated × t=0 → omitted)
    sub["did_pre"]  = sub["treated"] * sub["t_minus1"]  # pre-trend test
    sub["did_post"] = sub["treated"] * sub["t_plus1"]   # treatment effect

    # Hedonic controls (within-period variation only; listing FE would eat these)
    hedonic = ["log_carpet", "bhk", "floor", "building_age_years", "amenities_count"]
    cat     = pd.get_dummies(sub[["furnishing", "property_type"]], drop_first=True, dtype=float)

    feats = pd.concat([
        sub[["treated", "t_minus1", "t_plus1", "did_pre", "did_post"] + hedonic],
        cat,
    ], axis=1)
    return sm.add_constant(feats), sub["log_rent_panel"]


def run_did(panel: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Event-study DiD on all 3 periods with HC3 standard errors."""
    X, y = _build_X(panel, [-1, 0, 1])
    return sm.OLS(y, X).fit(cov_type="HC3")


def parallel_trends_test(results) -> dict:
    """Extract the pre-trend coefficient and test H₀: β_pre = 0."""
    coef = results.params.get("did_pre", np.nan)
    se   = results.bse.get("did_pre",   np.nan)
    pval = results.pvalues.get("did_pre", np.nan)
    ci   = results.conf_int().loc["did_pre"] if "did_pre" in results.params else [np.nan, np.nan]
    return {
        "coef":      coef,
        "se":        se,
        "pval":      pval,
        "ci_lo":     ci[0],
        "ci_hi":     ci[1],
        "pass":      pval > 0.05,   # fail to reject H₀ = good for us
    }


def treatment_effect(results) -> dict:
    """Extract DiD treatment-effect coefficient."""
    coef = results.params.get("did_post", np.nan)
    se   = results.bse.get("did_post",   np.nan)
    pval = results.pvalues.get("did_post", np.nan)
    ci   = results.conf_int().loc["did_post"]
    return {
        "coef":         coef,
        "se":           se,
        "pval":         pval,
        "ci_lo":        ci[0],
        "ci_hi":        ci[1],
        "pct_effect":   (np.exp(coef) - 1) * 100,
        "pct_ci_lo":    (np.exp(ci[0]) - 1) * 100,
        "pct_ci_hi":    (np.exp(ci[1]) - 1) * 100,
        "planted_pct":  (np.exp(TRUE_LOG_EFFECT) - 1) * 100,
        "recovered":    abs(coef - TRUE_LOG_EFFECT) < 2 * se,
    }


# ── descriptive table ────────────────────────────────────────────────────────

def group_means(panel: pd.DataFrame) -> pd.DataFrame:
    summary = (
        panel.groupby(["treated", "period"])["log_rent_panel"]
        .mean()
        .unstack("period")
        .rename(index={0: "Control (far)", 1: "Treatment (near)"})
    )
    summary.columns = [f"t={c}" for c in summary.columns]
    # DiD calculation row
    diff = summary.diff().iloc[-1]
    did_row = diff.diff().iloc[-1]
    return summary


# ── event study plot ─────────────────────────────────────────────────────────

def save_event_study_plot(results, output_path: Path) -> None:
    periods      = [-1, 0, 1]
    period_names = ["Pre-pre (t=−1)", "Pre (t=0) [ref]", "Post (t=+1)"]

    coefs = [
        results.params.get("did_pre",  0.0),   # t=-1
        0.0,                                     # t=0 reference, set to 0
        results.params.get("did_post", 0.0),   # t=+1
    ]
    cis_lo = [
        results.conf_int().loc["did_pre",  0] if "did_pre"  in results.params else 0.0,
        0.0,
        results.conf_int().loc["did_post", 0] if "did_post" in results.params else 0.0,
    ]
    cis_hi = [
        results.conf_int().loc["did_pre",  1] if "did_pre"  in results.params else 0.0,
        0.0,
        results.conf_int().loc["did_post", 1] if "did_post" in results.params else 0.0,
    ]
    errors_lo = [c - lo for c, lo in zip(coefs, cis_lo)]
    errors_hi = [hi - c for c, hi in zip(coefs, cis_hi)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(TRUE_LOG_EFFECT, color="#16A34A", linewidth=1.0, linestyle=":",
               label=f"Planted effect ({(np.exp(TRUE_LOG_EFFECT)-1)*100:.0f}%)")
    ax.axvline(0.5, color="#DC2626", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Station opens")

    ax.errorbar(
        periods, coefs,
        yerr=[errors_lo, errors_hi],
        fmt="o", color="#2563EB", capsize=5, linewidth=1.5, markersize=7,
        label="DiD coefficient (95% CI)",
    )
    ax.set_xticks(periods)
    ax.set_xticklabels(period_names, fontsize=9)
    ax.set_ylabel("Estimated effect on log(rent)", fontsize=10)
    ax.set_title("Event Study: Powai Lake Metro (Line 6)\nTreatment = within 1.5 km",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main entry ───────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, output_dir: Path, seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)

    print("\n--- Difference-in-Differences: Metro Line 6 Opening ---")
    print("  [NOTE: Methodology demonstration on SYNTHETIC data]")

    panel = create_panel(df, rng)

    n_treated = panel[panel["period"] == 0]["treated"].sum()
    n_control = panel[panel["period"] == 0]["treated"].eq(0).sum()
    print(f"\n  Event     : {EVENT_STATION_NAME}  (opening 2026-03-31)")
    print(f"  Threshold : {TREATMENT_DIST_M:,} m walking catchment")
    print(f"  N treated : {n_treated:,}  |  N control : {n_control:,}  (Powai listings only)")
    print(f"  Planted treatment effect : {(np.exp(TRUE_LOG_EFFECT)-1)*100:.1f}% rent uplift")
    print(f"  Common market drift      : {(np.exp(COMMON_DRIFT)-1)*100:.1f}% per period")

    # ── descriptive mean table
    gm = group_means(panel)
    print("\n  Mean log-rent by group × period:")
    print(gm.round(4).to_string())
    diff_row = (gm.iloc[1] - gm.iloc[0])
    did_raw  = diff_row["t=1"] - diff_row["t=0"]
    print(f"\n  Raw DiD (table arithmetic) : {did_raw:.4f} log-pts"
          f"  ≈ {(np.exp(did_raw)-1)*100:.1f}%")

    # ── regression
    results = run_did(panel)

    # ── parallel trends
    pt = parallel_trends_test(results)
    print(f"\n  Parallel-Trends Test (H₀: pre-trend coefficient = 0):")
    print(f"    β_pre   = {pt['coef']:+.4f}  SE={pt['se']:.4f}  p={pt['pval']:.3f}"
          f"  [{('PASS — no pre-trend detected' if pt['pass'] else 'FAIL — pre-trend present')}]")
    print(f"    95% CI  : [{pt['ci_lo']:+.4f}, {pt['ci_hi']:+.4f}]")
    if pt["pass"]:
        print("    ✓ Parallel trends assumption: fail to reject H₀ (consistent with assumption)")
    else:
        print("    ✗ Parallel trends potentially violated — interpret DiD with caution")

    # ── treatment effect
    te = treatment_effect(results)
    print(f"\n  DiD Treatment Effect:")
    print(f"    β_post  = {te['coef']:+.4f}  SE={te['se']:.4f}  p={te['pval']:.4f}")
    print(f"    Estimated rent uplift : {te['pct_effect']:+.1f}%"
          f"  (95% CI: {te['pct_ci_lo']:+.1f}% to {te['pct_ci_hi']:+.1f}%)")
    print(f"    Planted effect        : {te['planted_pct']:+.1f}%")
    sig = "Yes" if te["pval"] < 0.05 else "No"
    rec = "Yes" if te["recovered"] else "No"
    print(f"    Statistically significant (p<0.05) : {sig}")
    print(f"    Within 2 SE of planted value       : {rec}")

    # ── full regression summary (selected rows)
    print(f"\n  Model: N={int(results.nobs):,}  R²={results.rsquared:.4f}  (HC3 SEs)")
    print(f"  Key coefficients:")
    for var in ["treated", "t_minus1", "t_plus1", "did_pre", "did_post"]:
        if var in results.params:
            row = f"    {var:<18} β={results.params[var]:+.4f}  p={results.pvalues[var]:.3f}"
            print(row)

    # ── threats to validity
    print(f"""
  Threats to Validity (on real data — satisfied by construction here):
  1. Parallel trends: Market forces or amenity improvements correlated
     with metro construction may violate the assumption. Testing with
     pre-pre period data (coefficient {pt['coef']:+.4f}, p={pt['pval']:.3f}).
  2. Anticipation: Landlords may raise rents upon the metro announcement
     rather than the opening — a pre-trend signature of this concern.
  3. SUTVA / spillovers: Listings just outside the 1,500 m boundary may
     also benefit, compressing the DiD estimate downward.
  4. Selection: High-quality listings near the future station may have
     been absorbed into the pre-period (survivorship), biasing upward.
  5. On synthetic data: all of the above are satisfied by construction;
     the DiD recovers the planted effect as a methodology verification.
""")

    # ── event study plot
    plot_path = output_dir / "did_event_study.png"
    save_event_study_plot(results, plot_path)
    print(f"  Event study plot saved → {plot_path}")

    return {
        "panel":         panel,
        "results":       results,
        "parallel_trends": pt,
        "treatment_effect": te,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    df = pd.read_parquet(root / "data" / "processed" / "listings_scored.parquet")
    run(df, root / "outputs")
