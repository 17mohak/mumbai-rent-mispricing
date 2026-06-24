"""
Mispricing detection and transit-arbitrage ranking.

DESIGN NOTE ON SIGNAL RECOVERY
───────────────────────────────
The planted Powai/Mulund biases are locality-level: every Powai flat is
generated +12% above its hedonic fair value.  A model trained WITH locality
fixed effects absorbs that +12% into the Powai FE and produces near-zero
residuals — it has "explained away" the signal.

To expose the planted signal we need a CROSS-MARKET model: OLS trained on
all data WITHOUT locality identity.  This model prices each flat from its
observable fundamentals (area, age, amenities, transit distances) only.
The locality-level average residual then reflects *how much each area
deviates from what its fundamentals should command* — that is the
mispricing signal.

ARBITRAGE FRAMING
─────────────────
Arbitrage candidates: listings near UC / planned stations whose actual rent
is BELOW (a) their cross-market fundamental fair value AND (b) the median
rent of operationally-served comparable flats in the same locality.
These flats are doubly undervalued: below-fundamentals today, plus they
carry an unpriced transit option that will crystallise when the line opens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rentlens.data.generate import PLANTED_BIAS
from rentlens.model.features import ols_Xy


# ── cross-market (no-locality) fundamental model ────────────────────────────

def fit_cross_market(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS on log-rent without locality dummies — prices purely from fundamentals."""
    X, y = ols_Xy(df, include_locality=False)
    return sm.OLS(y, X).fit()


def add_cross_market_residuals(
    df: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    X, _ = ols_Xy(df, include_locality=False)
    log_pred = model.predict(X)
    # ols_Xy drops rows with NaN in any required numeric feature (real
    # `floor` data has scattered gaps) — restrict to X's surviving index
    # rather than copying the full df, which would leave those rows with no
    # prediction (NaN) and break the int cast below.
    out = df.loc[X.index].copy()
    out["fundamental_fair_rent"] = np.exp(log_pred).round(0).astype(int)
    out["residual_cm"]     = (out["monthly_rent"] - out["fundamental_fair_rent"]).round(0).astype(int)
    out["residual_cm_pct"] = (out["residual_cm"] / out["fundamental_fair_rent"] * 100).round(2)
    return out


# ── locality-level aggregation ───────────────────────────────────────────────

def locality_mispricing(df: pd.DataFrame) -> pd.DataFrame:
    tbl = (
        df.groupby("locality")
        .agg(
            n               = ("listing_id",        "count"),
            median_rent     = ("monthly_rent",       "median"),
            fair_rent_cm    = ("fundamental_fair_rent", "median"),
            residual_pct    = ("residual_cm_pct",    "median"),
            pct_overpriced  = ("residual_cm_pct",    lambda x: (x > 0).mean() * 100),
            pct_outside_iv  = ("outside_interval",   lambda x: x.sum() / len(x) * 100),
        )
        .sort_values("residual_pct", ascending=False)
    )
    tbl["signal"] = tbl["residual_pct"].apply(
        lambda r: "OVERPRICED" if r > 5 else ("UNDERPRICED" if r < -5 else "FAIR")
    )
    return tbl.round({"median_rent": 0, "fair_rent_cm": 0, "residual_pct": 2,
                       "pct_overpriced": 1, "pct_outside_iv": 1})


def verify_planted_signal(tbl: pd.DataFrame) -> dict[str, dict]:
    """Cross-market residual check — informational only. See verify_planted_signal_gt."""
    results = {}
    for loc, expected in PLANTED_BIAS.items():
        observed = tbl.loc[loc, "residual_pct"] / 100
        passed = abs(observed - expected) < 0.05
        results[loc] = {
            "expected_pct": round(expected * 100, 1),
            "observed_pct": round(observed * 100, 2),
            "pass": passed,
        }
    return results


def verify_planted_signal_gt(df: pd.DataFrame) -> dict[str, dict]:
    """
    Ground-truth signal recovery using _fair_rent_gt (SYNTHETIC data only).

    The cross-market residual mixes (a) the locality's natural premium vs the
    market average (encoded in base_ppsf) with (b) the planted bias.  The
    unambiguous check compares actual rent directly to the formula-generated
    fair rent, isolating just the planted bias.
    """
    if "_fair_rent_gt" not in df.columns:
        raise ValueError("_fair_rent_gt column required for ground-truth check")
    results = {}
    for loc, expected in PLANTED_BIAS.items():
        sub = df[df["locality"] == loc]
        observed = ((sub["monthly_rent"] / sub["_fair_rent_gt"]) - 1).median()
        passed = abs(observed - expected) < 0.04  # within 4 pp
        results[loc] = {
            "expected_pct": round(expected * 100, 1),
            "observed_pct": round(float(observed) * 100, 2),
            "pass": passed,
        }
    return results


# ── operational-metro peer median ────────────────────────────────────────────

def _op_peer_median(
    df: pd.DataFrame,
    locality: str,
    bhk: int,
    op_thresh_m: float = 1_500,
) -> float | None:
    peers = df[
        (df["locality"] == locality)
        & (df["bhk"] == bhk)
        & (df["dist_nearest_operational_m"] <= op_thresh_m)
    ]["monthly_rent"]
    if len(peers) < 5:
        # widen to all BHKs in locality near op-metro
        peers = df[
            (df["locality"] == locality)
            & (df["dist_nearest_operational_m"] <= op_thresh_m)
        ]["monthly_rent"]
    return float(peers.median()) if len(peers) >= 3 else None


# ── arbitrage list ───────────────────────────────────────────────────────────

def build_arbitrage_list(
    df: pd.DataFrame,
    max_future_dist_m: float = 2_500,
) -> pd.DataFrame:
    """
    Returns listings that are:
      1. Within max_future_dist_m of a UC or planned station
      2. Priced BELOW their cross-market fundamental fair rent (residual_cm < 0)
    Ranked by the combined discount: actual vs both fundamental fair rent
    and vs operational-metro peers.
    """
    near_uc = df["dist_nearest_under_construction_m"] <= max_future_dist_m
    near_pl = df["dist_nearest_planned_m"] <= max_future_dist_m
    underpriced = df["residual_cm_pct"] < 0

    cands = df[underpriced & (near_uc | near_pl)].copy()

    # Pick the closer of UC / planned station as primary reference.
    # fillna(inf) so a missing distance (e.g. no planned stations in the
    # transit table at all) never wins a "<=" comparison against NaN, which
    # pandas always evaluates False and would mislabel every UC-near candidate.
    uc_closer = (
        cands["dist_nearest_under_construction_m"].fillna(np.inf)
        <= cands["dist_nearest_planned_m"].fillna(np.inf)
    )
    cands["future_transit_type"] = np.where(uc_closer, "under_construction", "planned")
    cands["future_station"]      = np.where(uc_closer,
                                             cands["nearest_uc_name"],
                                             cands["nearest_planned_name"])
    cands["future_line"]         = np.where(uc_closer,
                                             cands["nearest_uc_line"],
                                             cands["nearest_planned_line"])
    # np.fmin (not np.minimum) — ignores NaN rather than propagating it, so a
    # candidate with no nearby planned station (or no planned stations in the
    # transit table at all) still gets a real dist_future_m from its UC station.
    cands["dist_future_m"]       = np.fmin(
        cands["dist_nearest_under_construction_m"],
        cands["dist_nearest_planned_m"],
    ).round(0).astype(int)
    cands["opening_date"]        = np.where(
        uc_closer,
        cands["nearest_uc_opening_date"].astype(str).str[:7],
        cands["nearest_planned_opening_date"].astype(str).str[:7],
    )

    # Gap vs operational-metro peers within same locality
    op_peer_map = {
        (row["locality"], row["bhk"]): _op_peer_median(df, row["locality"], row["bhk"])
        for _, row in cands[["locality", "bhk"]].drop_duplicates().iterrows()
    }
    cands["op_peer_median"] = cands.apply(
        lambda r: op_peer_map.get((r["locality"], r["bhk"])), axis=1
    )
    cands["vs_op_peer_pct"] = (
        (cands["monthly_rent"] - cands["op_peer_median"]) / cands["op_peer_median"] * 100
    ).round(1)

    # Composite rank: heavier weight on fundamental discount
    cands["arb_score"] = cands["residual_cm_pct"] * 0.6 + cands["vs_op_peer_pct"].fillna(0) * 0.4

    keep_cols = [
        "listing_id", "locality", "bhk", "carpet_area_sqft", "furnishing",
        "monthly_rent", "fundamental_fair_rent", "residual_cm_pct",
        "op_peer_median", "vs_op_peer_pct",
        "future_station", "future_line", "dist_future_m",
        "opening_date", "future_transit_type",
        "outside_interval", "arb_score",
    ]
    return (
        cands[keep_cols]
        .sort_values("arb_score")           # most underpriced / best arb first
        .reset_index(drop=True)
    )


# ── main entry ───────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, output_dir: Path) -> tuple:
    print("\n--- Cross-market fundamental model (no locality dummies) ---")
    cm_model = fit_cross_market(df)
    print(f"  R² = {cm_model.rsquared:.4f}  (without locality; lower is expected)")
    df = add_cross_market_residuals(df, cm_model)

    # ── locality mispricing table
    loc_tbl = locality_mispricing(df)
    print("\n  Locality mispricing (residual vs cross-market fundamentals):")
    print(
        loc_tbl[["n", "median_rent", "fair_rent_cm", "residual_pct",
                  "pct_overpriced", "signal"]]
        .rename(columns={
            "median_rent": "Median Rent",
            "fair_rent_cm": "Fair Rent (CM)",
            "residual_pct": "Residual%",
            "pct_overpriced": "% Overpriced",
        })
        .to_string()
    )

    # ── planted-signal check (ground-truth, requires _fair_rent_gt)
    print(f"\n{'='*60}")
    print("  PLANTED-SIGNAL RECOVERY CHECK  (SYNTHETIC DATA ONLY)")
    print(f"{'='*60}")
    if "_fair_rent_gt" in df.columns:
        print("  Method: actual / formula-fair-rent − 1  (isolates planted bias)")
        print("  (Cross-market residual mixes locality premium + planted signal;")
        print("   this GT check strips out the locality-level premium exactly.)\n")
        gt_checks = verify_planted_signal_gt(df)
        all_pass = True
        for loc, res in gt_checks.items():
            tag = "PASS" if res["pass"] else "FAIL"
            if not res["pass"]:
                all_pass = False
            print(
                f"  {loc:<15}  expected={res['expected_pct']:+.1f}%  "
                f"observed={res['observed_pct']:+.2f}%  [{tag}]"
            )
        print(f"  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    else:
        print("  [SKIP] _fair_rent_gt not available — only possible on synthetic data")

    # ── per-listing interval flags
    n_outside = df["outside_interval"].sum()
    pct_outside = n_outside / len(df) * 100
    print(f"\n  Listings outside 80% prediction interval: "
          f"{n_outside:,} / {len(df):,}  ({pct_outside:.1f}%)"
          f"  [target ~20%]")

    # ── arbitrage list
    arb = build_arbitrage_list(df)
    print(f"\n{'='*60}")
    print(f"  TRANSIT ARBITRAGE LIST  ({len(arb):,} candidates within 2.5 km of future transit)")
    print(f"{'='*60}")
    print("\n  Top 15 opportunities (ranked by composite arb score):\n")
    display = arb.head(15)[
        ["locality", "bhk", "monthly_rent", "fundamental_fair_rent",
         "residual_cm_pct", "vs_op_peer_pct",
         "future_station", "dist_future_m", "opening_date"]
    ].rename(columns={
        "monthly_rent":         "Rent",
        "fundamental_fair_rent":"Fair(CM)",
        "residual_cm_pct":      "Resid%",
        "vs_op_peer_pct":       "vs OpPeer%",
        "future_station":       "Future Station",
        "dist_future_m":        "Dist(m)",
        "opening_date":         "Opens",
    })
    print(display.to_string(index=True))

    # locality-level arbitrage summary
    arb_summary = (
        arb.groupby("locality")
        .agg(
            n_listings      = ("listing_id",       "count"),
            median_discount = ("residual_cm_pct",  "median"),
            nearest_station = ("future_station",   lambda x: x.mode()[0]),
            earliest_open   = ("opening_date",     "min"),
        )
        .sort_values("median_discount")
    )
    print("\n  Locality-level arbitrage summary:")
    print(arb_summary.to_string())

    # ── save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    arb.to_csv(output_dir / "arbitrage_list.csv", index=False)

    # Enrich the main scored parquet with cross-market residuals. Joined on
    # listing_id (not positional .values) — df may have fewer rows than
    # scored if ols_Xy dropped any with NaN in a required feature; those
    # listings correctly get NaN residuals here rather than a length-mismatch
    # crash or a silently misaligned value.
    scored_path = output_dir.parent / "data" / "processed" / "listings_scored.parquet"
    if scored_path.exists():
        scored = pd.read_parquet(scored_path)
        merge_cols = ["listing_id", "fundamental_fair_rent", "residual_cm", "residual_cm_pct"]
        scored = scored.merge(df[merge_cols], on="listing_id", how="left")
        scored.to_parquet(scored_path, index=False)

    print(f"\n  Arbitrage list saved  → outputs/arbitrage_list.csv  ({len(arb):,} rows)")
    print(f"  listings_scored.parquet updated with cross-market residuals")

    return cm_model, loc_tbl, arb


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    df = pd.read_parquet(root / "data" / "processed" / "listings_scored.parquet")
    run(df, root / "outputs")
