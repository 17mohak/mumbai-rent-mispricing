"""
RentLens pipeline orchestrator.

Usage:
    python -m rentlens.pipeline --city mumbai [--phase 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _city_config(city: str) -> Path:
    p = ROOT / "config" / "cities" / f"{city}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No config found for city '{city}': {p}")
    return p


def phase1(city_config: Path) -> None:
    print("\n[Phase 1] Generating synthetic listings...")
    from rentlens.data.generate import run
    out = ROOT / "data" / "processed" / "listings.parquet"
    df = run(city_config, out)

    from rentlens.data.generate import planted_signal_check

    summary = (
        df.groupby("locality")
        .agg(
            n=("listing_id", "count"),
            median_rent=("monthly_rent", "median"),
        )
        .sort_values("median_rent", ascending=False)
    )
    print(f"\nListings written: {len(df):,}  →  {out}")
    print(summary.to_string())

    checks = planted_signal_check(df)
    print("\nPlanted-signal check:")
    for loc, r in checks.items():
        tag = "PASS" if r["pass"] else "FAIL"
        print(f"  {loc:<15} expected={r['expected_bias_pct']:+.1f}%  "
              f"observed={r['observed_bias_pct']:+.2f}%  [{tag}]")


def phase2(city_config: Path) -> None:
    import yaml
    with open(city_config) as fh:
        cfg = yaml.safe_load(fh)

    print("\n[Phase 2] Enriching listings with transit distances...")
    from rentlens.geo.transit import run

    listings_in  = ROOT / "data" / "processed" / "listings.parquet"
    transit_csv  = ROOT / cfg["transit_table"]
    listings_out = ROOT / "data" / "processed" / "listings_geo.parquet"
    df = run(listings_in, transit_csv, listings_out)

    summary = (
        df[["locality",
            "dist_nearest_operational_m",
            "dist_nearest_under_construction_m",
            "dist_nearest_planned_m"]]
        .groupby("locality").median().round(0)
        .sort_values("dist_nearest_operational_m")
    )
    print(f"\nListings enriched: {len(df):,}  ->  {listings_out}")
    print("\nMedian distance to nearest station by status (metres):")
    print(summary.to_string())


def phase3(city_config: Path) -> None:
    import joblib
    import pandas as pd

    print("\n[Phase 3] Training models + spatial validation...")
    print("=" * 65)
    print("RENTLENS — Phase 3: Models + Spatial Cross-Validation")
    print("=" * 65)

    df = pd.read_parquet(ROOT / "data" / "processed" / "listings_geo.parquet")

    from rentlens.model.hedonic import run as run_hedonic
    ols_model, ols_cv = run_hedonic(df)

    from rentlens.model.gbm import run as run_gbm
    output_dir = ROOT / "outputs"
    lgbm_model, lgbm_cv, shap_importance = run_gbm(df, output_dir)

    from rentlens.model.uncertainty import run as run_uncertainty
    q10, q50, q90, intervals = run_uncertainty(df)

    # Persist model artefacts
    models_dir = ROOT / "models" / "rentlens"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ols_model,   models_dir / "hedonic_ols.pkl")
    joblib.dump(lgbm_model,  models_dir / "lgbm_regressor.pkl")
    joblib.dump(q10,         models_dir / "lgbm_q10.pkl")
    joblib.dump(q50,         models_dir / "lgbm_q50.pkl")
    joblib.dump(q90,         models_dir / "lgbm_q90.pkl")

    # Merge predictions onto geo listings and save
    geo = pd.read_parquet(ROOT / "data" / "processed" / "listings_geo.parquet")
    scored = geo.merge(
        intervals[["listing_id", "fair_rent_pred", "interval_lower",
                    "interval_upper", "outside_interval"]],
        on="listing_id", how="left"
    )
    scored.to_parquet(ROOT / "data" / "processed" / "listings_scored.parquet", index=False)
    print(f"\n  Scored listings saved → data/processed/listings_scored.parquet")
    print(f"  Model artefacts saved → {models_dir}")


def phase4(city_config: Path) -> None:
    import pandas as pd
    print("\n[Phase 4] Mispricing detection + transit arbitrage...")
    print("=" * 65)
    print("RENTLENS — Phase 4: Mispricing + Transit Arbitrage")
    print("=" * 65)

    df = pd.read_parquet(ROOT / "data" / "processed" / "listings_scored.parquet")
    from rentlens.model.mispricing import run
    run(df, ROOT / "outputs")


def phase5(city_config: Path) -> None:
    import pandas as pd
    print("\n[Phase 5] Causal analysis — Difference-in-Differences...")
    print("=" * 65)
    print("RENTLENS — Phase 5: Causal DiD")
    print("=" * 65)
    df = pd.read_parquet(ROOT / "data" / "processed" / "listings_scored.parquet")
    from rentlens.causal.diff_in_diff import run
    run(df, ROOT / "outputs")


def phase6(city_config: Path) -> None:
    import yaml
    with open(city_config) as fh:
        cfg = yaml.safe_load(fh)

    print("\n[Phase 6] Building interactive map + README...")
    from rentlens.viz.map import run as run_map
    out = run_map(
        scored_path  = ROOT / "data"    / "processed" / "listings_scored.parquet",
        transit_path = ROOT / cfg["transit_table"],
        config_path  = city_config,
        output_path  = ROOT / "outputs" / "rentlens_mumbai_map.html",
    )
    print(f"  Map saved -> {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rentlens.pipeline")
    parser.add_argument("--city", default="mumbai")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only this phase (1-6). Omit to run all.")
    parser.add_argument(
        "--source", choices=["synthetic", "real"], default="synthetic",
        help="'synthetic' (default) regenerates data/processed/listings.parquet from "
             "rentlens.data.generate (Phase 1). 'real' skips Phase 1 entirely and starts "
             "from the already-scraped+cleaned listings.parquet (refresh it first via "
             "`python -m rentlens.scrape.run` then `python -m rentlens.data.clean`).",
    )
    args = parser.parse_args(argv)

    cfg = _city_config(args.city)
    phases_to_run = [args.phase] if args.phase else list(range(1, 7))

    if args.source == "real" and 1 in phases_to_run:
        print("[Phase 1] Skipped — --source real uses the existing data/processed/listings.parquet "
              "(generate.py would overwrite it with synthetic data).")
        phases_to_run = [p for p in phases_to_run if p != 1]

    if args.source == "real" and 5 in phases_to_run:
        print("[Phase 5] Skipped — causal/diff_in_diff.py fabricates its own before/after panel "
              "(plants a hardcoded effect into synthetic time periods); it cannot produce a real "
              "estimate from a single cross-sectional scrape. Run it explicitly with --phase 5 "
              "--source synthetic if you want the methodology demo.")
        phases_to_run = [p for p in phases_to_run if p != 5]

    dispatch = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6}

    for ph in phases_to_run:
        fn = dispatch.get(ph)
        if fn is None:
            print(f"[Phase {ph}] Not yet implemented — skipping.")
            continue
        fn(cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
