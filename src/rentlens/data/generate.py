"""
Synthetic Mumbai rental listing generator.

ALL DATA IS SYNTHETIC — generated programmatically to demonstrate the
RentLens hedonic model and transit-arbitrage methodology. No real
listings or personal data are included.

PLANTED MISPRICING (ground-truth, used to verify model recovery):
  Powai  → actual rents are +12% above hedonic fair value
  Mulund → actual rents are  -9% below hedonic fair value
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PLANTED_BIAS = {
    "Powai": 0.12,
    "Mulund": -0.09,
}

FURNISHING_MULTIPLIER = {
    "unfurnished": 1.00,
    "semi": 1.15,
    "furnished": 1.30,
}

FLOOR_PREMIUM_PER_FLOOR = 0.005   # +0.5 % per floor above ground
AGE_DISCOUNT_PER_YEAR = 0.008     # -0.8 % per year of building age
AMENITY_PREMIUM_PER_COUNT = 0.01  # +1 % per amenity
LOGNORMAL_SIGMA = 0.08            # idiosyncratic noise


def _load_city_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _jitter(center: float, scale: float, rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(center, scale, size=n)


def _deterministic_id(row_hash: str) -> str:
    return str(uuid.UUID(hashlib.md5(row_hash.encode()).hexdigest()))


def generate_listings(
    config_path: Path,
    n_total: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    cfg = _load_city_config(config_path)
    localities = cfg["localities"]
    rng = np.random.default_rng(seed)

    # Distribute listings proportionally; add a few extra to Powai/Bandra
    weights = np.array([1.3, 1.2, 1.1, 0.9, 1.0, 0.9, 0.8])
    weights = weights / weights.sum()
    counts = (weights * n_total).astype(int)
    counts[-1] += n_total - counts.sum()  # fix rounding

    frames: list[pd.DataFrame] = []

    for loc, n in zip(localities, counts):
        name = loc["name"]
        lat_c, lon_c = loc["lat"], loc["lon"]
        base_ppsf = loc["base_rent_per_sqft"]

        # Property attributes
        bhk = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.40, 0.28, 0.07])
        carpet = bhk * rng.uniform(350, 450, size=n) + rng.uniform(-60, 60, size=n)
        carpet = np.clip(carpet, 300, 2000).astype(int)

        bathrooms = np.clip(bhk - rng.integers(0, 2, size=n), 1, 4)
        floor = rng.integers(0, 18, size=n)
        total_floors = floor + rng.integers(0, 10, size=n)
        total_floors = np.maximum(total_floors, floor + 1)
        building_age = rng.integers(0, 30, size=n)
        amenities = rng.integers(0, 11, size=n)

        furnishing = rng.choice(
            ["unfurnished", "semi", "furnished"], size=n, p=[0.35, 0.40, 0.25]
        )
        property_type = rng.choice(["apartment", "independent"], size=n, p=[0.88, 0.12])

        furn_mult = np.array([FURNISHING_MULTIPLIER[f] for f in furnishing])
        floor_mult = 1 + floor * FLOOR_PREMIUM_PER_FLOOR
        age_mult = np.maximum(0.60, 1 - building_age * AGE_DISCOUNT_PER_YEAR)
        amenity_mult = 1 + amenities * AMENITY_PREMIUM_PER_COUNT

        # Fair rent (hedonic)
        fair_rent = (
            carpet * base_ppsf * furn_mult * floor_mult * age_mult * amenity_mult
        )

        # Lognormal idiosyncratic noise (mean-preserving)
        noise = rng.lognormal(mean=-0.5 * LOGNORMAL_SIGMA**2, sigma=LOGNORMAL_SIGMA, size=n)

        # Planted mispricing bias
        bias = 1 + PLANTED_BIAS.get(name, 0.0)

        monthly_rent = np.round(fair_rent * noise * bias / 100) * 100  # round to nearest ₹100

        # Deposit = 2–3 months rent
        deposit_months = rng.uniform(2, 3, size=n)
        deposit = np.round(monthly_rent * deposit_months / 1000) * 1000

        lats = _jitter(lat_c, 0.012, rng, n)
        lons = _jitter(lon_c, 0.012, rng, n)

        ts = datetime.now(tz=timezone.utc).isoformat()
        ids = [_deterministic_id(f"{name}{i}{seed}") for i in range(n)]

        df = pd.DataFrame(
            {
                "listing_id": ids,
                "source": "SYNTHETIC_GENERATED",
                "scrape_timestamp": ts,
                "locality": name,
                "latitude": lats.round(6),
                "longitude": lons.round(6),
                "carpet_area_sqft": carpet,
                "bhk": bhk,
                "bathrooms": bathrooms,
                "furnishing": furnishing,
                "floor": floor,
                "total_floors": total_floors,
                "building_age_years": building_age,
                "amenities_count": amenities,
                "property_type": property_type,
                "monthly_rent": monthly_rent.astype(int),
                "deposit": deposit.astype(int),
                # Ground-truth fair rent stored for verification (not used in modelling)
                "_fair_rent_gt": fair_rent.round(2),
            }
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def validate_schema(df: pd.DataFrame) -> None:
    required = {
        "listing_id", "source", "scrape_timestamp", "locality",
        "latitude", "longitude", "carpet_area_sqft", "bhk", "bathrooms",
        "furnishing", "floor", "total_floors", "building_age_years",
        "amenities_count", "property_type", "monthly_rent", "deposit",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")

    assert df["source"].eq("SYNTHETIC_GENERATED").all(), "source column mismatch"
    assert df["monthly_rent"].gt(0).all(), "non-positive rents found"
    assert df["latitude"].between(18.0, 20.0).all(), "latitude out of Mumbai range"
    assert df["longitude"].between(72.5, 73.5).all(), "longitude out of Mumbai range"
    assert df["furnishing"].isin(["unfurnished", "semi", "furnished"]).all()
    assert df["property_type"].isin(["apartment", "independent"]).all()
    assert df["bhk"].between(1, 5).all()
    assert df.duplicated("listing_id").sum() == 0, "duplicate listing_ids found"


def planted_signal_check(df: pd.DataFrame) -> dict[str, dict]:
    results = {}
    grand_median = df["monthly_rent"].median()
    for locality, expected_bias in PLANTED_BIAS.items():
        sub = df[df["locality"] == locality]
        fair = sub["_fair_rent_gt"]
        actual = sub["monthly_rent"]
        observed_pct = (actual / fair - 1).median()
        results[locality] = {
            "expected_bias_pct": expected_bias * 100,
            "observed_bias_pct": round(observed_pct * 100, 2),
            "pass": abs(observed_pct - expected_bias) < 0.04,  # within 4pp
        }
    return results


def run(config_path: Path, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_listings(config_path)
    validate_schema(df)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    cfg = root / "config" / "cities" / "mumbai.yaml"
    out = root / "data" / "processed" / "listings.parquet"
    df = run(cfg, out)

    print(f"\n{'='*60}")
    print(f"RENTLENS — Phase 1: Synthetic Data Generation")
    print(f"{'='*60}")
    print(f"Total listings : {len(df):,}")
    print(f"Output         : {out}")
    print(f"\nLocality rent summary (median monthly rent Rs.):")
    summary = (
        df.groupby("locality")
        .agg(
            n=("listing_id", "count"),
            median_rent=("monthly_rent", "median"),
            p25_rent=("monthly_rent", lambda x: x.quantile(0.25)),
            p75_rent=("monthly_rent", lambda x: x.quantile(0.75)),
            median_sqft=("carpet_area_sqft", "median"),
        )
        .sort_values("median_rent", ascending=False)
    )
    print(summary.to_string())

    print(f"\n{'='*60}")
    print("PLANTED-SIGNAL RECOVERY CHECK")
    print(f"{'='*60}")
    checks = planted_signal_check(df)
    for loc, res in checks.items():
        status = "PASS" if res["pass"] else "FAIL"
        print(
            f"  {loc:<15} expected={res['expected_bias_pct']:+.1f}%  "
            f"observed={res['observed_bias_pct']:+.2f}%  [{status}]"
        )
    print()
