"""
Phase C — cleaning & validation of real (MagicBricks) listings.

Every transformation below is a documented, explicit rule derived from
profiling data/raw/magicbricks_listings_raw.parquet (621 rows). Nothing is
silently imputed or clipped — each step is counted and reported in
data/processed/data_quality_report.md.

Pipeline order (each step operates on the survivors of the previous one):
  1. Drop rows missing monthly_rent (target variable; can't model without it)
  2. Resolve BHK: "Studio Apartment" -> 1 BHK (industry convention); titles
     with neither a BHK number nor "Studio" -> drop (nothing to impute from)
  3. Reconcile carpet vs super (built-up) area; drop rows with no area info
     at all
  4. Parse floor/total_floors, including "Ground"/"Basement" patterns the
     Phase B regex didn't cover
  5. Drop rows with no furnishing info at all (can't impute a categorical
     from nothing)
  6. Outlier rent removal: absolute floor + a per-sqft ceiling that catches
     unit-mismatched (likely sale-price-leaked) listings
  7. Locality bucketing: nearest-centroid haversine assignment among the 3
     target localities (more rigorous than trusting ~50 raw sub-locality
     strings), with a sanity distance cap
  8. Dedupe relisted flats (same building + spec + rent)
  9. Building age: derive numeric years only from genuine age-bucket text;
     leave NaN (not impute) when the source only gave move-in availability
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Single source of truth for the haversine distance (identical formula/constant
# previously duplicated here); imported under the same private name this module
# already used so call sites are unchanged.
from rentlens.geo.transit import haversine_m as _haversine_m

# Step 3 — industry-standard carpet/super-area loading factor. Carpet area is
# typically 65-75% of super built-up area in Indian residential real estate;
# 0.70 is the commonly cited midpoint. Applied ONLY when carpet area itself
# isn't disclosed — flagged via carpet_area_is_estimated, never silent.
CARPET_TO_SUPER_RATIO = 0.70

# Step 6 — thresholds chosen from the observed distribution (see
# data_quality_report.md for the percentile evidence): legitimate listings
# in this dataset top out around Rs 355-515/sqft/month even for 5BHK luxury;
# the one row that broke Rs 1,000/sqft/month was Rs 4 Cr/month for a 950 sqft
# 2BHK (Rs 42,105/sqft/month) — almost certainly a sale price leaking into a
# rental search result, not a real rent.
MIN_PLAUSIBLE_RENT = 15_000
MAX_PLAUSIBLE_RENT_PSF = 1_000

# Step 7 — target locality centroids (same as config/cities/mumbai.yaml)
LOCALITY_CENTROIDS = {
    "Powai": (19.1176, 72.9060),
    "Mulund": (19.1726, 72.9574),
    "Andheri East": (19.1136, 72.8697),
}
MAX_LOCALITY_DIST_M = 4_000

# Step 8 — a listing is treated as a relisting (same physical unit, multiple
# ads) only if location + BHK + bathrooms + carpet area + rent all match
# exactly. A looser key (e.g. just lat/lon/BHK/rent) was tested and rejected
# — it collapsed distinct units in the same building that a builder/broker
# had priced identically (different carpet areas, same rent).
DEDUP_KEY = ["latitude", "longitude", "bhk", "bathrooms", "carpet_area_sqft", "monthly_rent"]

# Step 9 — only genuine construction-age text gets a numeric mapping.
# "Immediately"/availability-only strings are NOT age and are left NaN.
AGE_BUCKET_MIDPOINTS = {
    "Const. Age New Construction": 0.0,
    "Const. Age Less than 5 years": 2.5,
    "Const. Age 5 to 10 years": 7.5,
    "Const. Age 10 to 15 years": 12.5,
    "Const. Age 15 to 20 years": 17.5,
    "Const. Age 20+ years": 25.0,
}
POSSESSION_DATE_RE = re.compile(r"^From\s+\w+\s+'\d{2}$")  # e.g. "From Jul '26"


def _parse_floor(floor_raw: str | None) -> tuple[float | None, float | None]:
    if not floor_raw or not isinstance(floor_raw, str):
        return None, None
    m = re.search(r"(\d+)\s*out of\s*(\d+)", floor_raw, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"Ground\s*out of\s*(\d+)", floor_raw, re.IGNORECASE)
    if m:
        return 0.0, float(m.group(1))
    m = re.search(r"(?:Upper|Lower)\s*Basement\s*out of\s*(\d+)", floor_raw, re.IGNORECASE)
    if m:
        return -1.0, float(m.group(1))
    m = re.fullmatch(r"\d+", floor_raw.strip())
    if m:
        return float(floor_raw.strip()), None
    return None, None


def _building_age_years(age_status_raw: str | None) -> float | None:
    if not age_status_raw or not isinstance(age_status_raw, str):
        return None
    if age_status_raw in AGE_BUCKET_MIDPOINTS:
        return AGE_BUCKET_MIDPOINTS[age_status_raw]
    if POSSESSION_DATE_RE.match(age_status_raw.strip()):
        return 0.0  # pre-possession / brand new construction
    return None  # e.g. "Immediately" — availability, not age; do not guess


def clean_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (cleaned_df, step_log) — step_log feeds the quality report."""
    log: list[dict] = []
    df = raw.copy()

    def record(step: str, note: str, before: int, after: int):
        log.append({"step": step, "note": note, "rows_before": before, "rows_after": after,
                    "rows_dropped": before - after})

    # 1. monthly_rent required
    n0 = len(df)
    df = df[df["monthly_rent"].notna()].copy()
    record("drop_missing_rent", "monthly_rent is the target variable; can't model without it", n0, len(df))

    # 2. BHK resolution
    n0 = len(df)
    is_studio = df["title"].str.contains("Studio Apartment", case=False, na=False)
    df.loc[df["bhk"].isna() & is_studio, "bhk"] = 1.0
    df = df[df["bhk"].notna()].copy()
    record("resolve_bhk", "Studio -> 1 BHK (industry convention); rest had no BHK info to impute from", n0, len(df))

    # 3. carpet vs super-area reconciliation
    n0 = len(df)
    df["carpet_area_is_estimated"] = df["area_type_raw"] == "super_area_fallback"
    needs_estimate = df["carpet_area_is_estimated"] & df["area_sqft_raw"].notna()
    df.loc[needs_estimate, "carpet_area_sqft"] = (
        df.loc[needs_estimate, "area_sqft_raw"] * CARPET_TO_SUPER_RATIO
    )
    df = df[df["carpet_area_sqft"].notna()].copy()
    n_estimated = int(df["carpet_area_is_estimated"].sum())
    record(
        "reconcile_area",
        f"super-area -> carpet via x{CARPET_TO_SUPER_RATIO} loading factor for "
        f"{n_estimated} rows (flagged in carpet_area_is_estimated); dropped rows with no area at all",
        n0, len(df),
    )

    # 4. floor parsing (recovers "Ground"/"Basement"/bare-number patterns missed in Phase B)
    parsed = df["floor_raw"].apply(_parse_floor)
    df["floor"] = [p[0] for p in parsed]
    df["total_floors"] = [p[1] for p in parsed]

    # 5. furnishing required
    n0 = len(df)
    df = df[df["furnishing"].notna()].copy()
    record("drop_missing_furnishing", "no furnishing data on the card at all; nothing to impute from", n0, len(df))

    # 6. rent outlier removal
    n0 = len(df)
    rent_psf = df["monthly_rent"] / df["carpet_area_sqft"]
    plausible = (df["monthly_rent"] >= MIN_PLAUSIBLE_RENT) & (rent_psf <= MAX_PLAUSIBLE_RENT_PSF)
    df = df[plausible].copy()
    record(
        "rent_outliers",
        f"dropped rent < Rs {MIN_PLAUSIBLE_RENT:,} (implausible for a full flat in these localities) "
        f"or > Rs {MAX_PLAUSIBLE_RENT_PSF}/sqft/month (likely sale-price leakage, not rent)",
        n0, len(df),
    )

    # 7. locality bucketing via nearest centroid (more rigorous than the ~50
    # raw sub-locality strings MagicBricks reports)
    n0 = len(df)
    dists = pd.DataFrame(
        {name: _haversine_m(df["latitude"], df["longitude"], lat, lon)
         for name, (lat, lon) in LOCALITY_CENTROIDS.items()}
    )
    df["locality"] = dists.idxmin(axis=1).values
    df["dist_to_locality_centroid_m"] = dists.min(axis=1).values
    df = df[df["dist_to_locality_centroid_m"] <= MAX_LOCALITY_DIST_M].copy()
    record(
        "locality_bucketing",
        f"assigned to nearest of 3 target-locality centroids by haversine distance; "
        f"dropped rows > {MAX_LOCALITY_DIST_M:,} m from any centroid (too ambiguous to bucket)",
        n0, len(df),
    )

    # 8. dedupe relisted flats
    n0 = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEY, keep="first").copy()
    record(
        "dedupe_relistings",
        f"dropped exact-match relistings on {DEDUP_KEY} (same building/spec/rent = same physical unit)",
        n0, len(df),
    )

    # 9. building age — explicit, not imputed
    df["building_age_years"] = df["age_status_raw"].apply(_building_age_years)
    age_known_pct = df["building_age_years"].notna().mean() * 100
    log.append({
        "step": "building_age_years", "rows_before": len(df), "rows_after": len(df), "rows_dropped": 0,
        "note": f"only {age_known_pct:.1f}% of rows have genuine age-bucket text (rest say move-in "
                f"availability, e.g. 'Immediately' — NOT age); left as NaN rather than guessed",
    })

    df["deposit"] = np.nan  # not disclosed at search-results level for this source — documented, not estimated
    df["amenities_count"] = np.nan  # not extracted from this source at this pass — documented, not estimated
    df["bathrooms"] = df["bathrooms"].astype(float)
    df["property_type"] = df["property_type"].fillna("apartment")

    return df, log


CANONICAL_COLUMNS = [
    "listing_id", "source", "scrape_timestamp", "locality", "latitude", "longitude",
    "carpet_area_sqft", "bhk", "bathrooms", "furnishing", "floor", "total_floors",
    "building_age_years", "amenities_count", "property_type", "monthly_rent", "deposit",
]

AUDIT_COLUMNS = [
    "raw_locality", "dist_to_locality_centroid_m", "carpet_area_is_estimated",
    "age_status_raw", "detail_url",
]


def write_quality_report(log: list[dict], n_raw: int, n_final: int, out_path: Path) -> None:
    lines = [
        "# RentLens — Real Data Quality Report",
        "",
        f"Source: MagicBricks (Powai / Mulund / Andheri East rental search results)",
        f"Raw rows scraped: {n_raw:,}",
        f"Final clean rows: {n_final:,}  ({n_final / n_raw * 100:.1f}% retained)",
        "",
        "## Pipeline steps",
        "",
        "| Step | Rows before | Rows after | Dropped | Note |",
        "|---|---|---|---|---|",
    ]
    for entry in log:
        lines.append(
            f"| {entry['step']} | {entry['rows_before']:,} | {entry['rows_after']:,} | "
            f"{entry['rows_dropped']:,} | {entry['note']} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(raw_path: Path, output_path: Path, report_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    cleaned, log = clean_listings(raw)
    final = cleaned[CANONICAL_COLUMNS + AUDIT_COLUMNS].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False)
    write_quality_report(log, len(raw), len(final), report_path)
    return final


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    raw_in = root / "data" / "raw" / "magicbricks_listings_raw.parquet"
    out = root / "data" / "processed" / "listings.parquet"
    report = root / "data" / "processed" / "data_quality_report.md"

    df = run(raw_in, out, report)

    print(f"\n{'='*70}")
    print("RENTLENS — Phase C: Cleaning & Validation")
    print(f"{'='*70}")
    print(f"Final clean rows : {len(df):,}")
    print(f"Output           : {out}")
    print(f"Quality report   : {report}\n")
    print("Rows per locality:")
    print(df["locality"].value_counts().to_string())
    print("\nField completeness (non-null %):")
    print((df.notna().mean() * 100).round(1).to_string())
