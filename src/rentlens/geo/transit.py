"""
Transit enrichment: haversine distances from each listing to the
nearest station, segmented by operational status.

Adds to listings DataFrame:
  dist_nearest_operational_m      – metres to closest open station
  nearest_operational_name        – station name
  nearest_operational_line        – line name
  dist_nearest_under_construction_m
  nearest_uc_name
  nearest_uc_line
  nearest_uc_opening_date
  dist_nearest_planned_m
  nearest_planned_name
  nearest_planned_line
  nearest_planned_opening_date
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Vectorised haversine distance in metres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _nearest_by_status(
    listings: pd.DataFrame,
    stations: pd.DataFrame,
    status: str,
) -> pd.DataFrame:
    sub = stations[stations["status"] == status].reset_index(drop=True)
    if sub.empty:
        prefix = {"operational": "operational", "under_construction": "uc", "planned": "planned"}[status]
        listings[f"dist_nearest_{status}_m"] = np.nan
        listings[f"nearest_{prefix}_name"] = None
        listings[f"nearest_{prefix}_line"] = None
        if status != "operational":
            listings[f"nearest_{prefix}_opening_date"] = None
        return listings

    # Shape: (n_listings, n_stations)
    lat_l = listings["latitude"].values[:, None]
    lon_l = listings["longitude"].values[:, None]
    lat_s = sub["latitude"].values[None, :]
    lon_s = sub["longitude"].values[None, :]

    dist_matrix = haversine_m(lat_l, lon_l, lat_s, lon_s)  # (n, k)
    nearest_idx = dist_matrix.argmin(axis=1)
    nearest_dist = dist_matrix[np.arange(len(listings)), nearest_idx]

    prefix = {"operational": "operational", "under_construction": "uc", "planned": "planned"}[status]
    listings[f"dist_nearest_{status}_m"] = nearest_dist.round(1)
    listings[f"nearest_{prefix}_name"] = sub["station_name"].values[nearest_idx]
    listings[f"nearest_{prefix}_line"] = sub["line"].values[nearest_idx]
    if status != "operational":
        listings[f"nearest_{prefix}_opening_date"] = sub["opening_date"].values[nearest_idx]

    return listings


def enrich(listings: pd.DataFrame, transit_path: Path) -> pd.DataFrame:
    stations = pd.read_csv(transit_path, parse_dates=["opening_date"])
    df = listings.copy()

    for status in ("operational", "under_construction", "planned"):
        df = _nearest_by_status(df, stations, status)

    return df


def run(listings_path: Path, transit_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(listings_path)
    df = enrich(df, transit_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    listings_in  = root / "data" / "processed" / "listings.parquet"
    transit_csv  = root / "data" / "reference"  / "transit_mumbai.csv"
    listings_out = root / "data" / "processed"  / "listings_geo.parquet"

    df = run(listings_in, transit_csv, listings_out)

    print(f"\n{'='*65}")
    print("RENTLENS — Phase 2: Transit Enrichment")
    print(f"{'='*65}")
    print(f"Listings enriched : {len(df):,}")
    print(f"Output            : {listings_out}\n")

    summary_cols = [
        "locality",
        "dist_nearest_operational_m",
        "dist_nearest_under_construction_m",
        "dist_nearest_planned_m",
    ]
    locality_summary = (
        df[summary_cols]
        .groupby("locality")
        .median()
        .sort_values("dist_nearest_operational_m")
        .round(0)
    )
    print("Median distance to nearest station by status (metres):")
    print(locality_summary.to_string())

    print("\nSample UC stations near listings (top 5 closest overall):")
    sample = (
        df[["locality", "nearest_uc_name", "nearest_uc_line",
            "nearest_uc_opening_date", "dist_nearest_under_construction_m"]]
        .sort_values("dist_nearest_under_construction_m")
        .drop_duplicates(subset=["nearest_uc_name"])
        .head(5)
    )
    print(sample.to_string(index=False))
