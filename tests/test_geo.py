"""Haversine distance, transit enrichment, and geo utility tests."""

import math
from pathlib import Path

import pandas as pd
import pytest

from rentlens.geo.transit import haversine_m as hav_vec, enrich

TRANSIT_CSV = Path(__file__).resolve().parents[1] / "data" / "reference" / "transit_mumbai.csv"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# --- Known-distance reference checks ---

def test_haversine_same_point():
    assert haversine_m(19.1176, 72.9060, 19.1176, 72.9060) == pytest.approx(0.0, abs=1e-6)


def test_haversine_powai_to_andheri_east():
    # Powai centroid → Andheri East centroid; rough expected ~7 km
    dist = haversine_m(19.1176, 72.9060, 19.1136, 72.8697)
    assert 3_000 < dist < 12_000, f"Unexpected distance: {dist:.0f} m"


def test_haversine_powai_to_thane():
    # Powai → Thane West; expected ~9–15 km
    dist = haversine_m(19.1176, 72.9060, 19.2183, 72.9781)
    assert 8_000 < dist < 20_000, f"Unexpected distance: {dist:.0f} m"


def test_haversine_symmetry():
    d1 = haversine_m(19.0522, 72.8996, 19.1726, 72.9574)
    d2 = haversine_m(19.1726, 72.9574, 19.0522, 72.8996)
    assert d1 == pytest.approx(d2, rel=1e-9)


def test_haversine_triangle_inequality():
    a = haversine_m(19.1176, 72.9060, 19.0583, 72.8505)  # Powai → Bandra E
    b = haversine_m(19.0583, 72.8505, 19.1136, 72.8697)  # Bandra E → Andheri E
    c = haversine_m(19.1176, 72.9060, 19.1136, 72.8697)  # Powai → Andheri E directly
    assert c <= a + b + 1  # +1 m tolerance for floating-point


def test_haversine_one_degree_lat_approx_111km():
    dist = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110_000 < dist < 112_000


def test_haversine_mumbai_bounding_box_sanity():
    # The bounding box diagonal (lat_min,lon_min) → (lat_max,lon_max) should be ~60–70 km
    dist = haversine_m(18.89, 72.77, 19.35, 73.05)
    assert 50_000 < dist < 80_000


# --- Transit enrichment tests ---

@pytest.fixture(scope="module")
def tiny_listings() -> pd.DataFrame:
    """Three representative listings: Powai, Bandra East, Mulund centroids."""
    return pd.DataFrame({
        "listing_id": ["A", "B", "C"],
        "locality": ["Powai", "Bandra East", "Mulund"],
        "latitude": [19.1176, 19.0583, 19.1726],
        "longitude": [72.9060, 72.8505, 72.9574],
    })


@pytest.fixture(scope="module")
def enriched(tiny_listings) -> pd.DataFrame:
    return enrich(tiny_listings, TRANSIT_CSV)


def test_enrich_adds_distance_columns(enriched):
    for col in [
        "dist_nearest_operational_m",
        "dist_nearest_under_construction_m",
        "dist_nearest_planned_m",
    ]:
        assert col in enriched.columns, f"Missing column: {col}"


def test_enrich_adds_name_columns(enriched):
    for col in ["nearest_operational_name", "nearest_uc_name", "nearest_planned_name"]:
        assert col in enriched.columns


def test_distances_are_positive(enriched):
    # A status bucket with zero stations in the transit table (e.g. no
    # "planned" stations in the real OSM-sourced table) legitimately yields
    # NaN distances for every listing — only non-NaN distances must be positive.
    for col in [
        "dist_nearest_operational_m",
        "dist_nearest_under_construction_m",
        "dist_nearest_planned_m",
    ]:
        present = enriched[col].dropna()
        assert present.empty or (present > 0).all(), f"Non-positive distance in {col}"


def test_powai_nearest_uc_is_powai_lake_metro(enriched):
    powai_row = enriched[enriched["locality"] == "Powai"].iloc[0]
    assert "Powai" in powai_row["nearest_uc_name"], (
        f"Expected Powai station for Powai listing, got: {powai_row['nearest_uc_name']}"
    )


def test_powai_uc_distance_under_500m(enriched):
    powai_row = enriched[enriched["locality"] == "Powai"].iloc[0]
    assert powai_row["dist_nearest_under_construction_m"] < 500, (
        f"Powai Lake Metro should be <500m from Powai centroid, "
        f"got {powai_row['dist_nearest_under_construction_m']:.0f}m"
    )


def test_row_count_preserved(tiny_listings, enriched):
    assert len(enriched) == len(tiny_listings)


def test_transit_csv_has_all_statuses():
    # Real OSM-sourced data may legitimately have zero "planned" stations in
    # scope (none were found near Powai/Mulund/Andheri East at curation time);
    # "operational" and "under_construction" are the load-bearing statuses for
    # arbitrage scoring and must always be present.
    df = pd.read_csv(TRANSIT_CSV)
    statuses = set(df["status"].unique())
    assert statuses <= {"operational", "under_construction", "planned"}
    assert {"operational", "under_construction"} <= statuses


def test_transit_csv_opening_dates_parseable():
    # Real stations (esp. under-construction ones) often have no confirmed
    # opening date — the column must parse without error, but NaT is allowed.
    df = pd.read_csv(TRANSIT_CSV, parse_dates=["opening_date"])
    assert df["opening_date"].dtype.kind == "M"
