"""Regression tests for the real-data cleaning pipeline (rentlens.data.clean).

These tests pin the *current* behavior of clean_listings and its helpers so that
future refactors cannot silently change cleaning outcomes. They do not assert any
new or "desired" behavior — only what the code does today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rentlens.data.clean import (
    AGE_BUCKET_MIDPOINTS,
    CARPET_TO_SUPER_RATIO,
    DEDUP_KEY,
    LOCALITY_CENTROIDS,
    MAX_PLAUSIBLE_RENT_PSF,
    MIN_PLAUSIBLE_RENT,
    _building_age_years,
    _haversine_m,
    _parse_floor,
    clean_listings,
)

POWAI = LOCALITY_CENTROIDS["Powai"]
MULUND = LOCALITY_CENTROIDS["Mulund"]


def _base_row(**overrides) -> dict:
    """A single listing that survives every cleaning step unchanged."""
    row = {
        "listing_id": "MB1",
        "source": "MAGICBRICKS",
        "scrape_timestamp": "2026-06-23T00:00:00+00:00",
        "raw_locality": "Powai",
        "detail_url": "https://example.test/1",
        "title": "2 BHK Flat for Rent in Powai",
        "bhk": 2.0,
        "area_type_raw": "carpet",
        "area_sqft_raw": 600.0,
        "carpet_area_sqft": 600.0,
        "floor_raw": "5 out of 10",
        "furnishing": "semi",
        "latitude": POWAI[0],
        "longitude": POWAI[1],
        "bathrooms": 2.0,
        "age_status_raw": "Immediately",
        "property_type": "apartment",
        "monthly_rent": 50_000.0,
    }
    row.update(overrides)
    return row


def _clean(rows: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    return clean_listings(pd.DataFrame(rows))


def _step(log: list[dict], name: str) -> dict:
    return next(e for e in log if e["step"] == name)


# ── helper: _haversine_m ──────────────────────────────────────────────────────

def test_haversine_zero_for_identical_points():
    d = _haversine_m(np.array([19.1]), np.array([72.9]), 19.1, 72.9)
    assert float(d[0]) == pytest.approx(0.0, abs=1e-6)


def test_haversine_one_degree_lat_about_111km():
    d = _haversine_m(np.array([0.0]), np.array([0.0]), 1.0, 0.0)
    assert 110_000 < float(d[0]) < 112_000


# ── helper: _parse_floor ──────────────────────────────────────────────────────

def test_parse_floor_out_of():
    assert _parse_floor("5 out of 10") == (5.0, 10.0)


def test_parse_floor_ground():
    assert _parse_floor("Ground out of 8") == (0.0, 8.0)


def test_parse_floor_basement():
    assert _parse_floor("Lower Basement out of 3") == (-1.0, 3.0)


def test_parse_floor_bare_number():
    assert _parse_floor("7") == (7.0, None)


def test_parse_floor_unparseable_is_none():
    assert _parse_floor("Penthouse") == (None, None)
    assert _parse_floor(None) == (None, None)


# ── helper: _building_age_years ───────────────────────────────────────────────

def test_building_age_known_bucket():
    assert _building_age_years("Const. Age 5 to 10 years") == 7.5
    assert _building_age_years("Const. Age 5 to 10 years") == AGE_BUCKET_MIDPOINTS["Const. Age 5 to 10 years"]


def test_building_age_possession_date_is_zero():
    assert _building_age_years("From Jul '26") == 0.0


def test_building_age_availability_is_none():
    assert _building_age_years("Immediately") is None
    assert _building_age_years(None) is None


# ── clean_listings: row-level rules ───────────────────────────────────────────

def test_drop_missing_rent():
    cleaned, log = _clean([_base_row(), _base_row(listing_id="MB2", monthly_rent=np.nan)])
    assert _step(log, "drop_missing_rent")["rows_dropped"] == 1
    assert len(cleaned) == 1


def test_studio_resolves_to_1_bhk():
    cleaned, _ = _clean([
        _base_row(listing_id="S1", bhk=np.nan,
                  title="Studio Apartment for Rent in Powai"),
    ])
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["bhk"] == 1.0


def test_missing_bhk_non_studio_dropped():
    cleaned, log = _clean([
        _base_row(listing_id="N1", bhk=np.nan, title="Flat for Rent in Powai"),
    ])
    assert len(cleaned) == 0
    assert _step(log, "resolve_bhk")["rows_dropped"] == 1


def test_super_area_estimated_via_loading_factor():
    cleaned, _ = _clean([
        _base_row(listing_id="SA1", area_type_raw="super_area_fallback",
                  area_sqft_raw=1000.0, carpet_area_sqft=np.nan),
    ])
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["carpet_area_sqft"] == pytest.approx(1000.0 * CARPET_TO_SUPER_RATIO)
    assert bool(cleaned.iloc[0]["carpet_area_is_estimated"]) is True


def test_carpet_row_not_flagged_estimated():
    cleaned, _ = _clean([_base_row()])
    assert bool(cleaned.iloc[0]["carpet_area_is_estimated"]) is False


def test_no_area_dropped():
    cleaned, _ = _clean([
        _base_row(listing_id="NA1", area_type_raw=None,
                  area_sqft_raw=np.nan, carpet_area_sqft=np.nan),
    ])
    assert len(cleaned) == 0


def test_rent_per_sqft_outlier_dropped():
    # 40,000,000 / 950 sqft ~ 42,105 per sqft -> well above the psf ceiling
    cleaned, log = _clean([
        _base_row(listing_id="O1", monthly_rent=40_000_000.0,
                  area_sqft_raw=950.0, carpet_area_sqft=950.0),
    ])
    assert len(cleaned) == 0
    assert _step(log, "rent_outliers")["rows_dropped"] == 1


def test_low_rent_dropped():
    cleaned, _ = _clean([_base_row(listing_id="L1", monthly_rent=10_000.0)])
    assert MIN_PLAUSIBLE_RENT == 15_000
    assert len(cleaned) == 0


def test_far_from_all_centroids_dropped():
    cleaned, log = _clean([_base_row(listing_id="F1", latitude=19.30, longitude=73.20)])
    assert len(cleaned) == 0
    assert _step(log, "locality_bucketing")["rows_dropped"] == 1


def test_locality_assigned_to_nearest_centroid():
    cleaned, _ = _clean([
        _base_row(listing_id="P1"),  # at Powai centroid
        _base_row(listing_id="M1", latitude=MULUND[0], longitude=MULUND[1]),
    ])
    by_id = cleaned.set_index("listing_id")["locality"]
    assert by_id["P1"] == "Powai"
    assert by_id["M1"] == "Mulund"


def test_dedupe_exact_relistings():
    cleaned, log = _clean([_base_row(listing_id="D1"), _base_row(listing_id="D2")])
    assert DEDUP_KEY == ["latitude", "longitude", "bhk", "bathrooms",
                          "carpet_area_sqft", "monthly_rent"]
    assert len(cleaned) == 1
    assert _step(log, "dedupe_relistings")["rows_dropped"] == 1


def test_building_age_left_nan_for_availability_text():
    cleaned, _ = _clean([_base_row(age_status_raw="Immediately")])
    assert pd.isna(cleaned.iloc[0]["building_age_years"])


def test_building_age_mapped_for_bucket_text():
    cleaned, _ = _clean([_base_row(age_status_raw="Const. Age 5 to 10 years")])
    assert cleaned.iloc[0]["building_age_years"] == 7.5


def test_deposit_and_amenities_left_nan():
    cleaned, _ = _clean([_base_row()])
    assert pd.isna(cleaned.iloc[0]["deposit"])
    assert pd.isna(cleaned.iloc[0]["amenities_count"])


def test_max_plausible_psf_constant_unchanged():
    # Guards the documented cleaning threshold against accidental drift.
    assert MAX_PLAUSIBLE_RENT_PSF == 1_000
