"""
MagicBricks adapter — Phase B real-data source.

Picked over 99acres / Housing.com (blocked the bare robots.txt request —
403 / 406, clear bot-detection at the edge) and NoBroker / SquareYards
(robots.txt explicitly disallows their listing/search paths:
`/property/listing/` and `/rental/search*` respectively). MagicBricks'
robots.txt disallows admin/profile/image/map sub-paths but not the rental
search pages (`/flats-for-rent-in-<locality>-mumbai-pppfr`) or the general
property-detail pages used here.

Each search-results page is server-rendered (no JS execution needed) and
embeds two independent, parseable sources of truth per listing:
  1. A JSON-LD array of Apartment objects (precise lat/lon, BHK, locality)
     in page/DOM order.
  2. HTML cards (div#cardid<N>) with labelled data-summary fields
     (carpet-area / super-area, status, floor, furnishing, facing, parking,
     bathroom, balcony) and a rupee-formatted price block.

The JSON-LD array is matched to cards positionally (both are emitted in the
same server-side render order) — if the counts ever disagree, geo fields are
left null for that page rather than risking a mis-assigned lat/lon.

Fields NOT reliably available at search-results level (deposit, exact
building age, full amenity list) are intentionally left as raw/None here —
Phase C (cleaning) makes deliberate, documented decisions about them rather
than this module guessing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from rentlens.scrape.base import ScraperAdapter

FURNISHING_MAP = {
    "unfurnished": "unfurnished",
    "semi-furnished": "semi",
    "furnished": "furnished",
}

# Vocabulary observed in MagicBricks card titles (e.g. "4 BHK Flat for Rent
# in Powai, Mumbai" / "3 BHK Independent House for Rent in ...") — checked in
# this order, first match wins.
PROPERTY_TYPE_TITLE_TOKENS = [
    ("Independent House", "independent"),
    ("Villa", "independent"),
    ("Builder Floor", "independent"),
    ("Studio Apartment", "apartment"),
    ("Penthouse", "apartment"),
    ("Flat", "apartment"),
    ("Apartment", "apartment"),
]


def _rupee_to_number(text: str) -> float | None:
    """'₹3.5 Lac' -> 350000.0, '₹45,000' -> 45000.0, '₹1.2 Cr' -> 12000000.0."""
    if not text:
        return None
    t = text.replace("₹", "").replace(",", "").strip()
    m = re.search(r"([\d.]+)\s*(Lac|Lakh|Cr|Crore)?", t, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("lac", "lakh"):
        val *= 100_000
    elif unit in ("cr", "crore"):
        val *= 10_000_000
    return val


def _floor_to_pair(text: str) -> tuple[float | None, float | None]:
    m = re.search(r"(\d+)\s*out of\s*(\d+)", text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _property_type_from_title(title: str) -> str | None:
    for token, mapped in PROPERTY_TYPE_TITLE_TOKENS:
        if token.lower() in title.lower():
            return mapped
    return None


class MagicBricksAdapter(ScraperAdapter):
    source_name = "MAGICBRICKS"

    def search_url(self, locality: str, page: int) -> str:
        slug = locality.lower().replace(" ", "-")
        base = f"https://www.magicbricks.com/flats-for-rent-in-{slug}-mumbai-pppfr"
        return base if page == 1 else f"{base}?page={page}"

    def _extract_apartment_jsonld(self, soup: BeautifulSoup) -> list[dict]:
        for script in soup.find_all("script", type="application/ld+json"):
            raw = (script.string or "").strip()
            if not raw or not raw.startswith("["):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data and isinstance(data, list) and data[0].get("@type") in ("Apartment", "Residence"):
                return data
        return []

    def parse_search_page(self, html: str, locality: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        cards = soup.find_all("div", id=re.compile(r"^cardid\d+$"))
        jsonld = self._extract_apartment_jsonld(soup)
        geo_by_position = jsonld if len(jsonld) == len(cards) else [None] * len(cards)

        raw_listings = []
        for card, geo in zip(cards, geo_by_position):
            card_id = card.get("id", "")
            property_id = re.search(r"\d+", card_id)
            property_id = property_id.group(0) if property_id else None

            summary = {}
            for item in card.find_all(attrs={"data-summary": True}):
                key = item.get("data-summary")
                value_el = item.find(class_="mb-srp__card__summary--value")
                summary[key] = value_el.get_text(strip=True) if value_el else None

            title_el = card.find(class_="mb-srp__card--title")
            title = title_el.get("title") or title_el.get_text(strip=True) if title_el else None

            price_el = card.find(class_="mb-srp__card__price--amount")
            price_text = price_el.get_text(strip=True) if price_el else None

            detail_link = card.find("a", class_="mb-srp__card__society--name") or card.find(
                "a", class_="mb-srp__card__developer--name"
            )
            detail_url = detail_link.get("href") if detail_link else None

            raw_listings.append(
                {
                    "property_id": property_id,
                    "title": title,
                    "summary": summary,
                    "price_text": price_text,
                    "detail_url": detail_url,
                    "search_locality": locality,
                    "geo": geo,
                }
            )
        return raw_listings

    def to_canonical(self, raw: dict) -> dict:
        summary = raw["summary"]
        title = raw["title"] or ""
        geo = raw.get("geo") or {}
        address = geo.get("address", {}) if isinstance(geo, dict) else {}
        geo_coords = geo.get("geo", {}) if isinstance(geo, dict) else {}

        bhk_match = re.search(r"(\d+)\s*BHK", title, re.IGNORECASE)
        bhk = float(bhk_match.group(1)) if bhk_match else (
            float(geo["numberOfRooms"]) if isinstance(geo, dict) and geo.get("numberOfRooms") else None
        )

        if summary.get("carpet-area"):
            area_text, area_type = summary["carpet-area"], "carpet"
        elif summary.get("super-area"):
            area_text, area_type = summary["super-area"], "super_area_fallback"
        else:
            area_text, area_type = None, None
        area_sqft = None
        if area_text:
            m = re.search(r"([\d,.]+)", area_text)
            if m:
                area_sqft = float(m.group(1).replace(",", ""))

        floor, total_floors = _floor_to_pair(summary.get("floor") or "")

        furnishing_raw = (summary.get("furnishing") or "").strip().lower()
        furnishing = FURNISHING_MAP.get(furnishing_raw)

        bathrooms_text = summary.get("bathroom")
        bathrooms = float(bathrooms_text) if bathrooms_text and bathrooms_text.isdigit() else None

        property_type = _property_type_from_title(title)

        return {
            "listing_id": f"MAGICBRICKS_{raw['property_id']}" if raw["property_id"] else None,
            "source": "MAGICBRICKS",
            "scrape_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "search_locality": raw["search_locality"],          # locality slug used in the query
            "raw_locality": address.get("addressLocality"),      # site's own locality tag — may be a sub-locality
            "title": title,
            "latitude": float(geo_coords["latitude"]) if geo_coords.get("latitude") else None,
            "longitude": float(geo_coords["longitude"]) if geo_coords.get("longitude") else None,
            "carpet_area_sqft": area_sqft if area_type == "carpet" else None,
            "area_sqft_raw": area_sqft,
            "area_type_raw": area_type,                          # "carpet" or "super_area_fallback" or None
            "bhk": bhk,
            "bathrooms": bathrooms,
            "furnishing": furnishing,
            "furnishing_raw": summary.get("furnishing"),
            "floor": floor,
            "total_floors": total_floors,
            "floor_raw": summary.get("floor"),
            "age_status_raw": summary.get("status"),             # e.g. "Const. Age Less than 5 years" — Phase C buckets this
            "property_type": property_type,
            "monthly_rent": _rupee_to_number(raw["price_text"]),
            "deposit": None,                                     # not disclosed at search-results level
            "detail_url": raw.get("detail_url"),
        }
