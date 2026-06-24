"""
Phase A — Real transit table builder.

Compiles a real Mumbai metro / suburban-rail station table from OpenStreetMap
data, scoped to the corridors serving the three target localities (Powai,
Mulund, Andheri East). The module issues an Overpass API query and caches the
raw response for audit (fetch_overpass); the table it emits is the manually-
curated `STATIONS` list below — transcribed from that OSM data — not a
programmatic parse of the Overpass response. Output preserves the exact schema
the rest of the pipeline already reads (geo/transit.py):

    station_name, line, latitude, longitude, status, opening_date

Two extra columns are appended for human review only — geo/transit.py does
not read them, so they do not affect downstream behaviour:

    confidence    — high / medium / low / FLAG
    review_note   — why a name/line/date might need correction

Raw Overpass responses are cached to data/raw/osm/ so the curation step is
re-runnable without re-hitting the API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "RentLens-research/0.1 (private analytical study; contact: jatinlilani2@gmail.com)"

# Bounding box covering Andheri East -> Powai -> Vikhroli -> Kanjurmarg -> Mulund -> Thane border
BBOX = (19.00, 72.80, 19.25, 73.00)  # lat_min, lon_min, lat_max, lon_max

OVERPASS_QUERY = f"""
[out:json][timeout:60];
(
  node["railway"="station"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["station"="subway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["public_transport"="station"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["railway"="station"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["station"="subway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["railway"="construction"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["construction"="station"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["proposed"="station"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out center tags;
"""


def fetch_overpass(cache_path: Path, force_refresh: bool = False) -> dict:
    """Fetch raw Overpass JSON, caching to disk so repeat runs don't re-hit the API."""
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.post(
        OVERPASS_URL,
        data={"data": OVERPASS_QUERY},
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    time.sleep(1)  # be polite even though this is a single call
    return data


# ---------------------------------------------------------------------------
# Curated station list.
#
# Names + coordinates are transcribed by hand from the OSM/Overpass data for
# each station and recorded with the source osm_id where known (some rows carry
# a placeholder/FLAG osm_id pending confirmation). "status" is derived from the
# OSM tag observed at curation time
# (railway=station -> operational, railway=construction/construction=* ->
# under_construction, proposed=*/railway=proposed -> planned).
#
# "line" attribution and "opening_date" are filled in only where there is a
# well-known public fact behind them (e.g. Metro Line 1 opened 2014-06-08).
# Anything else is left blank rather than guessed, with confidence="FLAG"
# and a review_note explaining what needs the user's confirmation.
# ---------------------------------------------------------------------------

LINE1_OPENED = "2014-06-08"  # Versova - Ghatkopar, well-documented public fact

STATIONS: list[dict] = [
    # --- Operational: Metro Line 1 (2014-06-08), Andheri East corridor ---
    dict(osm_id="node/213030669... see Andheri", station_name="Andheri Metro", line="Metro Line 1",
         latitude=19.1197, longitude=72.8464, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note=""),
    dict(osm_id="way/151205149", station_name="Ghatkopar Metro", line="Metro Line 1",
         latitude=19.0867, longitude=72.9081, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note=""),
    dict(osm_id="way/331815671", station_name="Western Express Highway", line="Metro Line 1",
         latitude=19.1159, longitude=72.8564, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note=""),
    dict(osm_id="way/331824153", station_name="Chakala - J.B. Nagar", line="Metro Line 1",
         latitude=19.1120, longitude=72.8676, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note="Within Andheri East."),
    dict(osm_id="way/311841874", station_name="Airport Road", line="Metro Line 1",
         latitude=19.1101, longitude=72.8742, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note="Within Andheri East."),
    dict(osm_id="way/311841876", station_name="Marol Naka (Line 1)", line="Metro Line 1",
         latitude=19.1082, longitude=72.8795, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note="Within Andheri East."),
    dict(osm_id="way/331824155", station_name="Saki Naka", line="Metro Line 1",
         latitude=19.1035, longitude=72.8880, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note=""),
    dict(osm_id="way/331827626", station_name="Asalpha", line="Metro Line 1",
         latitude=19.0964, longitude=72.8949, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note=""),
    dict(osm_id="way/311841875", station_name="Jagruti Nagar", line="Metro Line 1",
         latitude=19.0926, longitude=72.9019, status="operational", opening_date=LINE1_OPENED,
         confidence="high", review_note=""),

    # --- Operational: Metro Line 7 (Dahisar E - Andheri E), Andheri East ---
    dict(osm_id="node/10576622687", station_name="Gundavali", line="Metro Line 7",
         latitude=19.1145, longitude=72.8552, status="operational", opening_date=None,
         confidence="FLAG", review_note="Line 7 opened in phases in 2019; exact opening date for "
         "this terminus station not independently verified — please confirm."),

    # --- Operational: Metro Line 3 (Aqua Line), Andheri East / airport corridor ---
    dict(osm_id="node/5787704195", station_name="SEEPZ (Line 3)", line="Metro Line 3",
         latitude=19.1260, longitude=72.8737, status="operational", opening_date=None,
         confidence="FLAG", review_note="Line 3 opened in phases (Aarey-BKC ~Oct 2024 per public "
         "reporting at training time); whether this station and the full line are operational as "
         "of the scrape date needs confirmation — current date is past model knowledge cutoff."),
    dict(osm_id="node/5787704196", station_name="MIDC - Andheri", line="Metro Line 3",
         latitude=19.1174, longitude=72.8736, status="operational", opening_date=None,
         confidence="FLAG", review_note="Same Line 3 phasing caveat as SEEPZ (Line 3)."),
    dict(osm_id="node/5787704197", station_name="Marol Naka (Line 3)", line="Metro Line 3",
         latitude=19.1085, longitude=72.8786, status="operational", opening_date=None,
         confidence="FLAG", review_note="Same Line 3 phasing caveat as SEEPZ (Line 3)."),
    dict(osm_id="node/12074909132", station_name="Sahar Road", line="Metro Line 3",
         latitude=19.1022, longitude=72.8652, status="operational", opening_date=None,
         confidence="FLAG", review_note="Same Line 3 phasing caveat as SEEPZ (Line 3)."),

    # --- Operational: Suburban rail (Central Railway), Powai/Mulund corridor ---
    dict(osm_id="node/4258411797", station_name="Vikhroli", line="Central Railway",
         latitude=19.1115, longitude=72.9280, status="operational", opening_date=None,
         confidence="medium", review_note="Central Line corridor opened 1854-04-16 (Bombay-Thane, "
         "first passenger train in India); exact date this individual station was added not verified."),
    dict(osm_id="node/619992025", station_name="Kanjur Marg", line="Central Railway",
         latitude=19.1287, longitude=72.9280, status="operational", opening_date=None,
         confidence="medium", review_note="Same Central Railway dating caveat as Vikhroli."),
    dict(osm_id="node/3248305759", station_name="Bhandup", line="Central Railway",
         latitude=19.1428, longitude=72.9377, status="operational", opening_date=None,
         confidence="medium", review_note="Same Central Railway dating caveat as Vikhroli."),
    dict(osm_id="node/2629791007", station_name="Nahur", line="Central Railway",
         latitude=19.1546, longitude=72.9467, status="operational", opening_date=None,
         confidence="medium", review_note="Same Central Railway dating caveat as Vikhroli."),
    dict(osm_id="node/1643351703", station_name="Mulund", line="Central Railway",
         latitude=19.1721, longitude=72.9567, status="operational", opening_date=None,
         confidence="medium", review_note="Same Central Railway dating caveat as Vikhroli."),

    # --- Operational: Suburban rail (Western Railway), Andheri East border ---
    dict(osm_id="node/12189363305", station_name="Jogeshwari (Western Line)", line="Western Railway",
         latitude=19.1361, longitude=72.8489, status="operational", opening_date=None,
         confidence="medium", review_note="Western Line corridor opened 1867; exact date this "
         "individual station was added not verified."),

    # --- Under construction: Metro Line 6 (Swami Samarth Nagar - Vikhroli), POWAI core ---
    dict(osm_id="node/5219519239", station_name="Powai Lake Station", line="Metro Line 6",
         latitude=19.1200, longitude=72.9060, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source; please supply target "
         "date if known (README references Line 6 / Powai Lake Metro at 2026-03-31 in the synthetic "
         "model — confirm if that still applies to the real station)."),
    dict(osm_id="node/9132708011", station_name="Powai Udyan Station", line="Metro Line 6",
         latitude=19.1216, longitude=72.8990, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/9132708012", station_name="IIT Powai Station", line="Metro Line 6",
         latitude=19.1242, longitude=72.9137, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/5219518819", station_name="Saki Vihar Road Station", line="Metro Line 6",
         latitude=19.1262, longitude=72.8905, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/5219518806", station_name="SEEPZ Station (Line 6)", line="Metro Line 6",
         latitude=19.1294, longitude=72.8826, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="Name collides with the operational Line 1/3 SEEPZ stations "
         "~1km away — OSM tags this as a separate Line 6 construction node; please confirm this is "
         "not a duplicate/mistag before treating it as a distinct station."),
    dict(osm_id="node/5219518799", station_name="Mahakali Caves Station", line="Metro Line 6",
         latitude=19.1335, longitude=72.8747, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/5219518752", station_name="Adarsh Nagar Station", line="Metro Line 6",
         latitude=19.1418, longitude=72.8336, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source; western end of Line 6, "
         "far from Powai core — low impact on Powai nearest-station distances."),
    dict(osm_id="node/5219518747", station_name="Lokhandwala Station", line="Metro Line 6",
         latitude=19.1407, longitude=72.8272, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source; western end of Line 6, "
         "far from Powai core — low impact on Powai nearest-station distances."),
    dict(osm_id="node/5219518738", station_name="Swami Samarth Nagar Station", line="Metro Line 6",
         latitude=19.1437, longitude=72.8197, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source; western terminus of "
         "Line 6, far from Powai core — low impact on Powai nearest-station distances."),
    dict(osm_id="node/5219519273", station_name="Vikhroli Station (Line 6/4 interchange)", line="Metro Line 6",
         latitude=19.1247, longitude=72.9365, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="OSM places this as the shared Line 6 / Line 4 interchange "
         "construction node at Vikhroli; please confirm interchange status."),

    # --- Under construction: Metro Line 4 (Wadala - Kasarvadavali via Ghatkopar/Mulund/Thane) ---
    dict(osm_id="node/5225899386", station_name="Kanjurmarg Station (Line 4)", line="Metro Line 4",
         latitude=19.1254, longitude=72.9253, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/5225899381", station_name="Naval Housing Station", line="Metro Line 4",
         latitude=19.1329, longitude=72.9277, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/5225899378", station_name="Bhandup Mahapalika Station", line="Metro Line 4",
         latitude=19.1388, longitude=72.9308, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/9126268646", station_name="Mulund Fire Station Metro", line="Metro Line 4",
         latitude=19.1757, longitude=72.9425, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source. README's synthetic "
         "model cites Bhandup Metro (Line 4) target 2026-09-30 — confirm if that applies to this "
         "or a different Line 4 station near Mulund."),
    dict(osm_id="node/9126268645", station_name="Sonapur Station", line="Metro Line 4",
         latitude=19.1662, longitude=72.9384, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/5225899340", station_name="Mulund Check Naka Station", line="Metro Line 4",
         latitude=19.1834, longitude=72.9508, status="under_construction", opening_date=None,
         confidence="FLAG", review_note="No confirmed opening date in source."),
    dict(osm_id="node/9121318237", station_name="Teen Hath Naka Station", line="Metro Line 4",
         latitude=19.1871, longitude=72.9616, status="under_construction", opening_date=None,
         confidence="medium", review_note="Line attribution confirmed via OSM network='Line 4' tag; "
         "opening date not confirmed."),
]


def build_curated_table() -> pd.DataFrame:
    df = pd.DataFrame(STATIONS)
    df["opening_date"] = pd.to_datetime(df["opening_date"])
    return df[
        ["station_name", "line", "latitude", "longitude", "status", "opening_date",
         "confidence", "review_note", "osm_id"]
    ]


def run(output_path: Path, raw_cache_path: Path, force_refresh: bool = False) -> pd.DataFrame:
    fetch_overpass(raw_cache_path, force_refresh=force_refresh)  # ensures raw cache exists for audit
    df = build_curated_table()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    raw_cache = root / "data" / "raw" / "osm" / "overpass_mumbai_stations_raw.json"
    out_csv = root / "data" / "reference" / "transit_mumbai.csv"

    df = run(out_csv, raw_cache)

    print(f"\n{'='*70}")
    print("RENTLENS — Phase A: Real Transit Table (OSM/Overpass)")
    print(f"{'='*70}")
    print(f"Stations written : {len(df)}")
    print(f"Output            : {out_csv}")
    print(f"Raw cache         : {raw_cache}")
    print(f"\nBy status:")
    print(df["status"].value_counts().to_string())
    print(f"\nBy confidence:")
    print(df["confidence"].value_counts().to_string())
    flagged = df[df["confidence"] == "FLAG"]
    print(f"\n{len(flagged)} stations flagged for your review:")
    print(flagged[["station_name", "line", "status", "review_note"]].to_string(index=False))
