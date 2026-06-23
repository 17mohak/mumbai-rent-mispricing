"""
Interactive Mumbai rental-mispricing map.

Layers (all toggleable via the layer control):
  1. Listings          — CircleMarkers coloured by cross-market residual %
                         Red = overpriced vs fundamentals
                         Green = underpriced vs fundamentals
  2. Operational transit — solid blue station markers
  3. Under-construction   — orange station markers (opening date in popup)
  4. Planned transit     — grey station markers
  5. Locality mispricing — large semi-transparent circles (choropleth proxy)
  6. Arbitrage picks     — gold star markers (top 30 candidates)

Output: standalone HTML file (~4 MB), no server required.
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import yaml
from folium import FeatureGroup, LayerControl
from folium.plugins import MarkerCluster


# ── colour helpers ───────────────────────────────────────────────────────────

def _residual_hex(pct: float) -> str:
    if pct < -20:   return "#14532D"   # very underpriced — deep green
    if pct < -10:   return "#16A34A"
    if pct < -5:    return "#4ADE80"
    if pct <  5:    return "#94A3B8"   # roughly fair — slate
    if pct < 10:    return "#F97316"
    if pct < 20:    return "#DC2626"
    return          "#7F1D1D"          # very overpriced — deep red


def _locality_hex(pct: float) -> str:
    if pct < -10:   return "#16A34A"
    if pct <  0:    return "#86EFAC"
    if pct <  10:   return "#FCA5A5"
    return          "#DC2626"


_TRANSIT_STYLE = {
    "operational":       {"color": "#1D4ED8", "fill": "#3B82F6", "label": "OPR"},
    "under_construction":{"color": "#C2410C", "fill": "#FB923C", "label": "UC"},
    "planned":           {"color": "#4B5563", "fill": "#9CA3AF", "label": "PLN"},
}


# ── layer builders ───────────────────────────────────────────────────────────

def _listing_popup(row: pd.Series) -> str:
    rcolor = _residual_hex(row["residual_cm_pct"])
    return f"""
<div style="font-family:sans-serif;font-size:12px;min-width:220px;">
  <b>{row['locality']} | {int(row['bhk'])} BHK</b><br>
  {int(row['carpet_area_sqft']):,} sqft &mdash; {row['furnishing']}<br>
  <b>Rent: Rs.{int(row['monthly_rent']):,}/mo</b><br>
  Fair (CM): Rs.{int(row['fundamental_fair_rent']):,}/mo<br>
  <span style="color:{rcolor};font-weight:bold;">
    Residual: {row['residual_cm_pct']:+.1f}%
  </span><br>
  <hr style="margin:4px 0">
  Nearest UC: {row.get('nearest_uc_name','N/A')}
  ({row.get('dist_nearest_under_construction_m', 0):.0f} m)
</div>"""


def build_listing_layer(df: pd.DataFrame) -> FeatureGroup:
    fg = FeatureGroup(name="Listings (by residual %)", show=True)
    cluster = MarkerCluster(
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 13}
    )
    # A few real listings have no fundamental_fair_rent/residual (ols_Xy
    # dropped them upstream — e.g. missing floor data) — this layer's whole
    # purpose is residual-based coloring, so skip rather than fabricate a
    # color or crash on the NaN.
    df = df[df["residual_cm_pct"].notna()]
    for _, row in df.iterrows():
        c = _residual_hex(row["residual_cm_pct"])
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.75,
            weight=0.8,
            popup=folium.Popup(_listing_popup(row), max_width=260),
            tooltip=f"{row['locality']} | Rs.{int(row['monthly_rent']):,} | {row['residual_cm_pct']:+.1f}%",
        ).add_to(cluster)
    cluster.add_to(fg)
    return fg


def build_transit_layer(transit: pd.DataFrame, status: str) -> FeatureGroup:
    label_map = {
        "operational":        "Transit — Operational",
        "under_construction": "Transit — Under Construction",
        "planned":            "Transit — Planned",
    }
    fg = FeatureGroup(name=label_map[status], show=True)
    style = _TRANSIT_STYLE[status]
    sub = transit[transit["status"] == status]

    for _, row in sub.iterrows():
        opening = (
            "" if pd.isnull(row.get("opening_date"))
            else f"<br>Opening: {str(row['opening_date'])[:10]}"
        )
        popup_html = f"""
<div style="font-family:sans-serif;font-size:12px;">
  <b>{row['station_name']}</b><br>
  {row['line']}<br>
  Status: <b>{status.replace('_',' ').title()}</b>{opening}
</div>"""
        # Outer ring to make UC stations visually pop
        if status == "under_construction":
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=16,
                color=style["color"],
                fill=False,
                weight=2,
                dash_array="6 4",
                opacity=0.7,
            ).add_to(fg)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=9 if status == "operational" else 7,
            color=style["color"],
            fill=True,
            fill_color=style["fill"],
            fill_opacity=0.95,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['station_name']} ({status.replace('_',' ')})",
        ).add_to(fg)

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:7px;font-weight:bold;'
                     f'color:{style["color"]};margin-top:-4px;margin-left:10px;">'
                     f'{style["label"]}</div>',
                icon_size=(30, 12),
            ),
        ).add_to(fg)

    return fg


def build_locality_layer(df: pd.DataFrame, localities: list[dict]) -> FeatureGroup:
    """Semi-transparent circles at locality centroids, coloured by median residual."""
    fg = FeatureGroup(name="Locality mispricing (choropleth)", show=True)

    stats = (
        df.groupby("locality")
        .agg(median_resid=("residual_cm_pct", "median"),
             n=("listing_id", "count"),
             median_rent=("monthly_rent", "median"))
        .to_dict("index")
    )
    for loc in localities:
        name = loc["name"]
        if name not in stats:
            continue
        s = stats[name]
        color = _locality_hex(s["median_resid"])
        radius = max(30, min(70, s["n"] / 5))

        popup_html = f"""
<div style="font-family:sans-serif;font-size:12px;">
  <b>{name}</b><br>
  Listings: {s['n']}<br>
  Median rent: Rs.{int(s['median_rent']):,}/mo<br>
  Mispricing: <span style="color:{color};font-weight:bold;">
    {s['median_resid']:+.1f}%</span>
</div>"""
        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.12,
            weight=2.5,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{name}: {s['median_resid']:+.1f}%",
        ).add_to(fg)

        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:{color};">'
                     f'{name}<br>{s["median_resid"]:+.0f}%</div>',
                icon_size=(100, 30),
                icon_anchor=(50, 0),
            ),
        ).add_to(fg)

    return fg


def build_arbitrage_layer(df: pd.DataFrame, top_n: int = 30) -> FeatureGroup:
    fg = FeatureGroup(name=f"Top {top_n} arbitrage candidates", show=True)

    arb_path = Path(__file__).resolve().parents[3] / "outputs" / "arbitrage_list.csv"
    if not arb_path.exists():
        return fg

    arb = pd.read_csv(arb_path).head(top_n)
    arb_ids = set(arb["listing_id"])
    sub = df[df["listing_id"].isin(arb_ids)].merge(
        arb[["listing_id", "future_station", "opening_date", "dist_future_m", "vs_op_peer_pct"]],
        on="listing_id", how="left"
    )

    for _, row in sub.iterrows():
        popup_html = f"""
<div style="font-family:sans-serif;font-size:12px;min-width:230px;">
  <b style="color:#B45309;">ARBITRAGE CANDIDATE</b><br>
  {row['locality']} | {int(row['bhk'])} BHK | {int(row['carpet_area_sqft']):,} sqft<br>
  Rent: Rs.{int(row['monthly_rent']):,}/mo<br>
  Fair: Rs.{int(row['fundamental_fair_rent']):,}/mo<br>
  <b>Discount: {row['residual_cm_pct']:+.1f}% vs fundamentals</b><br>
  vs op-peer: {row.get('vs_op_peer_pct', float('nan')):+.1f}%<br>
  <hr style="margin:4px 0">
  Future station: <b>{row.get('future_station','N/A')}</b><br>
  Distance: {row.get('dist_future_m', 0):.0f} m &nbsp;|&nbsp;
  Opens: {str(row.get('opening_date','?'))[:7]}
</div>"""
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.DivIcon(
                html='<div style="font-size:18px;color:#D97706;'
                     'text-shadow:0 0 3px #fff;">&#9733;</div>',
                icon_size=(22, 22),
                icon_anchor=(11, 11),
            ),
            popup=folium.Popup(popup_html, max_width=270),
            tooltip=f"ARB: {row['locality']} | {row['residual_cm_pct']:+.1f}%",
        ).add_to(fg)

    return fg


# ── legend ───────────────────────────────────────────────────────────────────

_LEGEND_HTML = """
<div style="
  position: fixed; bottom: 30px; left: 20px; z-index: 1000;
  background: white; padding: 12px 16px; border-radius: 8px;
  border: 1px solid #ccc; font-family: sans-serif; font-size: 12px;
  box-shadow: 2px 2px 6px rgba(0,0,0,0.2); min-width: 180px;">
  <b style="font-size:13px;">RentLens &mdash; Mumbai</b>
  <div style="margin-top:6px;font-weight:bold;font-size:11px;">Listing residual %</div>
  <div><span style="background:#7F1D1D;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> &gt;+20% Overpriced</div>
  <div><span style="background:#DC2626;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> +10 to +20%</div>
  <div><span style="background:#F97316;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> +5 to +10%</div>
  <div><span style="background:#94A3B8;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> &plusmn;5% Fair</div>
  <div><span style="background:#4ADE80;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> &minus;5 to &minus;10%</div>
  <div><span style="background:#16A34A;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> &minus;10 to &minus;20%</div>
  <div><span style="background:#14532D;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> &lt;&minus;20% Underpriced</div>
  <div style="margin-top:6px;font-weight:bold;font-size:11px;">Transit</div>
  <div><span style="background:#3B82F6;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> Operational</div>
  <div><span style="background:#FB923C;width:12px;height:12px;display:inline-block;
    border-radius:50%;border:2px dashed #C2410C;"></span> Under construction</div>
  <div><span style="background:#9CA3AF;width:12px;height:12px;display:inline-block;
    border-radius:50%;"></span> Planned</div>
  <div><span style="font-size:14px;color:#D97706;">&#9733;</span> Arbitrage pick</div>
  <div style="margin-top:8px;font-size:10px;color:#6B7280;">
    ALL DATA SYNTHETIC &mdash; RentLens v0.1
  </div>
</div>"""


def add_legend(m: folium.Map) -> None:
    m.get_root().html.add_child(folium.Element(_LEGEND_HTML))


# ── main entry ───────────────────────────────────────────────────────────────

def build_map(
    scored: pd.DataFrame,
    transit: pd.DataFrame,
    config_path: Path,
) -> folium.Map:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    bb = cfg["bounding_box"]
    center = [
        (bb["lat_min"] + bb["lat_max"]) / 2,
        (bb["lon_min"] + bb["lon_max"]) / 2,
    ]

    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles="CartoDB positron",
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Layers (order matters — bottom to top)
    build_locality_layer(scored, cfg["localities"]).add_to(m)

    for status in ("operational", "under_construction", "planned"):
        build_transit_layer(transit, status).add_to(m)

    build_listing_layer(scored).add_to(m)
    build_arbitrage_layer(scored).add_to(m)

    LayerControl(collapsed=False).add_to(m)
    add_legend(m)

    return m


def run(
    scored_path: Path,
    transit_path: Path,
    config_path: Path,
    output_path: Path,
) -> Path:
    scored  = pd.read_parquet(scored_path)
    transit = pd.read_csv(transit_path, parse_dates=["opening_date"])

    m = build_map(scored, transit, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    out = run(
        scored_path  = root / "data"      / "processed"  / "listings_scored.parquet",
        transit_path = root / "data"      / "reference"  / "transit_mumbai.csv",
        config_path  = root / "config"    / "cities"     / "mumbai.yaml",
        output_path  = root / "outputs"   / "rentlens_mumbai_map.html",
    )
    print(f"Map saved -> {out}")
