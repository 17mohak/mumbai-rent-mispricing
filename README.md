# RentLens — Mumbai Rental Mispricing & Transit-Arbitrage Engine

**The finding in one sentence:** across 573 real Powai/Mulund/Andheri East rental
listings (scraped from MagicBricks, June 2026), **Mulund rents trade ~23.5% below**
what their carpet area, BHK, furnishing and building fundamentals justify, while
Powai (+14.9%) and Andheri East (+11.4%) trade **above** fundamentals — and neither
locality shows a statistically significant rent relationship with distance to the
under-construction Metro Line 4 (Mulund) or Line 6 (Powai) stations yet, consistent
with that transit optionality not being priced in.

---

## Data Provenance

| | |
|---|---|
| Source | [MagicBricks](https://www.magicbricks.com) rental search results — chosen over 99acres/Housing.com (blocked even a bare robots.txt request) and NoBroker/SquareYards (robots.txt explicitly disallows their listing/search paths) |
| Localities | Powai, Mulund, Andheri East (chosen for depth, not breadth) |
| Scrape date | 2026-06-23 |
| Raw rows scraped | 621 |
| Clean rows modelled | 573 (92.3% retained) — see [`data/processed/data_quality_report.md`](data/processed/data_quality_report.md) for every drop/transform rule |
| Transit data | Real OpenStreetMap/Overpass station coordinates — [`data/reference/transit_mumbai.csv`](data/reference/transit_mumbai.csv) (37 stations; **opening dates for under-construction stations are unconfirmed** — OSM doesn't carry them, flagged for manual review, never guessed) |
| Geocoding | Not needed — MagicBricks listings carry lat/lon directly (building/society-level precision, not per-unit) |

**Known data limitations (read before trusting a number too far):**
- **Small sample, 3 localities.** Spatial leave-one-locality-out cross-validation
  is honestly poor (R² ranges from +0.16 down to **−2.8** for Mulund) — holding out
  1 of only 3 localities is a brutal generalization test. Treat the within-sample
  hedonic fit (R²=0.80) as the reliable number; treat spatial generalization as
  an open question this sample can't answer well.
- **Carpet area is estimated for 23.4%** of listings (site gave only super/built-up
  area; converted via the standard ~0.70 carpet/super loading factor, flagged in
  `carpet_area_is_estimated`).
- **`building_age_years` is unusable** — 94.6% of listings report move-in
  availability ("Immediately"), not construction age. Left as NaN, excluded from
  every model rather than guessed.
- **`deposit` and `amenities_count` are not disclosed** at search-results level by
  this source at all (0% coverage) — not estimated, not modelled.
- **One snapshot, not a panel.** This is rents observed on one day. There is no
  real before/after metro-opening data, so a genuine causal (DiD) estimate isn't
  possible from this source — see the Causal Analysis section below.

> ### Data publication policy
>
> Per this project's scraping rules, **individual real listings are never
> republished publicly — only aggregates ship in this repo.** `data/raw/`, the
> cleaned per-listing parquets, `outputs/arbitrage_list.csv`, and the interactive
> map (its popups render every individual listing) are all git-ignored. Every
> number in the **Key Findings** table below, the data quality report, and the
> SHAP/diagnostic plots are aggregate statistics and are committed. Clone the
> repo and run the Phase B/C/E commands under "Reproducing" to regenerate the
> per-listing map and arbitrage list locally.

---

## Interactive Map

Generated locally at `outputs/rentlens_mumbai_map.html` after running the
real-data pipeline (see "Reproducing" below) — **not committed to this public
repo** (see Data publication policy above). Built from the real 573-listing
dataset when regenerated.

| Symbol | Meaning |
|--------|---------|
| 🔴 Deep red marker | Flat priced > 20 % above fundamentals |
| 🟠 Orange marker   | Flat priced 5–20 % above fundamentals |
| ⬜ Grey marker     | Fairly priced (±5 %) |
| 🟢 Green marker    | Flat priced 5–20 % below fundamentals |
| 💚 Deep green      | Flat priced > 20 % below fundamentals |
| 🔵 Blue circle     | Operational metro / suburban rail station (real OSM data) |
| 🟠 Dashed orange   | Under-construction metro station (real OSM data) |
| ⭐ Gold star       | Top-30 transit-arbitrage candidates |
| Large halo         | Locality mispricing choropleth |

(No "planned"-status stations are currently in scope — none were found in the
Overpass query near these 3 localities, so that layer renders empty.)

---

## Key Findings (real data)

| Metric | Value |
|--------|-------|
| Powai vs cross-market fundamentals | **+14.9%** (OVERPRICED — 73.9% of listings above fair value) |
| Andheri East vs cross-market fundamentals | **+11.4%** (OVERPRICED — 66.9% above fair value) |
| Mulund vs cross-market fundamentals | **−23.5%** (UNDERPRICED — only 10.1% above fair value) |
| Cross-market hedonic R² (no locality FE) | 0.703 |
| Full hedonic OLS R² (with locality FE) | 0.799 (N=561) |
| Locality fixed effect: Mulund vs Andheri East (baseline) | **−37.9%*** (p<0.001) — real, well-identified |
| Locality fixed effect: Powai vs Andheri East (baseline) | −3.1% (not significant) |
| Distance-to-UC-metro effect on rent (within locality) | +0.11%/100m (not significant) — **no evidence the Line 4/6 optionality is priced in yet** |
| Distance-to-operational-metro effect (within locality) | +0.49%/100m (p<0.05) — small, counterintuitive; likely confounded with building vintage, not a strong causal claim |
| LightGBM spatial CV MAPE (held out 1 of 3 localities) | 42.2% (highly variable by locality — see caveats) |
| Quantile interval in-sample coverage | 76.1% (target 80%) |
| Transit-arbitrage candidates (within 2.5km of a UC station, underpriced) | **235** (Mulund 133, Powai 54, Andheri East 48) |
| Test suite | **63 / 63 passing** |

### Why Mulund, and why caution on Powai/Andheri East

The cross-market model prices each flat from observable fundamentals — carpet
area, BHK, furnishing, building age (where known), transit distances — without
using locality identity. Mulund's real listings sit **23.5% below** what those
fundamentals justify; this is a strong, statistically well-supported result (the
locality fixed effect is significant at p<0.001 even controlling for everything
else). Powai and Andheri East trading *above* fundamentals is the inverse of what
the original synthetic demonstration assumed — that's not an error, it's what the
real June 2026 MagicBricks data actually shows. With only 3 localities and ~570
listings, take the *direction* of these effects seriously and the *exact magnitude*
with more caution.

**On the transit-pricing question specifically:** within each locality, distance
to the nearest under-construction station (Line 4 near Mulund, Line 6 near Powai)
has **no statistically significant relationship** with current rent (β=+0.11%
per 100m, p>0.05). Read plainly, that means the market is **not yet pricing in**
the upcoming metro access — consistent with (not proof of) the transit-arbitrage
thesis. The operational-metro coefficient is statistically significant but tiny
and the *wrong sign* (rent rises slightly with distance from existing stations);
this is most plausibly explained by newer/larger developments sitting slightly
farther from older legacy station footprints in this sample, not a real transit
penalty — flagged here rather than over-interpreted.

---

## Methodology

> **Note on "Phase" numbering.** The `### Phase N` headings below describe the
> *conceptual* analytical stages and do **not** map one-to-one onto the
> `--phase N` flags of `rentlens.pipeline`. The orchestrator's CLI phases are:
> `1` = synthetic generation, `2` = transit enrichment, `3` = models,
> `4` = mispricing + arbitrage, `5` = causal DiD demo, `6` = interactive map.
> (Real-data ingestion and cleaning — the conceptual "Phase 1/2" headings — run
> as the separate `rentlens.scrape.run` and `rentlens.data.clean` commands, not
> as `--phase` steps.) So, e.g., the causal demo under "Phase 6 — Causal
> analysis" below is invoked with `--phase 5`, and the map with `--phase 6`,
> exactly as shown in "Reproducing".

### Phase 1 — Real data ingestion (replaces synthetic generation as the default path)

**Source selection** (`src/rentlens/scrape/`): checked robots.txt for 5 candidate
sites. 99acres and Housing.com blocked even a plain robots.txt request (403/406 —
active bot detection, so per project rules: don't fight it, switch sources).
NoBroker's robots.txt explicitly disallows `/property/listing/`; SquareYards
disallows `/rental/search*`. MagicBricks disallows only admin/profile/image/map
sub-paths — its rental search and listing-detail pages are fair game. The adapter
(`scrape/magicbricks.py`) implements the pluggable `ScraperAdapter` interface
(`scrape/base.py`), which any future source can also implement.

**Scraping discipline:** ≥5s between live requests, honest identifiable
User-Agent, every raw response cached to `data/raw/magicbricks/` (cache presence
*is* the resume checkpoint — a crash never restarts from zero), no CAPTCHA
solving, no login, no bot-detection evasion. 621 raw rows pulled across 3
localities before the scraper's own duplicate-detection stopped each locality
(MagicBricks' claimed inventory count is inflated by relistings beyond true
distinct supply).

**The synthetic generator is kept** (`src/rentlens/data/generate.py`,
`pyproject.toml` script entry unchanged) as a fallback — run
`python -m rentlens.pipeline --city mumbai` (default `--source synthetic`) to
reproduce the original methodology demonstration with planted ground-truth bias
recovery. Use `--source real` to run on the scraped data instead (see
"Reproducing" below).

### Phase 2 — Cleaning & validation (`src/rentlens/data/clean.py`)

Every drop and transform is a documented, counted rule — see
[`data/processed/data_quality_report.md`](data/processed/data_quality_report.md)
for the full step-by-step table. Highlights: rent-outlier removal caught one
listing showing ₹4 crore/month for a 950 sqft 2BHK (almost certainly a sale price
leaking into the rental search results, not a real rent); locality assignment
uses nearest-centroid haversine distance rather than trusting ~50 raw
sub-locality strings MagicBricks reports (e.g. "Sane Guruji Nagar", "Mulund
Colony - Mulund West"); relisting dedup requires an exact match on
building+BHK+bathrooms+carpet-area+rent (a looser key was tested and rejected —
it collapsed legitimately distinct units in the same building).

### Phase 3 — Real transit table (`src/rentlens/geo/build_transit_table.py`)

37 real stations compiled from OpenStreetMap, scoped to the
corridors serving the 3 target localities. The builder issues an Overpass API
query and caches the raw response to `data/raw/osm/` for audit; the shipped table
is then a manually-curated station list whose names/coordinates/construction-status
are transcribed from that OSM data (each row records its source `osm_id` where
known). This curation is deliberate — OSM's tagging for under-construction lines is
inconsistent, so opening dates and some line attributions are **left blank rather
than guessed** where OSM doesn't carry them, with an explicit `confidence`/`review_note` column flagging
21 of 37 stations for manual confirmation (these columns are for human review only
— the pipeline doesn't read them). The old synthetic transit table is preserved at
[`data/reference/transit_mumbai_synthetic_backup.csv`](data/reference/transit_mumbai_synthetic_backup.csv).

### Phase 4 — Models (`src/rentlens/model/`)

| Model | Purpose | Key metric (real data) |
|-------|---------|------------|
| **Hedonic OLS** | Interpretable log-rent baseline with locality FE | R² = 0.799, N=561 |
| **LightGBM** | Non-linear hedonic + SHAP importance | Spatial CV MAPE 42.2% |
| **Quantile regression** (q10/q50/q90) | 80% prediction intervals | 76.1% in-sample coverage |

**Top SHAP features:** `log_carpet` (0.20) › `locality` (0.17) › `bhk` (0.14) ›
`floor` (0.08) › `bathrooms` (0.05) › transit distances (0.037–0.040) ›
`furnishing` (0.03). `property_type` carries ~zero weight (94% of real listings
are plain "Flat" — little variation to learn from).

Real data exposed two latent bugs the synthetic table's full coverage had been
masking: (1) NaN-vs-NaN comparisons in the arbitrage ranking that silently
mislabeled candidates whenever the "planned" station bucket was empty, and
(2) the feature pipeline had no tolerance for a numeric column (like real
`building_age_years`) being mostly missing. Both are fixed and covered by the
existing test suite; `features.py` now drops any numeric feature below 50%
non-null coverage instead of crashing or guessing.

### Phase 5 — Mispricing & arbitrage (`src/rentlens/model/mispricing.py`)

Cross-market model (OLS without locality dummies, R²=0.703) prices each flat from
pure fundamentals; the residual is the mispricing signal. **235 real listings**
within 2.5km of an under-construction station are priced below their fundamental
fair rent (composite score: 60% fundamental discount, 40% discount vs.
operational-metro peers in the same locality). The full ranked, per-listing list
is generated locally at `outputs/arbitrage_list.csv` — not committed to this
public repo (see Data publication policy above).

### Phase 6 — Causal analysis: kept as a synthetic-only methodology demo, NOT run on real data

[`src/rentlens/causal/diff_in_diff.py`](src/rentlens/causal/diff_in_diff.py) is a
**difference-in-differences methodology demonstration**, not a general-purpose
causal estimator: it manufactures its own 3-period panel by injecting a hardcoded
8% treatment effect into synthetic time periods, then verifies the DiD estimator
recovers that planted number. That's a legitimate way to demonstrate the
methodology works — but running it against real rents would just inject the same
fabricated 8% effect into real cross-sectional data and "recover" it again,
which would misrepresent a scripted number as a real finding.

A genuine causal estimate of the metro-opening effect requires actual
before/after rent observations          `, which a single scrape cannot provide. This module
is therefore **excluded from the real-data pipeline run** (`pipeline.py` skips
Phase 5 automatically under `--source real`) and remains available under
`--source synthetic --phase 5` purely as a methodology demonstration. The closest
honest real-data substitute is the within-locality distance-to-UC-station
coefficient reported above (correlational, not causal, and not significant here).

---

## Reproducing the Project End-to-End

```bash
# 1. Clone and navigate, then create venv + install
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .

# 2a. REAL DATA (default localities: Powai, Mulund, Andheri East)
python -m rentlens.scrape.run              # scrape MagicBricks -> data/raw/magicbricks_listings_raw.parquet
python -m rentlens.data.clean              # clean + validate  -> data/processed/listings.parquet
python -m rentlens.geo.build_transit_table # write curated OSM-sourced transit table + refresh Overpass audit cache (table content changes only when the curated STATIONS list is edited)
python -m rentlens.pipeline --city mumbai --source real    # phases 2-4, 6 (phase 1 & 5 auto-skipped)

# 2b. OR fall back to the original synthetic methodology demonstration
python -m rentlens.pipeline --city mumbai --source synthetic   # all 6 phases, planted-bias recovery

# 3. Run a single phase (e.g. rebuild just the map)
python -m rentlens.pipeline --city mumbai --source real --phase 6

# 4. Run the test suite
pytest

# 5. Open the map
start outputs\rentlens_mumbai_map.html   # Windows
```

### Output files

Columns marked **local-only** are git-ignored per the data publication policy
above — they contain individual real listings and are regenerated by running
the commands in this section, not shipped in the repo.

| File | Description | In repo? |
|------|-------------|---|
| `data/raw/magicbricks/` | Cached raw HTML per (locality, page) — resumability checkpoint | local-only |
| `data/raw/magicbricks_listings_raw.parquet` | 621 raw scraped listings, pre-cleaning | local-only |
| `data/processed/data_quality_report.md` | Every cleaning step, counted, with rationale | ✅ committed |
| `data/processed/listings.parquet` | 573 clean real listings (canonical schema) | local-only |
| `data/processed/listings_geo.parquet` | + real transit distances | local-only |
| `data/processed/listings_scored.parquet` | + fair rent, intervals, residuals | local-only |
| `data/reference/transit_mumbai.csv` | 37 real OSM-sourced stations (+ review flags) | ✅ committed |
| `data/reference/transit_mumbai_synthetic_backup.csv` | Original synthetic transit table, preserved | ✅ committed |
| `outputs/rentlens_mumbai_map.html` | Interactive folium map (real data) | local-only |
| `outputs/arbitrage_list.csv` | 235 ranked real arbitrage candidates | local-only |
| `outputs/shap_importance.png` | SHAP feature importance bar chart (aggregate) | ✅ committed |
| `models/rentlens/` | Serialised OLS + LightGBM + quantile models (real-data fit) | local-only |

---

## City-as-Config Scalability

Every city-specific constant still lives in **`config/cities/mumbai.yaml`** (used
by the synthetic generator and the map's bounding box/locality list). The real
scraper/cleaner currently hardcode the 3 target localities and their centroids
(`scrape/run.py`, `data/clean.py`) rather than reading them from this YAML — that
config-driven wiring is the natural next step if a second real-data city or a 4th
Mumbai locality is added.

---

## Repository Structure

```
.
├── src/rentlens/
│   ├── scrape/
│   │   ├── base.py               # ScraperAdapter interface + CachedFetcher (throttled, resumable)
│   │   ├── magicbricks.py        # MagicBricks adapter
│   │   └── run.py                # Real-data volume pull (3 localities)
│   ├── data/
│   │   ├── generate.py           # Synthetic listing generator (fallback, unchanged)
│   │   └── clean.py              # Real-data cleaning/validation + quality report
│   ├── geo/
│   │   ├── build_transit_table.py # Real OSM/Overpass transit table builder
│   │   └── transit.py            # Haversine enrichment (schema-compatible, unchanged)
│   ├── model/
│   │   ├── features.py           # Shared feature engineering (now missingness-aware)
│   │   ├── hedonic.py            # OLS + elasticities + spatial CV
│   │   ├── gbm.py                # LightGBM + SHAP
│   │   ├── uncertainty.py        # Quantile prediction intervals
│   │   └── mispricing.py         # Residuals + arbitrage ranking
│   ├── causal/diff_in_diff.py    # DiD — SYNTHETIC METHODOLOGY DEMO ONLY, see Phase 6 above
│   ├── viz/map.py                # Folium interactive map
│   └── pipeline.py               # End-to-end orchestrator (--source synthetic|real)
├── config/cities/mumbai.yaml     # City-specific constants (synthetic path + map)
├── data/{raw,reference,processed}/
├── models/rentlens/               # Serialised model artefacts
├── outputs/                       # HTML map, plots, CSVs
├── tests/                         # 63 pytest tests (cover both synthetic and real-data edge cases)
└── pyproject.toml                 # Deps + build config
```

---

## Stack

Python 3.11+ · pandas · pyarrow · numpy · scikit-learn · lightgbm · shap ·
statsmodels · geopy · folium · scipy · matplotlib · seaborn · pytest ·
beautifulsoup4 · lxml · requests

---

*RentLens v0.2 — real Mumbai rental data (Powai / Mulund / Andheri East, scraped
2026-06-23) with a synthetic-data fallback for methodology demonstration.*
