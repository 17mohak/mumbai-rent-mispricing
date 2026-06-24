# Contributing to RentLens

Thanks for working on RentLens. This guide covers local setup, repository
layout, and the conventions to follow. It is descriptive of the project as it is
today — it does not change any behavior, models, or findings.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e .
pytest
```

Requires **Python 3.11+** (see `requires-python` in `pyproject.toml`).

> **Heads-up on dependencies.** `pyproject.toml` pins fairly aggressive minimums
> (e.g. `pandas>=3.0.3`, `numpy>=2.4.6`, `scikit-learn>=1.9.0`, `scipy>=1.18.0`).
> If `pip install -e .` cannot resolve those on your platform, install the stack
> with compatible versions of the same libraries and run `pytest` to confirm a
> green baseline before starting work. Do **not** change the pins as part of an
> unrelated change — dependency/version policy is tracked separately (see
> `HARDENING_LOG.md`, item B4).

## Running tests

```bash
pytest                      # full suite
pytest tests/test_clean.py  # a single module
```

The suite is self-contained: it runs on **synthetic** data plus small HTML/CSV
fixtures, and needs no network access and no scraped data on disk.

## Repository layout

```
src/rentlens/
  scrape/      base.py (adapter interface + cached/throttled fetcher),
               magicbricks.py (adapter), run.py (volume pull)
  data/        generate.py (synthetic generator), clean.py (real-data cleaning)
  geo/         build_transit_table.py (transit table), transit.py (haversine enrichment)
  model/       features.py (shared feature engineering), hedonic.py, gbm.py,
               uncertainty.py, mispricing.py
  causal/      diff_in_diff.py (synthetic-only DiD methodology demo)
  viz/         map.py (Folium map)
  pipeline.py  end-to-end orchestrator (--source synthetic|real, --phase 1..6)
config/cities/ mumbai.yaml (synthetic-generator + map constants)
tests/         pytest suite (synthetic + fixture based)
```

Start reading at `README.md` → `pipeline.py` → `model/features.py` (the shared
feature contract every model depends on) → `model/mispricing.py`.

## Pipeline phases (CLI)

The `--phase N` flags of `rentlens.pipeline` are: `1` synthetic generation,
`2` transit enrichment, `3` models, `4` mispricing + arbitrage, `5` causal DiD
demo, `6` interactive map. The "Phase N" headings in the README Methodology
section describe *conceptual* stages and are numbered independently — see the
note at the top of that section.

## Data publication policy (important)

Per the project's scraping rules, **individual real listings are never committed
or republished** — only aggregates. The following are git-ignored and must stay
that way: `data/raw/`, the per-listing `data/processed/listings*.parquet`,
`outputs/arbitrage_list.csv`, and the regenerated real-data map
`outputs/rentlens_mumbai_map.html`. The committed map under `docs/` is the
synthetic-data version. Never add real per-listing data to a commit.

## Conventions

- **Tests pin existing behavior.** Add regression tests when you touch a module;
  do not change expected outputs to make a test pass.
- **Small, focused commits.** One logical change per commit, with a clear
  subject line (e.g. `test(clean): ...`, `docs(readme): ...`, `refactor(geo): ...`).
- **Scraping discipline** (when working in `scrape/`): keep the throttle, the
  on-disk cache (it is the resume checkpoint), and the honest User-Agent; do not
  add CAPTCHA solving, login, or bot-detection evasion.
- Respect the boundaries in `HARDENING_LOG.md`: research methodology, model
  logic, scoring/ranking, cleaning thresholds, and statistical assumptions are
  out of scope for hardening changes — document, don't silently alter.
