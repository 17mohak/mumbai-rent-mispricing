# RentLens — Repository Hardening Log

This log records a repository-hardening pass: documentation accuracy, consistency,
reproducibility, test coverage, CI, and code-health improvements made **without**
altering the project's methodology, models, scoring, rankings, statistical
assumptions, cleaning thresholds, synthetic-data assumptions, findings, or any
other behavior.

**Validation baseline (before any change):** `python -m pytest -q` → **63 passed**
(deps installed without upgrading the present `numpy` 2.4.2 / `pandas` 2.2.2).
Every code change below was re-validated against this baseline.

Branch: `repo-hardening`.

---

## Backlog & categorization

### Category A — safe, no behavior change (implemented)

| # | Issue | Fix |
|---|---|---|
| A1 | Version drift: `pyproject` `0.1.0` vs README footer `v0.2` | Bump `pyproject` → `0.2.0` |
| A2 | README "Python 3.14" vs `requires-python = ">=3.11"` | README → "Python 3.11+" |
| A3 | README Methodology "Phase 1–6" ≠ `--phase N` CLI numbering | Add clarifying mapping note (no renumber) |
| A4 | No `clean.py` tests | Add `tests/test_clean.py` |
| A5 | No scraper parser tests | Add `tests/test_scrape.py` |
| A6 | Unused imports in 4 modules | Remove |
| A7 | Duplicate haversine (`clean.py` vs `geo/transit.py`) | `clean.py` imports `haversine_m` |
| A8 | No CI | Add `.github/workflows/ci.yml` |
| A9 | No contributor doc | Add `CONTRIBUTING.md` |

### Category B — could affect outputs / human judgment (documented, NOT changed)

- **B1** `viz/map.py` legend hardcodes `ALL DATA SYNTHETIC — RentLens v0.1`. Editing
  changes rendered HTML output; the "SYNTHETIC" wording is semantically load-bearing
  on real-data runs. Left untouched.
- **B2** README states the transit table is "pulled from OpenStreetMap via the Overpass
  API", but `geo/build_transit_table.py` fetches the Overpass response only to populate
  an audit cache and then **discards it**, emitting a hardcoded `STATIONS[]` literal
  (coordinates hand-transcribed from OSM; first row's `osm_id` is a placeholder string).
  Correcting the provenance wording is desirable, but the precise phrasing is a human
  judgment call. Left untouched.
- **B3** `model/mispricing.py` imports the synthetic `PLANTED_BIAS` constant on the real
  path. Decoupling risks touching scoring/verification logic. Left untouched.
- **B4** Aggressive dependency pins (`pandas>=3.0.3`, `numpy>=2.4.6`, `scikit-learn>=1.9.0`,
  `scipy>=1.18.0`) with no lockfile. Changing pins affects reproducibility/numerics.
  Left untouched; flagged for human decision.

### Category C — insufficient evidence (investigate, NOT changed)

- **C1** User-Agent contact `jatinlilani2@gmail.com` (`scrape/base.py`,
  `geo/build_transit_table.py`) vs git author *Mohak Mandwani*. Needs human confirmation
  of the intended contact before any edit.
- **C2** Git-ignored confidential `data/Updated_Data.xlsx` (Sagitec pension-system export).
  Confirm it was never pushed to a remote. No code change.
- **C3** Real-data findings are not reproducible from the repo (raw/processed per-listing
  data git-ignored and absent). Research/process concern, outside hardening scope.

---

## Completed changes

All entries dated 2026-06-24. Validation baseline before the pass: `63 passed`.
After the pass (with the two new test modules): `112 passed`.

| Commit | Category | Issue | Root cause | Files | Validation |
|---|---|---|---|---|---|
| `38d9a54` | — | Hardening log + backlog | n/a | `HARDENING_LOG.md` | n/a |
| `6b5fa2d` | A1 | Package version `0.1.0` ≠ README `v0.2` | pyproject never bumped after real-data support | `pyproject.toml` | `pytest` → 63 passed |
| `069c47e` | A2 | README "Python 3.14" ≠ `requires-python=">=3.11"` | prose drift from authoritative constraint | `README.md` | doc-only |
| `8b75617` | A3 | Methodology "Phase N" labels ≠ `--phase N` CLI flags | two independent numbering schemes; "Phase 6 — Causal" instructs `--phase 5` | `README.md` | doc-only (no renumber) |
| `5b226a0` | A4 | `clean.py` had no tests | missing coverage | `tests/test_clean.py` | 25 new tests pass; full 88 passed |
| `ef71e86` | A5 | scraper had no tests | missing coverage | `tests/test_scrape.py` | 24 new tests pass; full 112 passed |
| `346c38c` | A6 | unused imports in 4 modules | leftover imports | `mispricing.py`, `diff_in_diff.py`, `pipeline.py`, `generate.py` | imports OK; 112 passed |
| `8656be0` | A7 | duplicate haversine in `clean.py` | copy of `geo/transit.haversine_m` | `data/clean.py` | numeric bit-equivalence proven; 112 passed |
| `b478c29` | A8 | no CI | none configured | `.github/workflows/ci.yml` | YAML validated |
| `cbabb06` | A9 | no contributor doc | missing onboarding | `CONTRIBUTING.md` | doc-only |

### Validation notes

- The model modules (`hedonic`, `gbm`, `uncertainty`, `mispricing`) and `causal`
  require `statsmodels` / `lightgbm`; these were installed for validation
  **without upgrading** the pre-existing `numpy` 2.4.2 / `pandas` 2.2.2, so the
  numerical baseline is unchanged.
- `A7` was gated on an explicit numerical equivalence check (Series and array
  inputs, non-contiguous index) confirming the two haversine implementations
  return identical values before the duplicate was removed.
- No change touched model features, training, scoring, ranking, arbitrage,
  causal methodology, synthetic assumptions, planted-bias values, cleaning
  thresholds, or any reported metric/finding.
