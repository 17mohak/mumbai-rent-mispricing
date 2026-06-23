"""
Phase B — pull real listing volume from MagicBricks for the three target
localities (Powai, Mulund, Andheri East). Raw (uncleaned) canonical-shaped
rows are written to data/raw/magicbricks_listings_raw.parquet; Phase C
(clean.py) is responsible for validation, dedup, and the data quality report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rentlens.scrape.base import CachedFetcher
from rentlens.scrape.magicbricks import MagicBricksAdapter

LOCALITIES = ["Powai", "Mulund", "Andheri East"]
MAX_PAGES_PER_LOCALITY = 15  # ~30 listings/page; early-stops on repeated/duplicate pages


def run(output_path: Path, cache_dir: Path) -> pd.DataFrame:
    adapter = MagicBricksAdapter()
    fetcher = CachedFetcher(cache_dir, min_delay_s=5.0)
    df = adapter.fetch_listings(LOCALITIES, fetcher, max_pages_per_locality=MAX_PAGES_PER_LOCALITY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    out = root / "data" / "raw" / "magicbricks_listings_raw.parquet"
    cache = root / "data" / "raw" / "magicbricks"

    df = run(out, cache)

    print(f"\n{'='*70}")
    print("RENTLENS — Phase B: MagicBricks real-data pull")
    print(f"{'='*70}")
    print(f"Total raw rows : {len(df):,}")
    print(f"Output         : {out}\n")
    print("Rows per locality:")
    print(df["search_locality"].value_counts().to_string())
    print("\nField completeness (non-null %):")
    print((df.notna().mean() * 100).round(1).to_string())
