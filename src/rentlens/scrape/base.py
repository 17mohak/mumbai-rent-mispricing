"""
Pluggable scraper interface.

Every site adapter (e.g. scrape/magicbricks.py) implements ScraperAdapter so
the rest of the pipeline never needs to know which source produced a listing
— it only needs rows in the canonical schema defined by
rentlens.data.generate.validate_schema.

CachedFetcher is the shared HTTP layer all adapters should use. It enforces
the non-negotiable scraping rules for this project:
  - every raw response is cached to disk (data/raw/<source>/...) before
    anything is parsed, so a crash mid-run never loses work and a re-run
    never re-fetches a page it already has (resumability)
  - a minimum delay is enforced between *live* requests (cache hits are free)
  - an honest, identifiable User-Agent is sent on every request
"""

from __future__ import annotations

import abc
import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_USER_AGENT = (
    "RentLens-research/0.1 (private analytical study; contact: jatinlilani2@gmail.com)"
)


class CachedFetcher:
    """Throttled, disk-cached GET. Cache presence *is* the resume checkpoint —
    re-running a scrape after a crash just re-fetches the cache, sees the
    files already exist, and moves on without hitting the network again.
    """

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str = DEFAULT_USER_AGENT,
        min_delay_s: float = 5.0,
        timeout_s: float = 25.0,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.min_delay_s = min_delay_s
        self.timeout_s = timeout_s
        self._last_request_ts: float | None = None

    def _cache_path(self, cache_key: str) -> Path:
        safe_key = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{safe_key}.html"

    def get(self, url: str, cache_key: str | None = None) -> str:
        """Return response text for `url`, using `cache_key` (default: url
        itself) as the on-disk cache identity. Only sleeps/hits the network
        on an actual cache miss.
        """
        path = self._cache_path(cache_key or url)
        if path.exists():
            return path.read_text(encoding="utf-8")

        if self._last_request_ts is not None:
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.min_delay_s:
                time.sleep(self.min_delay_s - elapsed)

        resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout_s)
        self._last_request_ts = time.monotonic()
        resp.raise_for_status()

        path.write_text(resp.text, encoding="utf-8")
        return resp.text


class ScraperAdapter(abc.ABC):
    """One implementation per source site."""

    source_name: str

    @abc.abstractmethod
    def search_url(self, locality: str, page: int) -> str:
        """Build the search-results URL for a given locality + 1-indexed page."""

    @abc.abstractmethod
    def parse_search_page(self, html: str, locality: str) -> list[dict]:
        """Parse one search-results page into a list of raw listing dicts
        (site-native field names — not yet mapped to the canonical schema).
        """

    @abc.abstractmethod
    def to_canonical(self, raw: dict) -> dict:
        """Map one raw listing dict onto the canonical schema fields. May
        include extra non-canonical fields (e.g. `area_type_raw`,
        `age_bucket_raw`) for the Phase C cleaning step to consume —
        cleaning, not this method, is where ambiguous fields get resolved.
        """

    def fetch_listings(
        self,
        localities: list[str],
        fetcher: CachedFetcher,
        max_pages_per_locality: int = 1,
        id_field: str = "listing_id",
    ) -> pd.DataFrame:
        """Fetch + parse listings for each locality, return canonical-shaped
        rows (still raw/uncleaned — Phase C does cleaning/validation).

        Stops paginating a locality early if a page yields zero *new* ids —
        listing sites commonly report inflated/relisted totals that exceed
        true distinct inventory, and re-fetching pages of duplicates wastes
        requests for no benefit.
        """
        rows: list[dict] = []
        for locality in localities:
            seen_ids: set = set()
            for page in range(1, max_pages_per_locality + 1):
                url = self.search_url(locality, page)
                html = fetcher.get(url, cache_key=f"{self.source_name}:{locality}:{page}")
                raw_listings = self.parse_search_page(html, locality)
                if not raw_listings:
                    break  # no more pages for this locality

                canonical = [self.to_canonical(r) for r in raw_listings]
                new_rows = [r for r in canonical if r.get(id_field) not in seen_ids]
                if not new_rows:
                    break  # page repeated already-seen listings — end of distinct inventory
                seen_ids.update(r[id_field] for r in new_rows)
                rows.extend(new_rows)
        return pd.DataFrame(rows)
