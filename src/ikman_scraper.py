"""
Scrape recent car advertisements from ikman.lk.

Walks https://ikman.lk/en/ads/sri-lanka/cars page by page (newest first),
collects every ad whose listing age is within --days, then fetches each ad's
detail page and keeps those whose posting date (adDate) is within the window.

Output: a CSV (Excel-friendly, UTF-8 with BOM) and a JSONL file with the raw
ad records. Detail pages are cached in a folder so a re-run resumes instead of
re-downloading.

Usage:
    python ikman_scraper.py                 # last 14 days, default paths
    python ikman_scraper.py --days 7 --out cars_7d.csv
    python ikman_scraper.py --max-pages 3 --max-ads 20   # quick smoke test

Only standard-library modules are used.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://ikman.lk"
LISTING_URL = BASE + "/en/ads/sri-lanka/cars?page={page}"
AD_URL = BASE + "/en/ad/{slug}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 ikman-research-scraper"
)
COLOMBO_TZ = timezone(timedelta(hours=5, minutes=30))
INITIAL_DATA_RE = re.compile(r"window\.initialData\s*=\s*(\{.*?\});?\s*</script>", re.S)

# Columns written to the CSV, in order. Property keys (brand, model, ...) are
# appended dynamically after these.
BASE_COLUMNS = [
    "id", "url", "title", "price_lkr", "price_text", "negotiable", "posted_at",
    "location", "district", "seller_name", "seller_type", "seller_verified",
    "views", "image_count", "description",
]

# ------------------------------------------------------------------ HTTP ---

_rate_lock = threading.Lock()
_last_request_at = 0.0


def _throttle(min_interval: float) -> None:
    """Global rate limiter shared by all worker threads."""
    global _last_request_at
    with _rate_lock:
        now = time.monotonic()
        wait = _last_request_at + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def fetch(url: str, min_interval: float, retries: int = 4) -> str:
    """GET a URL and return the decoded body, retrying on transient errors."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en",
            "Accept-Encoding": "gzip",
        },
    )
    for attempt in range(retries):
        _throttle(min_interval)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code in (429, 503):
                backoff = 10 * (attempt + 1)
            else:
                backoff = 2 * (attempt + 1)
        except (urllib.error.URLError, TimeoutError, OSError):
            backoff = 2 * (attempt + 1)
        time.sleep(backoff + random.random())
    raise RuntimeError(f"giving up on {url}")


def extract_initial_data(html: str) -> dict:
    m = INITIAL_DATA_RE.search(html)
    if not m:
        raise ValueError("window.initialData not found in page")
    return json.loads(m.group(1))


# -------------------------------------------------------------- parsing ---

_AGE_RE = re.compile(r"(\d+)\s*(minute|hour|day|week|month|year)s?", re.I)
_AGE_DAYS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}


def relative_age_days(text: str) -> float | None:
    """'3 days' -> 3.0, 'just now' -> 0, unknown -> None."""
    if not text:
        return None
    if text.strip().lower() == "just now":
        return 0.0
    m = _AGE_RE.search(text)
    if not m:
        return None
    return int(m.group(1)) * _AGE_DAYS[m.group(2).lower()]


def listing_effective_age_days(ad: dict) -> float | None:
    """Age that determines the ad's position in the listing.

    Non-bumped ads carry their posting age in `timeStamp`; bumped ads carry
    'bump_up' there and the bump age in `lastBumpUpDate`.
    """
    ts = ad.get("timeStamp") or ""
    if ts == "bump_up":
        return relative_age_days(ad.get("lastBumpUpDate") or "")
    return relative_age_days(ts)


def parse_price(text: str) -> int | None:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def parse_number(text: str) -> int | None:
    m = re.search(r"[\d,]+", text or "")
    return int(m.group(0).replace(",", "")) if m else None


def flatten_ad(ad: dict) -> dict:
    """Turn the detail-page ad record into one flat row."""
    props = {p.get("key") or p.get("label"): p.get("value") for p in ad.get("properties") or []}
    money = ad.get("money") or {}
    shop = ad.get("shop") or {}
    contact = ad.get("contactCard") or {}
    account = ad.get("account") or {}
    traits = account.get("traits") or {}
    if shop:
        seller_type = "dealer"
    elif ad.get("isMember") or traits.get("member"):
        seller_type = "member"
    else:
        seller_type = "private"
    images = ad.get("images") or {}
    image_ids = (images.get("meta") or images.get("ids")) if isinstance(images, dict) else None

    row = {
        "id": ad.get("id"),
        "url": AD_URL.format(slug=ad.get("slug")),
        "title": ad.get("title"),
        "price_lkr": parse_price(money.get("amount") or ""),
        "price_text": money.get("amount"),
        "negotiable": bool(money.get("negotiable")),
        "posted_at": ad.get("adDate"),
        "location": (ad.get("location") or {}).get("name"),
        "district": (ad.get("area") or {}).get("name"),
        "seller_name": shop.get("name") or contact.get("name"),
        "seller_type": seller_type,
        "seller_verified": bool(ad.get("isVerified") or traits.get("verified")),
        "views": (ad.get("statistics") or {}).get("views"),
        "image_count": len(image_ids) if image_ids else None,
        "description": ad.get("description"),
    }
    for key, value in props.items():
        if key == "mileage":
            row["mileage_km"] = parse_number(value)
        elif key in ("engine", "engine_capacity"):
            row["engine_cc"] = parse_number(value)
        else:
            row[key] = value
    return row


# ---------------------------------------------------------------- phases ---

def collect_candidates(days: float, min_interval: float, max_pages: int | None,
                       stop_after_stale_pages: int = 5) -> dict[str, dict]:
    """Phase 1: walk listing pages and return {ad_id: listing_record}."""
    candidates: dict[str, dict] = {}
    stale_pages = 0
    page = 1
    while True:
        if max_pages and page > max_pages:
            break
        html = fetch(LISTING_URL.format(page=page), min_interval)
        data = extract_initial_data(html)
        serp = data["serp"]["ads"]["data"]
        active_page = serp.get("paginationData", {}).get("activePage")
        if active_page != page:  # ran off the end: site redirects to page 1
            print(f"  page {page}: server returned page {active_page}, stopping")
            break
        ads = serp.get("ads") or []
        fresh = 0
        for ad in ads:
            age = listing_effective_age_days(ad)
            if age is None or age <= days:
                fresh += 1
                candidates.setdefault(ad["id"], ad)
        ages = [a for a in (listing_effective_age_days(x) for x in ads) if a is not None]
        oldest = max(ages) if ages else None
        print(f"  page {page}: {len(ads)} ads, {fresh} within window, oldest age {oldest:.1f} d"
              if oldest is not None else f"  page {page}: {len(ads)} ads, {fresh} within window")
        if not ads:
            break
        # Every page carries a promoted ad or two from outside the date order,
        # so a page counts as past the window when only a few ads are fresh.
        stale_pages = stale_pages + 1 if fresh <= 3 else 0
        if stale_pages >= stop_after_stale_pages:
            print(f"  {stop_after_stale_pages} consecutive pages past the window, stopping")
            break
        page += 1
    return candidates


def fetch_detail(slug: str, cache_dir: Path, min_interval: float) -> dict | None:
    cache_file = cache_dir / f"{slug}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    try:
        html = fetch(AD_URL.format(slug=slug), min_interval)
    except urllib.error.HTTPError as e:
        if e.code == 404:  # ad removed between listing and detail fetch
            return None
        raise
    data = extract_initial_data(html)
    ad = ((data.get("adDetail") or {}).get("data") or {}).get("ad")
    if not ad:
        return None
    cache_file.write_text(json.dumps(ad, ensure_ascii=False), encoding="utf-8")
    return ad


def collect_details(candidates: dict[str, dict], cache_dir: Path, workers: int,
                    min_interval: float, max_ads: int | None) -> list[dict]:
    slugs = [c["slug"] for c in candidates.values()]
    if max_ads:
        slugs = slugs[:max_ads]
    results: list[dict] = []
    failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_detail, s, cache_dir, min_interval): s for s in slugs}
        for fut in as_completed(futures):
            done += 1
            try:
                ad = fut.result()
                if ad:
                    results.append(ad)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  ! {futures[fut]}: {e}")
            if done % 100 == 0 or done == len(slugs):
                print(f"  details: {done}/{len(slugs)} fetched, {failed} failed")
    return results


# ------------------------------------------------------------------ main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=14, help="keep ads posted within this many days (default 14)")
    ap.add_argument("--out", default="ikman_cars_recent.csv", help="CSV output path")
    ap.add_argument("--cache-dir", default="ikman_cache", help="folder for cached ad detail JSON")
    ap.add_argument("--workers", type=int, default=4, help="parallel detail fetches (default 4)")
    ap.add_argument("--delay", type=float, default=0.4, help="minimum seconds between requests, global (default 0.4)")
    ap.add_argument("--max-pages", type=int, default=None, help="stop after this many listing pages (testing)")
    ap.add_argument("--max-ads", type=int, default=None, help="fetch at most this many detail pages (testing)")
    args = ap.parse_args()

    cutoff = datetime.now(COLOMBO_TZ) - timedelta(days=args.days)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out)
    out_jsonl = out_csv.with_suffix(".jsonl")

    print(f"Cutoff: ads posted on or after {cutoff:%Y-%m-%d %H:%M %z}")
    print("Phase 1: scanning listing pages")
    t0 = time.monotonic()
    candidates = collect_candidates(args.days, args.delay, args.max_pages)
    print(f"  {len(candidates)} candidate ads in {time.monotonic() - t0:.0f}s")

    print("Phase 2: fetching ad details")
    t0 = time.monotonic()
    ads = collect_details(candidates, cache_dir, args.workers, args.delay, args.max_ads)
    print(f"  {len(ads)} ad records in {time.monotonic() - t0:.0f}s")

    rows = []
    for ad in ads:
        posted = ad.get("adDate")
        try:
            posted_dt = datetime.fromisoformat(posted) if posted else None
        except ValueError:
            posted_dt = None
        if posted_dt is None or posted_dt >= cutoff:
            rows.append(flatten_ad(ad))
    rows.sort(key=lambda r: r.get("posted_at") or "", reverse=True)
    print(f"  {len(rows)} ads posted within the last {args.days:g} days "
          f"({len(ads) - len(rows)} older ads dropped, mostly bumped-up listings)")

    extra_cols = sorted({k for r in rows for k in r} - set(BASE_COLUMNS))
    columns = BASE_COLUMNS + extra_cols
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_csv} and {out_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
