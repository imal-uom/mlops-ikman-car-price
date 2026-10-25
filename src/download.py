"""Refresh the raw data by scraping ikman.lk.

The raw CSV is tracked with DVC rather than produced by a pipeline stage, so this is run by
hand when a fresh month of ads is wanted:

    uv run python -m src.download --days 33
    uv run dvc add data/raw/ikman_cars_last_month.csv && uv run dvc push
"""

import argparse
import sys

from src.config import ROOT, load_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=33)
    args = parser.parse_args()
    out = ROOT / load_params()["data"]["raw"]
    out.parent.mkdir(parents=True, exist_ok=True)

    from src import ikman_scraper

    sys.argv = [
        "ikman_scraper",
        "--days",
        str(args.days),
        "--out",
        str(out),
        "--cache-dir",
        str(ROOT / "data" / "raw" / "ikman_cache"),
    ]
    ikman_scraper.main()
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
