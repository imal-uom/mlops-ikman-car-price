"""Simulate production traffic against the live API, then send the real prices as feedback.

Modes:
  normal      random ads from the training period: what "no drift" looks like
  future      the newest ads, in date order: real drift
  engine-bug  like future, but an upstream system now sends engine_cc in litres (1.5, not 1500)

Example:
  uv run python replayer/replay.py https://<app-url> --mode future --rows 700
"""

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TARGET = "price_lkr"


def load_rows(mode: str, rows: int, start: int) -> pd.DataFrame:
    processed = ROOT / "data" / "processed"
    if mode == "normal":
        df = pd.read_csv(processed / "train.csv")
        df = df.sample(min(rows, len(df)), random_state=7)
    else:
        df = pd.read_csv(processed / "future.csv").iloc[start : start + rows]
    if mode == "engine-bug":
        df = df.assign(engine_cc=df["engine_cc"] / 1000)
    return df


def to_payload(row: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="base URL of the API, without a trailing slash")
    parser.add_argument("--mode", choices=["normal", "future", "engine-bug"], default="future")
    parser.add_argument("--rows", type=int, default=700)
    parser.add_argument("--start", type=int, default=0, help="first future row to send")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--feedback-delay",
        type=float,
        default=20,
        help="seconds to wait before sending the real prices",
    )
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args()

    df = load_rows(args.mode, args.rows, args.start)
    prices = df[TARGET].astype(float).tolist()
    payloads = [to_payload(r) for r in df.drop(columns=[TARGET]).to_dict(orient="records")]
    client = httpx.Client(base_url=args.url, timeout=30)

    def send(i: int):
        response = client.post("/predict", json=payloads[i])
        return i, response.status_code, response.json() if response.status_code == 200 else None

    print(f"Sending {len(payloads)} '{args.mode}' requests to {args.url} ...")
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(send, range(len(payloads))))
    ok = [(i, body) for i, status, body in results if status == 200]
    rejected = sum(1 for _, status, _ in results if status == 422)
    failed = len(results) - len(ok) - rejected
    print(
        f"done in {time.time() - started:.0f}s: {len(ok)} ok, {rejected} rejected by the "
        f"contract (422), {failed} other errors"
    )
    if ok:
        mean_pred = sum(body["predicted_price_lkr"] for _, body in ok) / len(ok)
        mean_actual = sum(prices[i] for i, _ in ok) / len(ok)
        inside = sum(body["lower_lkr"] <= prices[i] <= body["upper_lkr"] for i, body in ok)
        print(
            f"mean predicted price {mean_pred:,.0f}, mean actual {mean_actual:,.0f}, "
            f"{inside / len(ok):.0%} of actual prices inside the 90% range"
        )

    if args.no_feedback or not ok:
        return
    print(f"Waiting {args.feedback_delay:.0f}s before sending feedback (prices arrive later)...")
    time.sleep(args.feedback_delay)

    def send_feedback(item):
        i, body = item
        client.post(
            "/feedback", json={"request_id": body["request_id"], "actual_price_lkr": prices[i]}
        )

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(send_feedback, ok))
    print(f"Sent {len(ok)} feedback events.")
    sys.exit(0)


if __name__ == "__main__":
    main()
