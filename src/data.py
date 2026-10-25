"""Clean the scraped ads, validate them against a contract, and build train / eval / future."""

import pandas as pd
import pandera.pandas as pa

from src.config import ROOT, load_params
from src.features import RAW_INPUT, TARGET, clean

CONDITIONS = ["Brand New", "Reconditioned", "Used", "Import"]
FUELS = ["Petrol", "Diesel", "Hybrid", "Electric"]
TRANSMISSIONS = ["Automatic", "Manual", "Tiptronic"]
SELLERS = ["dealer", "private", "member"]

# The data contract on cleaned rows: every column the model reads, with type and allowed range.
CONTRACT = pa.DataFrameSchema(
    {
        "brand": pa.Column(str),
        "model": pa.Column(str),
        "edition": pa.Column(str, nullable=True),
        "fuel_type": pa.Column(str, pa.Check.isin(FUELS)),
        "transmission": pa.Column(str, pa.Check.isin(TRANSMISSIONS)),
        "body": pa.Column(str, nullable=True),
        "condition": pa.Column(str, pa.Check.isin(CONDITIONS)),
        "district": pa.Column(str),
        "seller_type": pa.Column(str, pa.Check.isin(SELLERS)),
        "model_year": pa.Column(int, pa.Check.in_range(1929, 2027)),
        "mileage_km": pa.Column(float, pa.Check.in_range(0, 500_000), nullable=True),
        "engine_cc": pa.Column(float, pa.Check.in_range(500, 7_000), nullable=True),
        "image_count": pa.Column(int, pa.Check.ge(0)),
        "negotiable": pa.Column(bool),
        "seller_verified": pa.Column(bool),
        "title": pa.Column(str, nullable=True),
        "description": pa.Column(str, nullable=True),
        "posted_at": pa.Column(str),
        TARGET: pa.Column(float, pa.Check.in_range(300_000, 300_000_000)),
    },
    coerce=True,
)
COLUMNS = RAW_INPUT + [TARGET]


def load_raw(path, p: dict) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str}, on_bad_lines="skip")
    return clean(raw, p["min_price"], p["max_price"])


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check cleaned rows against the contract. Raises pandera.errors.SchemaError on bad data."""
    return CONTRACT.validate(df[COLUMNS])


def split_by_day(df: pd.DataFrame, eval_frac: float, future_frac: float):
    """Chronological split by posting day: oldest = train, then eval, newest = future."""
    posted = pd.to_datetime(df["posted_at"], utc=True)
    df = df.assign(_day=posted.dt.floor("D")).sort_values("_day")
    eval_start = df["_day"].quantile(eval_frac, interpolation="higher")
    future_start = df["_day"].quantile(future_frac, interpolation="higher")
    train = df[df["_day"] < eval_start]
    evaluation = df[(df["_day"] >= eval_start) & (df["_day"] < future_start)]
    future = df[df["_day"] >= future_start]
    return tuple(part.drop(columns="_day") for part in (train, evaluation, future))


def build_sets(ads: pd.DataFrame, production: pd.DataFrame, p: dict):
    train, evaluation, future = split_by_day(ads, p["eval_frac"], p["future_frac"])
    if len(production) == 0:
        # First model: train on the oldest ads, judge it on the next ones.
        return train, evaluation, future
    # Retraining: the newest labelled production rows become the evaluation set,
    # and everything older (all ads + older production rows) is used for training.
    production = production.reset_index(drop=True)
    cut = int(len(production) * (1 - p["holdout_frac"]))
    older, holdout = production.iloc[:cut], production.iloc[cut:]
    history = pd.concat([train, evaluation], ignore_index=True)
    return pd.concat([history, older[COLUMNS]], ignore_index=True), holdout[COLUMNS], future


def main() -> None:
    p = load_params()["data"]
    ads = validate(load_raw(ROOT / p["raw"], p))
    production = pd.read_csv(ROOT / p["production"])
    if len(production):
        production = validate(production.sort_values("timestamp"))

    train, evaluation, future = build_sets(ads, production, p)
    out = ROOT / p["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    evaluation.to_csv(out / "eval.csv", index=False)
    future.to_csv(out / "future.csv", index=False)
    print(
        f"train={len(train)}  eval={len(evaluation)}  future={len(future)}  "
        f"production rows used={len(production)}"
    )
    print(
        f"median price: train={train[TARGET].median():,.0f}  "
        f"eval={evaluation[TARGET].median():,.0f}"
    )


if __name__ == "__main__":
    main()
