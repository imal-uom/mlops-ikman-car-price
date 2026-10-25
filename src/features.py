"""Cleaning rules and feature engineering shared by training and the API.

Only pandas and numpy are imported here, so the serving container can use this module
without the training dependencies. Everything mirrors the analysis notebook.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

TARGET = "price_lkr"
CAT_FEATURES = [
    "brand",
    "model",
    "edition",
    "fuel_type",
    "transmission",
    "body",
    "condition",
    "district",
    "seller_type",
]
FLAG_FEATURES = [
    "flag_unregistered",
    "flag_first_owner",
    "flag_finance",
    "flag_auction_sheet",
    "flag_accident_free",
    "flag_service_records",
    "flag_mint",
    "flag_agent_maintained",
]
NUM_FEATURES = [
    "model_year",
    "age",
    "mileage_km",
    "km_per_year",
    "engine_cc",
    "image_count",
    "negotiable",
    "seller_verified",
    *FLAG_FEATURES,
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# The columns an ad arrives with (the CSV, the API request, the replayer). Features are derived.
RAW_INPUT = [
    "brand",
    "model",
    "edition",
    "fuel_type",
    "transmission",
    "body",
    "condition",
    "district",
    "seller_type",
    "model_year",
    "mileage_km",
    "engine_cc",
    "image_count",
    "negotiable",
    "seller_verified",
    "title",
    "description",
    "posted_at",
]

# ----------------------------------------------------------------- price-basis screen ---
MONEY = (
    r"(?:rs\.?|lkr)?\s*(\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(millions?|mil\b|mn\b|m\b|lakhs?|lak\b|lacs?)?"
)
PART_LABEL = (
    r"(?:down\s*payment|initial\s*payment|deposit|monthly\s*(?:rental|instal?lments?|payment)"
    r"|per\s*month)"
)
RENTAL = r"\b(?:for\s+rent|for\s+hire|rent\s+a\s+car|daily\s+rental|rental\s+basis|per\s+day)\b"
LEASE_TRANSFER = r"\b(?:lease|leasing)\s*(?:transfer|takeover|take\s*over)\b|\bbalance\s+lease\b"
DROP_STATUS = ["rental_offer", "partial_payment", "lease_transfer"]


def parse_amounts(text: str, label: str) -> list[float]:
    amounts = []
    for m in re.finditer(label + r"[^\d\n]{0,20}?" + MONEY, text, re.I):
        n = float(re.sub(r"[ ,]", "", m.group(1)))
        unit = (m.group(2) or "").lower()
        if unit.startswith(("million", "mil", "mn")) or unit == "m":
            n *= 1_000_000
        elif unit.startswith(("lakh", "lak", "lac")):
            n *= 100_000
        if n >= 10_000 or unit:
            amounts.append(n)
    return amounts


def pricing_status(text: str, price: float) -> str:
    """Is the advertised number a full sale price, or a deposit, instalment or rental?"""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    if re.search(RENTAL, t) and not re.search(r"\bnot\s+for\s+(?:rent|hire)", t):
        return "rental_offer"
    parts = parse_amounts(t, PART_LABEL)
    if any(abs(a - price) <= max(1_000, 0.01 * price) or a >= 0.9 * price for a in parts):
        return "partial_payment"
    if re.search(LEASE_TRANSFER, t):
        return "lease_transfer"
    if re.search(PART_LABEL, t):
        return "payment_terms_mentioned"
    return "full_price"


# ------------------------------------------------------------------- name cleaning ---
MODEL_ALIASES = {
    "crv": "cr v",
    "chr": "c hr",
    "wagonr": "wagon r",
    "rav4": "rav 4",
    "hrv": "hr v",
    "xtrail": "x trail",
    "land cruiser prado": "prado",
}


def norm_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[-_/|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


def standardise_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case names, merge known spellings, fold Wagon R trims into the model's edition."""
    df = df.copy()
    df["brand"] = df["brand"].map(norm_name)
    df["model"] = df["model"].map(norm_name).map(lambda m: MODEL_ALIASES.get(m, m))
    df["edition"] = df["edition"].map(norm_name)
    edition = df["edition"].fillna("").astype(str)  # an all-missing column is float
    stingray = df["model"].eq("wagon r") & edition.str.contains("stingray", case=False)
    df.loc[stingray, "model"] = "wagon r stingray"
    trim = df["model"].fillna("").astype(str).str.extract(r"^wagon r (fx|fz|zx)$")[0]
    has_trim = trim.notna()
    if has_trim.any():
        old = edition[has_trim].str.strip()
        df["edition"] = df["edition"].astype(object)
        df.loc[has_trim, "edition"] = np.where(
            old.eq(""), trim[has_trim], trim[has_trim] + " " + old
        )
        df.loc[has_trim, "model"] = "wagon r"
    return df


def clean(raw: pd.DataFrame, min_price: int, max_price: int) -> pd.DataFrame:
    """Price fixes, price-basis screen, duplicates, junk values, name standardisation."""
    df = raw.copy()
    lakh = (df[TARGET] < 1_000) & (df[TARGET] * 100_000).between(min_price, max_price)
    df.loc[lakh, TARGET] = df.loc[lakh, TARGET] * 100_000
    df = df[df[TARGET].between(min_price, max_price)]
    text = df["title"].fillna("") + "\n" + df["description"].fillna("")
    status = pd.Series(
        [pricing_status(t, p) for t, p in zip(text, df[TARGET], strict=True)], index=df.index
    )
    df = df[~status.isin(DROP_STATUS)]
    df = df.sort_values("posted_at", ascending=False).drop_duplicates(
        ["title", TARGET, "seller_name", "model_year", "mileage_km"]
    )
    bad_cc = ~df["engine_cc"].between(500, 7000) | df["fuel_type"].eq("Electric")
    df.loc[bad_cc, "engine_cc"] = np.nan
    df.loc[df["mileage_km"] > 500_000, "mileage_km"] = np.nan
    df = df[df["posted_at"].notna()].copy()
    return standardise_names(df)


# ---------------------------------------------------------------------- features ---
NEGATION = r"\b(?:not|no|never|without|isn'?t|non)\s+(?:\w+\s+){0,2}$"
CLAIM_PATTERNS = {
    "flag_unregistered": r"\bunregist",
    "flag_first_owner": r"\b(?:1st|first|single|one)\s+owner",
    "flag_finance": r"\bleas|" + PART_LABEL,
    "flag_auction_sheet": r"auction\s*sheet|\bgrade\s*(?:4|4\.5|5)\b",
    "flag_accident_free": r"accident[\s-]*free|\bno\s+(?:major\s+)?accidents?",
    "flag_service_records": r"service\s+(?:history|records?)|maintenance\s+records?",
    "flag_mint": r"\b(?:mint|immaculate|excellent|showroom)\s+condition",
    "flag_agent_maintained": r"\b(?:agent|company)\s*(?:maintained|serviced?)",
}


def claim(text: str, pattern: str) -> int:
    """1 if the pattern occurs and is not negated in the 40 characters before it."""
    for m in re.finditer(pattern, text):
        if not re.search(NEGATION, text[max(0, m.start() - 40) : m.start()]):
            return 1
    return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns from the raw ad columns. Identical for training and inference."""
    out = df.copy()
    posted = pd.to_datetime(out["posted_at"], utc=True, errors="coerce")
    year = posted.dt.tz_convert("Asia/Colombo").dt.year
    year = year.fillna(pd.Timestamp.now(tz="Asia/Colombo").year)
    out["age"] = (year - out["model_year"]).clip(lower=0)
    out["km_per_year"] = out["mileage_km"] / (out["age"] + 1)
    text = (out["title"].fillna("") + " " + out["description"].fillna("")).map(
        lambda s: unicodedata.normalize("NFKC", str(s)).lower()
    )
    for name, pattern in CLAIM_PATTERNS.items():
        out[name] = [claim(t, pattern) for t in text]
    out["negotiable"] = out["negotiable"].astype(bool).astype(int)
    out["seller_verified"] = out["seller_verified"].astype(bool).astype(int)
    return out


def encode(df: pd.DataFrame, encoders: dict | None = None):
    """Categorical text -> integer codes for LightGBM. Fit when encoders is None, else apply.

    Rare models (< 5 ads) and editions (< 10) are grouped as "other" when fitting; at inference
    an unseen value maps to "other" if that bucket exists, otherwise to missing.
    """
    X = df[FEATURES].copy()
    fitting = encoders is None
    encoders = {} if fitting else dict(encoders)
    for c in CAT_FEATURES:
        if fitting:
            counts = X[c].value_counts()
            min_n = {"model": 5, "edition": 10}.get(c, 1)
            keep = counts[counts >= min_n].index
            X[c] = X[c].where(X[c].isin(keep) | X[c].isna(), "other")
            encoders[c] = sorted(X[c].dropna().unique().tolist())
        cats = encoders[c]
        fallback = "other" if "other" in cats else np.nan
        X[c] = X[c].where(X[c].isin(cats) | X[c].isna(), fallback)
        codes = pd.Categorical(X[c], categories=cats).codes.astype("float32")
        X[c] = np.where(codes < 0, np.nan, codes)
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X, encoders


def prepare_inputs(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Raw ad columns -> model matrix, for serving. Missing optional columns are filled."""
    frame = df.copy()
    for c in RAW_INPUT:
        if c not in frame.columns:
            frame[c] = np.nan
    frame["posted_at"] = frame["posted_at"].fillna(pd.Timestamp.now(tz="Asia/Colombo").isoformat())
    frame["image_count"] = frame["image_count"].fillna(5)
    frame["negotiable"] = frame["negotiable"].fillna(True)
    frame["seller_verified"] = frame["seller_verified"].fillna(False)
    X, _ = encode(build_features(standardise_names(frame)), encoders)
    return X
