import pandas as pd
import pandera.errors
import pytest

from src.data import build_sets, validate
from src.features import (
    MODEL_FEATURES,
    apply_spec,
    build_features,
    claim,
    clean,
    fit_spec,
    pricing_status,
    standardise_names,
)


def test_contract_accepts_valid_rows(rows):
    assert len(validate(standardise_names(rows))) == len(rows)


def test_contract_rejects_bad_year(rows):
    rows.loc[0, "model_year"] = 1800
    with pytest.raises(pandera.errors.SchemaError):
        validate(standardise_names(rows))


def test_contract_rejects_unknown_condition(rows):
    rows.loc[0, "condition"] = "Scrap"
    with pytest.raises(pandera.errors.SchemaError):
        validate(standardise_names(rows))


def test_lakh_price_is_converted(rows):
    rows.loc[0, "price_lkr"] = 85  # typed in lakhs
    cleaned = clean(rows, 300_000, 300_000_000)
    assert 8_500_000 in cleaned["price_lkr"].values


def test_deposit_price_is_quarantined():
    assert pricing_status("Minimum Down Payment - Rs. 1,300,000/-", 400_000) == "partial_payment"
    assert pricing_status("Monthly rental 85,000", 85_000) == "partial_payment"
    assert pricing_status("Lease transfer, 24 months remaining", 1_500_000) == "lease_transfer"
    assert pricing_status("Leasing can be arranged.", 8_000_000) == "full_price"
    assert (
        pricing_status("Full price 8.5M, down payment 2M", 8_500_000) == "payment_terms_mentioned"
    )


def test_negation_aware_flags():
    pattern = r"accident[\s-]*free|\bno\s+(?:major\s+)?accidents?"
    assert claim("100% accident free", pattern) == 1
    assert claim("not accident free", pattern) == 0


def test_wagon_r_trims_fold_into_the_model():
    df = standardise_names(
        pd.DataFrame([{"brand": "Suzuki", "model": "Wagon R FX", "edition": None}])
    )
    assert df.loc[0, "model"] == "wagon r"
    assert df.loc[0, "edition"] == "fx"


def test_spec_handles_unseen_values(rows):
    df = build_features(standardise_names(rows))
    spec = fit_spec(df)
    X = apply_spec(df, spec)
    assert list(X.columns) == MODEL_FEATURES
    new = rows.head(1).copy()
    new.loc[0, "brand"] = "Lamborghini"
    X_new = apply_spec(build_features(standardise_names(new)), spec)
    assert X_new.shape == (1, len(MODEL_FEATURES))
    # an unseen brand gets the global typical price and no model support
    assert X_new.loc[0, "typical_price_brand"] == pytest.approx(spec["typical"]["global_median"])
    assert X_new.loc[0, "model_support"] == 0


def test_typical_price_is_shrunk_towards_the_brand(rows):
    df = build_features(standardise_names(rows))
    spec = fit_spec(df, prior=5)
    X = apply_spec(df, spec)
    row = X.iloc[0]
    brand, model = row["brand"], row["model"]
    brand_median, brand_n = spec["typical"]["brand"][brand]
    model_median, model_n = spec["typical"]["model"][f"{brand}|{model}"]
    glob = spec["typical"]["global_median"]
    brand_typ = (brand_median * brand_n + glob * 5) / (brand_n + 5)
    assert row["typical_price_brand"] == pytest.approx(brand_typ)
    assert row["typical_price_model"] == pytest.approx(
        (model_median * model_n + brand_typ * 5) / (model_n + 5)
    )
    assert row["model_support"] == model_n


def test_first_model_splits_chronologically(rows):
    p = {"eval_frac": 0.7, "future_frac": 0.9, "holdout_frac": 0.3}
    train, evaluation, future = build_sets(rows, pd.DataFrame(), p)
    assert len(train) + len(evaluation) + len(future) == len(rows)
    assert train["posted_at"].max() <= evaluation["posted_at"].min()
    assert evaluation["posted_at"].max() <= future["posted_at"].min()


def test_retraining_evaluates_on_newest_production_rows(rows):
    p = {"eval_frac": 0.7, "future_frac": 0.9, "holdout_frac": 0.3}
    production = validate(standardise_names(rows.head(100)))
    train, evaluation, _ = build_sets(rows, production, p)
    assert len(evaluation) == 30  # newest 30% of production rows
    assert len(train) > 100
