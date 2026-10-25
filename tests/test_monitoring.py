import json

from monitoring.export import parse_events
from tests.conftest import make_rows


def test_parse_events_joins_feedback():
    row = make_rows(1).drop(columns=["price_lkr", "seller_name"]).iloc[0].to_dict()
    row = {k: (None if v != v else v) for k, v in row.items()}
    lines = [
        'INFO:     127.0.0.1 - "POST /predict HTTP/1.1" 200 OK',  # non-JSON lines are skipped
        json.dumps(
            {
                "event": "prediction",
                "request_id": "a",
                "predicted_price_lkr": 9_000_000,
                "lower_lkr": 7_000_000,
                "upper_lkr": 11_000_000,
                "model_version": "v1",
                "timestamp": "2026-01-01T00:00:01",
                "features": row,
            }
        ),
        json.dumps(
            {
                "event": "prediction",
                "request_id": "b",
                "predicted_price_lkr": 12_000_000,
                "lower_lkr": 9_000_000,
                "upper_lkr": 15_000_000,
                "model_version": "v1",
                "timestamp": "2026-01-01T00:00:02",
                "features": row,
            }
        ),
        json.dumps({"event": "feedback", "request_id": "a", "actual_price_lkr": 9_500_000}),
    ]
    df = parse_events(lines)
    assert list(df["request_id"]) == ["a", "b"]
    assert df.loc[0, "price_lkr"] == 9_500_000
    assert df["price_lkr"].isna().sum() == 1  # b has no feedback yet
