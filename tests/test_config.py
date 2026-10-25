from src.config import ROOT, load_params


def test_params_file_exists():
    assert (ROOT / "params.yaml").exists()


def test_split_fractions_are_sensible():
    data = load_params()["data"]
    assert 0 < data["eval_frac"] < data["future_frac"] < 1
    assert 0 < data["holdout_frac"] < 1


def test_gate_metric_is_lower_is_better():
    e = load_params()["evaluate"]
    assert e["gate_metric"] == "median_ape"
    assert e["gate_lower_is_better"] is True
