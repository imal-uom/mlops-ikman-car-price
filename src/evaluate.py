"""Score the trained model on the evaluation set, size the price range, explain with SHAP."""

import json

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src import tracking  # noqa: E402
from src.config import ROOT, load_params  # noqa: E402
from src.features import FEATURES, TARGET, build_features, encode  # noqa: E402
from src.metrics import score  # noqa: E402


def load_model(folder):
    booster = lgb.Booster(model_file=str(folder / "model.txt"))
    encoders = json.loads((folder / "encoders.json").read_text())["encoders"]
    return booster, encoders


def conformal_width(y_price, pred_log, alpha: float) -> float:
    """Split-conformal: the (1 - alpha) quantile of absolute log error, finite-sample corrected."""
    resid = np.abs(np.log(np.asarray(y_price, dtype=float)) - np.asarray(pred_log))
    rank = min(len(resid), int(np.ceil((len(resid) + 1) * (1 - alpha))))
    return float(np.sort(resid)[rank - 1])


def main() -> None:
    tracking.configure()
    params = load_params()
    alpha = params["evaluate"]["conformal_alpha"]
    df = build_features(pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv"))
    booster, encoders = load_model(ROOT / "models")
    X, _ = encode(df, encoders)
    y = df[TARGET].to_numpy(dtype=float)
    pred_log = booster.predict(X)

    metrics = score(y, pred_log)
    metrics["eval_rows"] = len(df)
    # newline="\n": DVC hashes these files, and Windows would otherwise write CRLF
    with open(ROOT / "eval_metrics.json", "w", newline="\n") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    q = conformal_width(y, pred_log, alpha)
    conformal = {"alpha": alpha, "q_log": q, "factor": float(np.exp(q))}
    with open(ROOT / "models" / "conformal.json", "w", newline="\n") as f:
        f.write(json.dumps(conformal, indent=2) + "\n")
    print(f"90% price range: prediction x/÷ {np.exp(q):.2f}")

    # SHAP: how much each feature pushed each prediction up or down (in log price).
    sample = X.sample(min(500, len(X)), random_state=0)
    explanation = shap.TreeExplainer(booster)(sample)
    explanation.feature_names = FEATURES
    plot_path = ROOT / "plots" / "shap_summary.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=14, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close("all")

    run_id = json.loads((ROOT / "run_info.json").read_text())["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{name}": value for name, value in metrics.items()})
        mlflow.log_metric("conformal_q_log", q)
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(ROOT / "models" / "conformal.json"))


if __name__ == "__main__":
    main()
