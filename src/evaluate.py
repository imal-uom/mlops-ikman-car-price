"""Score the trained model on the evaluation set, size the price range, plot what it learned."""

import json

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import tracking  # noqa: E402
from src.config import ROOT, load_params  # noqa: E402
from src.features import TARGET, apply_spec, build_features  # noqa: E402
from src.metrics import score  # noqa: E402


def load_model(folder):
    model = joblib.load(folder / "ebm.joblib")
    spec = json.loads((folder / "spec.json").read_text())
    return model, spec


def conformal_width(y_price, pred_log, alpha: float) -> float:
    """Split-conformal: the (1 - alpha) quantile of absolute log error, finite-sample corrected."""
    resid = np.abs(np.log(np.asarray(y_price, dtype=float)) - np.asarray(pred_log))
    rank = min(len(resid), int(np.ceil((len(resid) + 1) * (1 - alpha))))
    return float(np.sort(resid)[rank - 1])


def term_importance_plot(model, path, top: int = 20) -> None:
    """The EBM's global explanation: mean |contribution| of every term to log price."""
    data = model.explain_global().data()
    importance = pd.Series(data["scores"], index=data["names"]).sort_values().tail(top)
    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="#177e89")
    plt.xlabel("mean |term contribution| to log price")
    plt.title(f"EBM term importance ({top} of {len(data['names'])} terms)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")


def main() -> None:
    tracking.configure()
    params = load_params()
    alpha = params["evaluate"]["conformal_alpha"]
    df = build_features(pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv"))
    model, spec = load_model(ROOT / "models")
    X = apply_spec(df, spec)
    y = df[TARGET].to_numpy(dtype=float)
    pred_log = model.predict(X)

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

    plot_path = ROOT / "plots" / "term_importance.png"
    plot_path.parent.mkdir(exist_ok=True)
    term_importance_plot(model, plot_path)

    run_id = json.loads((ROOT / "run_info.json").read_text())["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{name}": value for name, value in metrics.items()})
        mlflow.log_metric("conformal_q_log", q)
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(ROOT / "models" / "conformal.json"))


if __name__ == "__main__":
    main()
