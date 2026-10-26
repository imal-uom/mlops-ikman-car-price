"""Train the production model, an Explainable Boosting Machine, and log it to MLflow."""

import hashlib
import json
import os
import subprocess
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from mlflow.models import infer_signature

from src import tracking
from src.config import ROOT, load_params
from src.features import TARGET, apply_spec, build_features, fit_spec


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def file_md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    params = load_params()
    m = params["model"]
    train_path = ROOT / params["data"]["processed_dir"] / "train.csv"
    df = build_features(pd.read_csv(train_path))
    spec = fit_spec(df, prior=m["typical_prior"])
    X = apply_spec(df, spec)
    y = np.log(df[TARGET].to_numpy(dtype=float))

    if not tracking.configure().startswith("http"):
        print("WARNING: DAGSHUB_OWNER / DAGSHUB_TOKEN not set, logging to a local MLflow store.")
    mlflow.set_experiment(m["experiment"])

    with mlflow.start_run(run_name="ebm-pipeline") as run:
        mlflow.set_tags({"git_sha": git_sha(), "source": "dvc-pipeline", "model_family": "ebm"})
        mlflow.log_params({**m["ebm"], "typical_prior": m["typical_prior"], "train_rows": len(df)})
        mlflow.log_params({"train_data_md5": file_md5(train_path)})

        started = time.time()
        # The EBM keeps its own validation slice (15% of train) for early stopping.
        model = ExplainableBoostingRegressor(**m["ebm"])
        model.fit(X, y)
        mlflow.log_metrics(
            {"n_terms": len(model.term_names_), "fit_seconds": round(time.time() - started, 1)}
        )

        out = ROOT / "models"
        out.mkdir(exist_ok=True)
        joblib.dump(model, out / "ebm.joblib")
        # newline="\n" everywhere: DVC hashes these files, and Windows would otherwise write CRLF
        with open(out / "spec.json", "w", newline="\n") as f:
            json.dump(spec, f, indent=1)

        sample = X.head(200)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=X.head(3),
            # cloudpickle: MLflow's default (skops) refuses classes outside scikit-learn
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        mlflow.log_artifact(str(out / "spec.json"))
        run_info = {"run_id": run.info.run_id, "model_uri": info.model_uri}
        with open(ROOT / "run_info.json", "w", newline="\n") as f:
            f.write(json.dumps(run_info, indent=2) + "\n")
        print(f"Trained an EBM with {len(model.term_names_)} terms on {len(df)} ads")
        print(f"Logged run {run.info.run_id} -> {info.model_uri}")


if __name__ == "__main__":
    main()
