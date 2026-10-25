"""Train the production LightGBM model and log it to MLflow."""

import hashlib
import json
import os
import subprocess

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from src import tracking
from src.config import ROOT, load_params
from src.features import CAT_FEATURES, FEATURES, TARGET, build_features, encode


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
    X, encoders = encode(df)
    y = np.log(df[TARGET].to_numpy(dtype=float))
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=m["validation_frac"], random_state=m["lightgbm"]["random_state"]
    )

    if not tracking.configure().startswith("http"):
        print("WARNING: DAGSHUB_OWNER / DAGSHUB_TOKEN not set, logging to a local MLflow store.")
    mlflow.set_experiment(m["experiment"])

    with mlflow.start_run(run_name="lightgbm-pipeline") as run:
        mlflow.set_tags({"git_sha": git_sha(), "source": "dvc-pipeline"})
        mlflow.log_params({**m["lightgbm"], "train_rows": len(df)})
        mlflow.log_params({"train_data_md5": file_md5(train_path)})

        model = lgb.LGBMRegressor(**m["lightgbm"])
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=CAT_FEATURES,
            callbacks=[lgb.early_stopping(m["early_stopping_rounds"], verbose=False)],
        )
        mlflow.log_metric("best_iteration", model.best_iteration_)

        out = ROOT / "models"
        out.mkdir(exist_ok=True)
        model.booster_.save_model(str(out / "model.txt"))  # LightGBM's own text format
        spec = {"features": FEATURES, "categorical": CAT_FEATURES, "encoders": encoders}
        # newline="\n" everywhere: DVC hashes these files, and Windows would otherwise write CRLF
        with open(out / "encoders.json", "w", newline="\n") as f:
            json.dump(spec, f, indent=1)

        sample = X.head(200)
        info = mlflow.lightgbm.log_model(
            model.booster_,
            name="model",
            signature=infer_signature(sample, model.booster_.predict(sample)),
            input_example=X.head(3),
        )
        mlflow.log_artifact(str(out / "encoders.json"))
        run_info = {"run_id": run.info.run_id, "model_uri": info.model_uri}
        with open(ROOT / "run_info.json", "w", newline="\n") as f:
            f.write(json.dumps(run_info, indent=2) + "\n")
        print(f"Trained {model.best_iteration_} trees on {len(df)} ads")
        print(f"Logged run {run.info.run_id} -> {info.model_uri}")


if __name__ == "__main__":
    main()
