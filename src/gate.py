"""Quality gate: the newly trained model (challenger) must beat the @champion.

Both models are scored on the same evaluation set, which neither was trained on. The metric
is median absolute percentage error, so lower is better.
Exit code 0 = passed, 1 = failed. A markdown report is written for the pull request.
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src import tracking
from src.config import ROOT, load_params
from src.evaluate import load_model
from src.features import TARGET, build_features, encode
from src.metrics import GATE_KEYS, score


def load_champion(name: str):
    """The champion's trees plus the encoders it was trained with (logged as a run artifact)."""
    try:
        version = MlflowClient().get_model_version_by_alias(name, "champion")
        booster = mlflow.lightgbm.load_model(f"models:/{name}@champion")
        path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="encoders.json"
        )
        return booster, json.loads(Path(path).read_text())["encoders"]
    except MlflowException as err:
        print(f"No champion found ({err.error_code}); the first model passes automatically.")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="gate_report.md")
    args = parser.parse_args()

    tracking.configure()
    params = load_params()
    e = params["evaluate"]
    df = build_features(pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv"))
    y = df[TARGET].to_numpy(dtype=float)

    challenger, challenger_encoders = load_model(ROOT / "models")
    X_new, _ = encode(df, challenger_encoders)
    new = score(y, challenger.predict(X_new))

    champion = load_champion(params["model"]["name"])
    old = None
    if champion:
        booster, champion_encoders = champion
        X_old, _ = encode(df, champion_encoders)
        old = score(y, booster.predict(X_old))

    metric, delta = e["gate_metric"], e["gate_min_delta"]
    if old is None:
        passed = True
    elif e.get("gate_lower_is_better", True):
        passed = new[metric] <= old[metric] + delta
    else:
        passed = new[metric] >= old[metric] + delta

    lines = [
        f"## Model quality gate: {'PASSED' if passed else 'FAILED'}",
        "",
        f"Evaluation set: {len(df)} ads, median price {pd.Series(y).median():,.0f} LKR. "
        f"Rule: challenger `{metric}` must be at most champion + {delta} (lower is better).",
        "",
        "| Metric | Champion | Challenger |",
        "|---|---|---|",
    ]
    for key in GATE_KEYS:
        champ_value = f"{old[key]:.4f}" if old else "none yet"
        lines.append(f"| {key} | {champ_value} | {new[key]:.4f} |")
    report = "\n".join(lines) + "\n"
    (ROOT / args.report).write_text(report)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
