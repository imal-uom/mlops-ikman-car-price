"""Point MLflow at your DagsHub repository when DAGSHUB_OWNER and DAGSHUB_TOKEN are set.

Codespaces secrets and GitHub Actions both provide these values, so every script
calls configure() before talking to MLflow. Without them, MLflow logs to a local store.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file() -> None:
    """Read KEY=value lines from a git-ignored .env (written by scripts/dvc_auth.py) when the
    variables are not already set. Lets the pipeline run from any local shell."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def configure() -> str:
    load_env_file()
    owner, token = os.getenv("DAGSHUB_OWNER"), os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO", "mlops-ikman-car-price")
    if owner and token and not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{owner}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    return os.getenv("MLFLOW_TRACKING_URI", "")
