"""Windows equivalent of: make repro && dvc push && dvc status -c && gate && registry promote.

Run after scripts\\dvc_auth.py, with DAGSHUB_OWNER and DAGSHUB_TOKEN set so MLflow logs to DagsHub.

    .venv\\Scripts\\python.exe scripts\\run_pipeline.py [--force]
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
DVC = ROOT / ".venv" / "Scripts" / "dvc.exe"


def run(*cmd: str) -> None:
    print(f"\n=== {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault(
        "PYTHONUTF8", "1"
    )  # MLflow prints emoji; Windows consoles default to cp1252
    force = ["--force"] if "--force" in sys.argv else []
    run(str(DVC), "repro", *force)
    run(str(DVC), "push")
    run(str(DVC), "status", "-c")
    run(str(PY), "-m", "src.gate", "--report", "gate_report.md")
    run(str(PY), "-m", "src.registry", "promote")
    print("\n=== done")


if __name__ == "__main__":
    main()
