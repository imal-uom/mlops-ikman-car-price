"""Windows equivalent of `make dvc-auth`: point DVC at your DagsHub repo and store your login
in .dvc/config.local (never committed). Needs DAGSHUB_OWNER and DAGSHUB_TOKEN in the environment.

    .venv\\Scripts\\python.exe scripts\\dvc_auth.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DVC = ROOT / ".venv" / "Scripts" / "dvc.exe"


def main() -> None:
    owner, token = os.getenv("DAGSHUB_OWNER"), os.getenv("DAGSHUB_TOKEN")
    if not owner or not token:
        sys.exit("Set DAGSHUB_OWNER and DAGSHUB_TOKEN first.")
    repo = os.getenv("DAGSHUB_REPO", "mlops-ikman-car-price")
    url = f"https://dagshub.com/{owner}/{repo}.dvc"
    subprocess.run([str(DVC), "remote", "add", "-d", "-f", "origin", url], cwd=ROOT, check=True)
    for args in (["url", url], ["auth", "basic"], ["user", owner], ["password", token]):
        subprocess.run(
            [str(DVC), "remote", "modify", "origin", "--local", *args], cwd=ROOT, check=True
        )
    # Also keep the login in a git-ignored .env so MLflow logging works from any local shell.
    (ROOT / ".env").write_text(
        f"DAGSHUB_OWNER={owner}\nDAGSHUB_TOKEN={token}\nDAGSHUB_REPO={repo}\n"
    )
    print(f"DVC remote set to {url} (login stored in .dvc/config.local and .env, both git-ignored)")


if __name__ == "__main__":
    main()
