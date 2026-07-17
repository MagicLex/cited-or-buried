"""Clone python-feature-pipeline into cited-content-env and install fastembed.

The F2 content job and the serving predictor both embed text, so both run on this
env. Run once (idempotent-ish: skips create if it already exists).
Run: python deploy_env.py
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

ENV_NAME = "cited-content-env"
BASE = "python-feature-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
REQ = f"{_rel}/requirements-pipeline.txt"


def main() -> None:
    proj = hopsworks.login()
    ea = proj.get_environment_api()
    try:
        env = ea.get_environment(ENV_NAME)
    except Exception:
        env = None
    if env is None:
        print(f"creating {ENV_NAME} from {BASE} ...", flush=True)
        env = ea.create_environment(ENV_NAME, base_environment_name=BASE, await_creation=True)
        if env is None:
            env = ea.get_environment(ENV_NAME)
        print("created", flush=True)
    else:
        print(f"env {ENV_NAME} exists, reusing", flush=True)
    print(f"installing {REQ} ...", flush=True)
    env.install_requirements(REQ, await_installation=True)
    print(f"env {ENV_NAME} ready", flush=True)


if __name__ == "__main__":
    main()
