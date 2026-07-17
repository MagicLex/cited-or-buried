"""Register the T1 job `train-ranker` as a Hopsworks PYTHON job.

Runs on pandas-training-pipeline (sklearn + matplotlib). Ships cited_features.py +
encoder.py (encoder is bundled into the model artifact for the predictor, not
imported here). Run: hops job run train-ranker
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "train-ranker"
ENV_NAME = "pandas-training-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()
    base = f"hdfs:///Projects/{project.name}/{_rel}"

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"{base}/train_ranker.py"
    cfg["environmentName"] = ENV_NAME
    cfg["files"] = ",".join(f"{base}/{f}" for f in ("cited_features.py", "encoder.py"))
    cfg["resourceConfig"]["memory"] = 8192

    job = ja.get_job(JOB_NAME)
    if job is not None:
        job.delete()
        print(f"deleted stale {JOB_NAME}", flush=True)
    job = ja.create_job(JOB_NAME, cfg)
    print(f"created job {job.name} on {ENV_NAME}", flush=True)


if __name__ == "__main__":
    main()
