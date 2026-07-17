"""Register the F2 job `fetch-pages` (content features) as a Hopsworks PYTHON job.

Runs on cited-content-env (fastembed). Ships cited_features.py + encoder.py +
queries.csv alongside. Run: hops job run fetch-pages
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "fetch-pages"
ENV_NAME = "cited-content-env"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
_data = str(_here.parent / "data")


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()
    base = f"hdfs:///Projects/{project.name}/{_rel}"

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"{base}/fetch_pages.py"
    cfg["environmentName"] = ENV_NAME
    cfg["defaultArgs"] = f"--data-dir {_data} --workers 32"
    cfg["files"] = ",".join(f"{base}/{f}" for f in ("cited_features.py", "encoder.py"))
    cfg["resourceConfig"]["memory"] = 12288
    cfg["resourceConfig"]["cores"] = 2

    job = ja.get_job(JOB_NAME)
    if job is not None:
        job.delete()
        print(f"deleted stale {JOB_NAME}", flush=True)
    job = ja.create_job(JOB_NAME, cfg)
    print(f"created job {job.name} on {ENV_NAME}", flush=True)


if __name__ == "__main__":
    main()
