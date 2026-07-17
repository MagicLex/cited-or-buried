"""Register the I1 deployment `citedscorer` (KServe, the GEO coach).

Ensures cited-serve-env (pandas-inference-pipeline + fastembed), uploads
predictor.py next to the champion model's files, deploys with platform inference
logging, starts it. Run: python deploy_serving.py
"""
from __future__ import annotations

from pathlib import Path

import hopsworks
from hsml.inference_logger import InferenceLogger
from hsml.resources import PredictorResources, Resources
from hsml.scaling_config import PredictorScalingConfig, ScaleMetric

DEPLOY_NAME = "citedscorer"
ENV_NAME = "cited-serve-env"
BASE = "pandas-inference-pipeline"
MODEL_NAME = "cited_ranker"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
REQ = f"{_rel}/requirements-pipeline.txt"


def ensure_env(project) -> None:
    ea = project.get_environment_api()
    try:
        env = ea.get_environment(ENV_NAME)
    except Exception:
        env = None
    if env is None:
        print(f"creating {ENV_NAME} from {BASE} ...", flush=True)
        env = ea.create_environment(ENV_NAME, base_environment_name=BASE, await_creation=True)
        if env is None:
            env = ea.get_environment(ENV_NAME)
        env.install_requirements(REQ, await_installation=True)
        print(f"{ENV_NAME} ready", flush=True)
    else:
        print(f"{ENV_NAME} exists", flush=True)


def main() -> None:
    project = hopsworks.login()
    ensure_env(project)

    mr = project.get_model_registry()
    model = max(mr.get_models(MODEL_NAME), key=lambda m: m.version)
    print(f"deploying {MODEL_NAME} v{model.version}", flush=True)

    script_dir = f"/Projects/{project.name}/Models/{model.name}/{model.version}/Files"
    project.get_dataset_api().upload(str(_here.parent / "predictor.py"), script_dir, overwrite=True)

    ms = project.get_model_serving()
    existing = ms.get_deployment(DEPLOY_NAME)
    if existing is not None:
        existing.stop(await_stopped=180)
        existing.delete()
        print("deleted stale deployment", flush=True)

    deployment = model.deploy(
        name=DEPLOY_NAME,
        description="cited-or-buried GEO coach: {query, urls} -> live fetch + feature + citation score",
        script_file=f"{script_dir}/predictor.py",
        resources=PredictorResources(
            requests=Resources(cores=1, memory=2048, gpus=0),
            limits=Resources(cores=2, memory=4096, gpus=0),
        ),
        scaling_configuration=PredictorScalingConfig(
            min_instances=1, max_instances=2,
            scale_metric=ScaleMetric.CONCURRENCY, target=16,
        ),
        environment=ENV_NAME,
        inference_logger=InferenceLogger(mode="ALL"),
    )
    deployment.start(await_running=600)
    print(f"running: {deployment.get_inference_url()}", flush=True)


if __name__ == "__main__":
    main()
