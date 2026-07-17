"""Batch inference for the gallery: score every (query, url) in the corpus with
the registered ranker and write geo_scored, so the app opens instant.

  geo_scored (query_id, url) @ scored_at
    rank, cited (actual), model_score, title, query, page structural echo

The app reads this to draw GOOGLE RANKS vs AI CITES vs MODEL PREDICTS per query.
Runs on pandas-training-pipeline (sklearn). Deploy: python deploy_score.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

import hopsworks
from cited_features import FEATURES, feature_row, now_utc

MODEL_NAME = "cited_ranker"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    models = mr.get_models(MODEL_NAME)
    model = max(models, key=lambda m: m.version)  # champion = latest (register-all, serve-best)
    mdir = model.download()
    clf = joblib.load(Path(mdir) / "model.joblib")
    print(f"scoring with {MODEL_NAME} v{model.version}", flush=True)

    serp = fs.get_feature_group("serp_capture", version=1).read()
    serp = serp.sort_values("captured_at").drop_duplicates(["query_id", "url"], keep="last")
    pages = fs.get_feature_group("page_features", version=1).read().sort_values("fetched_at").drop_duplicates("url", keep="last")
    queries = fs.get_feature_group("query_features", version=1).read().sort_values("fetched_at").drop_duplicates("query_id", keep="last")
    df = serp.merge(pages, on="url", how="left").merge(queries, on="query_id", how="left")

    X = pd.DataFrame([feature_row(r) for r in df.to_dict("records")], columns=list(FEATURES)).astype("float32")
    df["model_score"] = clf.predict_proba(X)[:, 1]

    keep = ["query_id", "query", "url", "rank", "title", "cited", "model_score",
            "fetch_ok", "word_count", "intro_words", "n_h2", "n_list_items",
            "has_schema_org", "has_faq_schema", "freshness_days"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["cited"] = out["cited"].astype(int)
    out["scored_at"] = now_utc()
    print(f"scored {len(out)} rows, {out['query_id'].nunique()} queries", flush=True)
    if args.dry_run:
        print(out.head(10).to_string())
        return

    fg = fs.get_or_create_feature_group(
        name="geo_scored", version=1,
        description="Batch-scored corpus: raw rank, actual citation, model citation score",
        primary_key=["query_id", "url"], event_time="scored_at",
        online_enabled=False, statistics_config=False,
    )
    fg.insert(out)
    print(f"wrote geo_scored: {len(out)} rows", flush=True)


if __name__ == "__main__":
    main()
