"""Create the cited_fv feature view: the training/serving read contract.

serp_capture (label + rank) LEFT JOIN page_features (structural SEO) on url,
LEFT JOIN query_features (query meta) on query_id. Embeddings stay in their FGs
(read directly for the cosine feature); the FV carries the structural contract.
The no-skew guarantee is cited_features.feature_row(), shared by trainer + serving.

Run: python build_fv.py
"""
from __future__ import annotations

import hopsworks
from cited_features import PAGE_NUMERIC


def get_or_create_fv(fs):
    serp = fs.get_feature_group("serp_capture", version=1)
    pages = fs.get_feature_group("page_features", version=1)
    queries = fs.get_feature_group("query_features", version=1)
    query = (
        serp.select(["query_id", "url", "rank", "cited"])
        .join(pages.select(["fetch_ok", *PAGE_NUMERIC]), on=["url"], join_type="left")
        .join(queries.select(["query_words", "query_type"]), on=["query_id"], join_type="left")
    )
    return fs.get_or_create_feature_view(
        name="cited_fv",
        version=1,
        query=query,
        labels=["cited"],
        description="SERP citation label + raw rank + structural SEO features + query meta",
    )


def main() -> None:
    fs = hopsworks.login().get_feature_store()
    fv = get_or_create_fv(fs)
    print(f"feature view {fv.name} v{fv.version} ready", flush=True)


if __name__ == "__main__":
    main()
