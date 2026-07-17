"""F1 insert side: flatten a geo-serp-capture workflow output into serp_capture.

The fleet (capture_workflow.js) runs in the session and writes its result to a
JSON file; this reads that file, flattens to one row per (query_id, url) with the
cited label, snapshots to data/serp/, and inserts the serp_capture FG.

  serp_capture  (query_id, url) @ captured_at  -- rank + cited label, the SEO proxy

Offline FG inserts APPEND across runs (unstarred scar); dedupe at read in
training. Run: python capture_serp.py --from <workflow_output.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import hopsworks
from cited_features import SERP_DOC, now_utc

DATA = Path(__file__).resolve().parent / "data"


def _result_list(payload) -> list[dict]:
    """Accept either the raw workflow result array or the wrapped
    {result: [...]} task-output shape."""
    if isinstance(payload, dict) and "result" in payload:
        return payload["result"]
    if isinstance(payload, list):
        return payload
    raise ValueError("unrecognized workflow output shape")


def flatten(items: list[dict], captured) -> pd.DataFrame:
    rows = []
    for it in items:
        if not it:
            continue
        qid = str(it.get("query_id", "")).strip()
        if not qid:
            continue
        cited = {u for u in it.get("cited_urls", []) if isinstance(u, str)}
        ranked = it.get("ranked_urls", []) or []
        seen = set()
        for r in ranked:
            url = (r.get("url") or "").strip()
            if not url or url in seen:  # a page appears once per query at its best rank
                continue
            seen.add(url)
            rows.append({
                "query_id": qid,
                "url": url,
                "rank": int(r.get("rank") or 0),
                "title": (r.get("title") or "")[:300],
                "cited": 1 if url in cited else 0,
                "captured_at": captured,
            })
    df = pd.DataFrame(rows, columns=["query_id", "url", "rank", "title", "cited", "captured_at"])
    return df


def write_fg(df: pd.DataFrame) -> None:
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="serp_capture",
        version=1,
        description="Per-query search results with raw rank and the AI-citation label",
        primary_key=["query_id", "url"],
        event_time="captured_at",
        online_enabled=False,
        statistics_config=False,
    )
    fg.insert(df)
    for name, desc in SERP_DOC.items():
        if name in df.columns:
            fg.update_feature_description(name, desc)
    print(f"inserted serp_capture: {len(df)} rows, "
          f"{df['query_id'].nunique()} queries, {df['url'].nunique()} urls, "
          f"cited rate {df['cited'].mean():.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="workflow output JSON path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = json.loads(Path(args.src).read_text())
    items = _result_list(payload)
    captured = now_utc()
    df = flatten(items, captured)
    print(f"flattened {len(df)} rows from {len(items)} queries; "
          f"cited rate {df['cited'].mean():.3f}; rank range {df['rank'].min()}-{df['rank'].max()}", flush=True)

    (DATA / "serp").mkdir(parents=True, exist_ok=True)
    snap = DATA / "serp" / f"serp_{captured.strftime('%Y%m%dT%H%M%S')}.parquet"
    df.to_parquet(snap, index=False)
    print(f"snapshot -> {snap}", flush=True)

    if args.dry_run:
        print(df.head(12).to_string())
        return
    write_fg(df)


if __name__ == "__main__":
    main()
