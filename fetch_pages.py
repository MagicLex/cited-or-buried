"""F2 content pipeline: fetch every URL in serp_capture, compute structural MITs
+ a page embedding, and embed the queries. Writes:

  page_features   (url) @ fetched_at    structural SEO features + 384-d embedding
  query_features  (query_id) @ fetched_at   query metadata + 384-d embedding

Deterministic, no LLM. Runs on cited-content-env (fastembed). Fetches are threaded
with bounded retries; a page that will not fetch still gets a fetch_ok=0 row so the
FV can impute and rank still carries signal.

Run as a Hopsworks job. Deploy: python deploy_fetch.py ; run: hops job run fetch-pages
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

import hopsworks
from cited_features import (
    PAGE_DOC, QUERY_DOC, now_utc, page_features, page_text_for_embedding, query_row,
)
from encoder import DIM, embed

DATA = Path(__file__).resolve().parent / "data"
UA = "Mozilla/5.0 (compatible; cited-or-buried/013; +https://github.com/MagicLex/cited-or-buried)"
MAX_BYTES = 3_000_000  # skip / truncate monster pages
TIMEOUT = 25


def fetch(url: str) -> tuple[str, str, int]:
    """(url, html, status). status 0 = request failed; non-HTML -> empty html."""
    try:
        r = requests.get(
            url, timeout=TIMEOUT, stream=True,
            headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"},
        )
        status = r.status_code
        ctype = r.headers.get("Content-Type", "")
        if status != 200 or "html" not in ctype.lower():
            r.close()
            return url, "", status
        chunks, total = [], 0
        for chunk in r.iter_content(65536, decode_unicode=False):
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_BYTES:
                break
        r.close()
        html = b"".join(chunks).decode("utf-8", errors="replace")
        return url, html, status
    except requests.RequestException:
        return url, "", 0


def _process(url: str, asof) -> tuple[dict, str]:
    """Fetch + extract in the worker so the raw HTML is discarded here and never
    crosses the queue. Returns (small feature row, capped embedding text)."""
    url, html, status = fetch(url)
    return page_features(url, html, status, asof), page_text_for_embedding(html)


def build_pages(urls: list[str], workers: int) -> pd.DataFrame:
    asof = now_utc()
    rows, texts = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process, u, asof) for u in urls]
        for i, f in enumerate(as_completed(futs)):  # consume as they finish, no HL buffering
            row, text = f.result()
            rows.append(row)
            texts.append(text)
            if i % 100 == 0:
                print(f"fetched {i}/{len(urls)}", flush=True)
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c not in ("url",)]:
        if df[c].dtype == bool:
            df[c] = df[c].astype("int64")
        elif df[c].dtype == object:
            pass
        else:
            df[c] = df[c].astype("float64")
    print(f"embedding {len(texts)} pages ...", flush=True)
    vecs = embed(texts)
    df["page_embedding"] = [v.tolist() for v in vecs]
    df["fetched_at"] = asof
    print(f"pages: {len(df)}, fetch_ok {int(df['fetch_ok'].sum())}/{len(df)}", flush=True)
    return df


def build_queries(query_ids: set[str]) -> pd.DataFrame:
    asof = now_utc()
    text = {r["query_id"]: r["query"] for r in csv.DictReader((DATA / "queries.csv").open())}
    rows = [query_row(qid, text.get(qid, ""), asof) for qid in query_ids if text.get(qid)]
    df = pd.DataFrame(rows)
    vecs = embed(df["query"].tolist())
    df["query_embedding"] = [v.tolist() for v in vecs]
    print(f"queries: {len(df)}", flush=True)
    return df


def write_fgs(pages: pd.DataFrame, queries: pd.DataFrame) -> None:
    fs = hopsworks.login().get_feature_store()
    pfg = fs.get_or_create_feature_group(
        name="page_features", version=1,
        description="Structural SEO features + 384-d content embedding per URL",
        primary_key=["url"], event_time="fetched_at", online_enabled=True,
        statistics_config=False,
    )
    pfg.insert(pages)
    for name, desc in PAGE_DOC.items():
        if name in pages.columns:
            pfg.update_feature_description(name, desc)

    qfg = fs.get_or_create_feature_group(
        name="query_features", version=1,
        description="Query metadata + 384-d query embedding",
        primary_key=["query_id"], event_time="fetched_at", online_enabled=True,
        statistics_config=False,
    )
    qfg.insert(queries)
    for name, desc in QUERY_DOC.items():
        if name in queries.columns:
            qfg.update_feature_description(name, desc)
    print(f"wrote page_features {len(pages)}, query_features {len(queries)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="cap unique urls (debug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)

    fs = hopsworks.login().get_feature_store()
    serp = fs.get_feature_group("serp_capture", version=1).read()
    serp = serp.sort_values("captured_at").drop_duplicates(["query_id", "url"], keep="last")
    urls = sorted(serp["url"].unique().tolist())
    if args.limit:
        urls = urls[: args.limit]
    qids = set(serp["query_id"].unique().tolist())
    print(f"serp_capture: {len(serp)} rows -> {len(urls)} urls, {len(qids)} queries", flush=True)

    pages = build_pages(urls, args.workers)
    queries = build_queries(qids)
    assert len(pages.iloc[0]["page_embedding"]) == DIM, "embedding dim mismatch"
    if args.dry_run:
        print(pages.drop(columns=["page_embedding"]).head().to_string())
        print(queries.drop(columns=["query_embedding"]).head().to_string())
        return
    write_fgs(pages, queries)


if __name__ == "__main__":
    main()
