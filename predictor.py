"""I1: KServe predictor for the citation ranker (the GEO coach).

Request: {"instances": [{"query": "...", "urls": ["https://..."]}]}. For each URL
the predictor fetches the live page, computes the SAME structural + semantic
features training used (cited_features.feature_row, bundled in the artifact),
embeds the query with the SAME encoder, and scores citation probability. No web
search and no SERP key: the caller supplies the query and the candidate URLs.

Returns per-URL {url, cited_prob, reasons} where reasons are plain-word,
number-carrying explanations derived from the feature values.
"""

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

UA = "Mozilla/5.0 (compatible; cited-or-buried/013; +https://github.com/MagicLex/cited-or-buried)"
TIMEOUT = 20
MAX_BYTES = 3_000_000


def model_files_root() -> str:
    for root in (os.environ.get("MODEL_FILES_PATH"), os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root and glob.glob(f"{root}/**/model.joblib", recursive=True):
            return os.path.dirname(glob.glob(f"{root}/**/model.joblib", recursive=True)[0])
    raise FileNotFoundError("model.joblib not found under the model/artifact mounts")


def _fetch(url: str):
    try:
        r = requests.get(url, timeout=TIMEOUT, stream=True,
                         headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})
        status = r.status_code
        if status != 200 or "html" not in r.headers.get("Content-Type", "").lower():
            r.close()
            return url, "", status
        chunks, total = [], 0
        for c in r.iter_content(65536):
            chunks.append(c); total += len(c)
            if total >= MAX_BYTES:
                break
        r.close()
        return url, b"".join(chunks).decode("utf-8", errors="replace"), status
    except requests.RequestException:
        return url, "", 0


def reasons(feat: dict, page: dict) -> list:
    """Plain-word, number-carrying explanations. Positives first, then the fixes."""
    out = []
    if feat["cosine"] >= 0.75:
        out.append(f"strong topic match (semantic {feat['cosine']:.2f})")
    elif feat["cosine"] < 0.55:
        out.append(f"weak topic match (semantic {feat['cosine']:.2f}) — page may be off-intent")
    if not page.get("fetch_ok"):
        out.append("page did not fetch (blocked or JS-only) — the engine likely cannot read it either")
        return out
    intro, words = int(page["intro_words"]), int(page["word_count"])
    if intro <= 60:
        out.append(f"answer up top ({intro} words before the first heading)")
    elif intro >= 200:
        out.append(f"answer buried ({intro} words of intro before the first heading)")
    if page["has_schema_org"]:
        out.append("carries schema.org markup" + (" incl. FAQ/QA" if page["has_faq_schema"] else ""))
    else:
        out.append("no schema.org markup")
    if page["n_h2"] >= 4 or page["n_list_items"] >= 10:
        out.append(f"well structured ({int(page['n_h2'])} H2s, {int(page['n_list_items'])} list items)")
    else:
        out.append("thin structure (few headings/lists)")
    fresh = int(page["freshness_days"])
    if 0 <= fresh <= 365:
        out.append(f"fresh (~{fresh} days old)")
    elif fresh > 1095:
        out.append(f"stale (~{fresh // 365} years old)")
    if words < 300:
        out.append(f"very short ({words} words)")
    return out


class Predict:
    def __init__(self):
        import joblib

        root = model_files_root()
        sys.path.insert(0, root)
        import cited_features  # bundled
        import encoder  # bundled

        self.cf = cited_features
        self.encoder = encoder
        self.model = joblib.load(f"{root}/model.joblib")
        self.encoder.embed(["warmup"])  # load the ONNX model once
        print("cited_ranker loaded, encoder warm", flush=True)

    def predict(self, inputs):
        insts = inputs.get("instances", inputs) if isinstance(inputs, dict) else inputs
        out = []
        for inst in insts:
            query = inst.get("query", "")
            urls = inst.get("urls") or ([inst["url"]] if inst.get("url") else [])
            q_emb = self.encoder.embed([query])[0].tolist()
            q_words = len(query.split())
            q_type = self.cf.query_row("0", query, self.cf.now_utc())["query_type"]

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(urls)))) as ex:
                fetched = list(ex.map(_fetch, urls))
            scored = []
            for rank, (url, html, status) in enumerate(fetched, start=1):
                now = self.cf.now_utc()
                page = self.cf.page_features(url, html, status, now)
                p_emb = self.encoder.embed([self.cf.page_text_for_embedding(html)])[0].tolist()
                rec = {**page, "rank": inst.get("rank", rank), "query_words": q_words,
                       "query_type": q_type, "page_embedding": p_emb, "query_embedding": q_emb}
                feat = self.cf.feature_row(rec)
                X = [[feat[k] for k in self.cf.FEATURES]]
                prob = float(self.model.predict_proba(X)[0][1])
                scored.append({"url": url, "cited_prob": round(prob, 4),
                               "reasons": reasons(feat, page)})
            scored.sort(key=lambda s: s["cited_prob"], reverse=True)
            out.append({"query": query, "predictions": scored})
        return out
