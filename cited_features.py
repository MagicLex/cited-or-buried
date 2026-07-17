"""Shared, pure feature extraction for cited-or-buried.

One module imported by the F2 content pipeline, the trainer, and the serving
predictor, so training and serving cannot skew. Structural SEO features are
computed here from raw page HTML with the stdlib only (no bs4/lxml dep): the
model-independent transforms (MITs). Embeddings live in `encoder.py` so cheap
paths that only need structure do not import torch.

The features are the actionable GEO levers: where the answer sits, how
structured the page is, whether it carries schema markup, how fresh it is. The
label (cited) is captured separately by the F1 fleet; nothing here touches it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html.parser import HTMLParser

import numpy as np

# ---- FG grains (write these down before the code that fills them) -----------

SERP_DOC = {
    "query_id": "MS MARCO query id (primary key with url)",
    "query": "the natural-language search query",
    "url": "a result URL for the query (primary key with query_id)",
    "rank": "raw search rank of this URL for the query, 1-based (the SEO proxy)",
    "title": "result title as returned by search, '' when absent",
    "cited": "1 if the grounded AI answer cited this URL, else 0 (the label)",
    "captured_at": "fleet capture timestamp (event time)",
}

PAGE_DOC = {
    "url": "page URL (primary key)",
    "fetch_ok": "the raw HTML fetched with status 200",
    "status": "HTTP status of the fetch (0 when the request failed outright)",
    "word_count": "visible words in the page body",
    "intro_words": "words before the first heading (answer-depth proxy; low = answer up top)",
    "n_headings": "h1-h6 count",
    "n_h2": "h2 count",
    "n_lists": "ul+ol count",
    "n_list_items": "li count",
    "n_tables": "table count",
    "question_headings": "headings phrased as a question (ending in ?)",
    "has_schema_org": "any ld+json schema.org block present",
    "has_faq_schema": "ld+json declares FAQPage or QAPage",
    "has_tldr": "an explicit TL;DR / summary / key-takeaways marker present",
    "meta_desc_len": "length of the meta description, 0 when absent",
    "external_link_count": "outbound <a> links",
    "freshness_days": "age in days from a published/modified date in the page, -1 when none found",
    "title_len": "length of the <title>, 0 when absent",
    "fetched_at": "content fetch timestamp (event time)",
}

QUERY_DOC = {
    "query_id": "MS MARCO query id (primary key)",
    "query": "the natural-language search query",
    "query_words": "token count of the query",
    "query_type": "informational wh-class: what/how/why/who/where/when/which/yesno/other",
    "fetched_at": "capture timestamp (event time)",
}

# numeric page-feature order, used verbatim by trainer and predictor
PAGE_NUMERIC = (
    "word_count", "intro_words", "n_headings", "n_h2", "n_lists", "n_list_items",
    "n_tables", "question_headings", "has_schema_org", "has_faq_schema",
    "has_tldr", "meta_desc_len", "external_link_count", "freshness_days", "title_len",
)

_WS = re.compile(r"\s+")
_DATE_RX = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_WH = ("what", "how", "why", "who", "where", "when", "which")


class _Extract(HTMLParser):
    """Single-pass structural counts + visible-text accumulation."""

    _SKIP = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.n_h2 = self.n_lists = self.n_list_items = 0
        self.n_tables = self.n_links = 0
        self.title = ""
        self.meta_desc = ""
        self.ld_json: list[str] = []
        self._text: list[str] = []
        self._stack: list[str] = []
        self._grab_heading: str | None = None
        self._grab_title = False
        self._grab_ld = False
        self._body_words = 0        # visible words seen so far
        self.intro_words = -1       # body words before the first heading (-1 = no heading yet)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._stack.append(tag)
        a = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self.intro_words < 0:
                self.intro_words = self._body_words
            self._grab_heading = ""
            if tag == "h2":
                self.n_h2 += 1
        elif tag in ("ul", "ol"):
            self.n_lists += 1
        elif tag == "li":
            self.n_list_items += 1
        elif tag == "table":
            self.n_tables += 1
        elif tag == "a" and a.get("href", "").startswith("http"):
            self.n_links += 1
        elif tag == "title":
            self._grab_title = True
        elif tag == "meta" and a.get("name", "").lower() == "description":
            self.meta_desc = a.get("content", "") or self.meta_desc
        elif tag == "script" and a.get("type", "").lower() == "application/ld+json":
            self._grab_ld = True

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._grab_heading is not None:
            self.headings.append(_WS.sub(" ", self._grab_heading).strip())
            self._grab_heading = None
        elif tag == "title":
            self._grab_title = False
        elif tag == "script":
            self._grab_ld = False

    def handle_data(self, data: str) -> None:
        if self._grab_ld:
            self.ld_json.append(data)
            return
        if self._grab_title:
            self.title += data
        if self._grab_heading is not None:
            self._grab_heading += data
        if not (set(self._stack) & self._SKIP):
            self._text.append(data)
            if self._grab_heading is None:  # body prose, not heading text
                self._body_words += len(data.split())

    def text(self) -> str:
        return _WS.sub(" ", " ".join(self._text)).strip()


def _freshness_days(html: str, ex: _Extract, asof: datetime) -> int:
    """Age in days from a published/modified date, -1 when none is found.
    Looks at common meta/time carriers, then any ISO date in the head."""
    for pat in (
        r'property=["\']article:(?:published|modified)_time["\']\s+content=["\']([^"\']+)',
        r'itemprop=["\'](?:datePublished|dateModified)["\']\s+content=["\']([^"\']+)',
        r'<time[^>]+datetime=["\']([^"\']+)',
        r'"date(?:Published|Modified)"\s*:\s*"([^"]+)"',
    ):
        m = re.search(pat, html, re.I)
        if m:
            d = _DATE_RX.search(m.group(1))
            if d:
                try:
                    dt = datetime(int(d[1]), int(d[2]), int(d[3]), tzinfo=timezone.utc)
                    return max((asof - dt).days, 0)
                except ValueError:
                    pass
    return -1


def page_features(url: str, html: str, status: int, asof: datetime) -> dict:
    """Raw HTML -> one page_features FG row. Pure: bytes in, flat row out.
    Empty html (fetch failure) yields a fetch_ok=0 row with zeroed structure so
    the FV can impute; rank still carries signal for those."""
    ex = _Extract()
    ok = bool(html) and status == 200
    if ok:
        try:
            ex.feed(html)
        except Exception:  # malformed markup: keep whatever parsed
            pass
    text = ex.text()
    words = text.split()
    intro = ex.intro_words if ex.intro_words >= 0 else len(words)  # no heading: whole page is "intro"
    ld = " ".join(ex.ld_json)
    return {
        "url": url,
        "fetch_ok": ok,
        "status": int(status),
        "word_count": len(words),
        "intro_words": min(intro, len(words)) if words else 0,
        "n_headings": len(ex.headings),
        "n_h2": ex.n_h2,
        "n_lists": ex.n_lists,
        "n_list_items": ex.n_list_items,
        "n_tables": ex.n_tables,
        "question_headings": sum(1 for h in ex.headings if h.endswith("?")),
        "has_schema_org": bool(ex.ld_json) or ("schema.org" in html.lower() if ok else False),
        "has_faq_schema": bool(re.search(r'"@type"\s*:\s*"(FAQPage|QAPage)"', ld, re.I)),
        "has_tldr": bool(re.search(r"tl;?dr|key takeaways|in short|in summary|quick answer", text, re.I)),
        "meta_desc_len": len(ex.meta_desc),
        "external_link_count": ex.n_links,
        "freshness_days": _freshness_days(html, ex, asof) if ok else -1,
        "title_len": len(ex.title.strip()),
    }


def page_text_for_embedding(html: str, max_chars: int = 4000) -> str:
    """Visible text for the page embedding, capped. Title first (it carries the
    topic), then body."""
    ex = _Extract()
    try:
        ex.feed(html or "")
    except Exception:
        pass
    return (ex.title.strip() + ". " + ex.text())[:max_chars]


def query_row(query_id: str, query: str, asof: datetime) -> dict:
    first = query.strip().lower().split()[0] if query.strip() else ""
    if first in _WH:
        qtype = first
    elif first in ("is", "are", "can", "do", "does", "did", "will", "should"):
        qtype = "yesno"
    else:
        qtype = "other"
    return {
        "query_id": str(query_id),
        "query": query,
        "query_words": len(query.split()),
        "query_type": qtype,
        "fetched_at": asof,
    }


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---- ranker feature contract (the no-skew guarantee) ------------------------
# One function turns a joined (serp x page x query) record into the model's
# feature row. Trainer and predictor both call it, so they cannot skew.

QUERY_TYPES = ("what", "how", "why", "who", "where", "when", "which", "yesno", "other")

# final model feature order, used verbatim by trainer and predictor
FEATURES = ("rank",) + PAGE_NUMERIC + ("fetch_ok", "query_words", "query_type_id", "cosine")


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def query_type_id(qt: str) -> int:
    return QUERY_TYPES.index(qt) if qt in QUERY_TYPES else QUERY_TYPES.index("other")


def feature_row(rec: dict) -> dict:
    """Joined record -> the model's feature dict, in FEATURES order. `rec` carries
    rank, the PAGE_NUMERIC structural fields, fetch_ok, query_words, query_type,
    and page_embedding / query_embedding lists. Missing -> neutral (0)."""
    out = {"rank": float(rec.get("rank") or 0.0)}
    for k in PAGE_NUMERIC:
        out[k] = float(rec.get(k) or 0.0)
    out["fetch_ok"] = float(rec.get("fetch_ok") or 0.0)
    out["query_words"] = float(rec.get("query_words") or 0.0)
    out["query_type_id"] = float(query_type_id(rec.get("query_type") or "other"))
    out["cosine"] = cosine(rec.get("page_embedding"), rec.get("query_embedding"))
    return out
