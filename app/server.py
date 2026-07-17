"""cited-or-buried: the GEO scope. Two columns per query, GOOGLE RANKS vs AI CITES,
plus a live coach that scores any page.

Server-rendered FastAPI (no SPA): the gallery is in the initial payload. The coach
tab posts a {query, url} to the citedscorer deployment and renders the citation
probability with plain-word reasons. No live LLM in the app: "AI CITES" is the
precomputed label, the model score is precomputed for the gallery and live for the
coach. The demo shows what the model knows that raw rank does not.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

import hopsworks

state: dict = {}


def boot() -> None:
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = fs.get_feature_group("geo_scored", version=1).read()
    df = df.sort_values("scored_at").drop_duplicates(["query_id", "url"], keep="last")
    state["df"] = df
    state["by_q"] = {qid: g.copy() for qid, g in df.groupby("query_id")}
    cited = df[df["cited"] == 1]
    state["beyond3"] = float((cited["rank"] > 3).mean()) if len(cited) else 0.0
    state["not1"] = float((cited["rank"] > 1).mean()) if len(cited) else 0.0
    # queries ranked by how much the model disagrees with raw rank (the interesting ones first)
    order = []
    for qid, g in state["by_q"].items():
        div = _divergence(g)
        order.append((qid, g["query"].iloc[0], div, int(g["cited"].sum()), len(g)))
    state["queries"] = sorted(order, key=lambda x: -x[2])

    try:
        state["deployment"] = project.get_model_serving().get_deployment("citedscorer")
    except Exception:
        state["deployment"] = None

    mr = project.get_model_registry()
    model = max(mr.get_models("cited_ranker"), key=lambda m: m.version)
    state["model_version"] = model.version
    state["metrics"] = {re.sub(r"_+", "_", k): v for k, v in (model.training_metrics or {}).items()}
    print(f"boot: {len(state['by_q'])} queries, model v{model.version}, "
          f"deployment {'up' if state['deployment'] else 'down'}", flush=True)


def _divergence(g) -> float:
    """How far the actual citations sit from 'trust rank 1-3'. High = the answer
    engine reached past the top results, the story this project tells."""
    cited_ranks = g[g["cited"] == 1]["rank"].tolist()
    if not cited_ranks:
        return 0.0
    return float(np.mean([max(0, r - 1) for r in cited_ranks]))


def why(r) -> list:
    out = []
    if not r["fetch_ok"]:
        return ["page did not fetch (blocked or JS-only)"]
    intro, words = int(r["intro_words"]), int(r["word_count"])
    if intro <= 60:
        out.append(f"answer up top ({intro}w intro)")
    elif intro >= 200:
        out.append(f"answer buried ({intro}w intro)")
    if r["has_schema_org"]:
        out.append("schema markup" + (" + FAQ" if r["has_faq_schema"] else ""))
    if int(r["n_h2"]) >= 4 or int(r["n_list_items"]) >= 10:
        out.append(f"structured ({int(r['n_h2'])} H2 · {int(r['n_list_items'])} li)")
    fresh = int(r["freshness_days"])
    if 0 <= fresh <= 365:
        out.append(f"fresh ~{fresh}d")
    elif fresh > 1095:
        out.append(f"stale ~{fresh // 365}y")
    if words and words < 300:
        out.append(f"short ({words}w)")
    return out


CSS = """
:root{--bg:#f6f5f1;--bg2:#ffffff;--panel:#ffffff;--panel2:#f1efe9;--ink:#1b1b20;
--dim:#6a6a74;--faint:#9c9ca6;--line:#e7e4dc;--line2:#d9d6cc;--gold:#a9750d;
--gold-soft:#a9750d12;--gold-line:#a9750d33;--bad:#c0392b;--good:#1a7f5a;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1100px 520px at 15% -12%,#efe9db 0%,transparent 62%),var(--bg);
color:var(--ink);font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--gold);text-decoration:none}a:hover{text-decoration:underline}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1080px;margin:0 auto;padding:18px 22px 90px}
h1{font-size:20px;margin:0;font-weight:650;letter-spacing:-.01em}
h1 a{color:var(--ink)}.gold{color:var(--gold)}
.crumb{color:var(--faint)}.sub{color:var(--dim);max-width:680px;font-size:14.5px}
.tabs{display:flex;gap:6px;margin:16px 0 22px;border-bottom:1px solid var(--line)}
.tabs a{color:var(--dim);border-bottom:2px solid transparent;padding:9px 4px;margin-right:16px;
font-weight:500;font-size:14px}
.tabs a.active{color:var(--ink);border-bottom-color:var(--gold)}
.hero h2{font-size:clamp(26px,4vw,40px);line-height:1.1;font-weight:680;letter-spacing:-.02em;
margin:26px 0 12px;max-width:18ch}.hero h2 em{font-style:normal;color:var(--gold)}
.lede{font-size:16px;color:var(--dim);max-width:660px;margin:0 0 22px}
.flex{background:linear-gradient(180deg,#ffffff 0%,#f3efe6 100%);border:1px solid var(--line);
border-radius:14px;padding:22px 24px;margin:26px 0;box-shadow:0 1px 2px #0000000a}
.flex .big{font-family:var(--mono);font-size:clamp(34px,6vw,56px);font-weight:700;color:var(--gold);
line-height:1;letter-spacing:-.03em}
.flex .cap{color:var(--dim);max-width:620px;margin:8px 0 0;font-size:14px}
.qlist{display:flex;flex-direction:column;gap:2px;margin-top:14px}
.qrow{display:grid;grid-template-columns:1fr 92px 78px;align-items:center;gap:12px;
padding:9px 12px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}
.qrow:hover{border-color:var(--line2)}
.qrow .q{font-size:14.5px}.qrow .m{font-family:var(--mono);font-size:12px;color:var(--faint);text-align:right}
.qrow .m b{color:var(--gold)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.col h3{font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;
color:var(--dim);margin:0 0 10px;display:flex;justify-content:space-between}
.item{border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-bottom:9px;background:var(--panel)}
.item.cited{border-color:var(--gold-line);background:linear-gradient(180deg,var(--gold-soft),transparent)}
.item .u{font-size:13px;word-break:break-all}.item .u .r{color:var(--faint);font-family:var(--mono);margin-right:7px}
.item .t{color:var(--dim);font-size:12.5px;margin-top:3px}
.item .row{display:flex;align-items:center;gap:8px;margin-top:9px}
.bar{flex:1;height:5px;background:#0000000d;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#f5c451,#ffdd7a)}
.sc{font-family:var(--mono);font-size:12px;color:var(--gold);min-width:34px;text-align:right}
.star{color:var(--gold)}.why{margin-top:7px;display:flex;gap:6px;flex-wrap:wrap}
.why span{font-family:var(--mono);font-size:11px;color:var(--faint);border:1px solid var(--line2);
border-radius:5px;padding:0 6px}
form.coach{display:flex;flex-direction:column;gap:10px;max-width:620px;margin:8px 0 24px}
input[type=text]{background:var(--bg2);border:1px solid var(--line2);border-radius:9px;padding:11px 14px;
color:var(--ink);font-size:15px;width:100%}
input:focus{outline:none;border-color:var(--gold);box-shadow:0 0 0 3px var(--gold-soft)}
button.cta{background:var(--gold);color:#1a1405;border:0;border-radius:9px;padding:11px 18px;
font-weight:650;font-size:14px;cursor:pointer;align-self:flex-start}
.verdict{border:1px solid var(--gold-line);border-radius:12px;padding:18px 20px;margin:8px 0;
background:linear-gradient(180deg,var(--gold-soft),transparent);max-width:620px}
.verdict .p{font-family:var(--mono);font-size:40px;color:var(--gold);font-weight:700;line-height:1}
.err{color:var(--bad)}
footer{margin-top:48px;color:var(--faint);font-size:12px;border-top:1px solid var(--line);padding-top:16px;line-height:1.7}
"""


def shell(title: str, body: str, tab: str) -> HTMLResponse:
    def t(name, label):
        return f'<a class="{"active" if tab == name else ""}" href="{name if name != "gallery" else "."}">{label}</a>'
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1><a href="."><span class="gold">◆</span> cited or buried</a></h1>
<nav class="tabs">{t("gallery","the gap")}{t("coach","score a page")}</nav>
{body}
<footer>cited-or-buried #013 · does the AI answer quote you, or just rank you · citation is a
directional GEO signal from one answer engine, not a Google-AIO oracle ·
<a href="https://github.com/MagicLex/cited-or-buried">source</a></footer>
</div></body></html>""")


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def gallery():
    m = state["metrics"]
    auroc = float(m.get("auroc", 0))
    corr = float(m.get("rank_citation_corr", 0))
    beyond3 = state["beyond3"] * 100
    not1 = state["not1"] * 100
    rows = "".join(
        f"""<a class="qrow" href="q/{qid}"><span class="q">{html.escape(q)}</span>
<span class="m">{ncited}/{n} cited</span><span class="m">gap <b>{div:.1f}</b></span></a>"""
        for qid, q, div, ncited, n in state["queries"][:120])
    body = f"""
<div class="hero"><h2>Search ranks you. The AI answer <em>quotes</em> someone else.</h2>
<p class="lede">For each query: the raw search order on the left, the pages the AI answer actually
cited on the right. A model trained on {len(state['by_q'])} queries scores which page earns the
citation, from its structure and topic match.</p></div>
<div class="flex"><div class="big">{beyond3:.0f}%</div>
<p class="cap">of AI citations go to a page that wasn't even in the <b>top-3</b> search results
({not1:.0f}% aren't the #1 result). Rank correlates with citation only <b>{corr:.2f}</b>: the
answer engine reaches past the top of the page. The model reads that page and calls the citation
at <b>AUROC {auroc:.2f}</b>.</p></div>
<p class="sub">Queries ranked by how far the citations sit from the top results. Click one.</p>
<div class="qlist">{rows}</div>"""
    return shell("cited or buried", body, "gallery")


@app.get("/q/{qid}", response_class=HTMLResponse)
def query_view(qid: str):
    g = state["by_q"].get(qid)
    if g is None:
        return shell("cited or buried", '<p class="err">query not found</p>', "gallery")
    query = g["query"].iloc[0]
    by_rank = g.sort_values("rank")
    by_model = g.sort_values("model_score", ascending=False)

    def item(r, show_star: bool) -> str:
        cited = int(r["cited"]) == 1
        w = int(round(float(r["model_score"]) * 100))
        chips = "".join(f"<span>{html.escape(x)}</span>" for x in why(r))
        star = ' <span class="star">★ cited</span>' if (show_star and cited) else ""
        return f"""<div class="item{' cited' if cited else ''}">
<div class="u"><span class="r">#{int(r['rank'])}</span>
<a href="{html.escape(r['url'])}" target="_blank" rel="noopener">{html.escape(r['url'][:70])}</a>{star}</div>
<div class="t">{html.escape((r['title'] or '')[:90])}</div>
<div class="row"><span class="mono" style="font-size:11px;color:var(--faint)">MODEL</span>
<div class="bar"><i style="width:{max(4,w)}%"></i></div><span class="sc">{w}%</span></div>
<div class="why">{chips}</div></div>"""

    left = "".join(item(r, False) for _, r in by_rank.iterrows())
    right = "".join(item(r, True) for _, r in by_model.iterrows())
    body = f"""
<p class="sub" style="margin-top:14px"><a href=".">← the gap</a> &nbsp; query: <b>{html.escape(query)}</b></p>
<div class="cols">
<div class="col"><h3><span>google ranks</span><span>raw search order</span></h3>{left}</div>
<div class="col"><h3><span>model predicts cited</span><span>★ = actually cited</span></h3>{right}</div>
</div>"""
    return shell(f"cited or buried / {query[:40]}", body, "gallery")


@app.get("/coach", response_class=HTMLResponse)
def coach_form():
    down = state["deployment"] is None
    note = '<p class="err">scorer offline; try the gap gallery</p>' if down else ""
    body = f"""
<div class="hero"><h2>Would the AI answer <em>quote your page</em>?</h2>
<p class="lede">Paste a query and a URL. The model fetches the page live, reads its structure and
topic match, and scores the citation probability with plain-word reasons. No web search, your page
against the query.</p></div>
{note}
<form class="coach" action="coach" method="post">
<input type="text" name="query" placeholder="the search query" required>
<input type="text" name="url" placeholder="https://your-page" required>
<button class="cta">score it</button></form>"""
    return shell("cited or buried / coach", body, "coach")


@app.post("/coach", response_class=HTMLResponse)
def coach_run(query: str = Form(...), url: str = Form(...)):
    dep = state["deployment"]
    if dep is None:
        return shell("cited or buried / coach", '<p class="err">scorer offline</p>', "coach")
    try:
        res = dep.predict(inputs=[{"query": query, "urls": [url]}])
        pred = (res["predictions"] if isinstance(res, dict) else res)[0]["predictions"][0]
    except Exception as e:
        return shell("cited or buried / coach", f'<p class="err">score error: {html.escape(str(e))}</p>', "coach")
    prob = int(round(pred["cited_prob"] * 100))
    chips = "".join(f"<span>{html.escape(x)}</span>" for x in pred.get("reasons", []))
    body = f"""
<p class="sub" style="margin-top:14px"><a href="coach">← score another</a></p>
<div class="verdict"><div class="p">{prob}%</div>
<p class="cap" style="margin-top:8px">estimated chance the AI answer for <b>{html.escape(query)}</b>
cites <a href="{html.escape(url)}" target="_blank" rel="noopener">this page</a></p>
<div class="why" style="margin-top:12px">{chips}</div></div>"""
    return shell("cited or buried / coach", body, "coach")


@app.get("/health")
def health():
    return {"ok": True, "queries": len(state.get("by_q", {})),
            "model_version": state.get("model_version"), "scorer": state.get("deployment") is not None}


_PROXY_MOUNT = re.compile(r"^/hopsworks-api/pythonapp/[^/]+/[^/]+")


class StripForwardedPrefix:
    """Strip the Hopsworks proxy mount (no APP_BASE_URL_PATH, no X-Forwarded-Prefix
    on this cluster) so routes match. Without it the app 404s forever."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            prefix = dict(scope.get("headers") or {}).get(b"x-forwarded-prefix", b"").decode().rstrip("/")
            if not prefix:
                m = _PROXY_MOUNT.match(scope["path"])
                prefix = m.group(0) if m else ""
            if prefix and scope["path"].startswith(prefix):
                scope = dict(scope)
                scope["path"] = scope["path"][len(prefix):] or "/"
                scope["root_path"] = prefix
        await self.inner(scope, receive, send)


application = StripForwardedPrefix(app)


if __name__ == "__main__":
    boot()
    uvicorn.run(application, host="0.0.0.0", port=8000)
