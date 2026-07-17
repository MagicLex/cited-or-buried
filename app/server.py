"""cited-or-buried: does the AI answer quote your page, or just rank it?

Server-rendered FastAPI (no SPA). Two things a visitor should get in five seconds:
the gap (search ranks a page #1, the AI answer quotes someone lower) and the tool
(paste a page, get its citation odds). The gallery is precomputed from geo_scored;
the coach posts {query, url} to the citedscorer deployment and scores live.

"Quoted" is the captured label from one answer engine, a directional GEO signal,
not a Google-AIO oracle. The model score is the prediction; the citation is the
truth. We show the truth first.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

state: dict = {}


def _domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return url[:30]


def _divergence(g) -> float:
    """Mean how-far-below-#1 the cited pages sit. High = the answer engine reached
    past the top of the search results, which is the whole story."""
    cited_ranks = g[g["cited"] == 1]["rank"].tolist()
    if not cited_ranks:
        return 0.0
    return float(np.mean([max(0, r - 1) for r in cited_ranks]))


def _pick_hero(by_q) -> str | None:
    """One clean, real example for the top of the page: search's #1 was not quoted,
    yet a page well below it was, and both pages actually fetched (so the reasons are
    real). Deterministic, model-independent: the gap is a fact about the data."""
    best, best_key = None, (-1, -1.0)
    for qid, g in by_q.items():
        g = g.sort_values("rank")
        top = g.iloc[0]
        cited = g[g["cited"] == 1]
        if len(cited) == 0 or int(top["cited"]) == 1 or not top["fetch_ok"]:
            continue
        first_cited = cited.sort_values("rank").iloc[0]
        r = int(first_cited["rank"])
        if not first_cited["fetch_ok"] or r < 4 or r > 7:
            continue
        if g["fetch_ok"].sum() < 6 or not (4 <= len(str(top["query"]).split()) <= 12):
            continue
        # lead with an example the model clearly caught (high odds on a below-top
        # citation), so the hero shows the tool working, not just the gap
        key = (float(first_cited["model_score"]), r)
        if key > best_key:
            best_key, best = key, qid
    return best


def boot() -> None:
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = fs.get_feature_group("geo_scored", version=1).read()
    df = df.sort_values("scored_at").drop_duplicates(["query_id", "url"], keep="last")
    _index(df)

    try:
        state["deployment"] = project.get_model_serving().get_deployment("citedscorer")
    except Exception:
        state["deployment"] = None

    mr = project.get_model_registry()
    model = max(mr.get_models("cited_ranker"), key=lambda m: m.version)
    state["model_version"] = model.version
    state["metrics"] = {re.sub(r"_+", "_", k): v for k, v in (model.training_metrics or {}).items()}
    print(f"boot: {len(state['by_q'])} queries, hero={state['hero']}, model v{model.version}, "
          f"deployment {'up' if state['deployment'] else 'down'}", flush=True)


def _index(df) -> None:
    """Build the in-memory state the views read. Split out so a local preview can
    feed a parquet without hopsworks."""
    state["df"] = df
    state["by_q"] = {qid: g.copy() for qid, g in df.groupby("query_id")}
    cited = df[df["cited"] == 1]
    state["beyond3"] = float((cited["rank"] > 3).mean()) if len(cited) else 0.0
    state["not1"] = float((cited["rank"] > 1).mean()) if len(cited) else 0.0
    rows = []
    for qid, g in state["by_q"].items():
        c = g[g["cited"] == 1]
        if len(c) == 0 or g["fetch_ok"].sum() < 4:
            continue  # curate: skip no-citation and all-blocked queries
        rows.append({
            "qid": qid, "query": g["query"].iloc[0], "div": _divergence(g),
            "cited_ranks": sorted(int(r) for r in c["rank"]),
            "n": len(g), "ncited": len(c),
        })
    state["queries"] = sorted(rows, key=lambda r: -r["div"])
    state["hero"] = _pick_hero(state["by_q"])


# ---- human-language reasons -------------------------------------------------

def why(r) -> list:
    """Short, plain reasons a citation did or did not land, from the page features."""
    if not r["fetch_ok"]:
        return [("dim", "page blocked our reader")]
    out = []
    intro, words = int(r["intro_words"]), int(r["word_count"])
    if intro <= 60:
        out.append(("good", "answer up top"))
    elif intro >= 220:
        out.append(("bad", "answer buried deep"))
    if r["has_faq_schema"]:
        out.append(("good", "FAQ schema"))
    elif r["has_schema_org"]:
        out.append(("good", "schema markup"))
    if int(r["n_h2"]) >= 4 or int(r["n_list_items"]) >= 10:
        out.append(("good", "well structured"))
    fresh = int(r["freshness_days"])
    if 0 <= fresh <= 365:
        out.append(("good", "fresh"))
    elif fresh > 1095:
        out.append(("bad", f"~{fresh // 365}y old"))
    if words and words < 300:
        out.append(("bad", "thin content"))
    return out[:4]


def _fav(url: str, size: int = 32) -> str:
    d = html.escape(_domain(url))
    return (f'<img class="fav" width="{size}" height="{size}" loading="lazy" '
            f'src="https://www.google.com/s2/favicons?domain={d}&sz=64" '
            f'onerror="this.style.visibility=\'hidden\'" alt="">')


# ---- styling ----------------------------------------------------------------

CSS = """
:root{--bg:#faf9f6;--card:#fff;--ink:#17171c;--dim:#5f5f6b;--faint:#9a9aa6;
--line:#eae7df;--line2:#ddd9cf;--gold:#a26c00;--gold2:#c98a10;--goldbg:#fbf4e4;
--goldln:#e7d3a3;--slate:#3b4b66;--slatebg:#eef1f6;--bad:#b23b34;--good:#1f7a52;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15.5px/1.62 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1000px;margin:0 auto;padding:20px 22px 100px}
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;
padding-bottom:14px;border-bottom:1px solid var(--line)}
.brand{font-size:17px;font-weight:680;letter-spacing:-.01em}
.brand .d{color:var(--gold)}
.nav a{color:var(--dim);font-size:14px;font-weight:550;margin-left:20px;padding-bottom:2px}
.nav a.on{color:var(--ink);border-bottom:2px solid var(--gold)}
.fav{border-radius:5px;vertical-align:middle;background:#fff;flex:none}

/* hero */
.hero{margin:34px 0 8px}
.kick{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
color:var(--gold);font-weight:600}
.hero h1{font-size:clamp(30px,5vw,46px);line-height:1.08;font-weight:720;letter-spacing:-.022em;
margin:12px 0 14px;max-width:16ch}
.hero h1 em{font-style:normal;color:var(--gold)}
.hero .lede{font-size:17px;color:var(--dim);max-width:620px;margin:0}

/* the concrete example card */
.demo{background:var(--card);border:1px solid var(--line);border-radius:16px;
box-shadow:0 1px 3px #0000000d,0 8px 24px -18px #0000002e;padding:22px 24px;margin:26px 0}
.demo .q{font-size:13px;color:var(--faint);margin:0 0 16px}
.demo .q b{color:var(--ink);font-weight:600}
.duel{display:grid;grid-template-columns:1fr;gap:12px}
.side{border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.side.lose{background:#fbfbfa}
.side.win{background:var(--goldbg);border-color:var(--goldln)}
.side .lab{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
display:flex;align-items:center;gap:8px;margin-bottom:10px}
.side.lose .lab{color:var(--slate)}.side.win .lab{color:var(--gold)}
.side .lab .tag{margin-left:auto;font-weight:600}
.doc{display:flex;align-items:center;gap:11px}
.doc .fav{width:26px;height:26px}
.doc .txt{min-width:0}
.doc .dom{font-weight:620;font-size:14.5px}
.doc .ti{color:var(--dim);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rankchip{font-family:var(--mono);font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;
background:var(--slatebg);color:var(--slate);flex:none}
.side.win .rankchip{background:#fff;color:var(--gold);border:1px solid var(--goldln)}
.more{margin-top:9px;font-size:12.5px;color:var(--dim)}
.arrow{text-align:center;color:var(--faint);font-size:13px;margin:2px 0}

/* stat strip */
.strip{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.stat{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:15px 17px}
.stat .n{font-family:var(--mono);font-size:26px;font-weight:700;color:var(--gold);letter-spacing:-.02em}
.stat .c{font-size:12.5px;color:var(--dim);margin-top:3px;line-height:1.4}

/* CTA */
.cta-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:26px 0 8px}
.btn{background:var(--gold);color:#fff;font-weight:620;font-size:14.5px;padding:11px 20px;
border-radius:10px;border:0;cursor:pointer;display:inline-block}
.btn:hover{background:var(--gold2)}
.btn.ghost{background:#fff;color:var(--ink);border:1px solid var(--line2)}
.hint{color:var(--faint);font-size:13px}

/* browse list */
.sechead{font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
color:var(--faint);margin:40px 0 12px;display:flex;justify-content:space-between;align-items:baseline}
.qlist{display:flex;flex-direction:column;gap:7px}
.qrow{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);
border-radius:11px;padding:12px 15px}
.qrow:hover{border-color:var(--line2);box-shadow:0 2px 10px -6px #00000024}
.qrow .q{flex:1;font-size:15px;font-weight:520;min-width:0}
.qrow .chips{display:flex;gap:5px;flex:none}
.qrow .qc{font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--gold);
background:var(--goldbg);border:1px solid var(--goldln);border-radius:999px;padding:2px 8px}
.qrow .go{color:var(--faint);font-size:16px;flex:none}

/* query detail */
.back{color:var(--dim);font-size:13.5px;font-weight:550}
.qtitle{font-size:24px;font-weight:680;letter-spacing:-.015em;margin:14px 0 4px}
.qverdict{color:var(--dim);font-size:15px;margin:0 0 22px;max-width:640px}
.qverdict b{color:var(--gold)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:780px){.cols{grid-template-columns:1fr}}
.colh{font-family:var(--mono);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
color:var(--faint);margin:0 0 12px}
.rankrow{display:flex;align-items:center;gap:11px;padding:9px 4px;border-bottom:1px solid var(--line)}
.rankrow .r{font-family:var(--mono);font-size:13px;color:var(--faint);width:26px;flex:none;text-align:right}
.rankrow.hit .r{color:var(--gold);font-weight:700}
.rankrow .txt{min-width:0;flex:1}
.rankrow .dom{font-size:13.5px;font-weight:560;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rankrow .qd{font-size:11px;color:var(--gold);font-weight:600}
.card{background:var(--card);border:1px solid var(--goldln);border-radius:13px;padding:15px 16px;
margin-bottom:12px;box-shadow:0 1px 2px #0000000a}
.card .hd{display:flex;align-items:center;gap:11px;margin-bottom:8px}
.card .hd .txt{min-width:0;flex:1}
.card .hd .dom{font-weight:640;font-size:15px}
.card .hd .ti{font-size:12.5px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .badge{font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--gold);
background:var(--goldbg);border:1px solid var(--goldln);border-radius:999px;padding:3px 9px;flex:none}
.why{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.why span{font-size:12px;border-radius:6px;padding:1px 8px;font-weight:500}
.why .good{background:#eaf5ef;color:var(--good)}
.why .bad{background:#fbeceb;color:var(--bad)}
.why .dim{background:#f2f1ec;color:var(--dim)}
.pred{font-size:12px;color:var(--faint);margin-top:9px}
.pred b{color:var(--ink);font-family:var(--mono)}
.empty{color:var(--faint);font-size:13.5px;padding:10px 0}

/* coach */
.coachwrap{max-width:600px}
form.coach{display:flex;flex-direction:column;gap:11px;margin:22px 0}
label.f{font-size:13px;color:var(--dim);font-weight:550;margin-bottom:-4px}
input[type=text]{background:#fff;border:1px solid var(--line2);border-radius:10px;padding:12px 14px;
color:var(--ink);font-size:15px;width:100%;font-family:var(--sans)}
input:focus{outline:none;border-color:var(--gold);box-shadow:0 0 0 3px #a26c001f}
.gauge{background:var(--card);border:1px solid var(--goldln);border-radius:16px;padding:24px;margin:8px 0}
.gauge .big{font-family:var(--mono);font-size:60px;font-weight:750;line-height:1;letter-spacing:-.03em;color:var(--gold)}
.gauge .verd{font-size:17px;font-weight:640;margin:4px 0 14px}
.track{height:9px;background:#f0eee7;border-radius:6px;overflow:hidden;border:1px solid var(--line)}
.track i{display:block;height:100%;background:linear-gradient(90deg,var(--gold2),var(--gold))}
.gauge .sub{font-size:13px;color:var(--dim);margin:14px 0 0}
.lever{display:flex;align-items:center;gap:9px;font-size:14px;padding:7px 0;border-top:1px solid var(--line)}
.lever .ic{font-family:var(--mono);font-weight:700;width:18px;flex:none}
.lever.y .ic{color:var(--good)}.lever.n .ic{color:var(--bad)}.lever.d .ic{color:var(--faint)}
.err{color:var(--bad);font-size:14px;background:#fbeceb;border:1px solid #f0d3d0;border-radius:10px;padding:12px 14px}
footer{margin-top:56px;color:var(--faint);font-size:12px;border-top:1px solid var(--line);
padding-top:16px;line-height:1.7}
footer a{color:var(--gold)}
"""


def shell(title: str, body: str, tab: str) -> HTMLResponse:
    def nav(name, label, href):
        return f'<a class="{"on" if tab == name else ""}" href="{href}">{label}</a>'
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<div class="top"><a class="brand" href="."><span class="d">◆</span> cited or buried</a>
<nav class="nav">{nav("gallery","the gap",".")}{nav("coach","score a page","coach")}</nav></div>
{body}
<footer>cited or buried · does the AI answer quote your page, or just rank it · "quoted" is a
directional signal captured from one answer engine, not a Google-AIO oracle ·
<a href="https://github.com/MagicLex/cited-or-buried">source</a></footer>
</div></body></html>""")


app = FastAPI()


# ---- gallery ----------------------------------------------------------------

def _doc_line(r, win: bool) -> str:
    dom = html.escape(_domain(r["url"]))
    ti = html.escape((r["title"] or "")[:64])
    chip = f'<span class="rankchip">#{int(r["rank"])} in search</span>'
    return (f'<div class="doc">{_fav(r["url"])}<div class="txt">'
            f'<div class="dom">{dom}</div><div class="ti">{ti}</div></div>{chip}</div>')


@app.get("/", response_class=HTMLResponse)
def gallery():
    m = state["metrics"]
    auroc = float(m.get("auroc", 0))
    beyond3, not1 = state["beyond3"] * 100, state["not1"] * 100

    hero_html = ""
    hid = state.get("hero")
    if hid is not None:
        g = state["by_q"][hid].sort_values("rank")
        query = g["query"].iloc[0]
        top = g.iloc[0]
        cited = g[g["cited"] == 1].sort_values("rank")
        winner = cited.iloc[0]
        extra = len(cited) - 1
        more = (f'<div class="more">+ {extra} more source{"s" if extra > 1 else ""} '
                f'the answer pulled, none of them search\'s #1</div>' if extra > 0 else "")
        hero_html = f"""
<div class="demo">
<p class="q">query &nbsp;<b>{html.escape(query)}</b></p>
<div class="duel">
<div class="side lose"><div class="lab">search's top result <span class="tag">not quoted</span></div>
{_doc_line(top, False)}</div>
<div class="arrow">the AI answer skipped it and reached down for ↓</div>
<div class="side win"><div class="lab">what the AI actually quoted <span class="tag">✓ cited</span></div>
{_doc_line(winner, True)}{more}</div>
</div></div>"""

    rows = ""
    for q in state["queries"][:80]:
        chips = "".join(f'<span class="qc">#{r}</span>' for r in q["cited_ranks"][:4])
        rows += (f'<a class="qrow" href="q/{q["qid"]}"><span class="q">{html.escape(q["query"])}</span>'
                 f'<span class="chips">{chips}</span><span class="go">›</span></a>')

    body = f"""
<div class="hero"><div class="kick">generative engine optimization</div>
<h1>Search ranks you. The AI answer <em>quotes</em> someone else.</h1>
<p class="lede">Ranking #1 is not the same as getting cited. Here is who the AI answer actually
quotes, per query, and a model that predicts whether it will quote your page.</p></div>
{hero_html}
<div class="strip">
<div class="stat"><div class="n">{beyond3:.0f}%</div><div class="c">of AI citations go to a page
that was <b>not</b> in the top-3 search results</div></div>
<div class="stat"><div class="n">{not1:.0f}%</div><div class="c">of citations are not even the
#1 result the search engine returned</div></div>
<div class="stat"><div class="n">{auroc:.2f}</div><div class="c">AUROC: the model calls which page
gets quoted, held out on {len(state['by_q'])} queries</div></div>
</div>
<div class="cta-row"><a class="btn" href="coach">Score your page →</a>
<span class="hint">paste a query and a URL, get its citation odds</span></div>
<div class="sechead"><span>browse the gap</span><span>chips = the search ranks the AI quoted</span></div>
<div class="qlist">{rows}</div>"""
    return shell("cited or buried", body, "gallery")


# ---- query detail -----------------------------------------------------------

@app.get("/q/{qid}", response_class=HTMLResponse)
def query_view(qid: str):
    g = state["by_q"].get(qid)
    if g is None:
        return shell("cited or buried", '<p class="empty">query not found</p>', "gallery")
    query = g["query"].iloc[0]
    by_rank = g.sort_values("rank")
    cited = g[g["cited"] == 1].sort_values("rank")
    cited_ranks = [int(r) for r in cited["rank"]]
    top_cited = 1 in cited_ranks

    verdict = f"Search returned {len(g)} pages. "
    if len(cited) == 0:
        verdict += "The AI answer quoted none of them."
    elif top_cited and len(cited) == 1:
        verdict += "The AI quoted the #1 result. Rank and citation agreed here."
    else:
        where = ", ".join(f"#{r}" for r in cited_ranks)
        verdict += f"The AI answer was built from <b>{where}</b>" + (
            ", reaching past the top of the page." if not top_cited else ".")

    left = ""
    for _, r in by_rank.iterrows():
        hit = int(r["cited"]) == 1
        tag = '<span class="qd">✓ quoted</span>' if hit else ""
        left += (f'<div class="rankrow {"hit" if hit else ""}"><span class="r">#{int(r["rank"])}</span>'
                 f'{_fav(r["url"], 20)}<div class="txt"><div class="dom">{html.escape(_domain(r["url"]))}</div></div>'
                 f'{tag}</div>')

    if len(cited) == 0:
        right = '<p class="empty">The AI answer cited no page from this result set.</p>'
    else:
        right = ""
        for _, r in cited.iterrows():
            chips = "".join(f'<span class="{c}">{html.escape(t)}</span>' for c, t in why(r))
            pct = int(round(float(r["model_score"]) * 100))
            right += f"""<div class="card">
<div class="hd">{_fav(r['url'])}<div class="txt">
<a class="dom" href="{html.escape(r['url'])}" target="_blank" rel="noopener">{html.escape(_domain(r['url']))}</a>
<div class="ti">{html.escape((r['title'] or '')[:70])}</div></div>
<span class="badge">was #{int(r['rank'])} in search</span></div>
<div class="why">{chips}</div>
<div class="pred">our model gave this page <b>{pct}%</b> citation odds</div></div>"""

    body = f"""
<p style="margin:18px 0 0"><a class="back" href=".">← the gap</a></p>
<h2 class="qtitle">{html.escape(query)}</h2>
<p class="qverdict">{verdict}</p>
<div class="cols">
<div><p class="colh">what search ranked</p>{left}</div>
<div><p class="colh">what the AI quoted</p>{right}</div>
</div>"""
    return shell(f"cited or buried / {query[:40]}", body, "gallery")


# ---- coach ------------------------------------------------------------------

@app.get("/coach", response_class=HTMLResponse)
def coach_form():
    down = state.get("deployment") is None
    note = '<div class="err">the live scorer is offline right now; browse the gap gallery meanwhile</div>' if down else ""
    body = f"""
<div class="coachwrap">
<div class="hero"><div class="kick">the coach</div>
<h1>Would the AI answer <em>quote your page</em>?</h1>
<p class="lede">Paste a query and a URL. The model fetches the page live, reads where the answer
sits, how it is structured, its schema and freshness, and its match to the query, then scores the
citation odds with the levers you can actually pull.</p></div>
{note}
<form class="coach" action="coach" method="post">
<label class="f">the search query</label>
<input type="text" name="query" placeholder="e.g. what is mixed connective tissue disease" required>
<label class="f">your page URL</label>
<input type="text" name="url" placeholder="https://your-page" required>
<button class="btn" type="submit">Score it →</button></form></div>"""
    return shell("cited or buried / coach", body, "coach")


def _verdict_word(p: float) -> str:
    if p >= 0.6:
        return "Likely to be quoted"
    if p >= 0.35:
        return "In the running"
    if p >= 0.18:
        return "Long shot"
    return "Unlikely to be quoted"


@app.post("/coach", response_class=HTMLResponse)
def coach_run(query: str = Form(...), url: str = Form(...)):
    dep = state.get("deployment")
    if dep is None:
        return shell("cited or buried / coach", '<div class="coachwrap"><div class="err">the live scorer is offline</div></div>', "coach")
    try:
        res = dep.predict(inputs=[{"query": query, "urls": [url]}])
        pred = (res["predictions"] if isinstance(res, dict) else res)[0]["predictions"][0]
    except Exception as e:
        return shell("cited or buried / coach",
                     f'<div class="coachwrap"><div class="err">score error: {html.escape(str(e))}</div></div>', "coach")
    prob = float(pred["cited_prob"])
    pct = int(round(prob * 100))
    levers = ""
    for reason in pred.get("reasons", []):
        rl = reason.lower()
        cls = "n" if any(w in rl for w in ("buried", "blocked", "stale", "thin", "no ", "did not")) else "y"
        ic = "✓" if cls == "y" else "✕"
        levers += f'<div class="lever {cls}"><span class="ic">{ic}</span><span>{html.escape(reason)}</span></div>'
    if not levers:
        levers = '<div class="lever d"><span class="ic">–</span><span>no strong structural signal either way</span></div>'
    body = f"""
<div class="coachwrap">
<p style="margin:18px 0 0"><a class="back" href="coach">← score another</a></p>
<div class="gauge"><div class="big">{pct}%</div>
<div class="verd">{_verdict_word(prob)}</div>
<div class="track"><i style="width:{max(3, pct)}%"></i></div>
<p class="sub">estimated chance the AI answer for <b>{html.escape(query)}</b> cites
<a href="{html.escape(url)}" target="_blank" rel="noopener" style="color:var(--gold)">{html.escape(_domain(url))}</a></p>
{levers}</div></div>"""
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
                mm = _PROXY_MOUNT.match(scope["path"])
                prefix = mm.group(0) if mm else ""
            if prefix and scope["path"].startswith(prefix):
                scope = dict(scope)
                scope["path"] = scope["path"][len(prefix):] or "/"
                scope["root_path"] = prefix
        await self.inner(scope, receive, send)


application = StripForwardedPrefix(app)


if __name__ == "__main__":
    boot()
    uvicorn.run(application, host="0.0.0.0", port=8000)
