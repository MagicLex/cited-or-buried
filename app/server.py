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
import math
import re
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


def _showcase(by_q, n: int = 6) -> list:
    """Real gap examples for the rotating hero: search's #1 was not quoted, yet a
    page well below it was, both fetched, and the model caught it (high odds on the
    below-top citation) so each slide shows the tool working, not just the gap.
    Deterministic; the gap itself is a fact about the data, not the model."""
    cands = []
    for qid, g in by_q.items():
        g = g.sort_values("rank")
        top = g.iloc[0]
        cited = g[g["cited"] == 1]
        if len(cited) == 0 or int(top["cited"]) == 1 or not top["fetch_ok"]:
            continue
        first_cited = cited.sort_values("rank").iloc[0]
        r = int(first_cited["rank"])
        if not first_cited["fetch_ok"] or r < 4 or r > 8:
            continue
        if g["fetch_ok"].sum() < 6 or not (4 <= len(str(top["query"]).split()) <= 12):
            continue
        cands.append((float(first_cited["model_score"]), r, qid))
    cands.sort(reverse=True)
    seen, out = set(), []
    for _, _, qid in cands:  # de-dup near-identical topics by first cited domain
        dom = _domain(by_q[qid][by_q[qid]["cited"] == 1].sort_values("rank").iloc[0]["url"])
        if dom in seen:
            continue
        seen.add(dom)
        out.append(qid)
        if len(out) >= n:
            break
    return out


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
    print(f"boot: {len(state['by_q'])} queries, showcase={state['showcase']}, model v{model.version}, "
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
    state["showcase"] = _showcase(state["by_q"])


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
:root{--bg:#f7f6f2;--card:#fff;--ink:#17171c;--dim:#5f5f6b;--faint:#9a9aa6;
--line:#eae7df;--line2:#ddd9cf;--gold:#a26c00;--gold2:#c98a10;--goldbg:#fbf4e4;
--goldln:#e7d3a3;--slate:#3b4b66;--slate2:#7c8aa3;--slatebg:#eef1f6;
--bad:#b23b34;--good:#1f7a52;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:
radial-gradient(1200px 480px at 12% -10%,#fdf6e6 0%,transparent 60%),var(--bg);
color:var(--ink);font:15.5px/1.62 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1000px;margin:0 auto;padding:20px 22px 100px}
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;
padding-bottom:14px;border-bottom:1px solid var(--line)}
.brand{font-size:17px;font-weight:680;letter-spacing:-.01em}
.brand .d{color:var(--gold)}
.nav a{color:var(--dim);font-size:14px;font-weight:550;margin-left:20px;padding-bottom:2px}
.nav a:hover{color:var(--ink)}
.nav a.on{color:var(--ink);border-bottom:2px solid var(--gold)}
.fav{border-radius:5px;vertical-align:middle;background:#fff;flex:none}
.btn{background:var(--gold);color:#fff;font-weight:620;font-size:14.5px;padding:11px 20px;
border-radius:10px;border:0;cursor:pointer;display:inline-block;
transition:background .15s,transform .12s,box-shadow .15s}
.btn:hover{background:var(--gold2);transform:translateY(-1px);box-shadow:0 6px 16px -8px #a26c0099}
.hint{color:var(--faint);font-size:13px}
.back{color:var(--dim);font-size:13.5px;font-weight:550}.back:hover{color:var(--gold)}

/* reveal-on-load (progressive: content is already in the DOM) */
.rise{opacity:0;transform:translateY(10px);animation:rise .55s cubic-bezier(.2,.7,.3,1) forwards}
@keyframes rise{to{opacity:1;transform:none}}

/* hero */
.hero{margin:32px 0 6px}
.herogrid{display:grid;grid-template-columns:1.02fr 1.05fr;gap:34px;align-items:center;margin:30px 0 4px}
.herogrid .show{margin:0}
@media(max-width:900px){.herogrid{grid-template-columns:1fr;gap:20px}}
.kick{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
color:var(--gold);font-weight:600}
.heroleft h1{font-size:clamp(30px,4.4vw,44px);line-height:1.07;font-weight:730;letter-spacing:-.024em;
margin:12px 0 14px;text-wrap:balance}
.heroleft h1 em{font-style:normal;color:var(--gold)}
.heroleft .lede{font-size:16.5px;color:var(--dim);margin:0 0 20px}
.hero h1{font-size:clamp(28px,4vw,40px);line-height:1.07;font-weight:720;letter-spacing:-.022em;
margin:12px 0 14px;max-width:17ch;text-wrap:balance}
.hero h1 em{font-style:normal;color:var(--gold)}
.hero .lede{font-size:16.5px;color:var(--dim);max-width:600px;margin:0}

/* rotating showcase: one real query at a time, as a live bar chart */
.show{background:var(--card);border:1px solid var(--line);border-radius:18px;
box-shadow:0 1px 3px #0000000d,0 16px 40px -28px #0000003a;padding:20px 22px 16px;margin:26px 0 10px;
position:relative;overflow:hidden}
.show::before{content:"";position:absolute;inset:0 0 auto 0;height:3px;
background:linear-gradient(90deg,var(--gold),#e6cd8a,transparent)}
.slides{position:relative;min-height:238px}
.slide{position:absolute;inset:0;opacity:0;visibility:hidden;
transition:opacity .5s ease;pointer-events:none}
.slide.on{position:relative;opacity:1;visibility:visible;pointer-events:auto}
.slide .q{font-size:15.5px;font-weight:600;margin:0 0 3px;letter-spacing:-.01em}
.slide .q .lead{font-family:var(--mono);font-size:11px;color:var(--faint);font-weight:600;
letter-spacing:.1em;text-transform:uppercase;display:block;margin-bottom:5px}
.slide .say{font-size:13.5px;color:var(--dim);margin:2px 0 16px}
.slide .say b{color:var(--gold);font-weight:640}.slide .say s{color:var(--slate);text-decoration:none;font-weight:600}

/* the bar chart: one bar per result, in search order, height = model odds */
.chart{display:flex;align-items:flex-end;gap:7px;height:150px;padding-top:20px}
.bar{flex:1;display:flex;flex-direction:column;align-items:center;gap:0;height:100%;
justify-content:flex-end;min-width:0}
.bar .col{width:100%;max-width:46px;border-radius:6px 6px 3px 3px;position:relative;
height:var(--h);background:linear-gradient(180deg,var(--slate2),var(--slate));
transform-origin:bottom;transition:filter .15s}
.on .bar .col{animation:grow .6s cubic-bezier(.2,.8,.3,1) backwards;animation-delay:var(--d)}
@keyframes grow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.bar.hit .col{background:linear-gradient(180deg,var(--gold2),var(--gold));
box-shadow:0 0 0 1px #fff,0 6px 14px -6px #a26c0080}
.bar .col .tick{position:absolute;top:-19px;left:50%;transform:translateX(-50%);
font-family:var(--mono);font-size:11px;font-weight:700;color:var(--gold);white-space:nowrap}
.bar .rk{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:7px}
.bar.hit .rk{color:var(--gold);font-weight:700}
.bar .col:hover{filter:brightness(1.06)}
.dots{display:flex;gap:6px;justify-content:center;margin-top:8px}
.dots b{width:6px;height:6px;border-radius:50%;background:var(--line2);cursor:pointer;
transition:background .2s,transform .2s}
.dots b.on{background:var(--gold);transform:scale(1.25)}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;
color:var(--faint);letter-spacing:.05em;text-transform:uppercase;margin-top:4px}

/* stat strip */
.strip{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.stat{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:15px 17px;transition:transform .12s,box-shadow .15s}
.stat:hover{transform:translateY(-2px);box-shadow:0 8px 20px -14px #0000003a}
.stat .n{font-family:var(--mono);font-size:27px;font-weight:730;color:var(--gold);letter-spacing:-.02em}
.stat .c{font-size:12.5px;color:var(--dim);margin-top:3px;line-height:1.4}
.stat .c b{color:var(--ink)}

.cta-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:24px 0 8px}

/* browse list */
.sechead{font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;
color:var(--faint);margin:40px 0 12px;display:flex;justify-content:space-between;align-items:baseline}
.qlist{display:flex;flex-direction:column;gap:7px}
.qrow{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);
border-radius:11px;padding:12px 15px;transition:border-color .12s,transform .12s,box-shadow .15s}
.qrow:hover{border-color:var(--goldln);transform:translateX(3px);box-shadow:0 4px 14px -8px #0000002e}
.qrow .q{flex:1;font-size:15px;font-weight:520;min-width:0}
.qrow .chips{display:flex;gap:5px;flex:none}
.qrow .qc{font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--gold);
background:var(--goldbg);border:1px solid var(--goldln);border-radius:999px;padding:2px 8px}
.qrow .go{color:var(--faint);font-size:16px;flex:none;transition:transform .12s,color .12s}
.qrow:hover .go{color:var(--gold);transform:translateX(2px)}

/* query detail */
.qtitle{font-size:25px;font-weight:690;letter-spacing:-.017em;margin:14px 0 4px;text-wrap:balance}
.qverdict{color:var(--dim);font-size:15.5px;margin:0 0 20px;max-width:660px}
.qverdict b{color:var(--gold)}
.detchart{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:18px 20px 14px;margin:0 0 24px;box-shadow:0 1px 2px #0000000a}
.detchart .cap{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
color:var(--faint);margin:0 0 4px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:780px){.cols{grid-template-columns:1fr}.chart{height:120px}}
.colh{font-family:var(--mono);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
color:var(--faint);margin:0 0 12px}
.rankrow{display:flex;align-items:center;gap:11px;padding:9px 4px;border-bottom:1px solid var(--line);
transition:background .12s}
.rankrow:hover{background:#faf8f2}
.rankrow .r{font-family:var(--mono);font-size:13px;color:var(--faint);width:26px;flex:none;text-align:right}
.rankrow.hit .r{color:var(--gold);font-weight:700}
.rankrow .txt{min-width:0;flex:1}
.rankrow .dom{font-size:13.5px;font-weight:560;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rankrow .qd{font-size:11px;color:var(--gold);font-weight:600}
.card{background:var(--card);border:1px solid var(--goldln);border-radius:13px;padding:15px 16px;
margin-bottom:12px;box-shadow:0 1px 2px #0000000a;transition:transform .12s,box-shadow .15s}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 26px -16px #a26c0055}
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
.result{background:var(--card);border:1px solid var(--goldln);border-radius:16px;padding:24px;margin:8px 0;
box-shadow:0 1px 3px #0000000d,0 18px 44px -30px #a26c0055}
.dialwrap{display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.dial{position:relative;width:132px;height:132px;flex:none}
.dial svg{transform:rotate(-90deg)}
.dial .arc{stroke-dasharray:var(--c);stroke-dashoffset:var(--off);
animation:fill 1.1s cubic-bezier(.3,.7,.3,1) forwards}
@keyframes fill{from{stroke-dashoffset:var(--c)}to{stroke-dashoffset:var(--off)}}
.dial .n{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.dial .n b{font-family:var(--mono);font-size:34px;font-weight:760;letter-spacing:-.03em;color:var(--gold);line-height:1}
.dial .n span{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.09em;margin-top:2px}
.rlab .verd{font-size:20px;font-weight:680;letter-spacing:-.01em;margin:0 0 4px}
.rlab .sub{font-size:13.5px;color:var(--dim);margin:0;max-width:340px}
.rlab .sub b{color:var(--ink)}.rlab .sub a{color:var(--gold)}
.levers{margin-top:18px;border-top:1px solid var(--line);padding-top:6px}
.lever{display:flex;align-items:center;gap:10px;font-size:14px;padding:8px 0;border-bottom:1px solid var(--line)}
.lever:last-child{border-bottom:0}
.lever .ic{font-family:var(--mono);font-weight:800;width:18px;flex:none;text-align:center}
.lever.y .ic{color:var(--good)}.lever.n .ic{color:var(--bad)}.lever.d .ic{color:var(--faint)}
.err{color:var(--bad);font-size:14px;background:#fbeceb;border:1px solid #f0d3d0;border-radius:10px;padding:12px 14px}
footer{margin-top:56px;color:var(--faint);font-size:12px;border-top:1px solid var(--line);
padding-top:16px;line-height:1.7}
footer a{color:var(--gold)}

@media (prefers-reduced-motion:reduce){
*{animation:none!important;transition:none!important}
.rise{opacity:1;transform:none}
.slide{position:relative;opacity:1;visibility:visible}
.slide:not(:first-child){display:none}
.dial .arc{stroke-dashoffset:var(--off)}}
"""


# JS is progressive enhancement only: all content is already server-rendered.
JS = """
(function(){
 var rm=window.matchMedia&&matchMedia('(prefers-reduced-motion:reduce)').matches;
 // count-up the stat numbers
 document.querySelectorAll('[data-count]').forEach(function(el){
  var to=parseFloat(el.getAttribute('data-count')), dec=(el.getAttribute('data-dec')|0),
      suf=el.getAttribute('data-suf')||'';
  if(rm){el.textContent=to.toFixed(dec)+suf;return;}
  var t0=null,D=900;
  function step(t){t0=t0||t;var k=Math.min(1,(t-t0)/D);var e=1-Math.pow(1-k,3);
   el.textContent=(to*e).toFixed(dec)+suf; if(k<1)requestAnimationFrame(step);}
  requestAnimationFrame(step);
 });
 // rotating showcase
 var slides=[].slice.call(document.querySelectorAll('.slide'));
 var dots=[].slice.call(document.querySelectorAll('.dots b'));
 if(slides.length>1){
  var i=0,timer=null;
  function go(n){slides[i].classList.remove('on');dots[i]&&dots[i].classList.remove('on');
   i=(n+slides.length)%slides.length;
   slides[i].classList.add('on');dots[i]&&dots[i].classList.add('on');}
  function play(){if(rm)return;stop();timer=setInterval(function(){go(i+1);},4200);}
  function stop(){if(timer){clearInterval(timer);timer=null;}}
  dots.forEach(function(d,n){d.addEventListener('click',function(){go(n);play();});});
  var show=document.querySelector('.show');
  if(show){show.addEventListener('mouseenter',stop);show.addEventListener('mouseleave',play);}
  play();
 }
})();
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
</div><script>{JS}</script></body></html>""")


app = FastAPI()


# ---- the bar chart: one bar per result, in search order, height = model odds --

def _chart(g, active: bool = True) -> str:
    """Results in search order as bars. Height = the model's citation odds, gold
    bar + a ✓ tick for the pages the AI actually quoted, slate for the rest. Shows
    at a glance that the quoted pages are not the tallest or the leftmost."""
    gg = g.sort_values("rank").head(10)
    bars = ""
    for i, (_, r) in enumerate(gg.iterrows()):
        hit = int(r["cited"]) == 1
        h = max(7, int(round(float(r["model_score"]) * 100)))
        dom = html.escape(_domain(r["url"]))
        pct = int(round(float(r["model_score"]) * 100))
        tick = '<span class="tick">✓ quoted</span>' if hit else ""
        bars += (f'<div class="bar {"hit" if hit else ""}" title="#{int(r["rank"])} {dom} · {pct}% odds">'
                 f'<div class="col" style="--h:{h}%;--d:{i * 55}ms">{tick}</div>'
                 f'<div class="rk">#{int(r["rank"])}</div></div>')
    on = " on" if active else ""
    return (f'<div class="chart{on}">{bars}</div>'
            f'<div class="axis"><span>search rank 1 →</span><span>bar height = model citation odds</span></div>')


# ---- gallery ----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def gallery():
    m = state["metrics"]
    auroc = float(m.get("auroc", 0))
    beyond3, not1 = state["beyond3"] * 100, state["not1"] * 100

    slides, dots = "", ""
    for idx, qid in enumerate(state.get("showcase", [])):
        g = state["by_q"][qid].sort_values("rank")
        query = g["query"].iloc[0]
        top = g.iloc[0]
        winner = g[g["cited"] == 1].sort_values("rank").iloc[0]
        say = (f'Search ranked <s>{html.escape(_domain(top["url"]))}</s> #1, and the AI answer '
               f'skipped it. It quoted <b>{html.escape(_domain(winner["url"]))}</b>, '
               f'which sat at #{int(winner["rank"])}.')
        slides += (f'<div class="slide{" on" if idx == 0 else ""}">'
                   f'<p class="q"><span class="lead">the gap · live from the corpus</span>'
                   f'{html.escape(query)}</p><p class="say">{say}</p>{_chart(g, active=idx == 0)}</div>')
        dots += f'<b class="{"on" if idx == 0 else ""}"></b>'
    show = (f'<div class="show rise"><div class="slides">{slides}</div>'
            f'<div class="dots">{dots}</div></div>') if slides else ""

    rows = ""
    for q in state["queries"][:80]:
        chips = "".join(f'<span class="qc">#{r}</span>' for r in q["cited_ranks"][:4])
        rows += (f'<a class="qrow" href="q/{q["qid"]}"><span class="q">{html.escape(q["query"])}</span>'
                 f'<span class="chips">{chips}</span><span class="go">›</span></a>')

    body = f"""
<div class="herogrid">
<div class="heroleft rise"><div class="kick">generative engine optimization</div>
<h1>Search ranks you. The AI answer <em>quotes</em> someone else.</h1>
<p class="lede">Ranking #1 is not the same as getting cited. Watch who the AI answer actually
quotes, query after query, then score your own page against it.</p>
<div class="cta-row"><a class="btn" href="coach">Score your page →</a>
<span class="hint">paste a query + URL</span></div></div>
{show}
</div>
<div class="strip">
<div class="stat"><div class="n" data-count="{beyond3:.0f}" data-suf="%">0%</div><div class="c">of AI citations
go to a page that was <b>not</b> in the top-3 search results</div></div>
<div class="stat"><div class="n" data-count="{not1:.0f}" data-suf="%">0%</div><div class="c">of citations are
not even the <b>#1</b> result the search engine returned</div></div>
<div class="stat"><div class="n" data-count="{auroc:.2f}" data-dec="2">0.00</div><div class="c">AUROC: the model
calls which page gets quoted, held out on {len(state['by_q'])} queries</div></div>
</div>
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
<div class="detchart rise"><p class="cap">search order, and who got quoted</p>{_chart(g)}</div>
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


def _dial(prob: float) -> str:
    c = 2 * math.pi * 56
    off = c * (1 - min(1.0, max(0.0, prob)))
    pct = int(round(prob * 100))
    return (f'<div class="dial"><svg width="132" height="132" viewBox="0 0 132 132">'
            f'<circle cx="66" cy="66" r="56" fill="none" stroke="#eee7d5" stroke-width="12"/>'
            f'<circle class="arc" cx="66" cy="66" r="56" fill="none" stroke="url(#g)" stroke-width="12" '
            f'stroke-linecap="round" style="--c:{c:.1f};--off:{off:.1f}"/>'
            f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="#c98a10"/><stop offset="1" stop-color="#a26c00"/>'
            f'</linearGradient></defs></svg>'
            f'<div class="n"><b>{pct}%</b><span>citation odds</span></div></div>')


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
<div class="result rise"><div class="dialwrap">{_dial(prob)}
<div class="rlab"><p class="verd">{_verdict_word(prob)}</p>
<p class="sub">estimated chance the AI answer for <b>{html.escape(query)}</b> cites
<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(_domain(url))}</a></p></div></div>
<div class="levers">{levers}</div></div></div>"""
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
