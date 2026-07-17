# cited-or-buried (#013) -- spec

GEO recommender: predict whether an AI answer engine will **cite** a page for a
query, and expose the gap between what classic search ranks and what the answer
engine actually quotes. Ranking/retrieval problem (real recsys), served live,
zero API dollars to build.

## AI-system card

- **Prediction problem.** Ranking with a binary relevance label: for a
  (query, page) pair, will the AI answer for that query **cite** this page?
  Pointwise citation probability, ranked per query.
- **KPI it improves.** A page owner's AI-answer citation rate (share of target
  queries whose generated answer quotes the page). The system tells them which
  pages earn citations and what to change on the ones that do not.
- **ML proxy metric.** Citation AUC and precision@k per query, reported as
  **lift over the classic-rank baseline** ("rank #1 = most likely cited"). The
  lift is the whole thesis: if content features add nothing over rank, there is
  no GEO story. Secondary: rank-vs-citation correlation (expected weak), the
  divergence number the screen sells.
- **Data sources.** All new, all free, generated at build time (no existing
  feature groups to reuse; the project post-teardown is empty):
  1. **Query universe** -- a public informational-query set (ORCAS query slice
     or a curated how-to/what-is set; informational queries are where AI answers
     dominate, so the signal is densest there).
  2. **SERP capture + citation labels** -- a session subagent fleet (Workflow
     tool). Per query: one web search yields the ranked URL list (the classic-SEO
     proxy, no paid SERP API); a grounded answer written from those results
     records which URLs it cites. Rank and cited come from the **same call**.
     This is the unstarred fingerprint trick: LLM labels at build time, $0.
  3. **Page content** -- fetched HTML/text of each ranked URL (WebFetch), the
     raw material for content features.
- **ML-system type.** Real-time. Type a query, see the two columns live.
- **How predictions are consumed.** A custom app (the two-column GEO screen +
  per-page plain-word "why cited / why buried"), plus a precomputed gallery of
  queries so the demo is instant even when a live search is slow.
- **How it is monitored.** The inference pipeline logs each query, the ranked
  candidates, and the model's predicted-cited probabilities to a feature group.
  A scheduled job periodically re-captures a sample of logged queries (fresh
  subagent citations) to measure predicted-vs-actual drift as answer engines
  change.

## The load-bearing assumption -- SPIKE PASSED (2026-07-16)

The whole free-label economy rests on: **a subagent can run a web search, write a
grounded answer, and reliably report the URLs it cited.** A 10-query spike
(Workflow, 10 general-purpose agents, 34s, 217k tokens, $0) validated it:

- **Real / in-list:** every `cited_url` appeared verbatim in its `ranked_urls`.
  10/10, zero hallucinated URLs.
- **Subset held:** 10/10 (the `cited_urls subset of ranked_urls` schema held
  under real use).
- **Divergence is real:** rank-1 was cited only 8/10 times; citations reached
  down to rank 8 (two-tower, salary) and rank 5 (compost, background-removal).
  The rank-vs-citation gap -- the entire thesis -- is visible even at n=10.
- **Label balance:** ~2.7 cited of ~7.7 surfaced per query = ~35% positive.
  Healthy for a classifier.
- **Cost extrapolation:** ~22k tokens/query, $0 API. A 5k-query corpus ~= 110M
  subagent tokens, same order as unstarred's 66M. Feasible free.

**Two fixes the spike surfaced, mandatory for F1:**
1. **Capture the RAW search rank, do not let the agent re-rank.** On the
   northern-lights query, WebSearch injected two off-topic arXiv papers at ranks
   1 and 5 and the subagent silently excluded them. If the agent editorializes
   the ranked list, the classic-rank baseline (the SEO proxy) is contaminated by
   agent judgment. F1 records the ranked list verbatim (keep the noise, or mark
   `excluded=true` with a reason); `cited` is the only agent-judgment field.
2. **Tighten the schema** (`additionalProperties: false`): one agent emitted a
   stray `cited_urls_note` field. Cosmetic, but F1 wants clean rows.

## Honesty rules (loud in the app + README)

- Citations come from **one answer-engine proxy** (a web-grounded subagent, a
  specific model family), a snapshot in time. This is a directional GEO signal,
  not Google AI Overviews or Perplexity ground truth. Say so.
- Pages are labeled only among the **search-returned top-N**; a page search never
  surfaced cannot be scored (selection boundary). State it.
- The "classic rank" column is the **search API's ranking as a Google proxy**,
  not Google itself.
- Headline is **lift over the rank-only baseline**, never the absolute AUC alone.

## Pipelines (feature -> training -> inference)

### F1. serp-capture (feature pipeline, the label factory) -- blocks all

- **Source:** the query universe.
- **Work:** a subagent fleet (Workflow) captures, per query: ranked URLs + cited
  URLs + fetched page content. Writes rows to FG `serp_capture`
  (pk `query_id + url`, `event_time = fetched_at`): `query`, `url`, `rank`,
  `cited` (0/1), `raw_content`. Offline FG; dedupe at read (unstarred scar:
  offline inserts append across runs).
- **Skill:** hops-features -> hops-fg, hops-data-sources. The fleet itself is the
  Workflow tool (subagents with WebSearch/WebFetch).

### F2. content-features (feature pipeline, MITs) -- blocked-by F1

- **Model-independent transforms** on `raw_content` -> FG `page_features`
  (pk `url`): page content **embedding** (frozen text encoder, embed-at-the-door
  lineage), plus structural MITs that are the actionable SEO levers:
  answer-depth (how far down the direct answer sits), structuredness
  (headings/lists/tables), word count, entity density, freshness (last-modified),
  schema.org markup present, question-in-heading, TL;DR present, readability.
- **Query features** -> FG `query_features` (pk `query_id`): query embedding,
  query type (informational/navigational/transactional, derived), length.
- **Skill:** hops-features -> hops-fg, hops-transformations (@udf MITs).

### T1. train-citation-ranker (training pipeline) -- blocked-by F2

- **Feature view** `cited_fv`: `serp_capture` joined to `page_features` on `url`
  and `query_features` on `query_id`. Label = `cited`. Include `rank` as a
  feature (it is also the baseline).
- **MDTs on the FV** (train/serve identical): normalize numeric content features,
  encode query type. Attach as feature-view transformations, not precomputed.
- **Model.** Start with a gradient-boosted ranker (LightGBM/HistGB, LTR objective
  or pointwise logistic) as the honest bar; a two-tower query<->page head on the
  embeddings is the stage-2 lever, shipped only if it beats the GBM (the-untested
  staged-model discipline).
- **Split by `query_id`**, never by pair -- pages and vocabulary leak across a
  looser split (llm-tell-auditor scar).
- **Eval + registry:** AUC, precision@k, NDCG, and **lift over the rank-only
  baseline**; the rank-vs-citation correlation plot; per-feature importance
  (the "why cited" evidence). Register every run with card + images; pick the
  champion at serve time by advertised metric (register-all discipline).
- **Skill:** hops-eda (leakage/target analysis first), then hops-train.

### I1. serve-geo (real-time inference) -- blocked-by T1

- **KServe deployment** `citedscorer`. **No LLM at request time** (account is
  empty): the model predicts citation probability from content features alone.
- **On-demand transform (ODT):** at request time, web-search the query -> ranked
  URLs -> fetch + extract the same MITs -> the FV applies the MDTs -> the model
  scores each candidate. Returns the two columns (classic rank vs
  predicted-cited) + per-page plain-word reasons from feature attribution.
- **Precomputed gallery:** a batch inference run scores a fixed query set into FG
  `geo_scored` so the app opens instant; "try your own query" takes the live ODT
  path.
- **Log** query + candidates + predictions to `geo_inference_log` for monitoring.
- **Skill:** hops-online-inference (predictor + ODT), hops-batch-inference (the
  gallery), hops-environments (clone + pin the encoder/serving stack).

### App. cited-or-buried (custom app) -- blocked-by I1

- The two-column screen: type a query, GOOGLE RANKS vs AI CITES, stars on the
  cited, plain-word why-cited/why-buried per page, the divergence number up top.
  Series design rules: the screen answers ONE question ("would the AI quote you,
  and why not"), ranked attention, expected-vs-actual, sober aesthetic, external
  links to every page shown.
- Thin client where possible (calls `citedscorer`; loads no heavy model).
- **Skill:** hops-app.

## Monitoring -- blocked-by I1

- Descriptive stats + drift on `geo_inference_log`; a scheduled re-capture job
  refreshes citations for a sample of logged queries and compares predicted vs
  actual (answer engines drift, so the eval must be live). **Skill:**
  hops-monitoring.

## Parked v2 (do not build now)

- Model a **named engine** (Perplexity API / Google AI Overviews) once
  keys/credits exist -- the credible-ground-truth upgrade.
- **Multi-engine agreement**: does Claude cite what GPT/Gemini cite? The "which
  engine quotes what" study, a second labeling leg.
- **GEO coach / editable simulator**: paste your page, get the single edit that
  flips predicted-cited (graft the Rank-Radar live-simulator idea).
- **Feedback flywheel**: users submit URL+query, we score, later re-check actual
  citation, labeled example -> scheduled retrain.

## Build order

F1 -> F2 -> T1 -> I1 -> App, with Monitoring after I1. Repo forks the vaporware
structure (collect / shared extractor / pipelines / serving / app / Makefile /
README); add row #013 to awesome-ml-systems, bump the catalog badge. Repo:
github.com/MagicLex/cited-or-buried. Local: /hopsfs/Users/meb10000/013_cited_or_buried.
