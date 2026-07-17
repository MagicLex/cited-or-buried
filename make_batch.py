"""Emit a self-contained geo-serp-capture workflow script with a query slice
baked in, so the session can run F1 in batches via {scriptPath} (no giant args).

  python make_batch.py --start 0 --end 120 --out capture_batch.js
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

TEMPLATE = """export const meta = {{
  name: 'geo-serp-capture',
  description: 'F1 label factory: web-search -> raw ranked URLs + AI-cited subset (baked batch)',
  phases: [{{ title: 'Capture' }}],
}}

const BATCH = {batch_json}

const SCHEMA = {{
  type: 'object',
  additionalProperties: false,
  required: ['query_id', 'ranked_urls', 'cited_urls'],
  properties: {{
    query_id: {{ type: 'string' }},
    query: {{ type: 'string' }},
    ranked_urls: {{
      type: 'array',
      items: {{
        type: 'object',
        additionalProperties: false,
        required: ['rank', 'url'],
        properties: {{
          rank: {{ type: 'integer' }},
          url: {{ type: 'string' }},
          title: {{ type: 'string' }},
        }},
      }},
    }},
    cited_urls: {{ type: 'array', items: {{ type: 'string' }} }},
    answer_summary: {{ type: 'string' }},
  }},
}}

const results = await parallel(BATCH.map((item) => () =>
  agent(
    `You model an AI answer engine for one search query.\\n\\n` +
    `query_id: "${{item.query_id}}"\\n` +
    `Query: "${{item.query}}"\\n\\n` +
    `Steps:\\n` +
    `1. Use WebSearch on the query. Record the top 6-8 results as ranked_urls IN THE RAW ORDER the search returned them (rank starts at 1, include the title). ` +
    `Do NOT reorder, filter, or drop results, even ones that look off-topic or low quality. The raw order is the SEO-rank baseline and must stay honest.\\n` +
    `2. Optionally WebFetch one or two of the top results to ground yourself.\\n` +
    `3. Write a concise, factual answer to the query grounded in those results (answer_summary, 2-4 sentences).\\n` +
    `4. Record cited_urls: the EXACT subset of ranked_urls[].url you relied on for the answer. It MUST be a subset of ranked_urls, typically 1-4 URLs. Empty if unanswerable from the results.\\n\\n` +
    `Echo query_id back verbatim. Do NOT invent URLs; every cited_url must appear verbatim in ranked_urls.`,
    {{ label: `q:${{item.query_id}}`, phase: 'Capture', schema: SCHEMA, agentType: 'general-purpose' }}
  )
))

return results.filter(Boolean)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=120)
    ap.add_argument("--out", default="capture_batch.js")
    args = ap.parse_args()

    rows = list(csv.DictReader((DATA / "queries.csv").open()))[args.start:args.end]
    batch = [{"query_id": r["query_id"], "query": r["query"]} for r in rows]
    script = TEMPLATE.format(batch_json=json.dumps(batch, ensure_ascii=False))
    Path(args.out).write_text(script)
    print(f"wrote {args.out}: {len(batch)} queries [{args.start}:{args.end}]")


if __name__ == "__main__":
    main()
