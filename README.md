# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over one identical chunked corpus, compared on the same
queries with the same prompt — so the only variable is which three chunks reach
the LLM.

| | Pipeline 1 — Baseline | Pipeline 2 — Re-Ranking |
|---|---|---|
| Stage 1 | bi-encoder, top **10** by cosine | bi-encoder, top **10** by cosine *(the same 10)* |
| Stage 2 | — | cross-encoder scores all 10 `(query, chunk)` pairs |
| Context | top **3** by cosine | top **3** by cross-encoder score |
| Output | Answer 1 | Answer 2 |

**Document:** *D&D Basic Rules (2018)*, Wizards of the Coast — 180 pages,
~134k words, distributed free by the publisher. It is a rulebook in the strict
sense: general rules first, then the restrictions, penalties and "except
when…" clauses that qualify them, often 50+ pages away. That separation is
what breaks single-vector retrieval, which is what this assignment is about.

**Models**

| Component | Model |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `gemini-3-flash-preview`, temperature 0 |

## Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then paste your Gemini key into .env
jupyter notebook rag_baseline_vs_reranking.ipynb
```

The rulebook PDF (~11 MB) downloads on the first run. Retrieval, re-ranking,
the rank-shift analysis and the latency profiling **all run without an API
key** — only the two answer-generation cells need one.

To rebuild and re-execute the notebook from scratch:

```bash
python build_notebook.py
jupyter nbconvert --execute --inplace rag_baseline_vs_reranking.ipynb
```

## Layout

```
rag_baseline_vs_reranking.ipynb   the submission: all code inline, with outputs
build_notebook.py                 generates the notebook (single source of truth)
dev/                              the same logic as importable modules, used to
                                  iterate on the pipeline outside the notebook
  corpus.py                         PDF -> cleaned, page-tagged chunks
  pipeline.py                       DenseIndex, Reranker, prompt, generator
  queries.py                        stress-test pool + gold-chunk signatures
  run_experiment.py                 runs everything headless, writes results/
results/                          measured numbers the report quotes
figures/                          latency chart
report/                           evidence pack for the write-up
```

## How a retrieval failure is measured rather than asserted

For each stress query, the chunk that actually answers it is pinned up front by
a regex matching a phrase unique to that chunk. Every claim is then a number:

- `rank_100` — the gold chunk's rank in a top-**100** sweep. This separates two
  distinct failures: a chunk at rank 7 is a *ranking* problem the cross-encoder
  can fix, whereas a chunk at rank 60 never enters the 10-candidate set at all,
  so re-ranking **cannot** recover it. That recall ceiling is a real limit of
  the architecture, not a tuning issue.
- `base` / `rerank` — the gold chunk's rank before and after re-ranking.
- **rescued** — gold chunk outside the top 3 for the baseline, inside it after
  re-ranking.

The query pool spans the four failure modes the assignment asks for (keyword
trap, exception/corner case, negation/constraint, buried-but-vital) plus one
**control** query phrased in the rulebook's own vocabulary, where the baseline
is expected to succeed on its own.

## Note on the report

`report/` holds the measured evidence — tables, rank shifts, answer pairs —
organised under the four required headings. The analysis prose is written by
hand, per the assignment's authenticity requirement.
