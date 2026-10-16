"""Emit report/EVIDENCE.md -- the measured numbers, organised for the write-up.

This file contains data and no analysis. The assignment requires the report
prose to be written by hand, so this deliberately stops at the evidence and
leaves the interpretation to the author.

    PYTHONPATH=dev python dev/build_report_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
OUT = Path("report/EVIDENCE.md")

PLACEHOLDER = "> _Your analysis here._"


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |"
            for r in rows]
    return "\n".join(out)


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    ranks = load("rank_analysis.json")
    lat = load("latency.json")
    answers = load("answers.json")
    chunks = {c["chunk_id"]: c
              for c in json.loads(Path("data/processed/chunks.json").read_text())}

    def preview(cid: int, n: int = 150) -> str:
        return " ".join(chunks[cid]["text"].split())[:n]

    def outcome(r: dict) -> str:
        """Classified on top-3 membership, which is what reaches the LLM.

        A rank change inside the top 3 (or inside the tail) does not change
        the answer, so it is reported as a delta rather than as an outcome.
        """
        if r["baseline_rank"] is None:
            return "beyond recall"
        if r["in_reranked_context"] and not r["in_baseline_context"]:
            return "**rescued**"
        if r["in_baseline_context"] and not r["in_reranked_context"]:
            return "lost"
        if r["in_baseline_context"]:
            return "already in top 3"
        return "still missed"

    def delta(r: dict):
        if not r["baseline_rank"] or not r["reranked_rank"]:
            return None
        d = r["baseline_rank"] - r["reranked_rank"]
        return f"+{d}" if d > 0 else str(d)

    rescued = [r for r in ranks if outcome(r) == "**rescued**"]
    unreachable = [r for r in ranks if outcome(r) == "beyond recall"]
    still_missed = [r for r in ranks if outcome(r) == "still missed"]
    lost = [r for r in ranks if outcome(r) == "lost"]
    # Rank regressions, reported separately: some are harmless (both ranks sit
    # inside the top 3), others are not.
    regressed = [r for r in ranks
                 if r["baseline_rank"] and r["reranked_rank"]
                 and r["reranked_rank"] > r["baseline_rank"]]

    L = []
    A = L.append

    A("# Report evidence pack")
    A("")
    A("Measured output from `rag_baseline_vs_reranking.ipynb`. **Data only — no "
      "analysis.** The four headings below are the ones the assignment asks for; "
      "the prose under each is to be written by hand.")
    A("")
    A("Regenerate with `PYTHONPATH=dev python dev/build_report_data.py`.")
    A("")

    # ------------------------------------------------------------------ #
    A("## Setup")
    A("")
    A(table(
        ["Item", "Value"],
        [
            ["Document", "D&D Basic Rules (2018), Wizards of the Coast"],
            ["Pages", 180],
            ["Chunks", len(chunks)],
            ["Chunk size / overlap", "900 / 150 characters, page-bounded"],
            ["Bi-encoder", "`sentence-transformers/all-MiniLM-L6-v2` (384-d)"],
            ["Cross-encoder", "`cross-encoder/ms-marco-MiniLM-L-6-v2`"],
            ["Generator", "`gemini-3-flash-preview`, temperature 0"],
            ["Retrieve / context", "top 10 candidates, top 3 to the LLM"],
            ["Stress queries", len(ranks)],
        ],
    ))
    A("")

    # ------------------------------------------------------------------ #
    A("## 1. Architecture comparison")
    A("")
    A("Measurements that bear on this section:")
    A("")
    if lat:
        per_pair = lat["stages"]["cross_encode_10_pairs"]["mean_ms"] / 10
        A(table(
            ["Quantity", "Measured"],
            [
                ["Chunk vectors embedded once, reused per query",
                 f"{len(chunks)} x 384 floats"],
                ["Query embedding (1 forward pass)",
                 f"{lat['stages']['embed_query']['mean_ms']} ms"],
                ["Vector search over the whole corpus",
                 f"{lat['stages']['vector_search']['mean_ms']} ms"],
                ["Cross-encoder, 10 pairs",
                 f"{lat['stages']['cross_encode_10_pairs']['mean_ms']} ms"],
                ["Cross-encoder, cost per (query, chunk) pair",
                 f"{per_pair:.1f} ms"],
            ],
        ))
    A("")
    A(PLACEHOLDER)
    A("")

    # ------------------------------------------------------------------ #
    A("## 2. Qualitative & rank shift analysis")
    A("")
    A("### 2a. Full rank table")
    A("")
    A("`rank_100` = rank of the answer-bearing chunk in a top-100 bi-encoder "
      "sweep (diagnostic). `base` / `rerank` = its rank in the 10 candidates "
      "before and after the cross-encoder. `None` = absent from that set.")
    A("")
    A(table(
        ["qid", "category", "page", "rank_100", "base", "rerank", "shift",
         "in top-3 base", "in top-3 rerank", "outcome"],
        [
            [
                r["qid"], r["category"], ", ".join(map(str, r["gold_pages"])),
                r["rank_in_top100"], r["baseline_rank"], r["reranked_rank"],
                delta(r),
                "yes" if r["in_baseline_context"] else "no",
                "yes" if r["in_reranked_context"] else "no",
                outcome(r),
            ]
            for r in ranks
        ],
    ))
    A("")
    A(table(
        ["Summary", "Count"],
        [
            ["Queries", len(ranks)],
            ["Answer chunk in top 3 — baseline",
             sum(1 for r in ranks if r["in_baseline_context"])],
            ["Answer chunk in top 3 — re-ranked",
             sum(1 for r in ranks if r["in_reranked_context"])],
            ["Rescued by re-ranking", len(rescued)],
            ["Never retrieved — re-ranking cannot help", len(unreachable)],
            ["Retrieved but still outside top 3 after re-ranking",
             len(still_missed)],
            ["Dropped out of top 3 by re-ranking", len(lost)],
            ["Rank moved worse (harmless where both ranks are inside top 3)",
             len(regressed)],
        ],
    ))
    A("")

    # ------------------------------------------------------------------ #
    A("### 2b. Rank shifts, with what the baseline ranked instead")
    A("")
    for r in rescued:
        A(f"#### {r['qid']} — {r['category']} — "
          f"**Rank #{r['baseline_rank']} -> Rank #{r['reranked_rank']}**")
        A("")
        A(f"*Query:* {r['query']}")
        A("")
        A(f"*Why it is hard:* {r['note']}")
        A("")
        A(f"*Answer-bearing chunk:* {r['gold_chunk_ids']} (page "
          f"{', '.join(map(str, r['gold_pages']))})")
        A("")
        gold = set(r["gold_chunk_ids"])
        A("Baseline top 3 (cosine):")
        A("")
        A(table(["rank", "page", "chunk", "cosine", "is answer", "text"],
                [[i, h["page"], h["chunk_id"], h["score"],
                  "**YES**" if h["chunk_id"] in gold else "no",
                  preview(h["chunk_id"])]
                 for i, h in enumerate(r["baseline_top3"], 1)]))
        A("")
        A("Re-ranked top 3 (cross-encoder):")
        A("")
        A(table(["rank", "was", "page", "chunk", "cross score", "is answer", "text"],
                [[i, f"#{h['came_from_rank']}", h["page"], h["chunk_id"], h["score"],
                  "**YES**" if h["chunk_id"] in gold else "no",
                  preview(h["chunk_id"])]
                 for i, h in enumerate(r["reranked_top3"], 1)]))
        A("")

    # ------------------------------------------------------------------ #
    if unreachable or still_missed or lost or regressed:
        A("### 2c. Cases where re-ranking did not help")
        A("")
        for r in unreachable:
            where = (f"sits at rank {r['rank_in_top100']} of a top-100 sweep"
                     if r["rank_in_top100"]
                     else "does not appear in a top-100 sweep at all")
            A(f"- **{r['qid']}** ({r['category']}) — *beyond recall.* Answer "
              f"chunk {r['gold_chunk_ids']} on page "
              f"{', '.join(map(str, r['gold_pages']))} {where}, so it never "
              f"enters the 10 candidates and no re-ranker can reach it. "
              f"Query: *{r['query']}*")
        for r in still_missed:
            A(f"- **{r['qid']}** ({r['category']}) — *still missed.* Retrieved "
              f"at #{r['baseline_rank']} but re-ranked to "
              f"#{r['reranked_rank']}, still outside the top 3. "
              f"Query: *{r['query']}*")
        for r in lost:
            A(f"- **{r['qid']}** ({r['category']}) — *lost.* Was in the top 3 "
              f"at #{r['baseline_rank']}, re-ranked out to "
              f"#{r['reranked_rank']}. Query: *{r['query']}*")
        A("")
        harmless = [r for r in regressed
                    if r["in_baseline_context"] and r["in_reranked_context"]]
        if harmless:
            A("Rank regressions that did **not** change what the LLM saw "
              "(both ranks inside the top 3): "
              + ", ".join(f"{r['qid']} (#{r['baseline_rank']} -> "
                          f"#{r['reranked_rank']})" for r in harmless) + ".")
            A("")

    # ------------------------------------------------------------------ #
    A("### 2d. Answer 1 vs Answer 2")
    A("")
    if answers:
        for a in answers:
            A(f"#### {a['qid']} — {a['query']}")
            A("")
            A(f"**Answer 1** (baseline; context pages {a['baseline_pages']}, "
              f"chunks {a.get('baseline_chunks', '-')}):")
            A("")
            A("> " + a["answer_1_baseline"].replace("\n", "\n> "))
            A("")
            A(f"**Answer 2** (re-ranked; context pages {a['reranked_pages']}, "
              f"chunks {a.get('reranked_chunks', '-')}):")
            A("")
            A("> " + a["answer_2_reranked"].replace("\n", "\n> "))
            A("")
    else:
        A("**Not yet generated** — needs `GEMINI_API_KEY` in `.env`. Add it and "
          "re-run the answer cell in the notebook (or "
          "`PYTHONPATH=dev python dev/run_experiment.py`), then regenerate "
          "this file.")
        A("")
    A(PLACEHOLDER)
    A("")

    # ------------------------------------------------------------------ #
    A("## 3. Latency profiling")
    A("")
    if lat:
        A(f"{lat['n_queries']} queries x {lat['repeats_per_query']} repeats = "
          f"{lat['pipelines']['baseline_retrieval']['n']} measurements per "
          f"pipeline. One untimed warm-up pass precedes the run. The LLM call "
          f"is excluded: it is identical work in both pipelines and its network "
          f"variance is far larger than the retrieval times being compared.")
        A("")
        rows = []
        for label, key, where in [
            ("Query embedding (1 pass)", "embed_query", "stages"),
            ("Vector search (whole corpus)", "vector_search", "stages"),
            ("Cross-encoder (10 pairs)", "cross_encode_10_pairs", "stages"),
            ("**Pipeline 1 retrieval**", "baseline_retrieval", "pipelines"),
            ("**Pipeline 2 retrieval**", "reranked_retrieval", "pipelines"),
        ]:
            s = lat[where][key]
            rows.append([label, s["mean_ms"], s["median_ms"], s["p95_ms"],
                         s["min_ms"], s["max_ms"]])
        A(table(["Stage", "mean ms", "median ms", "p95 ms", "min ms", "max ms"], rows))
        A("")
        b = lat["pipelines"]["baseline_retrieval"]["mean_ms"]
        rr = lat["pipelines"]["reranked_retrieval"]["mean_ms"]
        A(f"Pipeline 2 retrieval costs **{lat['overhead_x']}x** Pipeline 1 "
          f"(**+{rr - b:.0f} ms** absolute).")
        A("")
        if answers:
            llm = [a["baseline_llm_ms"] for a in answers] + \
                  [a["reranked_llm_ms"] for a in answers]
            mean_llm = sum(llm) / len(llm)
            A(f"LLM call, measured separately: mean **{mean_llm:.0f} ms** over "
              f"{len(llm)} calls — so against generation, re-ranking adds "
              f"**{100 * (rr - b) / mean_llm:.1f}%** to end-to-end time "
              f"(the LLM dominates by roughly "
              f"**{mean_llm / (rr - b):.0f}x**).")
            A("")
    A(PLACEHOLDER)
    A("")

    # ------------------------------------------------------------------ #
    A("## 4. Scaling question")
    A("")
    if lat:
        per_pair = lat["stages"]["cross_encode_10_pairs"]["mean_ms"] / 10
        two_stage = lat["pipelines"]["reranked_retrieval"]["mean_ms"] / 1000
        A(f"Measured cross-encoder cost: **{per_pair:.1f} ms per (query, chunk) "
          f"pair**, linear in the number of pairs.")
        A("")
        A(table(
            ["Corpus", "Chunks", "Cross-encode everything, per query",
             "Two-stage, per query", "Ratio"],
            [
                [label, f"{n:,}", f"{n * per_pair / 1000:,.1f} s",
                 f"{two_stage:.3f} s", f"{(n * per_pair / 1000) / two_stage:,.0f}x"]
                for label, n in [
                    ("This rulebook", len(chunks)),
                    ("10k chunks", 10_000),
                    ("1M chunks", 1_000_000),
                ]
            ],
        ))
        A("")
    A("Relevant measured facts: the bi-encoder's chunk vectors are "
      "query-independent and were embedded once "
      f"({len(chunks)} x 384); a cross-encoder score exists only for a specific "
      "pair, so nothing is cached between queries. The `beyond recall` rows in "
      "the table in §2a show the ceiling this imposes.")
    A("")
    A(PLACEHOLDER)
    A("")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
    print(f"  rescued      : {[r['qid'] for r in rescued]}")
    print(f"  beyond recall: {[r['qid'] for r in unreachable]}")
    print(f"  still missed : {[r['qid'] for r in still_missed]}")
    print(f"  lost         : {[r['qid'] for r in lost]}")
    print(f"  rank worse   : {[r['qid'] for r in regressed]}")
    print(f"  answers      : {'yes' if answers else 'NOT YET (needs API key)'}")


if __name__ == "__main__":
    main()
