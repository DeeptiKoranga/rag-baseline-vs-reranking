"""Run both pipelines over the query pool and record what changed.

Retrieval and re-ranking need no API key, so this script runs them always and
only calls the LLM when a key is present. Everything it measures lands in
results/ as JSON so the notebook and the report read the same numbers.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from pipeline import (
    TOP_K_RETRIEVE,
    TOP_N_CONTEXT,
    DenseIndex,
    GeminiGenerator,
    RagSystem,
    Reranker,
    load_chunks,
)
from queries import QUERY_POOL

RESULTS = Path("results")
DIAGNOSTIC_K = 100  # wider sweep, only to see where the gold chunk really sits


def gold_chunk_ids(pattern: str, chunks: list[dict]) -> set[int]:
    return {c["chunk_id"] for c in chunks if re.search(pattern, c["text"], re.I)}


def rank_of_gold(hits, gold: set[int]) -> int | None:
    """1-based rank of the best-placed gold chunk, or None if absent."""
    for h in hits:
        if h.chunk_id in gold:
            return h.rank
    return None


def analyse(index: DenseIndex, reranker: Reranker, chunks: list[dict]) -> list[dict]:
    rows = []
    for q in QUERY_POOL:
        gold = gold_chunk_ids(q["gold"], chunks)

        # Wide sweep purely for diagnosis: is a miss a ranking problem inside
        # the top 10, or did the bi-encoder never surface the chunk at all?
        deep, _ = index.search(q["query"], k=DIAGNOSTIC_K)
        deep_rank = rank_of_gold(deep, gold)

        candidates = deep[:TOP_K_RETRIEVE]
        for i, h in enumerate(candidates, start=1):
            h.rank = i
            h.baseline_rank = i
        reranked, _ = reranker.rerank(q["query"], candidates)

        base_rank = rank_of_gold(candidates, gold)
        rr_rank = rank_of_gold(reranked, gold)

        rows.append(
            {
                "qid": q["qid"],
                "category": q["category"],
                "query": q["query"],
                "note": q["note"],
                "gold_chunk_ids": sorted(gold),
                "gold_pages": sorted({c["page"] for c in chunks if c["chunk_id"] in gold}),
                "rank_in_top100": deep_rank,
                "baseline_rank": base_rank,
                "reranked_rank": rr_rank,
                "in_baseline_context": bool(base_rank and base_rank <= TOP_N_CONTEXT),
                "in_reranked_context": bool(rr_rank and rr_rank <= TOP_N_CONTEXT),
                "baseline_top3": [
                    {"chunk_id": h.chunk_id, "page": h.page, "score": round(h.score, 4)}
                    for h in candidates[:TOP_N_CONTEXT]
                ],
                "reranked_top3": [
                    {
                        "chunk_id": h.chunk_id,
                        "page": h.page,
                        "score": round(h.score, 4),
                        "came_from_rank": h.baseline_rank,
                    }
                    for h in reranked[:TOP_N_CONTEXT]
                ],
            }
        )
    return rows


def profile_latency(index: DenseIndex, reranker: Reranker, repeats: int = 5) -> dict:
    """Time the retrieval stages. The LLM call is excluded on purpose: it is
    identical work for both pipelines and its variance would swamp everything.
    """
    qs = [q["query"] for q in QUERY_POOL]
    base, rr, embed, search, cross = [], [], [], [], []

    # One untimed pass so lazy model init / caches don't land in the numbers.
    index.search(qs[0], k=TOP_K_RETRIEVE)
    reranker.rerank(qs[0], index.search(qs[0], k=TOP_K_RETRIEVE)[0])

    for _ in range(repeats):
        for query in qs:
            cands, t = index.search(query, k=TOP_K_RETRIEVE)
            embed.append(t.embed_query_ms)
            search.append(t.vector_search_ms)
            base.append(t.retrieval_ms)

            _, cross_ms = reranker.rerank(query, cands)
            cross.append(cross_ms)
            rr.append(t.retrieval_ms + cross_ms)

    def stat(xs):
        return {
            "mean_ms": round(statistics.mean(xs), 2),
            "median_ms": round(statistics.median(xs), 2),
            "p95_ms": round(sorted(xs)[int(0.95 * len(xs)) - 1], 2),
            "min_ms": round(min(xs), 2),
            "max_ms": round(max(xs), 2),
            "n": len(xs),
        }

    return {
        "repeats_per_query": repeats,
        "n_queries": len(qs),
        "stages": {
            "embed_query": stat(embed),
            "vector_search": stat(search),
            "cross_encode_10_pairs": stat(cross),
        },
        "pipelines": {
            "baseline_retrieval": stat(base),
            "reranked_retrieval": stat(rr),
        },
        "overhead_x": round(statistics.mean(rr) / statistics.mean(base), 2),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    chunks = load_chunks()
    print(f"chunks: {len(chunks)}")

    index = DenseIndex(chunks)
    build_s = index.build()
    print(f"index built in {build_s:.1f}s (0.0 = loaded from cache)")
    reranker = Reranker()

    print("\n=== rank analysis ===")
    rows = analyse(index, reranker, chunks)
    hdr = f"{'qid':4} {'top100':>6} {'base':>5} {'rerank':>6}  category"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['qid']:4} {str(r['rank_in_top100']):>6} "
            f"{str(r['baseline_rank']):>5} {str(r['reranked_rank']):>6}  {r['category']}"
        )
    (RESULTS / "rank_analysis.json").write_text(json.dumps(rows, indent=1))

    rescued = [
        r for r in rows if not r["in_baseline_context"] and r["in_reranked_context"]
    ]
    print(f"\nrescued by re-ranking: {len(rescued)}/{len(rows)} -> "
          f"{[r['qid'] for r in rescued]}")

    print("\n=== latency ===")
    lat = profile_latency(index, reranker)
    (RESULTS / "latency.json").write_text(json.dumps(lat, indent=1))
    print(json.dumps(lat["pipelines"], indent=1))
    print(f"re-ranking costs {lat['overhead_x']}x baseline retrieval")

    # --- answers, only if a key is configured ---------------------------- #
    try:
        generator = GeminiGenerator()
    except RuntimeError as e:
        print(f"\n[skipped answer generation] {e}")
        return

    rag = RagSystem(index, reranker, generator)
    answers = []
    targets = rescued or rows[:4]
    print(f"\n=== generating answers for {[r['qid'] for r in targets]} ===")
    for r in targets:
        b = rag.baseline(r["query"])
        rr = rag.reranked(r["query"])
        answers.append(
            {
                "qid": r["qid"],
                "query": r["query"],
                "answer_1_baseline": b.answer,
                "answer_2_reranked": rr.answer,
                "baseline_pages": [h.page for h in b.context],
                "reranked_pages": [h.page for h in rr.context],
                "baseline_chunks": [h.chunk_id for h in b.context],
                "reranked_chunks": [h.chunk_id for h in rr.context],
                "baseline_llm_ms": round(b.timing.llm_ms, 1),
                "reranked_llm_ms": round(rr.timing.llm_ms, 1),
            }
        )
        print(f"\n--- {r['qid']}: {r['query']}")
        print(f"  A1 (pages {answers[-1]['baseline_pages']}): {b.answer[:220]}")
        print(f"  A2 (pages {answers[-1]['reranked_pages']}): {rr.answer[:220]}")
        time.sleep(1)  # stay well inside the free-tier rate limit

    (RESULTS / "answers.json").write_text(json.dumps(answers, indent=1))


if __name__ == "__main__":
    main()
