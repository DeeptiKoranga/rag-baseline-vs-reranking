"""Two-stage retrieval over one chunked corpus.

Pipeline 1 (baseline)   : bi-encoder -> top 10 -> take top 3 -> LLM
Pipeline 2 (re-ranking) : bi-encoder -> top 10 -> cross-encoder -> top 3 -> LLM

Both pipelines read the same chunks and use the same prompt template, so the
only variable between Answer 1 and Answer 2 is which three chunks arrive.
"""
from __future__ import annotations  # `X | None` hints on Python 3.9

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-3-flash-preview"

CHUNKS_PATH = Path("data/processed/chunks.json")
EMB_PATH = Path("data/processed/embeddings.npy")

TOP_K_RETRIEVE = 10  # candidates the bi-encoder hands downstream
TOP_N_CONTEXT = 3  # chunks that actually reach the LLM


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    chunk_id: int
    page: int
    chapter: str
    text: str
    score: float
    rank: int  # 1-based, in the stage that produced this hit
    baseline_rank: int | None = None  # where the bi-encoder had put it


@dataclass
class StageTiming:
    embed_query_ms: float = 0.0
    vector_search_ms: float = 0.0
    cross_encode_ms: float = 0.0
    llm_ms: float = 0.0

    @property
    def retrieval_ms(self) -> float:
        """Everything before the LLM — the part the two pipelines differ in."""
        return self.embed_query_ms + self.vector_search_ms + self.cross_encode_ms

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.llm_ms


@dataclass
class PipelineResult:
    query: str
    pipeline: str
    candidates: list[Hit] = field(default_factory=list)  # all 10
    context: list[Hit] = field(default_factory=list)  # the 3 sent to the LLM
    answer: str = ""
    timing: StageTiming = field(default_factory=StageTiming)


# --------------------------------------------------------------------------- #
# Stage 1: dense retrieval (bi-encoder)
# --------------------------------------------------------------------------- #
class DenseIndex:
    """Exact (flat) inner-product index over L2-normalised chunk embeddings.

    The corpus is ~1.1k chunks, so an exact scan is both instant and free of
    the recall loss an approximate index (HNSW/IVF) would add. Keeping it
    exact means any retrieval miss we observe is the embedding model's fault,
    not the index's — which matters for the failure analysis.
    """

    def __init__(self, chunks: list[dict], model_name: str = BI_ENCODER_NAME):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None

    def build(self, cache: Path | None = EMB_PATH, batch_size: int = 64) -> float:
        """Embed every chunk. Returns wall-clock seconds spent indexing."""
        if cache is not None and cache.exists():
            cached = np.load(cache)
            if cached.shape[0] == len(self.chunks):
                self.embeddings = cached
                return 0.0

        t0 = time.perf_counter()
        self.embeddings = self.model.encode(
            [c["text"] for c in self.chunks],
            batch_size=batch_size,
            normalize_embeddings=True,  # so a dot product *is* cosine similarity
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        elapsed = time.perf_counter() - t0

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.embeddings)
        return elapsed

    def search(self, query: str, k: int = TOP_K_RETRIEVE) -> tuple[list[Hit], StageTiming]:
        t = StageTiming()

        t0 = time.perf_counter()
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        t.embed_query_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        scores = (self.embeddings @ q[0]).astype(np.float32)
        # argpartition first: O(n) to isolate the top k, then sort just those k.
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        t.vector_search_ms = (time.perf_counter() - t0) * 1000

        hits = []
        for rank, idx in enumerate(top, start=1):
            c = self.chunks[int(idx)]
            hits.append(
                Hit(
                    chunk_id=c["chunk_id"],
                    page=c["page"],
                    chapter=c["chapter"],
                    text=c["text"],
                    score=float(scores[idx]),
                    rank=rank,
                    baseline_rank=rank,
                )
            )
        return hits, t


# --------------------------------------------------------------------------- #
# Stage 2: cross-encoder re-ranking
# --------------------------------------------------------------------------- #
class Reranker:
    """Scores each (query, chunk) pair jointly, then re-sorts."""

    def __init__(self, model_name: str = CROSS_ENCODER_NAME):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Hit]) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        scores = self.model.predict(
            [(query, h.text) for h in candidates],
            show_progress_bar=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        order = np.argsort(-np.asarray(scores))
        reranked = []
        for new_rank, i in enumerate(order, start=1):
            src = candidates[int(i)]
            reranked.append(
                Hit(
                    chunk_id=src.chunk_id,
                    page=src.page,
                    chapter=src.chapter,
                    text=src.text,
                    score=float(scores[int(i)]),
                    rank=new_rank,
                    baseline_rank=src.baseline_rank,
                )
            )
        return reranked, elapsed_ms


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are answering questions about a tabletop game rulebook.
Use ONLY the numbered context passages below. Do not use outside knowledge.

Rules for your answer:
- If the passages do not contain the answer, reply exactly: NOT IN CONTEXT.
- State any exception, restriction or penalty that the passages mention.
- Cite the page number for each rule you state, like (p. 46).
- Answer in at most 120 words.

CONTEXT
{context}

QUESTION
{query}

ANSWER"""


def build_prompt(query: str, context: list[Hit]) -> str:
    """Identical template for both pipelines; only `context` differs."""
    blocks = [
        f"[{i}] (page {h.page} | {h.chapter})\n{h.text}"
        for i, h in enumerate(context, start=1)
    ]
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), query=query)


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader so the API key never has to be pasted into code."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class GeminiGenerator:
    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai

        load_env()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key. Put GEMINI_API_KEY=... in rag_assignment/.env "
                "(see .env.example)."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, attempts: int = 6) -> tuple[str, float]:
        """Generate, retrying on the transient errors the free tier throws.

        429 (quota) and 503 (model busy) are routine here and would otherwise
        abort a multi-query run part-way. Only the successful call is timed,
        so retries do not contaminate the latency figures.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0,  # same context must give the same answer
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        for attempt in range(attempts):
            try:
                t0 = time.perf_counter()
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                return (resp.text or "").strip(), (time.perf_counter() - t0) * 1000
            except Exception as e:
                transient = any(c in str(e) for c in ("429", "503", "500", "UNAVAILABLE"))
                if not transient or attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt * 4)  # 4s, 8s, 16s, 32s, 64s
        raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# The two pipelines
# --------------------------------------------------------------------------- #
class RagSystem:
    def __init__(self, index: DenseIndex, reranker: Reranker, generator=None):
        self.index = index
        self.reranker = reranker
        self.generator = generator

    def _answer(self, result: PipelineResult) -> None:
        if self.generator is None:
            result.answer = "(no LLM configured)"
            return
        result.answer, result.timing.llm_ms = self.generator.generate(
            build_prompt(result.query, result.context)
        )

    def baseline(self, query: str, generate: bool = True) -> PipelineResult:
        """Pipeline 1: dense top-10, take top-3 as-is."""
        candidates, timing = self.index.search(query, k=TOP_K_RETRIEVE)
        r = PipelineResult(
            query=query,
            pipeline="baseline",
            candidates=candidates,
            context=candidates[:TOP_N_CONTEXT],
            timing=timing,
        )
        if generate:
            self._answer(r)
        return r

    def reranked(
        self, query: str, generate: bool = True, candidates: list[Hit] | None = None,
        timing: StageTiming | None = None,
    ) -> PipelineResult:
        """Pipeline 2: same dense top-10, re-sorted by the cross-encoder."""
        if candidates is None:
            candidates, timing = self.index.search(query, k=TOP_K_RETRIEVE)
        timing = timing or StageTiming()

        reranked, cross_ms = self.reranker.rerank(query, candidates)
        t = StageTiming(
            embed_query_ms=timing.embed_query_ms,
            vector_search_ms=timing.vector_search_ms,
            cross_encode_ms=cross_ms,
        )
        r = PipelineResult(
            query=query,
            pipeline="reranked",
            candidates=reranked,
            context=reranked[:TOP_N_CONTEXT],
            timing=t,
        )
        if generate:
            self._answer(r)
        return r


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    return json.loads(path.read_text())
