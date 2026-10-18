"""Generate report/report.tex -- the typeset report, data already filled in.

Every table, figure and number comes from results/, so the document cannot
drift from what the notebook actually measured. The four analysis sections
are left as marked gaps for the author to write: the assignment requires the
report prose to be the author's own.

    python3 dev/build_report_tex.py
    tectonic report/report.tex
"""
import json
import re
from pathlib import Path

RESULTS = Path("results")
OUT = Path("report/report.tex")

# How much chunk text to show in the walkthrough tables.
PREVIEW = 58


def esc(s):
    """Escape the LaTeX specials that appear in this corpus (D&D, 5%, ...)."""
    s = str(s).replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    # Curly quotes and dashes the PDF text is full of.
    for a, b in [("’", "'"), ("‘", "'"), ("“", "``"),
                 ("”", "''"), ("—", "---"), ("–", "--"),
                 ("•", r"$\bullet$"), (" ", " ")]:
        s = s.replace(a, b)
    return s


def esc_note(s):
    """Like esc(), but turns the notes' *markdown emphasis* into italics.

    `esc` leaves `*` alone (it is not a TeX special), so the markers survive
    escaping and can be matched here.
    """
    return re.sub(r"\*([^*]+)\*", r"\\textit{\1}", esc(s))


def preview(chunks, cid, n=PREVIEW):
    return esc(" ".join(chunks[cid]["text"].split())[:n])


def gap(prompt):
    """A clearly marked place for the author's own analysis."""
    return "\n".join([
        r"\begin{analysisgap}",
        f"% TO WRITE: {prompt}",
        r"\textit{[Your analysis here.]}",
        r"\end{analysisgap}",
        "",
    ])


def main():
    OUT.parent.mkdir(exist_ok=True)
    ranks = json.loads((RESULTS / "rank_analysis.json").read_text())
    lat = json.loads((RESULTS / "latency.json").read_text())
    answers = json.loads((RESULTS / "answers.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    chunks = {c["chunk_id"]: c
              for c in json.loads(Path("data/processed/chunks.json").read_text())}

    def outcome(r):
        if r["baseline_rank"] is None:
            return "beyond recall"
        if r["in_reranked_context"] and not r["in_baseline_context"]:
            return r"\textbf{rescued}"
        if r["in_baseline_context"] and not r["in_reranked_context"]:
            return "lost"
        return "already in top 3" if r["in_baseline_context"] else "still missed"

    def shift(r):
        if not r["baseline_rank"] or not r["reranked_rank"]:
            return "--"
        d = r["baseline_rank"] - r["reranked_rank"]
        return f"$+{d}$" if d > 0 else (f"${d}$" if d < 0 else "0")

    rescued = [r for r in ranks if outcome(r) == r"\textbf{rescued}"]
    unreachable = [r for r in ranks if outcome(r) == "beyond recall"]
    still_missed = [r for r in ranks if outcome(r) == "still missed"]

    per_pair = lat["stages"]["cross_encode_10_pairs"]["mean_ms"] / 10
    b_ms = lat["pipelines"]["baseline_retrieval"]["mean_ms"]
    rr_ms = lat["pipelines"]["reranked_retrieval"]["mean_ms"]
    llm = [a["baseline_llm_ms"] for a in answers] + \
          [a["reranked_llm_ms"] for a in answers]
    mean_llm = sum(llm) / len(llm)

    L = []
    A = L.append

    # ------------------------------ preamble ----------------------------- #
    A(r"""\documentclass[11pt,a4paper]{article}

\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{mdframed}
\usepackage{xurl}        % allow line breaks inside long URLs
\usepackage[hidelinks]{hyperref}

\newcolumntype{Y}{>{\raggedright\arraybackslash}X}

% Marks the sections the author writes. Delete the environment definition
% and the \begin/\end pairs once the prose is in.
\newmdenv[linecolor=gray!55,linewidth=0.5pt,backgroundcolor=gray!6,
          skipabove=6pt,skipbelow=6pt,leftmargin=0pt,rightmargin=0pt,
          innertopmargin=6pt,innerbottommargin=6pt]{analysisgap}

\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}

\title{\textbf{RAG: Baseline Retrieval vs.\ Cross-Encoder Re-Ranking}}
\author{Deepti Koranga}
\date{\today}

\begin{document}
\maketitle
""")

    # ------------------------------- setup ------------------------------- #
    doc, mod, ret = summary["document"], summary["models"], summary["retrieval"]
    A(r"\section*{Setup}")
    A(r"\begin{center}\begin{tabular}{ll}\toprule")
    A(r"\textbf{Item} & \textbf{Value} \\ \midrule")
    arrow = r"$\rightarrow$"
    for k, v in [
        ("Document", r"D\&D Basic Rules (2018), Wizards of the Coast"),
        ("Pages / chunks", f"{doc['pages']} pages {arrow} {doc['chunks']} chunks"),
        ("Chunking", f"{doc['mean_chunk_chars']:.0f} chars mean, 150 overlap, page-bounded"),
        ("Bi-encoder", r"\texttt{" + esc(mod["bi_encoder"]) + "}"),
        ("Cross-encoder", r"\texttt{" + esc(mod["cross_encoder"]) + "}"),
        ("Generator", r"\texttt{" + esc(mod["generator"]) + "}, temperature 0"),
        ("Retrieve / context", f"top {ret['top_k_retrieve']} candidates, "
                               f"top {ret['top_n_context']} to the LLM"),
        ("Stress queries", str(ret["queries"])),
    ]:
        A(esc(k) + " & " + v + r" \\")
    A(r"\bottomrule\end{tabular}\end{center}")
    A(r"""
Code and the executed notebook:
\url{https://github.com/DeeptiKoranga/rag-baseline-vs-reranking}
""")

    # --------------------------- 1. architecture -------------------------- #
    A(r"\section{Architecture Comparison}")
    A(r"""
Measured quantities relevant to this section:
""")
    A(r"\begin{center}\begin{tabular}{lr}\toprule")
    A(r"\textbf{Quantity} & \textbf{Measured} \\ \midrule")
    for k, v in [
        ("Chunk vectors embedded once, reused every query",
         f"${doc['chunks']} \\times 384$"),
        ("Query embedding (one forward pass)",
         f"{lat['stages']['embed_query']['mean_ms']:.2f} ms"),
        ("Vector search over the whole corpus",
         f"{lat['stages']['vector_search']['mean_ms']:.2f} ms"),
        ("Cross-encoder, 10 pairs",
         f"{lat['stages']['cross_encode_10_pairs']['mean_ms']:.2f} ms"),
        ("Cross-encoder, per (query, chunk) pair", f"{per_pair:.1f} ms"),
    ]:
        A(f"{esc(k)} & {v} \\\\")
    A(r"\bottomrule\end{tabular}\end{center}")
    A(gap("Bi-encoder = two independent embeddings compared by cosine; "
          "cross-encoder = one joint sequence, full self-attention across "
          "query and chunk tokens, single relevance score. Tie it to the "
          "numbers above: what is cacheable and what is not."))

    # ---------------------- 2. rank shift / qualitative ------------------- #
    A(r"\section{Qualitative and Rank Shift Analysis}")
    A(r"""
\subsection{How a failure is measured}

For each query the chunk that actually answers it is pinned in advance by a
regular expression matching a phrase unique to that chunk. Every claim below
is therefore a rank, not an impression. \textit{rank$_{100}$} is the
answer-bearing chunk's position in a top-100 bi-encoder sweep (diagnostic
only); \textit{base} and \textit{rerank} are its position among the 10
candidates before and after the cross-encoder. A dash means absent from that
set. A query is \textbf{rescued} when the answer chunk was outside the top 3
for the baseline and inside it after re-ranking.
""")

    # --- full rank table
    A(r"\subsection{Full rank table}")
    A(r"\begin{center}\footnotesize\begin{tabular}{llrrrrcl}\toprule")
    A(r"\textbf{ID} & \textbf{Category} & \textbf{p.} & "
      r"\textbf{rank$_{100}$} & \textbf{base} & \textbf{rerank} & "
      r"\textbf{shift} & \textbf{Outcome} \\ \midrule")
    for r in ranks:
        A(" & ".join([
            r["qid"],
            esc(r["category"].replace(" / ", "/").replace(
                "Control (baseline expected to succeed)", "Control")),
            ", ".join(map(str, r["gold_pages"])),
            str(r["rank_in_top100"] or "--"),
            str(r["baseline_rank"] or "--"),
            str(r["reranked_rank"] or "--"),
            shift(r),
            outcome(r),
        ]) + r" \\")
    A(r"\bottomrule\end{tabular}\end{center}")

    A(r"\begin{center}\begin{tabular}{lr}\toprule")
    A(r"\textbf{Summary} & \textbf{Count} \\ \midrule")
    for k, v in [
        ("Queries", len(ranks)),
        ("Answer chunk in top 3 -- baseline", ret["gold_in_top3_baseline"]),
        ("Answer chunk in top 3 -- re-ranked", ret["gold_in_top3_reranked"]),
        ("Rescued by re-ranking", len(rescued)),
        ("Never retrieved (re-ranking cannot help)", len(unreachable)),
        ("Retrieved but still outside top 3", len(still_missed)),
    ]:
        A(f"{esc(k)} & {v} \\\\")
    A(r"\bottomrule\end{tabular}\end{center}")

    # --- rescued walkthroughs
    A(r"\subsection{Rank shifts, with what the baseline ranked instead}")
    for r in rescued:
        A(r"\subsubsection*{%s --- %s --- Rank \#%s $\rightarrow$ Rank \#%s}"
          % (r["qid"], esc(r["category"]), r["baseline_rank"], r["reranked_rank"]))
        A(r"\textbf{Query:} \textit{%s}" % esc(r["query"]))
        A("")
        A(r"\textbf{Why it is hard:} %s" % esc_note(r["note"]))
        A("")
        gold = set(r["gold_chunk_ids"])
        for label, key, scorecol in [
            ("Baseline top 3 (cosine)", "baseline_top3", "cosine"),
            ("Re-ranked top 3 (cross-encoder)", "reranked_top3", "score"),
        ]:
            A(r"\textbf{%s}" % label)
            A(r"\begin{center}\footnotesize"
              r"\begin{tabularx}{\textwidth}{rrrrcY}\toprule")
            A(r"\textbf{\#} & \textbf{was} & \textbf{p.} & \textbf{chunk} & "
              r"\textbf{%s} & \textbf{text} \\ \midrule" % scorecol)
            for i, h in enumerate(r[key], 1):
                was = (f"\\#{h['came_from_rank']}" if "came_from_rank" in h else "--")
                mark = r"\textbf{$\leftarrow$ ANSWER}" if h["chunk_id"] in gold else ""
                A(" & ".join([
                    str(i), was, str(h["page"]), str(h["chunk_id"]),
                    f"{h['score']:.3f}",
                    preview(chunks, h["chunk_id"]) + " " + mark,
                ]) + r" \\")
            A(r"\bottomrule\end{tabularx}\end{center}")
        A("")

    # --- failures not fixed
    A(r"\subsection{Cases where re-ranking did not help}")
    A(r"\begin{itemize}[leftmargin=1.2em,itemsep=3pt]")
    for r in unreachable:
        where = (f"sits at rank {r['rank_in_top100']} of a top-100 sweep"
                 if r["rank_in_top100"]
                 else "does not appear in a top-100 sweep at all")
        A(r"\item \textbf{%s} (%s) --- \textit{beyond recall.} The answer chunk "
          r"on p.~%s %s, so it never enters the 10 candidates and no re-ranker "
          r"can reach it. Query: \textit{%s}"
          % (r["qid"], esc(r["category"]), ", ".join(map(str, r["gold_pages"])),
             where, esc(r["query"])))
    for r in still_missed:
        A(r"\item \textbf{%s} (%s) --- \textit{still missed.} Retrieved at "
          r"\#%s, re-ranked to \#%s, still outside the top 3. "
          r"Query: \textit{%s}"
          % (r["qid"], esc(r["category"]), r["baseline_rank"],
             r["reranked_rank"], esc(r["query"])))
    A(r"\end{itemize}")

    # --- answers
    A(r"\subsection{Answer 1 vs.\ Answer 2}")
    for a in answers:
        A(r"\subsubsection*{%s --- %s}" % (a["qid"], esc(a["query"])))
        for tag, key, pages, ch in [
            ("Answer 1 (baseline)", "answer_1_baseline",
             a["baseline_pages"], a["baseline_chunks"]),
            ("Answer 2 (re-ranked)", "answer_2_reranked",
             a["reranked_pages"], a["reranked_chunks"]),
        ]:
            A(r"\textbf{%s} --- context pages %s, chunks %s:"
              % (tag, esc(pages), esc(ch)))
            A(r"\begin{quote}\itshape %s\end{quote}"
              % esc(" ".join(a[key].split())))
        A("")
    A(gap("Per rescued query: which rank moved, and did the baseline "
          "hallucinate, omit a clause, or misstate a rule? Note that Q9/Q11 "
          "refused outright while Q14 answered confidently and wrongly -- "
          "those are different failure shapes. Then address the cases "
          "re-ranking did not fix."))

    # ----------------------------- 3. latency ----------------------------- #
    A(r"\section{Latency Profiling}")
    A(r"""
%d queries $\times$ %d repeats $=$ %d measurements per pipeline, after one
untimed warm-up pass. The LLM call is excluded from the pipeline comparison:
it is identical work in both pipelines and its variance is far larger than
the retrieval times under test. It is reported separately below.
""" % (lat["n_queries"], lat["repeats_per_query"],
       lat["pipelines"]["baseline_retrieval"]["n"]))

    A(r"\begin{center}\begin{tabular}{lrrrrr}\toprule")
    A(r"\textbf{Stage} & \textbf{mean} & \textbf{median} & \textbf{p95} & "
      r"\textbf{min} & \textbf{max} \\")
    A(r" & (ms) & (ms) & (ms) & (ms) & (ms) \\ \midrule")
    for label, key, where in [
        ("Query embedding (1 pass)", "embed_query", "stages"),
        ("Vector search (whole corpus)", "vector_search", "stages"),
        ("Cross-encoder (10 pairs)", "cross_encode_10_pairs", "stages"),
    ]:
        s = lat[where][key]
        A(f"{esc(label)} & {s['mean_ms']} & {s['median_ms']} & {s['p95_ms']} "
          f"& {s['min_ms']} & {s['max_ms']} \\\\")
    A(r"\midrule")
    for label, key in [("Pipeline 1 retrieval", "baseline_retrieval"),
                       ("Pipeline 2 retrieval", "reranked_retrieval")]:
        s = lat["pipelines"][key]
        A(f"\\textbf{{{esc(label)}}} & \\textbf{{{s['mean_ms']}}} & "
          f"{s['median_ms']} & {s['p95_ms']} & {s['min_ms']} & {s['max_ms']} \\\\")
    A(r"\bottomrule\end{tabular}\end{center}")

    A(r"""
Pipeline 2 retrieval costs $\mathbf{%.1f\times}$ Pipeline 1
($\mathbf{+%.0f}$ ms absolute). Measured separately, the LLM call averages
$\mathbf{%.0f}$ ms over %d calls, so against generation the re-ranker adds
$\mathbf{%.1f\%%}$ to end-to-end latency --- the generator dominates by
roughly $%.0f\times$.
""" % (lat["overhead_x"], rr_ms - b_ms, mean_llm, len(llm),
       100 * (rr_ms - b_ms) / mean_llm, mean_llm / (rr_ms - b_ms)))

    A(r"""
\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../figures/latency.png}
\caption{Per-stage cost (left) and end-to-end retrieval latency (right).}
\end{figure}
""")
    A(gap("Where the cost actually sits, and whether 6x matters once the "
          "LLM is in the loop. The two framings disagree -- say which one "
          "is the right one to design against, and why."))

    # ----------------------------- 4. scaling ----------------------------- #
    A(r"\section{Scaling: Why Not Cross-Encode Everything?}")
    A(r"""
The cross-encoder costs %.1f~ms per (query, chunk) pair and is linear in the
number of pairs, because a score exists only for a specific pair and nothing
can be precomputed between queries.
""" % per_pair)
    A(r"\begin{center}\begin{tabular}{lrrrr}\toprule")
    A(r"\textbf{Corpus} & \textbf{Chunks} & \textbf{Cross-encode all} & "
      r"\textbf{Two-stage} & \textbf{Ratio} \\")
    A(r" & & (s/query) & (s/query) & \\ \midrule")
    for label, n in [("This rulebook", doc["chunks"]),
                     ("10k chunks", 10_000),
                     ("1M chunks", 1_000_000)]:
        full = n * per_pair / 1000
        two = rr_ms / 1000
        A(f"{esc(label)} & {n:,} & {full:,.1f} & {two:.3f} & "
          f"{full / two:,.0f}$\\times$ \\\\")
    A(r"\bottomrule\end{tabular}\end{center}")
    A(gap("The arithmetic above is the cost argument. Add the structural "
          "one: bi-encoder chunk vectors are query-independent and "
          "precomputable, cross-encoder scores are not, so no index can "
          "prune the candidate set. Then close the loop -- the beyond-recall "
          "rows in Section 2 are the price this design charges."))

    A(r"\end{document}")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
    print(f"  analysis gaps to fill: {sum(1 for x in L if 'TO WRITE' in x)}")
    print(f"  rescued: {[r['qid'] for r in rescued]} | "
          f"beyond recall: {[r['qid'] for r in unreachable]} | "
          f"still missed: {[r['qid'] for r in still_missed]}")


if __name__ == "__main__":
    main()
