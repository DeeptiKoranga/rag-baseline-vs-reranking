"""Corpus preparation: PDF -> cleaned, section-aware chunks.

Kept as a dev module so the logic can be iterated on quickly; the final
deliverable notebook carries the same code inline.
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH = Path("data/raw/dnd_basic_rules_2018.pdf")
OUT_PATH = Path("data/processed/chunks.json")

# Printed on every single page of the source PDF; pure noise for retrieval.
BOILERPLATE = re.compile(
    r"D&D Basic Rules \(Version 1\.0\)\. Not for resale\. "
    r"Permission granted to print and photocopy this document for personal use only\.",
)
# Running header, e.g. " Chapter 8: Adventuring" or " Part 2 | Chapter 9".
CHAPTER_HEADER = re.compile(r"^\s*(?:Part \d+\s*\|\s*)?Chapter \d+:?.*$", re.M)
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$", re.M)


def page_chapter(text: str) -> str:
    """Read the chapter name off the page's running header."""
    m = re.search(r"Chapter (\d+): ([^\n]+)", text)
    return f"Chapter {m.group(1)}: {m.group(2).strip()}" if m else "Front matter"


def clean_page(text: str) -> str:
    text = BOILERPLATE.sub("", text)
    text = CHAPTER_HEADER.sub("", text)
    text = PAGE_NUMBER_LINE.sub("", text)
    # PyMuPDF emits a tab before each indented paragraph in this PDF.
    text = text.replace("\t", " ")
    # Rejoin words hyphenated across a line break ("Constitu-\ntion").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Soft-wrap inside a paragraph becomes a space; blank lines stay as breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[  ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path = PDF_PATH) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        raw = page.get_text("text")
        pages.append(
            {
                "page": i + 1,  # 1-based, matches the PDF viewer
                "chapter": page_chapter(raw),
                "text": clean_page(raw),
            }
        )
    doc.close()
    return pages


def tail_words(text: str, overlap: int) -> str:
    """Last <=`overlap` characters of `text`, trimmed to a word boundary.

    Slicing on a raw character count leaves chunks that start mid-word
    ("eted by a black pudding"), which is noise both in the embedding and in
    the context handed to the LLM.
    """
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    return tail if len(tail) == len(text) else tail.partition(" ")[2].strip()


def split_long_sentence(sentence: str, size: int) -> list[str]:
    """Last-resort word-level split for spans with no sentence boundary.

    The source has a few table dumps (deity lists, equipment tables) that
    arrive as one unbroken span; without this they blow past `size`.
    """
    out, cur = [], ""
    for word in sentence.split():
        if cur and len(cur) + len(word) + 1 > size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def split_paragraph(para: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware split for a paragraph that is longer than `size`."""
    sentences = []
    for s in re.split(r"(?<=[.!?”])\s+", para):
        sentences.extend([s] if len(s) <= size else split_long_sentence(s, size))

    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            out.append(cur.strip())
            # Carry the tail of the previous piece forward as context.
            cur = f"{tail_words(cur, overlap)} {s}".strip()
        else:
            cur = f"{cur} {s}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_pages(pages: list[dict], size: int = 900, overlap: int = 150) -> list[dict]:
    """Pack paragraphs into ~`size`-character chunks, never crossing a page.

    Staying inside a page keeps the page number on each chunk exact, which is
    what makes the citations in the generated answers checkable by hand.
    """
    chunks = []
    for p in pages:
        buf = ""
        pieces = []
        for para in [x.strip() for x in p["text"].split("\n") if x.strip()]:
            if len(para) > size:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.extend(split_paragraph(para, size, overlap))
                continue
            if buf and len(buf) + len(para) + 1 > size:
                pieces.append(buf)
                buf = para
            else:
                buf = f"{buf} {para}".strip()
        if buf:
            pieces.append(buf)

        for piece in pieces:
            # Drop fragments too short to answer anything (stray captions etc.).
            if len(piece) < 120:
                continue
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "page": p["page"],
                    "chapter": p["chapter"],
                    "text": piece,
                    "n_chars": len(piece),
                }
            )
    return chunks


if __name__ == "__main__":
    pages = extract_pages()
    chunks = chunk_pages(pages)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1))

    lengths = [c["n_chars"] for c in chunks]
    print(f"pages      : {len(pages)}")
    print(f"chunks     : {len(chunks)}")
    print(f"chars/chunk: mean {sum(lengths)/len(lengths):.0f}  "
          f"min {min(lengths)}  max {max(lengths)}")
    print(f"total chars: {sum(lengths):,}")
    print("\n--- sample chunk ---")
    s = chunks[len(chunks) // 2]
    print(f"[{s['chunk_id']}] p.{s['page']} | {s['chapter']}\n{s['text'][:400]}")
