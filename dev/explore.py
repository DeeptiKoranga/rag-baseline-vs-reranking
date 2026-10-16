"""Scratch harness: try many candidate queries, keep the ones that move.

Used to pick the stress queries empirically instead of assuming which ones
would trip the bi-encoder. Not part of the deliverable.
"""
from __future__ import annotations

import re
import sys

from pipeline import TOP_K_RETRIEVE, TOP_N_CONTEXT, DenseIndex, Reranker, load_chunks

# (label, query, gold regex)
CANDIDATES = [
    ("trap-cover", "Is it harder to hit someone crouching behind a low stone wall?",
     r"half cover|three-quarters cover"),
    ("trap-flee", "Do I get hurt if I walk away from a monster standing next to me?",
     r"opportunity attack"),
    ("trap-unseen", "What happens when I swing at something I cannot see?",
     r"Unseen Attackers and Targets"),
    ("trap-potion", "Can I drink a potion and still swing my sword in the same turn?",
     r"bonus action"),
    ("trap-tie", "What happens if two people roll the same initiative?",
     r"decides the order among tied"),
    ("trap-dim", "Why can I not tell colours apart in the dark?",
     r"can.t discern color in darkness"),
    ("trap-carry", "How much stuff can my character haul around before it is too heavy?",
     r"carrying capacity"),
    ("trap-hide-behind", "Can I hide if the enemy is looking straight at me?",
     r"can.t hide from a creature"),
    ("exc-crit", "When I score a critical hit, does my damage bonus double as well?",
     r"Critical Hits"),
    ("exc-deadheal", "Can a healing spell bring back someone who has already died?",
     r"A dead creature|creature that has died"),
    ("exc-twospells", "Can I cast two spells in one turn if one is a cantrip?",
     r"Bonus Action.{0,200}cast"),
    ("exc-armorprof", "What goes wrong if I put on armour I was never trained to use?",
     r"Strength or Dexterity, and you can.t cast spells"),
    ("neg-unconscious", "What is a knocked-out character unable to do?",
     r"Unconscious"),
    ("neg-prone", "What am I not able to do while lying flat on the ground?",
     r"Prone"),
    ("neg-ritual", "Which spells am I not allowed to cast as a ritual?",
     r"ritual tag"),
    ("bur-light", "How far does a torch actually let me see?",
     r"bright light for a 20-foot radius"),
    ("bur-forcedmarch", "What happens if the party keeps walking for more than eight hours?",
     r"forced march|Constitution saving throw.{0,80}exhaustion"),
    ("bur-exhaust", "What are the penalties for being completely worn out?",
     r"Exhaustion"),
    ("bur-starve", "How many days can someone go without food?",
     r"without food.{0,120}exhaustion|Food.{0,80}days"),
    ("bur-mount", "What happens to my horse if I get knocked off it?",
     r"Mounted Combat|fall off the mount"),
]


def main() -> None:
    chunks = load_chunks()
    index = DenseIndex(chunks)
    index.build()
    reranker = Reranker()

    print(f"{'label':18} {'base':>5} {'rr':>4}  outcome")
    print("-" * 52)
    keepers = []
    for label, query, gold in CANDIDATES:
        gold_ids = {c["chunk_id"] for c in chunks if re.search(gold, c["text"], re.I)}
        if not gold_ids:
            print(f"{label:18} {'--':>5} {'--':>4}  NO GOLD MATCH")
            continue
        if len(gold_ids) > 4:
            print(f"{label:18} {'--':>5} {'--':>4}  gold too broad ({len(gold_ids)} chunks)")
            continue

        hits, _ = index.search(query, k=TOP_K_RETRIEVE)
        rr, _ = reranker.rerank(query, hits)
        b = next((h.rank for h in hits if h.chunk_id in gold_ids), None)
        r = next((h.rank for h in rr if h.chunk_id in gold_ids), None)
        in_b = bool(b and b <= TOP_N_CONTEXT)
        in_r = bool(r and r <= TOP_N_CONTEXT)
        outcome = ("RESCUED" if in_r and not in_b else
                   "already ok" if in_b else
                   "unreachable" if b is None else "still missed")
        if outcome == "RESCUED":
            keepers.append((label, query, gold, b, r))
        print(f"{label:18} {str(b):>5} {str(r):>4}  {outcome}")

    print(f"\nrescued: {len(keepers)}")
    for label, query, gold, b, r in keepers:
        print(f"  {label}: #{b} -> #{r}\n      {query}\n      gold=/{gold}/")


if __name__ == "__main__":
    sys.exit(main())
