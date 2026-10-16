"""Stress-test query pool.

Each entry names the chunk that actually answers the question, via a regex
matching a phrase unique to that chunk (`gold`). That makes "the baseline
missed it" a measurement rather than an impression: we can report exactly
what rank the answer-bearing chunk held before and after re-ranking.

The queries are deliberately worded the way a player would ask -- with
everyday synonyms rather than the rulebook's own vocabulary. That mismatch is
what a bi-encoder is weakest at, and it is the whole point of the exercise.
"""

QUERY_POOL = [
    # ---------------------- keyword trap / distractor ---------------------- #
    {
        "qid": "Q1",
        "category": "Keyword trap / distractor",
        "query": "Does wearing heavy plate make it harder to sneak past guards unnoticed?",
        "gold": r"Stealth column",
        "note": (
            "The decisive rule is in the Equipment chapter and never says "
            "'sneak' or 'guards'. The Skills section does: it literally reads "
            "'slink past guards'. So the query's own words point at the wrong "
            "chunk."
        ),
    },
    {
        "qid": "Q2",
        "category": "Keyword trap / distractor",
        "query": "How much gold per day does a comfortable standard of living cost?",
        "gold": r"prices listed are per day",
        "note": (
            "'Comfortable' and 'wealthy' appear all over the personality-trait "
            "and background tables, which have nothing to do with expenses."
        ),
    },
    {
        "qid": "Q3",
        "category": "Keyword trap / distractor",
        "query": "What do I roll to spot an enemy trying to hide from me?",
        "gold": r"contest.{0,80}Wisdom \(Perception\)|Wisdom \(Perception\) check.{0,80}contest",
        "note": "'Hide' and 'spot' collide with dozens of rogue and spell entries.",
    },
    # ------------------------ exceptions / corner cases -------------------- #
    {
        "qid": "Q4",
        "category": "Exception / corner case",
        "query": "If a fight comes at us in several waves, can we rest in between to heal up?",
        "gold": r"between parts of a multipart encounter",
        "note": (
            "The general resting rules say a short rest restores hit points. "
            "The restriction that denies it mid-encounter sits ~96 pages away "
            "in the DM's encounter-building chapter."
        ),
    },
    {
        "qid": "Q5",
        "category": "Exception / corner case",
        "query": "After a full night of sleep, do I get all of my Hit Dice back?",
        "gold": r"regains spent Hit Dice, up to a number of dice equal to half",
        "note": (
            "The intuitive answer is 'yes, everything resets'. The actual rule "
            "grants only half, and that clause is one sentence inside a long "
            "passage about resting."
        ),
    },
    {
        "qid": "Q6",
        "category": "Exception / corner case",
        "query": "Can I wear a second shield on my other arm for more protection?",
        "gold": r"benefit from only one shield at a time",
        "note": "A one-clause restriction buried in a chunk about armor types.",
    },
    # --------------------------- negation / constraints -------------------- #
    {
        "qid": "Q7",
        "category": "Negation / constraint",
        "query": "What is a creature not allowed to do while another creature holds it in a grapple?",
        "gold": r"Escaping a Grapple",
        "note": (
            "Bi-encoders blur 'what you can do' and 'what you cannot do' -- "
            "both embed near the same region."
        ),
    },
    {
        "qid": "Q8",
        "category": "Negation / constraint",
        "query": "What will make me lose a spell I am trying to keep going?",
        "gold": r"factors can break concentration",
        "note": (
            "Over a hundred spell entries contain the word 'Concentration' in "
            "their duration line, swamping the one chunk that states the rule."
        ),
    },
    {
        "qid": "Q9",
        "category": "Negation / constraint",
        "query": "When am I prohibited from acting at all at the start of a battle?",
        "gold": r"can.t move or take an action on your first turn",
        "note": "Negative phrasing against a chapter full of 'you can' statements.",
    },
    # --------------------------- buried but vital -------------------------- #
    {
        "qid": "Q10",
        "category": "Buried vital chunk",
        "query": "How long can somebody stay underwater before they start dying?",
        "gold": r"rounds to reach air",
        "note": "Answered in the travel chapter, not anywhere near combat or healing.",
    },
    {
        "qid": "Q11",
        "category": "Buried vital chunk",
        "query": "Does scrambling up a wall or wading through a swamp slow me down?",
        "gold": r"climbing, swimming, or crawling",
        "note": "The movement-cost rule is phrased in feet, not in 'slow down'.",
    },
    {
        "qid": "Q12",
        "category": "Buried vital chunk",
        "query": "How far can I jump from a standing start without a run-up?",
        "gold": r"make a standing long jump",
        "note": "Jumping sits in the travel chapter, away from the movement rules.",
    },
    {
        "qid": "Q13",
        "category": "Buried vital chunk",
        "query": "How much damage do you take from a long drop?",
        "gold": r"bludgeoning damage for every 10 feet it fell",
        "note": "'Damage' retrieves combat and spell text far more strongly.",
    },
    {
        "qid": "Q14",
        "category": "Keyword trap / distractor",
        "query": "Can I attack with a weapon in each hand on the same turn?",
        "gold": r"different light melee weapon",
        "note": (
            "The Fighter class has a feature *called* 'Two-Weapon Fighting', "
            "which is not the rule -- the rule, with its 'light' restriction, "
            "is 50 pages later in the combat chapter."
        ),
    },
    {
        "qid": "Q15",
        "category": "Control (baseline expected to succeed)",
        "query": "How many failed death saving throws does it take to die?",
        "gold": r"third failure, you die",
        "note": (
            "Included as a control: the query uses the rulebook's exact "
            "terminology, so the bi-encoder should already rank it first. If "
            "re-ranking were simply 'better', it would improve this too."
        ),
    },
]


def verify(chunks: list) -> None:
    """Report how many chunks each gold pattern matches.

    A pattern matching 0 chunks means the query is unanswerable from this
    corpus and must be dropped; matching very many means the pattern is not
    specific enough to identify 'the' answer chunk.
    """
    import re

    print(f"{'qid':4} {'hits':>4}  chunk_ids (page)")
    for q in QUERY_POOL:
        hits = [c for c in chunks if re.search(q["gold"], c["text"], re.I)]
        ids = ", ".join(f"{c['chunk_id']}(p{c['page']})" for c in hits[:5])
        flag = "  <-- CHECK" if not 1 <= len(hits) <= 3 else ""
        print(f"{q['qid']:4} {len(hits):>4}  {ids}{flag}")


if __name__ == "__main__":
    import json
    from pathlib import Path

    verify(json.loads(Path("data/processed/chunks.json").read_text()))
