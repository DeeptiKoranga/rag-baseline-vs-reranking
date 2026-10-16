# Report evidence pack

Measured output from `rag_baseline_vs_reranking.ipynb`. **Data only — no analysis.** The four headings below are the ones the assignment asks for; the prose under each is to be written by hand.

Regenerate with `PYTHONPATH=dev python dev/build_report_data.py`.

## Setup

| Item | Value |
|---|---|
| Document | D&D Basic Rules (2018), Wizards of the Coast |
| Pages | 180 |
| Chunks | 1119 |
| Chunk size / overlap | 900 / 150 characters, page-bounded |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `gemini-3-flash-preview`, temperature 0 |
| Retrieve / context | top 10 candidates, top 3 to the LLM |
| Stress queries | 15 |

## 1. Architecture comparison

Measurements that bear on this section:

| Quantity | Measured |
|---|---|
| Chunk vectors embedded once, reused per query | 1119 x 384 floats |
| Query embedding (1 forward pass) | 11.01 ms |
| Vector search over the whole corpus | 0.07 ms |
| Cross-encoder, 10 pairs | 57.3 ms |
| Cross-encoder, cost per (query, chunk) pair | 5.7 ms |

> _Your analysis here._

## 2. Qualitative & rank shift analysis

### 2a. Full rank table

`rank_100` = rank of the answer-bearing chunk in a top-100 bi-encoder sweep (diagnostic). `base` / `rerank` = its rank in the 10 candidates before and after the cross-encoder. `None` = absent from that set.

| qid | category | page | rank_100 | base | rerank | shift | in top-3 base | in top-3 rerank | outcome |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | Keyword trap / distractor | 46 | 6 | 6 | 7 | -1 | no | no | still missed |
| Q2 | Keyword trap / distractor | 54 | 2 | 2 | 1 | +1 | yes | yes | already in top 3 |
| Q3 | Keyword trap / distractor | 63 | 1 | 1 | 2 | -1 | yes | yes | already in top 3 |
| Q4 | Exception / corner case | 166 | 98 |  |  |  | no | no | beyond recall |
| Q5 | Exception / corner case | 70 | 2 | 2 | 2 | 0 | yes | yes | already in top 3 |
| Q6 | Exception / corner case | 46 | 3 | 3 | 1 | +2 | yes | yes | already in top 3 |
| Q7 | Negation / constraint | 77 | 2 | 2 | 1 | +1 | yes | yes | already in top 3 |
| Q8 | Negation / constraint | 84 |  |  |  |  | no | no | beyond recall |
| Q9 | Negation / constraint | 72 | 5 | 5 | 1 | +4 | no | yes | **rescued** |
| Q10 | Buried vital chunk | 68 | 1 | 1 | 1 | 0 | yes | yes | already in top 3 |
| Q11 | Buried vital chunk | 67 | 6 | 6 | 1 | +5 | no | yes | **rescued** |
| Q12 | Buried vital chunk | 67 | 2 | 2 | 2 | 0 | yes | yes | already in top 3 |
| Q13 | Buried vital chunk | 68 | 5 | 5 | 8 | -3 | no | no | still missed |
| Q14 | Keyword trap / distractor | 77 | 5 | 5 | 1 | +4 | no | yes | **rescued** |
| Q15 | Control (baseline expected to succeed) | 79 | 1 | 1 | 1 | 0 | yes | yes | already in top 3 |

| Summary | Count |
|---|---|
| Queries | 15 |
| Answer chunk in top 3 — baseline | 8 |
| Answer chunk in top 3 — re-ranked | 11 |
| Rescued by re-ranking | 3 |
| Never retrieved — re-ranking cannot help | 2 |
| Retrieved but still outside top 3 after re-ranking | 2 |
| Dropped out of top 3 by re-ranking | 0 |
| Rank moved worse (harmless where both ranks are inside top 3) | 3 |

### 2b. Rank shifts, with what the baseline ranked instead

#### Q9 — Negation / constraint — **Rank #5 -> Rank #1**

*Query:* When am I prohibited from acting at all at the start of a battle?

*Why it is hard:* Negative phrasing against a chapter full of 'you can' statements.

*Answer-bearing chunk:* [510, 511] (page 72)

Baseline top 3 (cosine):

| rank | page | chunk | cosine | is answer | text |
|---|---|---|---|---|---|
| 1 | 72 | 509 | 0.4195 | no | nonplayer characters involved in combat, and each other player controls an adventurer. “You” can also mean the character or monster that you control.  |
| 2 | 72 | 508 | 0.3669 | no | The clatter of a sword striking against a shield. The terrible rending sound as monstrous claws tear through armor. A brilliant flash of light as a ba |
| 3 | 162 | 1031 | 0.3652 | no | until the start of its next turn. Actions Greataxe. Melee Weapon Attack: +5 to hit, reach 5 ft., one target. Hit: 9 (1d12 + 3) slashing damage. Hailin |

Re-ranked top 3 (cross-encoder):

| rank | was | page | chunk | cross score | is answer | text |
|---|---|---|---|---|---|---|
| 1 | #5 | 72 | 511 | -7.0014 | **YES** | If you’re surprised, you can’t move or take an action on your first turn of the combat, and you can’t take a reaction until that turn ends. A member o |
| 2 | #3 | 162 | 1031 | -8.2026 | no | until the start of its next turn. Actions Greataxe. Melee Weapon Attack: +5 to hit, reach 5 ft., one target. Hit: 9 (1d12 + 3) slashing damage. Hailin |
| 3 | #9 | 73 | 517 | -9.5334 | no | The DM might require you to use an action for any of these activities when it needs special care or when it presents an unusual obstacle. For instance |

#### Q11 — Buried vital chunk — **Rank #6 -> Rank #1**

*Query:* Does scrambling up a wall or wading through a swamp slow me down?

*Why it is hard:* The movement-cost rule is phrased in feet, not in 'slow down'.

*Answer-bearing chunk:* [468, 469] (page 67)

Baseline top 3 (cosine):

| rank | page | chunk | cosine | is answer | text |
|---|---|---|---|---|---|
| 1 | 66 | 459 | 0.3948 | no | Movement Swimming across a rushing river, sneaking down a dungeon corridor, scaling a treacherous mountain slope—all sorts of movement play a key role |
| 2 | 73 | 520 | 0.3837 | no | If you have more than one speed, such as your walking speed and a flying speed, you can switch back and forth between your speeds during your move. Wh |
| 3 | 73 | 522 | 0.3773 | no | A. You can drop prone without using any of your speed. Standing up takes more effort; doing so costs an amount of movement equal to half your speed. F |

Re-ranked top 3 (cross-encoder):

| rank | was | page | chunk | cross score | is answer | text |
|---|---|---|---|---|---|---|
| 1 | #6 | 67 | 468 | -8.1715 | **YES** | Difficult Terrain The travel speeds given in the Travel Pace table assume relatively simple terrain: roads, open plains, or clear dungeon corridors. B |
| 2 | #1 | 66 | 459 | -8.27 | no | Movement Swimming across a rushing river, sneaking down a dungeon corridor, scaling a treacherous mountain slope—all sorts of movement play a key role |
| 3 | #5 | 73 | 521 | -8.8794 | no | treacherous staircases—the setting of a typical fight contains difficult terrain. Every foot of movement in difficult terrain costs 1 extra foot. This |

#### Q14 — Keyword trap / distractor — **Rank #5 -> Rank #1**

*Query:* Can I attack with a weapon in each hand on the same turn?

*Why it is hard:* The Fighter class has a feature *called* 'Two-Weapon Fighting', which is not the rule -- the rule, with its 'light' restriction, is 50 pages later in the combat chapter.

*Answer-bearing chunk:* [549] (page 77)

Baseline top 3 (cosine):

| rank | page | chunk | cosine | is answer | text |
|---|---|---|---|---|---|
| 1 | 27 | 193 | 0.4027 | no | Two-Weapon Fighting When you engage in two-weapon fighting, you can add your ability modifier to the damage of the second attack. Second Wind You have |
| 2 | 77 | 550 | 0.3894 | no | unless that modifier is negative. If either weapon has the thrown property, you can throw the weapon, instead of making a melee attack with it. Grappl |
| 3 | 27 | 197 | 0.3808 | no | or (b) two handaxes • (a) a dungeoneer’s pack or (b) an explorer’s pack Fighting Style You adopt a particular style of fighting as your specialty. Cho |

Re-ranked top 3 (cross-encoder):

| rank | was | page | chunk | cross score | is answer | text |
|---|---|---|---|---|---|---|
| 1 | #5 | 77 | 549 | 2.1195 | **YES** | modifier—the same modifier used for the attack roll—to the damage. A spell tells you which dice to roll for damage and whether to add any modifiers. I |
| 2 | #1 | 27 | 193 | 1.577 | no | Two-Weapon Fighting When you engage in two-weapon fighting, you can add your ability modifier to the damage of the second attack. Second Wind You have |
| 3 | #7 | 73 | 516 | 1.4452 | no | Other Activity on Your Turn Your turn can include a variety of flourishes that require neither your action nor your move. You can communicate however  |

### 2c. Cases where re-ranking did not help

- **Q4** (Exception / corner case) — *beyond recall.* Answer chunk [1050] on page 166 sits at rank 98 of a top-100 sweep, so it never enters the 10 candidates and no re-ranker can reach it. Query: *If a fight comes at us in several waves, can we rest in between to heal up?*
- **Q8** (Negation / constraint) — *beyond recall.* Answer chunk [589] on page 84 does not appear in a top-100 sweep at all, so it never enters the 10 candidates and no re-ranker can reach it. Query: *What will make me lose a spell I am trying to keep going?*
- **Q1** (Keyword trap / distractor) — *still missed.* Retrieved at #6 but re-ranked to #7, still outside the top 3. Query: *Does wearing heavy plate make it harder to sneak past guards unnoticed?*
- **Q13** (Buried vital chunk) — *still missed.* Retrieved at #5 but re-ranked to #8, still outside the top 3. Query: *How much damage do you take from a long drop?*

Rank regressions that did **not** change what the LLM saw (both ranks inside the top 3): Q3 (#1 -> #2).

### 2d. Answer 1 vs Answer 2

#### Q14 — Can I attack with a weapon in each hand on the same turn?

**Answer 1** (baseline; context pages [27, 77, 27], chunks [193, 550, 197]):

> Based on the provided context, you can engage in two-weapon fighting. When doing so, you can add your ability modifier to the damage of the second attack, unless that modifier is negative (p. 27, 77). If either weapon has the thrown property, you may throw it instead of making a melee attack (p. 77). 
> 
> Additionally, if you are at least 2nd level, you can use Action Surge to take one additional action on your turn, which increases to two uses at 17th level (p. 27). You must finish a short or long rest before using Action Surge again (p. 27).

**Answer 2** (re-ranked; context pages [77, 27, 73], chunks [549, 193, 516]):

> Yes, when you take the Attack action with a light melee weapon in one hand, you can use a bonus action to attack with a different light melee weapon in the other hand (p. 77). If either weapon has the thrown property, you may throw it instead of making a melee attack (p. 77).
> 
> Restrictions and penalties include:
> *   Both weapons must be light melee weapons (p. 77).
> *   You do not add your ability modifier to the damage of the bonus attack unless the modifier is negative (p. 77). However, certain class features allow you to add the modifier to the second attack's damage (p. 27).

#### Q9 — When am I prohibited from acting at all at the start of a battle?

**Answer 1** (baseline; context pages [72, 72, 162], chunks [509, 508, 1031]):

> NOT IN CONTEXT.

**Answer 2** (re-ranked; context pages [72, 162, 73], chunks [511, 1031, 517]):

> You are prohibited from moving or taking an action on your first turn of combat if you are surprised (p. 72). Additionally, you are restricted from taking any reactions until that first turn ends (p. 72). While a reaction is an instant response to a trigger that can occur on your turn or someone else's, once you use a reaction, you are prohibited from taking another one until the start of your next turn (p. 73).

#### Q11 — Does scrambling up a wall or wading through a swamp slow me down?

**Answer 1** (baseline; context pages [66, 73, 73], chunks [459, 520, 522]):

> Yes. According to the rules for difficult terrain, every foot of movement costs 1 extra foot (p. 73). The context identifies boulder-strewn caverns, briar-choked forests, and treacherous staircases as examples of settings containing difficult terrain (p. 73). Additionally, scaling a treacherous mountain slope or swimming across a rushing river are noted as key types of movement in adventures (p. 66). While the DM may summarize movement between encounters without calculating exact distances, it is sometimes important to determine how long it takes to travel between spots in days, hours, or minutes (p. 66).

**Answer 2** (re-ranked; context pages [67, 66, 73], chunks [468, 459, 521]):

> Yes. Climbing and swimming (such as wading through a swamp) are special types of movement where each foot costs 1 extra foot of speed (p. 67). 
> 
> If the terrain is considered difficult terrain—such as deep swamps, steep mountains, or rubble—you move at half speed, meaning 1 foot of movement costs 2 feet of speed (p. 67). When climbing or swimming in difficult terrain, each foot costs 2 extra feet (p. 67). Additionally, the travel speeds for simple terrain assume you can cover only half the normal distance in a minute, hour, or day when in difficult terrain (p. 67). This rule applies even if multiple factors in a space count as difficult terrain (p. 73).

> _Your analysis here._

## 3. Latency profiling

15 queries x 5 repeats = 75 measurements per pipeline. One untimed warm-up pass precedes the run. The LLM call is excluded: it is identical work in both pipelines and its network variance is far larger than the retrieval times being compared.

| Stage | mean ms | median ms | p95 ms | min ms | max ms |
|---|---|---|---|---|---|
| Query embedding (1 pass) | 11.01 | 11.18 | 12.21 | 6.84 | 13.71 |
| Vector search (whole corpus) | 0.07 | 0.06 | 0.07 | 0.05 | 0.37 |
| Cross-encoder (10 pairs) | 57.3 | 48.34 | 110.96 | 37.45 | 114.11 |
| **Pipeline 1 retrieval** | 11.08 | 11.24 | 12.32 | 6.9 | 13.77 |
| **Pipeline 2 retrieval** | 68.38 | 59.99 | 121.62 | 44.87 | 125.68 |

Pipeline 2 retrieval costs **6.17x** Pipeline 1 (**+57 ms** absolute).

LLM call, measured separately: mean **34674 ms** over 6 calls — so against generation, re-ranking adds **0.2%** to end-to-end time (the LLM dominates by roughly **605x**).

> _Your analysis here._

## 4. Scaling question

Measured cross-encoder cost: **5.7 ms per (query, chunk) pair**, linear in the number of pairs.

| Corpus | Chunks | Cross-encode everything, per query | Two-stage, per query | Ratio |
|---|---|---|---|---|
| This rulebook | 1,119 | 6.4 s | 0.068 s | 94x |
| 10k chunks | 10,000 | 57.3 s | 0.068 s | 838x |
| 1M chunks | 1,000,000 | 5,730.0 s | 0.068 s | 83,796x |

Relevant measured facts: the bi-encoder's chunk vectors are query-independent and were embedded once (1119 x 384); a cross-encoder score exists only for a specific pair, so nothing is cached between queries. The `beyond recall` rows in the table in §2a show the ceiling this imposes.

> _Your analysis here._
