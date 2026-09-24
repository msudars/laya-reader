"""Choose the few papers worth rating today.

Rating everything is tedious, so each day asks for only a handful, each chosen
for what it teaches:
- a top pick: are the picks good? (top-pick precision)
- where Laya's yes/no and similarity disagree most: the most useful label for
  fine-tuning, and it shows which signal to trust
- a hidden paper: did the ranking miss something? (without these, AUC would
  only be measured on papers the tool chose to show)
- an "also close" paper, then another top pick
"""

from __future__ import annotations

import random

from .judge import HIDE, PICK, UNSURE

TOP_PICK = "one of your top picks"
DISAGREE = "Laya's yes/no and the similarity ranking disagree most on this one"
HIDDEN = "a random hidden paper, to check the ranking isn't missing things"
ALSO_CLOSE = "one from 'also close'"
SLOTS = [TOP_PICK, DISAGREE, HIDDEN, ALSO_CLOSE, TOP_PICK]


def _disagreement(rows) -> dict[str, int]:
    """Rank difference between similarity and Laya's P, over the whole shortlist."""
    shortlist = [r for r in rows if r["p_relevant"] is not None]
    by_sim = {r["paper_id"]: i for i, r in enumerate(sorted(shortlist, key=lambda r: -r["sim"]))}
    by_p = {r["paper_id"]: i for i, r in enumerate(sorted(shortlist, key=lambda r: -r["p_relevant"]))}
    return {pid: abs(by_sim[pid] - by_p[pid]) for pid in by_sim}


def choose(rows, n: int, rng: random.Random | None = None) -> list[tuple[object, str]]:
    """Up to `n` unrated rows from one digest, each with the reason it was chosen."""
    rng = rng or random.Random()
    gap = _disagreement(rows)
    unrated = [r for r in rows if r["rated"] is None]
    chosen: list[tuple[object, str]] = []
    taken: set[str] = set()

    def pool(bucket: str):
        return [r for r in unrated if r["bucket"] == bucket and r["paper_id"] not in taken]

    for reason in (SLOTS * (n // len(SLOTS) + 1))[:n]:
        if reason == DISAGREE:
            candidates = [r for r in pool(PICK) + pool(UNSURE) if gap.get(r["paper_id"], 0) > 0]
            row = max(candidates, key=lambda r: gap[r["paper_id"]], default=None)
        else:
            candidates = pool({TOP_PICK: PICK, HIDDEN: HIDE, ALSO_CLOSE: UNSURE}[reason])
            row = rng.choice(candidates) if candidates else None
        if row is not None:
            chosen.append((row, reason))
            taken.add(row["paper_id"])

    # A slot with no candidate (e.g. everything shown is rated) falls back to any unrated paper.
    rest = [r for r in unrated if r["paper_id"] not in taken]
    rng.shuffle(rest)
    chosen += [(r, "another paper from today") for r in rest[: n - len(chosen)]]
    return chosen
