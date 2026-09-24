"""A one-page "reader's digest" post from a day's ranking.

The top few papers get their title plus one key sentence, quoted verbatim from
the abstract and chosen by Laya (see judge.key_sentences); the rest are listed
by title. Nothing is generated: Laya ranks and chooses, it does not write.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

# Split after . ! ? followed by whitespace and a capital/bracket, but not after
# common abbreviations ("e.g. The", "et al. Show") or inside numbers ("2.3x").
_ABBREV = ("e.g", "i.e", "et al", "etc", "vs", "cf", "Fig", "Eq", "Sec", "No")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def split_sentences(text: str) -> list[str]:
    parts = _BOUNDARY.split(" ".join(text.split()))
    sentences: list[str] = []
    for part in parts:
        if sentences and sentences[-1].rstrip(".").endswith(_ABBREV):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return [s for s in sentences if s]


def ranked(rows, rank_by: str) -> list:
    """The day's papers best first: the shortlist in ranking order, then the rest by similarity."""
    shortlist = [r for r in rows if r["p_relevant"] is not None]
    key = (lambda r: -r["p_relevant"]) if rank_by == "laya" else (lambda r: -r["sim"])
    rest = [r for r in rows if r["p_relevant"] is None]
    return sorted(shortlist, key=key) + sorted(rest, key=lambda r: -r["sim"])


def _category(r) -> str:
    cats = json.loads(r["categories"])
    return cats[0] if cats else ""


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def render(
    top: list,
    key_sentences: list[str],
    also: list,
    digest_date: str,
    n_read: int,
    categories: list[str],
    rank_by: str,
) -> str:
    day = date.fromisoformat(digest_date).strftime("%A %-d %B %Y")
    lines = [
        f"# Reader's digest · {day}",
        "",
        f"_The {len(top) + len(also)} papers closest to your profile, out of the {n_read} read today "
        f"in {_join(categories)}._",
        "",
        f"## Today's top {len(top)}",
        "",
    ]
    for r, sentence in zip(top, key_sentences):
        lines += [f"**[{r['title']}]({r['url']})** · {_category(r)}  ", sentence, ""]
    if also:
        lines += ["## Also today", ""]
        lines += [f"{i}. [{r['title']}]({r['url']}) · {_category(r)}" for i, r in enumerate(also, len(top) + 1)]
        lines.append("")
    lines += [
        "---",
        f"_Ranked by {rank_by} with [Laya](https://github.com/NandhaKishorM/laya). Each key sentence "
        "is quoted word for word from the abstract: the one Laya rated most likely to say what the "
        "paper proposes or finds._",
    ]
    return "\n".join(lines) + "\n"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
