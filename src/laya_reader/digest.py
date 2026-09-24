"""Render a day's decisions as a terminal table and a Markdown file."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .judge import HIDE, PICK, UNSURE


def split(rows, rank_by: str = "similarity"):
    """(top picks, not sure, hidden count), each list in ranking order."""
    key = (lambda r: -r["p_relevant"]) if rank_by == "laya" else (lambda r: -r["sim"])
    picks = sorted((r for r in rows if r["bucket"] == PICK), key=key)
    unsure = sorted((r for r in rows if r["bucket"] == UNSURE), key=key)
    hidden = sum(1 for r in rows if r["bucket"] == HIDE)
    return picks, unsure, hidden


def _category(r) -> str:
    cats = json.loads(r["categories"])
    return cats[0] if cats else ""


def _laya(r) -> str:
    return "—" if r["p_relevant"] is None else f"{r['p_relevant']:.2f}"


def _table(title: str, rows, style: str) -> Table:
    t = Table(title=title, title_style=style, title_justify="left", expand=True)
    t.add_column("match", justify="right", width=5)
    t.add_column("Laya P", justify="right", width=6)
    t.add_column("cat", width=8)
    t.add_column("paper", ratio=1, overflow="fold")
    for r in rows:
        mark = {1: " ✓", 0: " ✗"}.get(r["rated"], "")
        t.add_row(
            f"{r['sim']:.3f}",
            _laya(r),
            _category(r),
            f"[link={r['url']}]{escape(r['title'])}[/link]{mark}\n[dim]{r['url']}[/dim]",
        )
    return t


def print_digest(console: Console, rows, digest_date: str, rank_by: str, footer: str = ""):
    picks, unsure, hidden = split(rows, rank_by)
    console.rule(f"[bold]What to read next · {digest_date}")
    if picks:
        console.print(_table(f"Top {len(picks)}", picks, "bold green"))
    if unsure:
        console.print(_table(f"Also close ({len(unsure)})", unsure, "bold yellow"))
    console.print(
        f"[dim]match = similarity to your profile (Laya encoder) · Laya P = its yes/no answer · "
        f"ranked by {rank_by}\n{hidden} more hidden · {len(rows)} read in total{footer}[/dim]"
    )


def write_markdown(path: Path, rows, digest_date: str, rank_by: str) -> Path:
    picks, unsure, hidden = split(rows, rank_by)
    lines = [f"# What to read next · {digest_date}", ""]

    def section(title: str, items):
        lines.extend([f"## {title}", ""])
        for r in items:
            lines.append(f"- **[{r['title']}]({r['url']})**  ")
            lines.append(f"  match {r['sim']:.3f} · Laya P(relevant) {_laya(r)} · {_category(r)}")
        lines.append("")

    if picks:
        section("Top picks", picks)
    if unsure:
        section("Also close", unsure)
    lines.append(
        f"_{hidden} more hidden · {len(rows)} read · ranked by {rank_by} with "
        "[Laya](https://github.com/NandhaKishorM/laya)_"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
