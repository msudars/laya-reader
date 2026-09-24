"""laya-reader: a daily "what to read next" digest for arXiv, ranked by Laya."""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import time
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.progress import Progress
from rich.table import Table

from . import config, digest, judge, metrics, post, rating, sources
from .store import Store

console = Console()


def cmd_today(args, cfg: config.Config, store: Store) -> int:
    on = date.fromisoformat(args.date) if args.date else None
    digest_date = (on or date.today()).isoformat()
    limit = args.limit or cfg.max_papers

    try:
        with console.status(f"Fetching up to {limit} papers from arXiv ({', '.join(cfg.categories)})…"):
            if on:
                papers = sources.fetch_submitted_on(cfg.categories, limit, on)
            else:
                papers = sources.fetch_today(cfg.categories, limit)
    except sources.FetchError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    new = store.unjudged(papers)
    console.print(f"Fetched {len(papers)} papers, {len(new)} not judged before.")

    elapsed = 0.0
    if new:
        start = time.perf_counter()
        total = len(new) + min(cfg.shortlist, len(new))
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task(f"Laya ({cfg.model}) is reading…", total=total)
            decisions = judge.judge(new, cfg, progress=lambda n: progress.advance(task, n))
        elapsed = time.perf_counter() - start
        store.save(new, decisions, digest_date)
        store.rebucket(digest_date, cfg.top_n, cfg.rank_by)

    rows = store.digest(digest_date)
    footer = f" · {elapsed:.0f}s for {len(new)} new" if new else ""
    digest.print_digest(console, rows, digest_date, cfg.rank_by, footer)
    if not args.no_md and rows:
        path = digest.write_markdown(cfg.digest_dir / f"{digest_date}.md", rows, digest_date, cfg.rank_by)
        console.print(f"[dim]Wrote {path}[/dim]")
    if rows:
        console.print(f"[dim]Teach it with {cfg.daily_ratings} quick ratings (~1 min): laya-reader rate[/dim]")
        console.print("[dim]No time to read? One-page summary: laya-reader post[/dim]")
    return 0


def cmd_rate(args, cfg: config.Config, store: Store) -> int:
    digest_date = args.date or store.latest_digest_date()
    if not digest_date:
        console.print("Nothing to rate yet. Run: laya-reader today")
        return 1
    rows = store.digest(digest_date)
    unrated = [r for r in rows if r["rated"] is None]

    if args.all:
        # Everything shown, plus a few hidden so the ranking isn't only judged on what it showed.
        shown = [(r, r["bucket"]) for r in unrated if r["bucket"] != judge.HIDE]
        hidden = [r for r in unrated if r["bucket"] == judge.HIDE]
        queue = shown + [(r, "hidden") for r in random.sample(hidden, min(args.hidden, len(hidden)))]
    else:
        quota = args.n or cfg.daily_ratings
        remaining = quota - (len(rows) - len(unrated))
        if remaining <= 0:
            console.print(
                f"You've rated {len(rows) - len(unrated)} papers from {digest_date}, that's today's "
                f"{quota}. Thanks! (Want more? laya-reader rate --all)"
            )
            return 0
        queue = rating.choose(rows, remaining)
    if not queue:
        console.print(f"Everything from {digest_date} is already rated.")
        return 0

    console.print(
        f"{len(queue)} papers from {digest_date} to rate. "
        "[bold]y[/bold] want to read · [bold]n[/bold] not for me · [bold]s[/bold] skip · "
        "[bold]q[/bold] quit."
    )
    done = 0
    for i, (r, reason) in enumerate(queue, 1):
        console.rule(f"{i}/{len(queue)} · {reason}")
        console.print(f"[bold]{escape(r['title'])}[/bold]\n[dim]{r['url']}[/dim]")
        console.print(escape(textwrap.shorten(r["abstract"], 600, placeholder=" …")))
        laya = "not asked" if r["p_relevant"] is None else f"{r['p_relevant']:.2f}"
        console.print(f"[cyan]Laya:[/cyan] match {r['sim']:.3f} · P(relevant) {laya} · {r['bucket']}")
        try:
            action = console.input("> ").strip().lower()[:1]
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if action == "q":
            break
        if action in ("y", "n"):
            store.rate(r["paper_id"], action == "y")
            done += 1
    console.print(f"Saved {done} ratings. See how Laya is doing: laya-reader stats")
    return 0


def cmd_post(args, cfg: config.Config, store: Store) -> int:
    digest_date = args.date or store.latest_digest_date()
    if not digest_date:
        console.print("Nothing to summarise yet. Run: laya-reader today")
        return 1
    rows = store.digest(digest_date)
    best = post.ranked(rows, cfg.rank_by)[: cfg.post_total]
    top, also = best[: cfg.post_top], best[cfg.post_top :]
    papers = [sources.Paper(r["paper_id"], r["title"], r["abstract"], r["url"], "", []) for r in top]
    with console.status("Laya is picking a key sentence for each top paper…"):
        sentences = judge.key_sentences(papers, cfg)
    text = post.render(top, sentences, also, digest_date, len(rows), cfg.categories, cfg.rank_by)
    console.print(Markdown(text))
    path = post.write(cfg.post_dir / f"{digest_date}.md", text)
    console.print(f"\n[dim]Wrote {path} ({len(text.split())} words)[/dim]")
    return 0


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def cmd_stats(args, cfg: config.Config, store: Store) -> int:
    counts = store.bucket_counts()
    total = sum(counts.values())
    if not total:
        console.print("No decisions yet. Run: laya-reader today")
        return 1
    rep = metrics.report(store.rated())

    console.rule("[bold]How is Laya doing?")
    console.print(
        f"Read [bold]{total}[/bold] papers: {counts.get('pick', 0)} picks · "
        f"{counts.get('unsure', 0)} also close · {counts.get('hide', 0)} hidden"
    )
    for r in store.latency_by_model():
        console.print(f"Checkpoint [bold]{r['model']}[/bold]: {r['n']} papers, {r['ms']:.0f} ms/paper on average")

    if not rep.n_rated:
        console.print("\nNo ratings yet. Run: laya-reader rate")
        return 0
    auc = lambda x: "— (need yes and no ratings)" if x is None else f"{x:.2f}"
    console.print(
        f"\nRated [bold]{rep.n_rated}[/bold] papers\n"
        f"  Top-pick precision (picks you wanted): [bold]{_pct(rep.pick_precision)}[/bold] "
        f"of {rep.n_rated_picks}\n"
        "  Ranking quality, AUC (0.5 = coin flip, 1.0 = perfect):\n"
        f"    similarity (Laya encoder):  [bold]{auc(rep.sim_auc)}[/bold] over {rep.n_rated}\n"
        f"    Laya yes/no P(relevant):    [bold]{auc(rep.laya_auc)}[/bold] over {rep.n_laya} shortlisted\n"
        f"  Laya yes/no accuracy (P ≥ 0.5 vs you): [bold]{_pct(rep.laya_accuracy)}[/bold]\n"
        f"  Laya calibration error (ECE, lower is better): [bold]"
        f"{'—' if rep.ece is None else f'{rep.ece:.3f}'}[/bold]"
    )
    if rep.sim_auc is not None and rep.laya_auc is not None and rep.laya_auc > rep.sim_auc and cfg.rank_by != "laya":
        console.print('  [green]Laya\'s yes/no is ranking better: try rank_by = "laya" in profile.toml[/green]')
    t = Table(title="Calibration: when Laya says P, how often did you say yes?", title_justify="left")
    for col in ("P(relevant)", "papers", "mean P", "you said yes"):
        t.add_column(col, justify="right")
    for b in rep.bins:
        t.add_row(
            f"{b.lo:.1f}–{b.hi:.1f}",
            str(b.n),
            f"{b.mean_p:.2f}" if b.n else "—",
            _pct(b.frac_yes) if b.n else "—",
        )
    console.print(t)
    return 0


def cmd_export(args, cfg: config.Config, store: Store) -> int:
    questions = judge.build_questions(cfg)
    rows = store.rated()
    out = Path(args.out)
    with out.open("w") as f:
        for r in rows:
            paper = sources.Paper(r["paper_id"], r["title"], r["abstract"], "", "", [])
            labels = {"relevant": bool(r["rated"])}
            f.write(json.dumps({"state": judge.build_state(cfg, paper), "questions": questions, "labels": labels}) + "\n")
    console.print(f"Wrote {len(rows)} labelled examples to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laya-reader", description=__doc__)
    parser.add_argument("--profile", help="path to profile.toml (default: ./profile.toml, then ~/.config/laya-reader/)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("today", help="fetch new arXiv papers and print today's digest")
    p.add_argument("--limit", type=int, help="max papers to fetch (default: max_papers in profile)")
    p.add_argument("--date", help="only papers submitted on YYYY-MM-DD")
    p.add_argument("--no-md", action="store_true", help="don't write the Markdown digest")
    p.set_defaults(func=cmd_today)

    p = sub.add_parser("rate", help="rate today's few most useful papers y/n (daily_ratings in profile)")
    p.add_argument("--date", help="digest date to rate (default: latest)")
    p.add_argument("-n", type=int, help="papers to rate today (default: daily_ratings in profile)")
    p.add_argument("--all", action="store_true", help="rate every shown paper instead of today's few")
    p.add_argument("--hidden", type=int, default=5, help="with --all: also rate N random hidden papers (default 5)")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("post", help="write a one-page reader's digest post from a day's ranking")
    p.add_argument("--date", help="digest date to summarise (default: latest)")
    p.set_defaults(func=cmd_post)

    p = sub.add_parser("stats", help="accuracy, calibration and latency from your ratings")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="write rated papers as JSONL for fine-tuning")
    p.add_argument("out", help="output .jsonl path")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    try:
        cfg = config.load(args.profile)
    except config.ConfigError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    store = Store()
    try:
        return args.func(args, cfg, store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
