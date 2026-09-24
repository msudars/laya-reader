# laya-reader

**What to read next.** A daily digest of new arXiv papers, ranked against your own reading profile by the open-source [Laya](https://github.com/NandhaKishorM/laya) decision model. It runs locally on a CPU, needs no API key, and uses only open data.

It is also a small experiment in how usable Laya is day to day: you rate its picks, and `laya-reader stats` tells you how good they were.

```
──────────────── What to read next · 2026-09-24 ────────────────
Top 5
 match  Laya P  cat    paper
 0.957    0.87  cs.AI  Silent Failures in Agent-Tool Interaction: An Audit of ToolUniverse
 0.954    0.82  cs.LG  ChipMEM: Verification-Grounded Memory for EDA Agents
 0.953    0.82  cs.AI  Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity
 ...
Also close (15)
 ...
40 more hidden · 60 read in total · 84s for 60 new
```

## Install

Needs Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:msudars/laya-reader.git
cd laya-reader
uv sync
```

The first run downloads Laya's English checkpoint (about 800 MB) from Hugging Face.

## Set up your profile

```bash
cp profile.example.toml profile.toml   # or ~/.config/laya-reader/profile.toml
```

Edit `profile.toml`:

- **`profile`**: 2–4 sentences about what you want to read. Every paper is compared against this text, so name concrete topics ("LLM agents, tool use, efficient inference") rather than general ones ("AI").
- **`categories`**: the arXiv categories to follow ([list](https://arxiv.org/category_taxonomy)). The example uses `cs.LG`, `cs.AI`, `cs.CL` and `stat.ML`.
- **`max_papers`**, **`shortlist`**, **`top_n`**: how many papers to read, how many get Laya's yes/no question, and how many are shown as top picks.

## Daily use

```bash
uv run laya-reader today              # fetch today's papers and print the digest
uv run laya-reader today --limit 50   # quick run on the first 50 papers
```

This prints the digest and saves it to `digests/YYYY-MM-DD.md`. Running it again the same day only reads papers it hasn't seen before.

- **match**: how similar the paper is to your profile, measured with Laya's encoder. This decides the order.
- **Laya P**: Laya's answer to "Is this paper about one of the reader's interests?", as a probability. Only the shortlist gets asked.
- **cat**: the paper's primary arXiv category.

arXiv announces new papers on weekdays (not Saturday or Sunday), so a morning cron job works well:

```cron
# 08:00 on weekdays
0 8 * * 1-5  cd ~/Projects/laya-reader && ~/.local/bin/uv run laya-reader today > /dev/null
```

## Teach it and measure it

```bash
uv run laya-reader rate     # go through the latest digest: y = want to read, n = not for me, s = skip, q = quit
uv run laya-reader stats    # how well is it doing?
```

`rate` shows the top picks and the "also close" papers, plus 5 random hidden ones (`--hidden N`), so the ranking isn't only judged on what it chose to show. `stats` reports:

- **Top-pick precision**: the share of the picks you actually wanted.
- **AUC for each signal**: how often a paper you wanted ranks above one you didn't. 0.5 is a coin flip; 1.0 is perfect. It compares Laya's encoder similarity with Laya's yes/no answer, and suggests `rank_by = "laya"` once the yes/no answer does better.
- **Calibration**: when Laya says 0.8, did you say yes about 80% of the time?
- **Speed**: milliseconds per paper.

```bash
uv run laya-reader export ratings.jsonl   # your ratings as {state, questions, labels} for fine-tuning
```

After a few weeks of ratings you have a labelled set that can go into Laya's [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb). That is the route to making its yes/no answer personal.

## How it works

1. **Fetch.** Today's announcements come from the arXiv RSS feed (`rss.arxiv.org/rss/cs.LG+cs.AI+…`), keeping new and cross-listed papers and skipping revisions. `--date YYYY-MM-DD` uses the arXiv search API instead.
2. **Similarity.** Laya's encoder embeds your profile and each paper (title + abstract), and papers are ranked by cosine similarity. This takes about 1 s per paper on a CPU.
3. **Decision.** The top `shortlist` papers also get Laya's typed `noul` (yes/no) question, which takes about 1 s more each.
4. **Store.** Papers, scores and ratings live in SQLite at `~/.local/share/laya-reader/db.sqlite` (override with `LAYA_READER_DB`).

## What we learned about Laya

Findings from building this, on Laya 0.3.20, with the English checkpoint on a 16-core CPU and no GPU:

- **Zero-shot yes/no relevance is close to chance on real papers.** On 16 hand-labelled papers, Laya's `noul` answer reached AUC 0.36–0.69 depending on wording, and every paper scored between 0.67 and 0.89. Laya is built to be fine-tuned, and its README says so. The ratings you collect are what fixes this.
- **Laya's encoder is the better zero-shot signal.** Profile↔paper similarity reached AUC 0.78 on the same papers. A small general-purpose embedding model (MiniLM) scored 0.73 in about 1/30 of the time, so if you only need similarity you don't need Laya for it.
- **Multi-option `choice` questions collapsed.** A topic question answered the first option for every paper, and a skim/deep question always said "deep". Both were removed. The digest shows arXiv's own category instead.
- **Plain-string states beat dict states** for this question (AUC 0.69 vs 0.50).
- **Cost grows with the number of questions.** Each question re-reads the whole state: about 1.2 s per question for a 450-token abstract on CPU. The README's 0.2–0.5 s CPU figures are for short texts.
- **The arXiv search API was returning HTTP 406** to uncached requests in September 2026, which is why today's papers come from RSS.

## Development

```bash
uv run pytest    # no model download needed: tests use a fake judge and saved arXiv responses
```
