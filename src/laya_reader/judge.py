"""Ask Laya about each paper. The only module that imports laya.

Two stages, both on Laya's `english` checkpoint:
1. Similarity: embed the profile and every paper with Laya's encoder and rank by cosine.
   On hand-labelled papers this ranked far better than asking the decision head zero-shot.
2. Decision: ask Laya's `noul` question "is this paper about the reader's interests?" on
   the top `shortlist` papers only (each question costs a full encoder pass per paper).
   Its P(relevant) is shown and measured by `stats`; it drives the ranking only with
   rank_by = "laya", which is worth trying once it has been fine-tuned on your ratings.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .config import Config
from .sources import Paper

PICK, UNSURE, HIDE = "pick", "unsure", "hide"
EMBED_MAX_LEN = 512  # 256 halves the time but ranked noticeably worse in testing


@dataclass
class Decision:
    paper_id: str
    sim: float  # cosine(profile, paper) with Laya's encoder
    p_relevant: float | None  # Laya's P(true); None if not shortlisted
    confidence: float | None  # calibrated answer_confidence of that answer
    bucket: str
    model: str
    latency_ms: float  # per paper: embedding + (if shortlisted) decision


def paper_text(paper: Paper) -> str:
    return f"{paper.title}. {paper.abstract}"


def build_state(cfg: Config, paper: Paper) -> str:
    # A plain string ranked better than a {reader, title, abstract} dict in testing.
    # Reader first: Laya right-truncates the state at its token budget.
    return f"Reader's interests: {cfg.profile}\n\nPaper: {paper_text(paper)}"


# Bump when scoring changes (questions, state format, embedding length) so old scores are redone.
SCORING_VERSION = 1


def profile_key(cfg: Config) -> str:
    """Fingerprint of everything that determines a paper's scores."""
    import hashlib
    import json

    blob = json.dumps(
        {"profile": cfg.profile, "model": cfg.model, "embed_max_len": EMBED_MAX_LEN, "v": SCORING_VERSION},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_questions(cfg: Config) -> dict:
    # No criteria: the README warns noul tends to follow true/false option labels.
    return {
        "relevant": {
            "type": "noul",
            "instructions": "Is this paper about one of the reader's interests?",
        }
    }


def assign_buckets(decisions: list[Decision], top_n: int, rank_by: str) -> None:
    """Top `top_n` of the shortlist are picks, the rest of the shortlist is "not sure"."""
    shortlist = [d for d in decisions if d.p_relevant is not None]
    shortlist.sort(key=lambda d: -(d.p_relevant if rank_by == "laya" else d.sim))
    for i, d in enumerate(shortlist):
        d.bucket = PICK if i < top_n else UNSURE


_agent = None


def _get_agent(model: str):
    global _agent
    if _agent is None:
        from laya import Router  # heavy (torch); import only when judging

        _agent = Router(max_loaded=1).load(model)
    return _agent


def _embed(agent, texts: list[str]):
    import numpy as np
    from laya import embed_fn_from_agent

    e = np.asarray(embed_fn_from_agent(agent, max_length=EMBED_MAX_LEN, batch_size=8)(texts))
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def judge(
    papers: list[Paper],
    cfg: Config,
    batch_size: int = 16,
    progress: Callable[[int], None] | None = None,
) -> list[Decision]:
    """`progress(n)` is called as work completes; total work is len(papers) + shortlist size."""
    if not papers:
        return []
    agent = _get_agent(cfg.model)
    profile = _embed(agent, [cfg.profile])[0]

    decisions: list[Decision] = []
    for i in range(0, len(papers), batch_size):
        chunk = papers[i : i + batch_size]
        start = time.perf_counter()
        sims = _embed(agent, [paper_text(p) for p in chunk]) @ profile
        ms = (time.perf_counter() - start) * 1000 / len(chunk)
        decisions += [
            Decision(p.id, round(float(s), 4), None, None, HIDE, cfg.model, round(ms, 1))
            for p, s in zip(chunk, sims)
        ]
        if progress:
            progress(len(chunk))

    by_id = {p.id: p for p in papers}
    shortlist = sorted(decisions, key=lambda d: -d.sim)[: cfg.shortlist]
    questions = build_questions(cfg)
    for i in range(0, len(shortlist), batch_size):
        chunk = shortlist[i : i + batch_size]
        start = time.perf_counter()
        results = agent.predict_batch(
            [build_state(cfg, by_id[d.paper_id]) for d in chunk], questions, batch_size=batch_size
        )
        ms = (time.perf_counter() - start) * 1000 / len(chunk)
        for d, res in zip(chunk, results):
            a = res["answers"]["relevant"]
            d.p_relevant, d.confidence = a["noul"], a["answer_confidence"]
            d.latency_ms = round(d.latency_ms + ms, 1)
        if progress:
            progress(len(chunk))

    assign_buckets(decisions, cfg.top_n, cfg.rank_by)
    return decisions


KEY_SENTENCE_QUESTION = {
    "finding": {
        "type": "noul",
        "instructions": "Does this sentence state what the paper proposes or finds?",
    }
}


def key_sentences(papers: list[Paper], cfg: Config, batch_size: int = 32) -> list[str]:
    """For each paper, the abstract sentence Laya rates most likely to say what it proposes or finds.

    One short row per sentence ("Paper: <title> / Sentence: <s>"), so this is cheap
    compared with judging a whole abstract.
    """
    from .post import split_sentences

    sentences = [split_sentences(p.abstract) or [p.abstract] for p in papers]
    states = [f"Paper: {p.title}\nSentence: {s}" for p, ss in zip(papers, sentences) for s in ss]
    if not states:
        return []
    results = _get_agent(cfg.model).predict_batch(states, KEY_SENTENCE_QUESTION, batch_size=batch_size)
    scores = iter(r["answers"]["finding"]["noul"] for r in results)
    return [pick_key_sentence(ss, [next(scores) for _ in ss]) for ss in sentences]


KEY_SENTENCE_MARGIN = 0.05


def pick_key_sentence(sentences: list[str], scores: list[float], margin: float = KEY_SENTENCE_MARGIN) -> str:
    """The earliest sentence scoring within `margin` of the best.

    Abstracts state the contribution ("We introduce X") before the details, and
    Laya often scores both about equally: 0.83 vs 0.84 on a real abstract, where
    the detail sentence would have been the worse summary.
    """
    best = max(scores)
    return next(s for s, p in zip(sentences, scores) if p >= best - margin)
