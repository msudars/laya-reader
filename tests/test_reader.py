import io
import random
import re
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from laya_reader import cli, config, judge, metrics, rating, sources
from laya_reader.judge import HIDE, PICK, UNSURE, Decision
from laya_reader.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


@pytest.fixture
def cfg(tmp_path):
    c = config.load(str(ROOT / "profile.example.toml"))
    c.digest_dir = tmp_path / "digests"
    c.shortlist, c.top_n = 4, 2
    return c


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_READER_DB", str(tmp_path / "db.sqlite"))


def make_paper(i: int) -> sources.Paper:
    return sources.Paper(f"2609.{i:05d}", f"Paper {i}", f"Abstract {i}", f"https://arxiv.org/abs/2609.{i:05d}", "", ["cs.LG"])


# ---- sources


def test_parse_rss_fixture():
    papers = sources.parse_rss((FIXTURES / "arxiv_rss.xml").read_text())
    assert len(papers) == 3  # 2 new + 1 cross; the "replace" item is skipped
    for p in papers:
        assert re.fullmatch(r"\d{4}\.\d{4,5}", p.id)  # version suffix stripped
        assert p.abstract and not p.abstract.startswith(("arXiv:", "Abstract"))
        assert p.url.startswith("https://arxiv.org/abs/") and p.categories
    assert len(sources.parse_rss((FIXTURES / "arxiv_rss.xml").read_text(), {"replace"})) == 1


def test_parse_atom_fixture():
    papers = sources.parse_atom((FIXTURES / "arxiv_3.xml").read_text())
    assert [p.id for p in papers] == ["2609.28471", "2609.28449", "2609.28442"]
    for p in papers:
        assert p.title and p.abstract and "\n" not in p.abstract
        assert p.url.startswith("http") and p.categories


def test_build_query():
    assert sources.build_query(["cs.LG"]) == "cat:cs.LG"
    q = sources.build_query(["cs.LG", "cs.CL"], date(2026, 9, 23))
    assert q == "(cat:cs.LG OR cat:cs.CL) AND submittedDate:[202609230000 TO 202609232359]"


def test_fetch_retries_then_gives_clear_error(monkeypatch):
    monkeypatch.setattr(sources, "RETRY_WAITS_S", (0, 0))
    calls = []

    def flaky(req, timeout):
        calls.append(req.full_url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 406, "Not Acceptable", {}, io.BytesIO())
        return io.BytesIO((FIXTURES / "arxiv_rss.xml").read_bytes())

    monkeypatch.setattr(sources.urllib.request, "urlopen", flaky)
    assert len(sources.fetch_today(["cs.CL"], 10)) == 3 and len(calls) == 3
    assert calls[0] == "https://rss.arxiv.org/rss/cs.CL"

    def always_406(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 406, "Not Acceptable", {}, io.BytesIO())

    monkeypatch.setattr(sources.urllib.request, "urlopen", always_406)
    with pytest.raises(sources.FetchError, match="rate-limits"):
        sources.fetch_today(["cs.CL"], 3)


# ---- config and judge


def test_example_profile_loads(cfg):
    assert cfg.categories and cfg.rank_by == "similarity"
    qs = judge.build_questions(cfg)
    assert qs["relevant"]["type"] == "noul" and "criteria" not in qs["relevant"]


def test_bad_rank_by_is_rejected(tmp_path):
    bad = tmp_path / "p.toml"
    bad.write_text('profile = "x"\ncategories = ["cs.LG"]\nrank_by = "vibes"\n')
    with pytest.raises(config.ConfigError, match="rank_by"):
        config.load(str(bad))


def D(i, sim, p=None):
    return Decision(f"2609.{i:05d}", sim, p, None if p is None else 0.8, HIDE, "fake", 1.0)


@pytest.mark.parametrize(
    "rank_by, expected_picks", [("similarity", {"2609.00000", "2609.00001"}), ("laya", {"2609.00001", "2609.00002"})]
)
def test_assign_buckets(rank_by, expected_picks):
    ds = [D(0, 0.9, 0.2), D(1, 0.8, 0.9), D(2, 0.7, 0.8), D(3, 0.1)]
    judge.assign_buckets(ds, top_n=2, rank_by=rank_by)
    assert {d.paper_id for d in ds if d.bucket == PICK} == expected_picks
    assert [d.bucket for d in ds][3] == HIDE  # not shortlisted
    assert sum(d.bucket == UNSURE for d in ds) == 1


# ---- store and metrics


def fake_judge_factory(calls):
    def fake_judge(ps, cfg, progress=None):
        calls.append(len(ps))
        ds = [D(int(p.id.split(".")[1]), 1 - int(p.id.split(".")[1]) / 10) for p in ps]
        for d in sorted(ds, key=lambda d: -d.sim)[: cfg.shortlist]:
            d.p_relevant, d.confidence = 0.8, 0.8
        judge.assign_buckets(ds, cfg.top_n, cfg.rank_by)
        return ds

    return fake_judge


def test_store_roundtrip_and_rebucket(db, cfg):
    store = Store()
    papers = [make_paper(i) for i in range(6)]
    store.save(papers, fake_judge_factory([])(papers, cfg), "2026-09-24")
    assert store.unjudged(papers + [make_paper(9)]) == [make_paper(9)]
    rows = store.digest("2026-09-24")
    assert [r["bucket"] for r in rows] == [PICK, PICK, UNSURE, UNSURE, HIDE, HIDE]
    store.rebucket("2026-09-24", top_n=1, rank_by="similarity")
    assert [r["bucket"] for r in store.digest("2026-09-24")][:2] == [PICK, UNSURE]
    store.rate(papers[0].id, True)
    store.rate(papers[5].id, False)
    assert {(r["paper_id"], r["rated"]) for r in store.rated()} == {(papers[0].id, 1), (papers[5].id, 0)}


def test_metrics():
    rows = [
        {"sim": 0.9, "p_relevant": 0.9, "rated": 1, "bucket": PICK},
        {"sim": 0.8, "p_relevant": 0.8, "rated": 0, "bucket": PICK},
        {"sim": 0.5, "p_relevant": 0.3, "rated": 1, "bucket": UNSURE},
        {"sim": 0.1, "p_relevant": None, "rated": 0, "bucket": HIDE},
    ]
    rep = metrics.report(rows)
    assert rep.pick_precision == 0.5 and rep.n_rated_picks == 2
    # similarity: wanted {0.9, 0.5} vs unwanted {0.8, 0.1} -> 3 of 4 pairs ordered right
    assert rep.sim_auc == 0.75
    # Laya: wanted {0.9, 0.3} vs unwanted {0.8} -> 1 of 2
    assert rep.laya_auc == 0.5 and rep.n_laya == 3
    assert rep.laya_accuracy == pytest.approx(1 / 3)
    # bins: [0.2-0.4) p=.3 y=1 · [0.8-1.0] p=.85 y=.5
    assert rep.ece == pytest.approx(1 / 3 * 0.7 + 2 / 3 * 0.35)


def test_auc_needs_both_classes():
    assert metrics.auc([(0.9, 1), (0.8, 1)]) is None


# ---- CLI end to end with a fake judge (no model download)


def test_cli_today_rerun_rate_stats_export(cfg, db, monkeypatch, tmp_path):
    papers = [make_paper(i) for i in range(8)]
    calls = []
    monkeypatch.setattr(sources, "fetch_today", lambda cats, limit: papers[:limit])
    monkeypatch.setattr(judge, "judge", fake_judge_factory(calls))
    monkeypatch.setattr(config, "load", lambda _p=None: cfg)

    assert cli.main(["today"]) == 0
    md = next(cfg.digest_dir.glob("*.md")).read_text()
    assert "## Top picks" in md and "Paper 0" in md and "Paper 7" not in md  # 7 is hidden

    assert cli.main(["today"]) == 0
    assert calls == [8]  # second run judged nothing new

    answers = iter(["y", "n", "y", "n"])
    monkeypatch.setattr(cli.console, "input", lambda prompt="": next(answers))
    assert cli.main(["rate"]) == 0  # daily_ratings = 4
    assert len(Store().rated()) == 4
    assert cli.main(["rate"]) == 0  # quota done: asks nothing (next(answers) would raise)
    assert len(Store().rated()) == 4
    assert cli.main(["stats"]) == 0

    out = tmp_path / "out.jsonl"
    assert cli.main(["export", str(out)]) == 0
    assert out.read_text().count("\n") == 4

    answers = iter(["y"] * 20)
    assert cli.main(["rate", "--all", "--hidden", "1"]) == 0
    store = Store()
    rows = store.digest(store.latest_digest_date())
    assert all(r["rated"] is not None for r in rows if r["bucket"] != HIDE)


def test_titles_with_brackets_survive_rendering():
    from rich.console import Console

    from laya_reader import digest

    row = {"bucket": PICK, "sim": 0.9, "p_relevant": 0.8, "rated": None, "categories": '["cs.LG"]',
           "url": "https://arxiv.org/abs/2609.00001", "title": "[RE] Reproducing [bold] claims"}
    console = Console(record=True, width=200)
    digest.print_digest(console, [row], "2026-09-24", "similarity")
    assert "[RE] Reproducing [bold] claims" in console.export_text()


# ---- choosing today's few ratings


def row(i, bucket, sim, p=None, rated=None):
    return {"paper_id": f"p{i}", "bucket": bucket, "sim": sim, "p_relevant": p, "rated": rated}


def digest_rows():
    # shortlist p0..p5 ranked by sim; Laya ranks p5 first and p0 last -> p5 and p0 disagree most
    ps = [0.60, 0.70, 0.75, 0.72, 0.71, 0.90]
    rows = [row(i, PICK if i < 3 else UNSURE, 0.99 - i / 100, ps[i]) for i in range(6)]
    return rows + [row(i, HIDE, 0.5 - i / 100) for i in range(6, 12)]


@pytest.mark.parametrize("n", [3, 4, 5])
def test_choose_fills_each_slot(n):
    chosen = rating.choose(digest_rows(), n, random.Random(0))
    reasons = [reason for _, reason in chosen]
    assert reasons == rating.SLOTS[:n]
    by_reason = dict((reason, r) for r, reason in chosen)
    assert by_reason[rating.TOP_PICK]["bucket"] == PICK
    assert by_reason[rating.HIDDEN]["bucket"] == HIDE
    assert by_reason[rating.DISAGREE]["paper_id"] in {"p0", "p5"}
    assert len({r["paper_id"] for r, _ in chosen}) == n  # no repeats


def test_choose_skips_rated_and_falls_back():
    rows = [dict(r, rated=1) if r["bucket"] != HIDE else r for r in digest_rows()]
    chosen = rating.choose(rows, 4, random.Random(0))
    assert len(chosen) == 4 and all(r["bucket"] == HIDE and r["rated"] is None for r, _ in chosen)
    assert rating.choose([dict(r, rated=0) for r in digest_rows()], 4) == []
