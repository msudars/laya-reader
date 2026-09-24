"""SQLite storage for papers, Laya's decisions and the reader's ratings."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .judge import PICK, UNSURE, Decision
from .sources import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    url TEXT NOT NULL,
    published TEXT NOT NULL,
    categories TEXT NOT NULL          -- JSON list
);
CREATE TABLE IF NOT EXISTS decisions (
    paper_id TEXT PRIMARY KEY REFERENCES papers(id),
    digest_date TEXT NOT NULL,        -- YYYY-MM-DD of the run that judged it
    sim REAL NOT NULL,                -- cosine(profile, paper), Laya encoder
    p_relevant REAL,                  -- Laya's P(relevant); NULL if not shortlisted
    confidence REAL,
    bucket TEXT NOT NULL,             -- pick | unsure | hide
    model TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ratings (
    paper_id TEXT PRIMARY KEY REFERENCES papers(id),
    relevant INTEGER NOT NULL,        -- 1 = wanted to read it, 0 = not
    rated_at TEXT NOT NULL
);
"""


def default_path() -> Path:
    if env := os.environ.get("LAYA_READER_DB"):
        return Path(env)
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "laya-reader" / "db.sqlite"


class Store:
    def __init__(self, path: Path | None = None):
        path = path or default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def unjudged(self, papers: list[Paper]) -> list[Paper]:
        judged = {r[0] for r in self.db.execute("SELECT paper_id FROM decisions")}
        return [p for p in papers if p.id not in judged]

    def save(self, papers: list[Paper], decisions: list[Decision], digest_date: str):
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                [(p.id, p.title, p.abstract, p.url, p.published, json.dumps(p.categories)) for p in papers],
            )
            self.db.executemany(
                """INSERT OR REPLACE INTO decisions VALUES
                   (:paper_id, :digest_date, :sim, :p_relevant, :confidence, :bucket, :model, :latency_ms)""",
                [{**asdict(d), "digest_date": digest_date} for d in decisions],
            )

    def rebucket(self, digest_date: str, top_n: int, rank_by: str):
        """Recompute picks over the whole day, so a second run the same day merges in."""
        order = "p_relevant" if rank_by == "laya" else "sim"
        ids = [r[0] for r in self.db.execute(
            f"SELECT paper_id FROM decisions WHERE digest_date = ? AND p_relevant IS NOT NULL ORDER BY {order} DESC",
            (digest_date,),
        )]
        with self.db:
            self.db.executemany(
                "UPDATE decisions SET bucket = ? WHERE paper_id = ?",
                [(PICK if i < top_n else UNSURE, pid) for i, pid in enumerate(ids)],
            )

    def digest(self, digest_date: str) -> list[sqlite3.Row]:
        """All papers judged on `digest_date`, most similar first."""
        return self.db.execute(
            """SELECT d.*, p.title, p.abstract, p.url, p.categories, r.relevant AS rated
               FROM decisions d JOIN papers p ON p.id = d.paper_id
               LEFT JOIN ratings r ON r.paper_id = d.paper_id
               WHERE d.digest_date = ?
               ORDER BY d.sim DESC""",
            (digest_date,),
        ).fetchall()

    def latest_digest_date(self) -> str | None:
        return self.db.execute("SELECT MAX(digest_date) FROM decisions").fetchone()[0]

    def rate(self, paper_id: str, relevant: bool):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO ratings VALUES (?, ?, ?)",
                (paper_id, int(relevant), datetime.now().isoformat(timespec="seconds")),
            )

    def rated(self) -> list[sqlite3.Row]:
        """Every rated paper joined with Laya's decision on it."""
        return self.db.execute(
            """SELECT d.*, p.title, p.abstract, r.relevant AS rated
               FROM ratings r JOIN decisions d ON d.paper_id = r.paper_id
               JOIN papers p ON p.id = r.paper_id
               ORDER BY r.rated_at"""
        ).fetchall()

    def bucket_counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT bucket, COUNT(*) FROM decisions GROUP BY bucket").fetchall())

    def latency_by_model(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT model, COUNT(*) AS n, AVG(latency_ms) AS ms FROM decisions GROUP BY model"
        ).fetchall()
