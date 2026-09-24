"""Load the reader profile (profile.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

SEARCH_PATHS = [
    Path("profile.toml"),
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "laya-reader" / "profile.toml",
]
RANK_BY = ("similarity", "laya")


@dataclass
class Config:
    profile: str
    categories: list[str]
    model: str = "english"  # Laya checkpoint: english | multilingual | typed-decisions
    max_papers: int = 400
    shortlist: int = 20  # papers that get Laya's yes/no question
    top_n: int = 5
    rank_by: str = "similarity"
    digest_dir: Path = Path("digests")


class ConfigError(Exception):
    pass


def find_profile(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ConfigError(f"Profile not found: {path}")
        return path
    for path in SEARCH_PATHS:
        if path.exists():
            return path
    raise ConfigError(
        "No profile.toml found. Copy profile.example.toml to ./profile.toml "
        "or ~/.config/laya-reader/profile.toml and edit it."
    )


def load(explicit: str | None = None) -> Config:
    path = find_profile(explicit)
    with path.open("rb") as f:
        data = tomllib.load(f)
    try:
        cfg = Config(
            profile=" ".join(data["profile"].split()),
            categories=list(data["categories"]),
            model=str(data.get("model", "english")),
            max_papers=int(data.get("max_papers", 400)),
            shortlist=int(data.get("shortlist", 20)),
            top_n=int(data.get("top_n", 5)),
            rank_by=str(data.get("rank_by", "similarity")),
            digest_dir=Path(data.get("digest_dir", "digests")).expanduser(),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{path}: missing or invalid field {e}") from e
    if not cfg.categories:
        raise ConfigError(f"{path}: 'categories' is empty")
    if cfg.rank_by not in RANK_BY:
        raise ConfigError(f"{path}: rank_by must be one of {RANK_BY}")
    if not 1 <= cfg.top_n <= cfg.shortlist:
        raise ConfigError(f"{path}: need 1 <= top_n <= shortlist")
    return cfg
