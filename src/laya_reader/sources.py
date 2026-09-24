"""Fetch papers from arXiv.

Today's papers come from the RSS feed (one request, exactly the day's announcements).
The Atom search API is used only for `--date`: in September 2026 export.arxiv.org
started answering uncached queries with an empty HTTP 406, so it can't be relied on.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

API_URL = "https://export.arxiv.org/api/query"
RSS_URL = "https://rss.arxiv.org/rss/"
# "new" = first announcement in its primary category, "cross" = cross-listed into one of ours.
# "replace" / "replace-cross" are revisions of older papers, so they are skipped.
NEW_TYPES = {"new", "cross"}
PAGE_SIZE = 100
PAGE_DELAY_S = 3.0  # arXiv asks for >= 3 s between calls
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "laya-reader/0.1 (+https://github.com/msudars/laya-reader)"


@dataclass
class Paper:
    id: str  # arXiv id without version, e.g. "2409.01234"
    title: str
    abstract: str
    url: str
    published: str  # ISO timestamp
    categories: list[str]


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", NS):
        raw_id = _clean(entry.findtext("a:id", namespaces=NS))
        if "/abs/" not in raw_id:  # arXiv returns an error "entry" for bad queries
            continue
        arxiv_id = raw_id.rsplit("/abs/", 1)[1].rsplit("v", 1)[0]
        url = next(
            (l.get("href") for l in entry.findall("a:link", NS) if l.get("rel") == "alternate"),
            raw_id,
        )
        papers.append(
            Paper(
                id=arxiv_id,
                title=_clean(entry.findtext("a:title", namespaces=NS)),
                abstract=_clean(entry.findtext("a:summary", namespaces=NS)),
                url=url,
                published=_clean(entry.findtext("a:published", namespaces=NS)),
                categories=[c.get("term") for c in entry.findall("a:category", NS)],
            )
        )
    return papers


def parse_rss(xml_text: str, announce_types: set[str] = NEW_TYPES) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for item in root.iter("item"):
        if item.findtext("arxiv:announce_type", namespaces=NS) not in announce_types:
            continue
        guid = _clean(item.findtext("guid"))  # oai:arXiv.org:2609.26811v1
        description = _clean(item.findtext("description"))
        papers.append(
            Paper(
                id=guid.rsplit(":", 1)[-1].rsplit("v", 1)[0],
                title=_clean(item.findtext("title")),
                abstract=description.split("Abstract:", 1)[-1].strip(),
                url=_clean(item.findtext("link")),
                published=_clean(item.findtext("pubDate")),
                categories=[_clean(c.text) for c in item.findall("category")],
            )
        )
    return papers


def fetch_today(categories: list[str], limit: int) -> list[Paper]:
    """Today's announced papers in `categories` (new + cross-listed), up to `limit`."""
    return parse_rss(_get_url(RSS_URL + "+".join(categories)))[:limit]


def build_query(categories: list[str], on: date | None = None) -> str:
    query = " OR ".join(f"cat:{c}" for c in categories)
    if len(categories) > 1:
        query = f"({query})"
    if on is not None:
        d = on.strftime("%Y%m%d")
        query += f" AND submittedDate:[{d}0000 TO {d}2359]"
    return query


class FetchError(Exception):
    pass


# arXiv answers a client that calls too often with 406/429/503 for a while.
RETRY_CODES = {406, 429, 503}
RETRY_WAITS_S = (10, 30, 60)


def _get(params: dict) -> str:
    return _get_url(f"{API_URL}?{urllib.parse.urlencode(params)}")


def _get_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for wait in (*RETRY_WAITS_S, None):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or wait is None:
                raise FetchError(
                    f"arXiv returned HTTP {e.code}. It rate-limits clients that call too "
                    "often (the search API used by --date is often down with 406); "
                    "wait a few minutes and try again."
                ) from e
            time.sleep(wait)
        except urllib.error.URLError as e:
            raise FetchError(f"Could not reach arXiv: {e.reason}") from e
    raise AssertionError("unreachable")


def fetch_submitted_on(categories: list[str], limit: int, on: date) -> list[Paper]:
    """Papers in `categories` submitted on `on`, via the search API, up to `limit`."""
    query = build_query(categories, on)
    papers: list[Paper] = []
    seen: set[str] = set()
    start = 0
    while len(papers) < limit:
        if start:
            time.sleep(PAGE_DELAY_S)
        page = parse_atom(
            _get(
                {
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "start": start,
                    "max_results": min(PAGE_SIZE, limit - len(papers)),
                }
            )
        )
        if not page:
            break
        for p in page:  # cross-listed papers can repeat across pages
            if p.id not in seen:
                seen.add(p.id)
                papers.append(p)
        start += len(page)
    return papers[:limit]
