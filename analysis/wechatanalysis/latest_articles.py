from __future__ import annotations

import sys

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from .analysis import fetch_latest_articles
except Exception:  # pragma: no cover
    from analysis import fetch_latest_articles  # type: ignore

_ALLOWED_SCOPES = {"auto", "account", "album"}


def _normalize_scope(scope: str) -> str:
    s = str(scope or "").strip().lower()
    return s if s in _ALLOWED_SCOPES else "auto"


def _compact_articles(rows: list[dict]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows or []:
        title = str((row or {}).get("title", "")).strip()
        url = str((row or {}).get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": title, "url": url})
    return out


def get_album_articles(
    article_url: str,
    limit: int = 5,
    album_keyword: str | None = None,
    latest_scope: str = "auto",
) -> list[dict[str, str]]:
    """
    Fetch latest WeChat articles from account/album HTTP APIs.

    - article_url: any WeChat article URL from the target account
    - limit: max article count
    - album_keyword: reserved argument (currently unused in API mode)
    - latest_scope: auto/account/album
    """
    _ = album_keyword
    rows, _meta = fetch_latest_articles(
        article_url=article_url,
        limit=max(1, int(limit or 1)),
        latest_scope=_normalize_scope(latest_scope),
    )
    return _compact_articles(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        astrbot_logger.info(
            'Usage: python latest_articles.py "article_url" [limit] [album_keyword] [latest_scope]'
        )
        sys.exit(1)

    url = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    album_keyword = sys.argv[3] if len(sys.argv) > 3 else None
    latest_scope = sys.argv[4] if len(sys.argv) > 4 else "auto"

    items = get_album_articles(
        url,
        limit=limit,
        album_keyword=album_keyword,
        latest_scope=latest_scope,
    )

    astrbot_logger.info("Fetched %s articles:", len(items))
    for i, a in enumerate(items, 1):
        astrbot_logger.info("%s. %s\n   %s", i, a["title"], a["url"])
