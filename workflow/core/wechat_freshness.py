from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urldefrag

WECHAT_SEEN_ARTICLES_LIMIT = 50
WECHAT_SEEN_BASELINE_TTL_HOURS = 48

_DATE_IN_TEXT_RE_ASCII = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
_DATE_IN_TEXT_RE = re.compile(r"(20\d{2})[-/.骞碷](\d{1,2})[-/.鏈圿](\d{1,2})")


def build_wechat_seed_key(
    source_url: str, album_keyword: str = "", latest_scope: str = "auto"
) -> str:
    return f"{str(source_url or '').strip()}||{str(album_keyword or '').strip()}||{str(latest_scope or 'auto').strip()}"


def article_create_ts(item: dict[str, Any]) -> int:
    try:
        raw = str((item or {}).get("create_time") or "").strip()
        if raw.isdigit():
            n = int(raw)
            if n > 10_000_000_000:
                n = n // 1000
            return n
    except Exception:
        pass
    return 0


def looks_like_today_item(
    item: dict[str, Any], *, now: datetime | None = None
) -> bool:
    today = (now or datetime.now()).date()
    for key in ("title", "name", "date_label", "published", "publish_time"):
        text = str((item or {}).get(key) or "").strip()
        if not text:
            continue
        m = _DATE_IN_TEXT_RE_ASCII.search(text) or _DATE_IN_TEXT_RE.search(text)
        if not m:
            continue
        try:
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
            if datetime(y, mo, d).date() == today:
                return True
        except Exception:
            continue
    return False


def build_article_fingerprint(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    raw_url = str(item.get("url") or item.get("link") or "").strip()
    if raw_url:
        normalized_url, _ = urldefrag(raw_url)
        if normalized_url:
            return f"url:{normalized_url}"

    title = str(item.get("title") or item.get("name") or "").strip()
    create_time = str(item.get("create_time") or item.get("published") or "").strip()
    if title and create_time:
        payload = f"{title}||{create_time}"
    elif title:
        payload = title
    else:
        return ""
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"hash:{digest}"


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def get_seen_articles(entry: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    if not isinstance(entry, dict):
        return {}
    raw_items = entry.get("seen_articles")
    if not isinstance(raw_items, list):
        return {}

    seen: dict[str, dict[str, str]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        fingerprint = str(raw.get("fingerprint") or "").strip()
        if not fingerprint:
            continue
        seen[fingerprint] = {
            "fingerprint": fingerprint,
            "url": str(raw.get("url") or "").strip(),
            "title": str(raw.get("title") or "").strip(),
            "create_time": str(raw.get("create_time") or "").strip(),
            "first_seen_at": str(raw.get("first_seen_at") or "").strip(),
            "last_seen_at": str(raw.get("last_seen_at") or "").strip(),
        }
    return seen


def has_recent_seen_baseline(
    entry: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    ttl_hours: int = WECHAT_SEEN_BASELINE_TTL_HOURS,
) -> bool:
    now_dt = now or datetime.now()
    seen = get_seen_articles(entry)
    if not seen:
        return False

    threshold = now_dt - timedelta(hours=max(1, int(ttl_hours or 0)))
    updated_at = _parse_iso_datetime((entry or {}).get("updated_at"))
    if updated_at is not None and updated_at >= threshold:
        return True

    for item in seen.values():
        last_seen = _parse_iso_datetime(item.get("last_seen_at"))
        if last_seen is not None and last_seen >= threshold:
            return True
    return False


def article_is_new_since_recent_baseline(
    item: dict[str, Any],
    entry: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    ttl_hours: int = WECHAT_SEEN_BASELINE_TTL_HOURS,
) -> bool:
    fingerprint = build_article_fingerprint(item)
    if not fingerprint:
        return False
    if not has_recent_seen_baseline(entry, now=now, ttl_hours=ttl_hours):
        return False
    return fingerprint not in get_seen_articles(entry)


def merge_seen_articles(
    entry: dict[str, Any] | None,
    articles: list[dict[str, Any]] | None,
    *,
    now: datetime | None = None,
    limit: int = WECHAT_SEEN_ARTICLES_LIMIT,
) -> dict[str, Any]:
    merged = dict(entry or {})
    now_dt = now or datetime.now()
    now_iso = now_dt.isoformat()
    seen = get_seen_articles(merged)

    for item in articles or []:
        if not isinstance(item, dict):
            continue
        fingerprint = build_article_fingerprint(item)
        if not fingerprint:
            continue
        existing = seen.get(fingerprint, {})
        seen[fingerprint] = {
            "fingerprint": fingerprint,
            "url": str(item.get("url") or existing.get("url") or "").strip(),
            "title": str(
                item.get("title") or item.get("name") or existing.get("title") or ""
            ).strip(),
            "create_time": str(
                item.get("create_time")
                or item.get("published")
                or existing.get("create_time")
                or ""
            ).strip(),
            "first_seen_at": str(existing.get("first_seen_at") or now_iso).strip(),
            "last_seen_at": now_iso,
        }

    def _sort_key(record: dict[str, str]) -> tuple[datetime, datetime]:
        last_seen = _parse_iso_datetime(record.get("last_seen_at")) or datetime.min
        first_seen = _parse_iso_datetime(record.get("first_seen_at")) or datetime.min
        return last_seen, first_seen

    items = sorted(seen.values(), key=_sort_key, reverse=True)[
        : max(1, int(limit or WECHAT_SEEN_ARTICLES_LIMIT))
    ]
    merged["seen_articles"] = items
    merged["updated_at"] = now_iso
    return merged
