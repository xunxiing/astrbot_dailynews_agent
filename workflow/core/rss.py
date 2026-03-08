from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp

_IMG_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _local_name(tag: Any) -> str:
    s = str(tag or "")
    if "}" in s:
        return s.rsplit("}", 1)[-1]
    return s


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = _WS_RE.sub(" ", text).strip()
    return text


def _short_text(value: Any, limit: int = 220) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _extract_image_urls(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        for match in _IMG_RE.finditer(text):
            url = str(match.group(1) or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(url)
    return out


def _parse_datetime(value: Any) -> tuple[str, int]:
    raw = str(value or "").strip()
    if not raw:
        return "", 0
    dt = None
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        dt = None
    if dt is None:
        normalized = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except Exception:
            dt = None
    if dt is None:
        return raw, 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M"), int(dt.timestamp())


def _child_text(elem: ET.Element, *names: str) -> str:
    target = {name.lower() for name in names}
    for child in list(elem):
        if _local_name(child.tag).lower() not in target:
            continue
        text = "".join(child.itertext()).strip()
        if text:
            return text
    return ""


def _iter_descendant_image_urls(elem: ET.Element) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for node in elem.iter():
        local = _local_name(node.tag).lower()
        if local in {"content", "thumbnail", "enclosure"}:
            url = str(node.attrib.get("url") or node.attrib.get("href") or "").strip()
            node_type = str(node.attrib.get("type") or "").strip().lower()
            if (
                local == "thumbnail"
                or local == "enclosure"
                or node_type.startswith("image/")
            ):
                if url and url not in seen:
                    seen.add(url)
                    out.append(url)
    return out


def _atom_link(entry: ET.Element) -> str:
    fallback = ""
    for child in list(entry):
        if _local_name(child.tag).lower() != "link":
            continue
        href = str(child.attrib.get("href") or "").strip()
        if not href:
            continue
        rel = str(child.attrib.get("rel") or "").strip().lower()
        if rel in {"", "alternate"}:
            return href
        if not fallback:
            fallback = href
    return fallback


def _dedupe_urls(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        url = str(value or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


async def fetch_rss_feed(
    url: str,
    *,
    limit: int = 10,
    timeout_s: int = 20,
    user_agent: str | None = None,
) -> dict[str, Any]:
    feed_url = str(url or "").strip()
    if not feed_url:
        raise ValueError("rss url is required")
    headers = {
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        "User-Agent": user_agent or "AstrBotDailyNewsAgent/1.0 (+rss)",
    }
    timeout = aiohttp.ClientTimeout(total=max(5, int(timeout_s)))
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(feed_url) as resp:
            resp.raise_for_status()
            text = await resp.text()

    root = ET.fromstring(text)
    root_name = _local_name(root.tag).lower()
    feed_title = ""
    feed_description = ""
    items: list[dict[str, Any]] = []

    if root_name in {"rss", "rdf", "rdf:rdf"}:
        channel = None
        for child in list(root):
            if _local_name(child.tag).lower() == "channel":
                channel = child
                break
        if channel is None:
            channel = root
        feed_title = _clean_text(_child_text(channel, "title"))
        feed_description = _clean_text(_child_text(channel, "description", "subtitle"))
        raw_items = [
            child for child in list(channel) if _local_name(child.tag).lower() == "item"
        ]
        for item in raw_items:
            title = _clean_text(_child_text(item, "title"))
            link = _child_text(item, "link").strip()
            guid = _child_text(item, "guid").strip()
            published_raw = _child_text(item, "pubDate", "published", "updated", "date")
            summary_html = _child_text(item, "description", "summary")
            content_html = _child_text(item, "encoded", "content")
            published, published_ts = _parse_datetime(published_raw)
            images = _dedupe_urls(
                _iter_descendant_image_urls(item)
                + _extract_image_urls(summary_html, content_html)
            )
            items.append(
                {
                    "title": title or link or guid or "Untitled",
                    "link": link,
                    "id": guid or link or title,
                    "published": published,
                    "published_ts": published_ts,
                    "summary": _short_text(summary_html or content_html, 220),
                    "content": _clean_text(content_html or summary_html),
                    "images": images,
                    "image": images[0] if images else "",
                }
            )
    elif root_name == "feed":
        feed_title = _clean_text(_child_text(root, "title"))
        feed_description = _clean_text(_child_text(root, "subtitle", "description"))
        raw_items = [
            child for child in list(root) if _local_name(child.tag).lower() == "entry"
        ]
        for entry in raw_items:
            title = _clean_text(_child_text(entry, "title"))
            link = _atom_link(entry)
            entry_id = _child_text(entry, "id").strip()
            published_raw = _child_text(entry, "published", "updated")
            summary_html = _child_text(entry, "summary")
            content_html = _child_text(entry, "content")
            published, published_ts = _parse_datetime(published_raw)
            images = _dedupe_urls(
                _iter_descendant_image_urls(entry)
                + _extract_image_urls(summary_html, content_html)
            )
            items.append(
                {
                    "title": title or link or entry_id or "Untitled",
                    "link": link,
                    "id": entry_id or link or title,
                    "published": published,
                    "published_ts": published_ts,
                    "summary": _short_text(summary_html or content_html, 220),
                    "content": _clean_text(content_html or summary_html),
                    "images": images,
                    "image": images[0] if images else "",
                }
            )
    else:
        raise ValueError(f"unsupported rss/atom root tag: {root.tag}")

    items.sort(key=lambda row: int(row.get("published_ts") or 0), reverse=True)
    if limit > 0:
        items = items[:limit]

    return {
        "feed_title": feed_title or "RSS Feed",
        "feed_description": feed_description,
        "feed_url": feed_url,
        "items": items,
    }


def format_rss_feed_for_tool(
    feed: dict[str, Any],
    *,
    limit: int = 8,
    focus: str = "",
    include_content: bool = False,
) -> str:
    title = str(feed.get("feed_title") or "RSS Feed").strip()
    feed_url = str(feed.get("feed_url") or "").strip()
    items = feed.get("items") if isinstance(feed.get("items"), list) else []
    rows = items[: max(1, int(limit))]

    lines = [f"RSS ??{title}"]
    if feed_url:
        lines.append(f"?????{feed_url}")
    desc = _short_text(feed.get("feed_description") or "", 160)
    if desc:
        lines.append(f"???{desc}")
    if focus:
        lines.append(f"????{focus}")
    lines.append(f"?????{len(rows)}")
    lines.append("")

    for idx, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        title_text = str(item.get("title") or "Untitled").strip()
        link = str(item.get("link") or "").strip()
        published = str(item.get("published") or "").strip()
        summary = _short_text(item.get("summary") or item.get("content") or "", 220)
        images = item.get("images") if isinstance(item.get("images"), list) else []
        head = f"{idx}. {title_text}"
        if published:
            head += f" [{published}]"
        lines.append(head)
        if link:
            lines.append(f"   ???{link}")
        if summary:
            lines.append(f"   ???{summary}")
        if include_content:
            content = _short_text(item.get("content") or "", 360)
            if content and content != summary:
                lines.append(f"   ?????{content}")
        if images:
            lines.append(f"   ???{len(images)} ?")
        lines.append("")

    return "\n".join(lines).strip()
