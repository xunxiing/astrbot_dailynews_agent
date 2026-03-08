from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

TZ = ZoneInfo("Asia/Shanghai")
OFFICIAL_SORT_TYPE = 2


def _import_skland_client() -> tuple[Any, Any, Any]:
    errors: list[str] = []
    candidates = [
        "skland_client",
        "workflow.core.skland_client",
        "data.plugins.astrbot_dailynews_agent.workflow.core.skland_client",
    ]
    plugin_root = Path(__file__).resolve().parents[2]
    root_str = str(plugin_root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["SklandClient"])
            return (
                getattr(mod, "SklandClient"),
                getattr(mod, "format_item"),
                getattr(mod, "rewrite_markdown_images_to_local"),
            )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Skland client module not found. Please provide `skland_client.py` with "
        "SklandClient / format_item / rewrite_markdown_images_to_local. Tried: "
        + " | ".join(errors)
    )


def parse_target_date(date_text: str | None) -> tuple[int, int, str]:
    if date_text:
        start = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=TZ)
    else:
        now = datetime.now(TZ)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp()), start.strftime("%Y-%m-%d")


def normalize_filters(games_text: str | None) -> set[str]:
    if not games_text:
        return set()
    return {part.strip() for part in str(games_text).split(",") if part.strip()}


def match_game(filters: set[str], game_id: int, game_name: str) -> bool:
    if not filters:
        return True
    return str(game_id) in filters or str(game_name) in filters


async def fetch_skland_official_grouped(
    *,
    d_id: str = "",
    thumbcache: str = "",
    date_text: str | None = None,
    page_size: int = 5,
    max_pages: int = 10,
    games_text: str | None = None,
) -> dict[str, Any]:
    SklandClient, format_item, _ = _import_skland_client()
    start_ts, end_ts, date_label = parse_target_date(date_text)
    game_filters = normalize_filters(games_text)
    grouped: list[dict[str, Any]] = []

    client = SklandClient(d_id=d_id or None, thumbcache=thumbcache or None)
    catalog = client.get_game_catalog()
    if catalog.get("code") != 0:
        raise RuntimeError(f"get_game_catalog failed: {catalog}")

    game_entries = catalog.get("data", {}).get("list", [])
    for game_entry in game_entries:
        game = game_entry.get("game") or {}
        game_id = game.get("gameId")
        game_name = game.get("name")
        if (
            not game_id
            or not game_name
            or not match_game(game_filters, game_id, game_name)
        ):
            continue

        official_cates = [
            cate for cate in (game_entry.get("cates") or []) if cate.get("name") == "官方"
        ]
        if not official_cates:
            continue

        category = official_cates[0]
        page_token: str | None = None
        game_results: list[dict[str, Any]] = []

        for _page in range(max(1, int(max_pages))):
            response = client.get_home_feed(
                game_id=game_id,
                cate_id=category["id"],
                page_size=max(1, min(int(page_size), 5)),
                page_token=page_token,
                sort_type=OFFICIAL_SORT_TYPE,
            )
            if response.get("code") != 0:
                raise RuntimeError(f"home/index failed for {game_name}: {response}")

            data = response.get("data") or {}
            entries = data.get("list") or []
            if not entries:
                break

            formatted = [format_item(entry) for entry in entries]
            for item in formatted:
                published_at_ts = int(item.get("published_at_ts") or 0)
                if not (start_ts <= published_at_ts < end_ts):
                    continue
                detail = client.parse_item_detail(item["item_id"])
                detail["game_name"] = game_name
                detail["cate_name"] = category["name"]
                detail.setdefault("published_at_ts", published_at_ts)
                game_results.append(detail)

            oldest_ts = min(
                (int(item.get("published_at_ts") or 0) for item in formatted), default=0
            )
            if oldest_ts and oldest_ts < start_ts:
                break
            if not data.get("hasMore"):
                break
            page_token = data.get("pageToken")
            if not page_token:
                break

        grouped.append(
            {
                "game_id": game_id,
                "game_name": game_name,
                "cate_id": category["id"],
                "cate_name": category["name"],
                "posts": game_results,
            }
        )

    return {"date": date_label, "games": grouped, "client": client}


def flatten_skland_grouped(grouped: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    date_label = str(grouped.get("date") or "")
    games = grouped.get("games") if isinstance(grouped.get("games"), list) else []
    for group in games:
        if not isinstance(group, dict):
            continue
        game_name = str(group.get("game_name") or "").strip()
        cate_name = str(group.get("cate_name") or "官方").strip()
        for post in group.get("posts") or []:
            if not isinstance(post, dict):
                continue
            item = dict(post)
            item.setdefault("game_name", game_name)
            item.setdefault("cate_name", cate_name)
            item.setdefault("date_label", date_label)
            out.append(item)
    out.sort(key=lambda row: int(row.get("published_at_ts") or 0), reverse=True)
    return out


def save_skland_markdown_posts(
    grouped: dict[str, Any],
    *,
    md_dir: str,
) -> list[str]:
    _, _, rewrite_markdown_images_to_local = _import_skland_client()
    client = grouped.get("client")
    if client is None:
        return []
    out_dir = Path(md_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for group in grouped.get("games") or []:
        if not isinstance(group, dict):
            continue
        game_name = str(group.get("game_name") or "Skland").strip() or "Skland"
        game_dir = out_dir / client.safe_filename(game_name)
        game_dir.mkdir(parents=True, exist_ok=True)
        for item in group.get("posts") or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("item_id") or "post")
            title = str(item.get("title") or "untitled").strip() or "untitled"
            markdown_full = str(item.get("markdown_full") or "").strip()
            if not markdown_full:
                continue
            filename = f"{item_id}_{client.safe_filename(title)}.md"
            path = game_dir / filename
            markdown = rewrite_markdown_images_to_local(
                markdown_full,
                path,
                client.session,
                client.timeout,
            )
            path.write_text(markdown, encoding="utf-8")
            saved.append(path.resolve().as_posix())
    return saved


def format_skland_posts_for_tool(
    grouped: dict[str, Any],
    *,
    limit: int = 8,
    focus: str = "",
) -> str:
    rows = flatten_skland_grouped(grouped)
    max_items = max(1, min(int(limit or 8), 20))
    picked = rows[:max_items]
    date_label = str(grouped.get("date") or "").strip()

    game_names: list[str] = []
    for group in grouped.get("games") or []:
        if not isinstance(group, dict):
            continue
        game_name = str(group.get("game_name") or "").strip()
        if game_name and game_name not in game_names:
            game_names.append(game_name)

    lines: list[str] = []
    lines.append("Source: 森空岛官方")
    if date_label:
        lines.append(f"Date: {date_label}")
    if game_names:
        lines.append(f"Games: {'、'.join(game_names[:10])}")
    if focus:
        lines.append(f"Focus: {focus.strip()}")
    lines.append(f"Posts: {len(rows)}")
    lines.append("")

    if not picked:
        lines.append("No official posts found for the given game/date.")
        return "\n".join(lines).strip()

    lines.append("Top Posts:")
    tool_images: list[str] = []
    seen_images: set[str] = set()
    for idx, item in enumerate(picked, start=1):
        title = str(item.get("title") or "未命名帖子").strip()
        game_name = str(item.get("game_name") or "未知游戏").strip()
        url = str(item.get("url") or item.get("article_url") or "").strip()
        published = str(
            item.get("published_at")
            or item.get("publish_time")
            or item.get("date_label")
            or ""
        ).strip()
        summary = str(item.get("summary") or item.get("content_text") or "").strip()
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > 180:
            summary = summary[:177].rstrip() + "..."
        image_count = len(item.get("image_urls") or [])

        head = f"{idx}. [{game_name}] {title}"
        if published:
            head += f" ({published})"
        lines.append(head)
        if url:
            lines.append(f"   URL: {url}")
        if summary:
            lines.append(f"   Summary: {summary}")
        if image_count:
            lines.append(f"   Images: {image_count}")
        preview_urls: list[str] = []
        for candidate in item.get("image_urls") or []:
            image_url = str(candidate or "").strip()
            if not image_url:
                continue
            if image_url not in seen_images:
                seen_images.add(image_url)
                tool_images.append(image_url)
            if len(preview_urls) < 2:
                preview_urls.append(image_url)
        for image_url in preview_urls:
            lines.append(f"   Image: {image_url}")

    if tool_images:
        lines.append("")
        lines.append(f"Image URLs ({len(tool_images)}):")
        for image_url in tool_images[:12]:
            lines.append(f"- {image_url}")
        lines.append("")
        lines.append("Image Markdown Samples:")
        for idx, image_url in enumerate(tool_images[:4], start=1):
            lines.append(f"![skland-image-{idx}]({image_url})")

    return "\n".join(lines).strip()
