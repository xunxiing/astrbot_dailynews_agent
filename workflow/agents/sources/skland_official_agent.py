from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.image_utils import get_plugin_data_dir
from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.skland_official import (
    fetch_skland_official_grouped,
    flatten_skland_grouped,
    save_skland_markdown_posts,
)


def _join_game_filters(value: Any) -> str | None:
    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(parts) if parts else None
    text = str(value or "").strip()
    return text or None


def _short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


class SklandOfficialSubAgent:
    """Skland official source agent."""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        meta = source.meta if isinstance(source.meta, dict) else {}
        try:
            grouped = await fetch_skland_official_grouped(
                d_id=str(meta.get("d_id") or "").strip(),
                thumbcache=str(meta.get("thumbcache") or "").strip(),
                date_text=str(meta.get("date") or "").strip() or None,
                page_size=max(1, min(int(meta.get("page_size") or 5), 5)),
                max_pages=max(1, min(int(meta.get("max_pages") or 10), 50)),
                games_text=_join_game_filters(meta.get("games")),
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] skland official fetch failed source=%s err=%s",
                source.name,
                e,
                exc_info=True,
            )
            return source.name, []

        flat = flatten_skland_grouped(grouped)
        md_dir = str(meta.get("md_dir") or "").strip()
        if md_dir:
            try:
                save_root = Path(md_dir)
                if not save_root.is_absolute():
                    save_root = get_plugin_data_dir("skland_markdown") / save_root
                saved = save_skland_markdown_posts(grouped, md_dir=str(save_root))
                saved_map = {Path(p).stem.split("_", 1)[0]: p for p in saved}
                for item in flat:
                    key = str(item.get("item_id") or "").strip()
                    if key and key in saved_map:
                        item["local_markdown_path"] = saved_map[key]
            except Exception:
                astrbot_logger.warning(
                    "[dailynews] skland markdown export failed source=%s",
                    source.name,
                    exc_info=True,
                )
        return source.name, flat[: max(1, int(source.max_articles or 20))]

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]], llm: LLMRunner
    ) -> dict[str, Any]:
        _ = llm
        titles: list[str] = []
        games: list[str] = []
        for item in articles or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            game_name = str(item.get("game_name") or "").strip()
            if title and title not in titles:
                titles.append(title)
            if game_name and game_name not in games:
                games.append(game_name)
        angle = (
            f"请聚焦森空岛各游戏官方当天动态，提炼运营活动、版本公告、联动情报与福利信息。"
            f"优先关注：{'、'.join(titles[:5]) or '当天暂无标题样本'}。"
            f"写作时保持信息密度高、概括准确，避免把社区二创或非官方内容混入。"
        )
        sample: list[dict[str, Any]] = []
        for item in (articles or [])[:5]:
            if not isinstance(item, dict):
                continue
            sample.append(
                {
                    "game_name": item.get("game_name") or "",
                    "title": item.get("title") or "",
                    "url": item.get("url") or item.get("jump_url") or "",
                    "published": item.get("publish_time")
                    or item.get("published_at")
                    or "",
                    "summary": _short_text(
                        item.get("content_text") or item.get("summary") or "", 160
                    ),
                }
            )
        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": int(source.priority or 1),
            "article_count": len(articles or []),
            "topics": games[:8] or ["skland", "official"],
            "quality_score": int(len(articles or []) * 4 + len(games) * 2),
            "today_angle": angle,
            "sample_articles": sample,
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: list[dict[str, Any]],
        llm: LLMRunner,
        user_config: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        _ = llm
        _ = user_config
        chosen = [item for item in (articles or []) if isinstance(item, dict)]
        chosen = chosen[: max(1, int(source.max_articles or 20))]
        if not chosen:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                images=None,
                error=None,
            )

        lines = [f"## {source.name}", "", f"> {instruction}", ""]
        key_points: list[str] = []
        image_urls: list[str] = []
        seen_images: set[str] = set()
        for item in chosen:
            title = str(item.get("title") or "未命名帖子").strip()
            url = str(item.get("url") or item.get("jump_url") or "").strip()
            game_name = str(item.get("game_name") or "未知游戏").strip()
            published = str(
                item.get("publish_time")
                or item.get("published_at")
                or item.get("date_label")
                or ""
            ).strip()
            summary = _short_text(
                item.get("content_text") or item.get("summary") or "", 220
            )
            head = (
                f"- **{game_name}** · [{title}]({url})"
                if url
                else f"- **{game_name}** · {title}"
            )
            if published:
                head += f"（{published}）"
            lines.append(head)
            if summary:
                lines.append(f"  - {summary}")
            md_path = str(item.get("local_markdown_path") or "").strip()
            if md_path:
                lines.append(f"  - 本地 Markdown：`{md_path}`")
            key_points.append(f"{game_name}：{title}")
            for candidate in item.get("image_urls") or []:
                image_url = str(candidate or "").strip()
                if not image_url or image_url in seen_images:
                    continue
                seen_images.add(image_url)
                image_urls.append(image_url)
        summary = f"森空岛官方共整理 {len(chosen)} 篇帖子"
        return SubAgentResult(
            source_name=source.name,
            content="\n".join(lines).strip(),
            summary=summary,
            key_points=key_points[:10],
            images=image_urls[:12] or None,
            error=None,
        )
