from __future__ import annotations

from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.rss import fetch_rss_feed


class RssSubAgent:
    """RSS/Atom source agent."""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        meta = source.meta if isinstance(source.meta, dict) else {}
        timeout_s = max(5, min(int(meta.get("timeout_s") or 20), 60))
        date_text = str(meta.get("date") or "").strip() or None
        try:
            feed = await fetch_rss_feed(
                source.url,
                limit=max(1, min(int(source.max_articles or 5), 30)),
                timeout_s=timeout_s,
                date_text=date_text,
                keep_only_report_day=True,
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] rss fetch failed source=%s url=%s err=%s",
                source.name,
                source.url,
                e,
            )
            return source.name, []

        items = feed.get("items") if isinstance(feed.get("items"), list) else []
        for item in items:
            if isinstance(item, dict):
                item.setdefault("feed_title", feed.get("feed_title") or source.name)
                item.setdefault("feed_url", feed.get("feed_url") or source.url)
                item.setdefault("report_date", feed.get("report_date") or "")
        return source.name, [x for x in items if isinstance(x, dict)]

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]], llm: LLMRunner
    ) -> dict[str, Any]:
        titles: list[str] = []
        for item in articles or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if title and title not in titles:
                titles.append(title)
        angle = (
            f"请围绕 RSS 源 {source.name} 的具体更新写内容，不要只写“有若干条更新”。"
            f"优先关注这些条目：{'、'.join(titles[:5]) or '最近更新'}。"
            f"每条尽量带上主题、链接、发布时间和一句话摘要。"
        )
        sample: list[dict[str, Any]] = []
        for item in (articles or [])[:5]:
            if not isinstance(item, dict):
                continue
            sample.append(
                {
                    "title": item.get("title") or "",
                    "link": item.get("link") or "",
                    "published": item.get("published") or "",
                    "summary": item.get("summary") or "",
                    "image": item.get("image") or "",
                }
            )
        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": int(source.priority or 1),
            "article_count": len(articles or []),
            "topics": ["rss", "feed"],
            "quality_score": int(len(articles or []) * 4),
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
        chosen = [item for item in (articles or []) if isinstance(item, dict)]
        chosen = chosen[: max(1, int(source.max_articles or 5))]
        if not chosen:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                images=None,
                error=None,
            )

        lines = [f"## {source.name}", ""]
        key_points: list[str] = []
        image_urls: list[str] = []
        seen_images: set[str] = set()
        for item in chosen:
            title = str(item.get("title") or "未命名条目").strip()
            link = str(item.get("link") or "").strip()
            published = str(item.get("published") or "").strip()
            summary = str(item.get("summary") or item.get("content") or "").strip()
            head = f"- [{title}]({link})" if link else f"- {title}"
            if published:
                head += f"（{published}）"
            lines.append(head)
            if summary:
                lines.append(f"  - {summary}")
            key_points.append(title)
            images = item.get("images") if isinstance(item.get("images"), list) else []
            for image in images[:2]:
                image_url = str(image or "").strip()
                if not image_url or image_url in seen_images:
                    continue
                seen_images.add(image_url)
                image_urls.append(image_url)
        summary = f"{source.name} 最近更新 {len(chosen)} 条"
        return SubAgentResult(
            source_name=source.name,
            content="\n".join(lines).strip(),
            summary=summary,
            key_points=key_points[:8],
            images=image_urls or None,
            error=None,
        )
