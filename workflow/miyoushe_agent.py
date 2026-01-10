import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from ..analysis.miyousheanalysis.analysis import fetch_miyoushe_post
    from ..analysis.miyousheanalysis.latest_posts import get_user_latest_posts
except Exception:  # pragma: no cover
    from analysis.miyousheanalysis.analysis import fetch_miyoushe_post  # type: ignore
    from analysis.miyousheanalysis.latest_posts import get_user_latest_posts  # type: ignore

from .llm import LLMRunner
from .models import NewsSourceConfig, SubAgentResult
from .utils import _json_from_text, _run_sync, ensure_section_links


class MiyousheSubAgent:
    """米游社子 Agent：抓取用户帖子列表 -> 抓取帖子正文 -> 写出小节"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, str]]]:
        limit = max(int(source.max_articles), 5)
        headless = bool(user_config.get("miyoushe_headless", True))
        sleep_between = float(user_config.get("miyoushe_sleep_between_s", 0.6) or 0.6)

        url = (source.url or "").strip()
        if not url:
            return source.name, []

        if "accountCenter/postList" in url:
            last_err: Optional[str] = None
            for attempt in range(1, 4):
                try:
                    posts = await _run_sync(
                        get_user_latest_posts,
                        url,
                        limit,
                        headless=headless,
                        sleep_between=sleep_between,
                    )
                    if posts:
                        return source.name, posts
                    last_err = "empty posts"
                except Exception as e:
                    last_err = str(e) or type(e).__name__
                    astrbot_logger.warning(
                        "[dailynews] get_user_latest_posts failed for %s (attempt %s/3): %s",
                        source.name,
                        attempt,
                        last_err,
                        exc_info=True,
                    )
                await asyncio.sleep(0.8 * attempt)

            astrbot_logger.warning(
                "[dailynews] %s miyoushe list still empty after retries: %s",
                source.name,
                last_err or "unknown",
            )

        astrbot_logger.warning(
            "[dailynews] %s miyoushe list empty; fallback to single url: %s",
            source.name,
            url,
        )
        return source.name, [{"title": "", "url": url}]

    async def analyze_source(
        self, source: NewsSourceConfig, articles: List[Dict[str, str]], llm: LLMRunner
    ) -> Dict[str, Any]:
        system_prompt = (
            "你是子Agent（信息侦察）。"
            "你将收到某个来源的最新文章标题与链接。"
            "请快速判断今日主要看点/主题，并给出可写作的角度建议。"
            "只输出 JSON，不要输出其它文本。"
        )
        prompt = {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "latest_articles": articles[:10],
            "output_schema": {
                "source_name": source.name,
                "source_type": source.type,
                "priority": source.priority,
                "article_count": len(articles),
                "topics": ["topic"],
                "quality_score": 0,
                "today_angle": "string",
            },
        }

        raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        data = _json_from_text(raw) or {}
        topics = data.get("topics", [])
        if not isinstance(topics, list):
            topics = []

        quality = data.get("quality_score")
        try:
            if isinstance(quality, float) and 0 <= quality <= 1:
                quality_score = int(quality * 100)
            elif isinstance(quality, str):
                q = quality.strip()
                qf = float(q[:-1]) if q.endswith("%") else float(q)
                quality_score = int(qf * 100) if 0 <= qf <= 1 else int(qf)
            else:
                quality_score = int(quality)
        except Exception:
            quality_score = len(articles) * 2 + len(topics)

        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": len(articles),
            "topics": [str(t) for t in topics[:8]],
            "quality_score": quality_score,
            "today_angle": str(data.get("today_angle") or ""),
            "sample_articles": articles[:3],
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: List[Dict[str, str]],
        llm: LLMRunner,
        user_config: Dict[str, Any] | None = None,
    ) -> SubAgentResult:
        if not articles:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                error="该来源未抓取到任何最新帖子",
            )

        chosen = articles[: max(1, int(source.max_articles))]
        max_fetch_concurrency = 2
        sem = asyncio.Semaphore(max_fetch_concurrency)

        async def _fetch_one(a: Dict[str, str]) -> Dict[str, Any]:
            url = (a.get("url") or "").strip()
            if not url:
                return {"title": (a.get("title") or "").strip(), "url": "", "error": "missing url"}

            last_err: Optional[str] = None
            for attempt in range(1, 3):
                try:
                    async with sem:
                        detail = await _run_sync(fetch_miyoushe_post, url)
                    content_text = (detail.get("content_text") or "").strip()
                    if len(content_text) > 1500:
                        content_text = content_text[:1500] + "…"
                    return {
                        "title": (detail.get("title") or a.get("title") or "").strip(),
                        "url": url,
                        "content_text": content_text,
                        "image_urls": detail.get("image_urls") or [],
                    }
                except Exception as e:
                    last_err = str(e) or type(e).__name__
                    astrbot_logger.warning(
                        "[dailynews] fetch_miyoushe_post failed (attempt %s/2): %s",
                        attempt,
                        last_err,
                        exc_info=True,
                    )
                    await asyncio.sleep(1.0 * attempt)
            return {"title": (a.get("title") or "").strip(), "url": url, "error": last_err or "unknown"}

        article_details = await asyncio.gather(*[_fetch_one(a) for a in chosen], return_exceptions=False)
        images: List[str] = []
        seen = set()
        for d in article_details:
            if not isinstance(d, dict):
                continue
            for u in d.get("image_urls") or []:
                if isinstance(u, str) and u and u not in seen:
                    seen.add(u)
                    images.append(u)
        if images:
            astrbot_logger.info("[dailynews] %s collected %s image urls", source.name, len(images))

        system_prompt = (
            "你是子Agent（写作）。"
            "你会收到：写作指令 + 多篇帖子正文摘录。"
            "请写出该来源在今日日报中的一段 Markdown 小节（含小标题、要点，尽量附上链接）。"
            "同时只输出 JSON，不要输出其它文本。"
        )
        prompt = {
            "source_name": source.name,
            "instruction": instruction,
            "articles": article_details,
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

        try:
            raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        except Exception as e:
            astrbot_logger.warning("[dailynews] miyoushe subagent write failed, fallback: %s", e, exc_info=True)
            lines = [f"## {source.name}", "", "（模型生成失败/超时，以下为自动回退摘要）", ""]
            for a in chosen:
                u = (a.get("url") or "").strip()
                t = (a.get("title") or "").strip()
                lines.append(f"- {t} ({u})" if t and u else (u or t))
            return SubAgentResult(
                source_name=source.name,
                content="\n".join([x for x in lines if x]).strip(),
                summary="",
                key_points=[],
                images=images or None,
                error=None,
            )

        data = _json_from_text(raw)
        if not isinstance(data, dict):
            return SubAgentResult(
                source_name=source.name,
                content=str(raw),
                summary="",
                key_points=[],
                images=images or None,
                error=None,
            )

        summary = str(data.get("summary") or "")
        key_points = data.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        section = str(data.get("section_markdown") or "")
        section = ensure_section_links(section, article_details)

        return SubAgentResult(
            source_name=source.name,
            content=section,
            summary=summary,
            key_points=[str(x) for x in key_points[:10]],
            images=images or None,
            error=None,
        )
