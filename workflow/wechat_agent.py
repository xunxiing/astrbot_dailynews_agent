import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from ..analysis.wechatanalysis.analysis import fetch_wechat_article
    from ..analysis.wechatanalysis.latest_articles import get_album_articles_chasing_latest_with_seed
except Exception:  # pragma: no cover
    from analysis.wechatanalysis.analysis import fetch_wechat_article  # type: ignore
    from analysis.wechatanalysis.latest_articles import (  # type: ignore
        get_album_articles_chasing_latest_with_seed,
    )

from .llm import LLMRunner
from .models import NewsSourceConfig, SubAgentResult
from .seed_store import _get_seed_state, _update_seed_entry
from .utils import _json_from_text, _run_sync, ensure_section_links


class WechatSubAgent:
    """公众号子 Agent：抓取最新文章列表、抓取正文并写出小节"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, str]]]:
        limit = max(int(source.max_articles), 5)
        max_hops = int(user_config.get("wechat_chase_max_hops", 6))
        persist_seed = bool(user_config.get("wechat_seed_persist", True))

        album_keyword = source.album_keyword
        key = f"{source.url}||{album_keyword or ''}"

        start_url = source.url
        if persist_seed:
            state = await _get_seed_state()
            entry = state.get(key) if isinstance(state, dict) else None
            if isinstance(entry, dict) and entry.get("seed_url"):
                start_url = str(entry.get("seed_url"))

        seed_url = start_url
        articles: List[Dict[str, str]] = []

        for attempt in range(1, 3):
            try:
                seed_url, articles = await _run_sync(
                    get_album_articles_chasing_latest_with_seed,
                    start_url,
                    limit,
                    album_keyword=album_keyword,
                    max_hops=max_hops,
                )
            except Exception as e:
                astrbot_logger.warning(
                    "[dailynews] chasing latest failed for %s (attempt %s/2): %s",
                    source.name,
                    attempt,
                    e,
                    exc_info=True,
                )
                seed_url, articles = start_url, []

            if articles:
                break
            await asyncio.sleep(0.8 * attempt)

        if not articles:
            astrbot_logger.warning(
                "[dailynews] %s has no album articles; fallback to configured URL as single article: %s",
                source.name,
                start_url,
            )
            seed_url = start_url
            articles = [{"title": "", "url": start_url}]

        if persist_seed and seed_url:
            await _update_seed_entry(
                key,
                {
                    "seed_url": seed_url,
                    "source_url": source.url,
                    "album_keyword": album_keyword or "",
                    "updated_at": datetime.now().isoformat(),
                },
            )

        return source.name, articles

    async def analyze_source(
        self, source: NewsSourceConfig, articles: List[Dict[str, str]], llm: LLMRunner
    ) -> Dict[str, Any]:
        system_prompt = (
            "你是子Agent（信息侦察）。"
            "你将收到某个公众号来源的最新文章标题与链接。"
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
                if q.endswith("%"):
                    quality_score = int(float(q[:-1]) * 1)
                else:
                    qf = float(q)
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
    ) -> SubAgentResult:
        if not articles:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                error="该来源未抓取到任何最新文章",
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
                        detail = await _run_sync(fetch_wechat_article, url)
                    content_text = (detail.get("content_text") or "").strip()
                    if len(content_text) > 1500:
                        content_text = content_text[:1500] + "…"
                    image_urls = detail.get("image_urls") or []
                    if not isinstance(image_urls, list):
                        image_urls = []
                    image_urls = [str(u) for u in image_urls if isinstance(u, str) and u.strip()][:30]
                    return {
                        "title": (detail.get("title") or a.get("title") or "").strip(),
                        "url": url,
                        "author": (detail.get("author") or "").strip(),
                        "publish_time": (detail.get("publish_time") or "").strip(),
                        "content_text": content_text,
                        "image_urls": image_urls,
                    }
                except Exception as e:
                    last_err = str(e) or type(e).__name__
                    astrbot_logger.warning(
                        "[dailynews] fetch_wechat_article failed (attempt %s/2): %s",
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
            "你会收到：写作指令+多篇公众号文章的正文摘录。"
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
            astrbot_logger.warning("[dailynews] subagent write failed, fallback: %s", e, exc_info=True)
            lines = [f"## {source.name}", "", "（模型生成失败/超时，以下为自动回退摘要）", ""]
            for a in chosen:
                t = (a.get("title") or "").strip()
                u = (a.get("url") or "").strip()
                if t and u:
                    lines.append(f"- {t} ({u})")
                elif u:
                    lines.append(f"- {u}")
            return SubAgentResult(
                source_name=source.name,
                content="\n".join(lines).strip(),
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
