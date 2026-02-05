from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import aiohttp

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.utils import _json_from_text
from ...pipeline.rendering import load_template

HUXIU_XIUXIU_API = (
    "https://api-data-mini.huxiu.com/hxgpt/agent/ai-product-daily/v3/detail-list"
)


def _pick_date(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        meta = {}
    raw = str(meta.get("date") or "").strip()
    if raw:
        return raw
    try:
        days_ago = int(meta.get("days_ago") or 0)
    except Exception:
        days_ago = 0
    if days_ago < 0:
        days_ago = 0
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _as_list_str(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


class XiuxiuAISubAgent:
    """
    虎嗅「AI 产品日报」(xiuxiu.huxiu.com) 信息源。
    - 抓取结构化事件列表（含 AI comment）
    - 输出一段可供后续 group-writer 复用的 Markdown 证据材料
    """

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        meta = source.meta or {}
        date_str = _pick_date(meta)
        page_size = max(5, int(source.max_articles or 20))

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://xiuxiu.huxiu.com",
            "Referer": "https://xiuxiu.huxiu.com/",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        payload = {
            "date": date_str,
            "platform": "www",
            "page_num": "1",
            "page_size": str(page_size),
        }

        timeout = aiohttp.ClientTimeout(total=20)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    HUXIU_XIUXIU_API, headers=headers, data=payload
                ) as resp:
                    if resp.status != 200:
                        astrbot_logger.warning(
                            "[dailynews] xiuxiu_ai http %s", resp.status
                        )
                        return source.name, []
                    data = await resp.json()
        except asyncio.TimeoutError:
            astrbot_logger.warning("[dailynews] xiuxiu_ai timeout")
            return source.name, []
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] xiuxiu_ai fetch failed: %s", e, exc_info=True
            )
            return source.name, []

        root = data.get("data") if isinstance(data, dict) else None
        event_list = root.get("event_list") if isinstance(root, dict) else None
        if not isinstance(event_list, list) or not event_list:
            return source.name, []

        out: list[dict[str, Any]] = []
        for group in event_list:
            if not isinstance(group, dict):
                continue
            category = str(
                group.get("dynamic_group") or group.get("dynamic_title") or ""
            ).strip()
            items = (
                group.get("group_list")
                if isinstance(group.get("group_list"), list)
                else []
            )
            for it in items:
                if not isinstance(it, dict):
                    continue
                share = (
                    it.get("share_info")
                    if isinstance(it.get("share_info"), dict)
                    else {}
                )
                share_url = str(share.get("share_url") or "").strip()
                out.append(
                    {
                        "category": category,
                        "title": str(it.get("title") or "").strip(),
                        "ai_comment": str(it.get("ai_comment") or "").strip(),
                        "product_name": _as_list_str(it.get("product_name")),
                        "industry": _as_list_str(it.get("industry")),
                        "publish_datetime": str(
                            it.get("publish_datetime") or ""
                        ).strip(),
                        "sub_dynamic": str(it.get("sub_dynamic") or "").strip(),
                        "share_url": share_url,
                        "event_id": str(it.get("event_id") or "").strip(),
                    }
                )

        return source.name, out[:page_size]

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]], llm: LLMRunner
    ) -> dict[str, Any]:
        cnt = len(articles or [])
        categories: list[str] = []
        seen = set()
        for a in articles[:20]:
            if not isinstance(a, dict):
                continue
            c = str(a.get("category") or "").strip()
            if c and c not in seen:
                seen.add(c)
                categories.append(c)
        angle = "AI 产品日报（虎嗅）" + (f" · {cnt} 条" if cnt else "")
        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": cnt,
            "topics": categories[:6] or ["ai"],
            "quality_score": cnt * 2,
            "today_angle": angle,
            "sample_articles": articles[:3],
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
        if not articles:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                images=None,
                error=None,
            )

        system_prompt = str(
            load_template("templates/prompts/xiuxiu_ai_agent_system.txt") or ""
        ).strip()
        prompt = {
            "source_name": source.name,
            "instruction": instruction,
            "items": articles[: max(1, int(source.max_articles or 20))],
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

        try:
            raw = await llm.ask(
                system_prompt=system_prompt,
                prompt=json.dumps(prompt, ensure_ascii=False),
            )
            data = _json_from_text(raw)
            if isinstance(data, dict):
                section = str(data.get("section_markdown") or "").strip()
                summary = str(data.get("summary") or "").strip()
                key_points = data.get("key_points", [])
                if not isinstance(key_points, list):
                    key_points = []
                return SubAgentResult(
                    source_name=source.name,
                    content=section,
                    summary=summary,
                    key_points=[str(x) for x in key_points[:10]],
                    images=None,
                    error=None,
                )
        except Exception:
            astrbot_logger.warning(
                "[dailynews] xiuxiu_ai write failed; fallback to deterministic markdown",
                exc_info=True,
            )

        # Fallback: deterministic markdown evidence (no raw URLs).
        lines: list[str] = [f"## {source.name}", ""]
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for a in articles:
            if not isinstance(a, dict):
                continue
            by_cat.setdefault(
                str(a.get("category") or "").strip() or "今日要点", []
            ).append(a)

        for cat, rows in list(by_cat.items())[:8]:
            if not rows:
                continue
            lines.append(f"### {cat}")
            for r in rows[:8]:
                title = str(r.get("title") or "").strip()
                comment = str(r.get("ai_comment") or "").strip()
                products = r.get("product_name") or []
                if not isinstance(products, list):
                    products = []
                p_txt = "、".join([str(x) for x in products if str(x).strip()][:3])
                url = str(r.get("share_url") or "").strip()
                head = title or "AI 动态"
                if p_txt:
                    head = f"{head}（涉及：{p_txt}）"
                if url:
                    head = f"{head} ( [阅读原文]({url}) )"
                lines.append(f"- **{head}**")
                if comment:
                    lines.append(f"  - 细节：{comment}")
            lines.append("")

        return SubAgentResult(
            source_name=source.name,
            content="\n".join(lines).strip(),
            summary="",
            key_points=[],
            images=None,
            error=None,
        )
