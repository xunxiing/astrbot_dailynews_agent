from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.astrbook_client import ASTRBOOK_API_BASE, AstrBookClient
from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.utils import _json_from_text, ensure_section_links


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\((https?://[^)\s]+)\)", flags=re.I)
_HTML_IMG_RE = re.compile(r"<img[^>]+src=[\"'](https?://[^\"'>\s]+)", flags=re.I)


def _valid_category(category: str | None) -> str | None:
    c = (category or "").strip().lower()
    if not c:
        return None
    if c in {"chat", "deals", "misc", "tech", "help", "intro", "acg"}:
        return c
    return None


def _extract_image_urls(*texts: str, max_items: int = 60) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        s = str(text or "")
        for m in _MD_IMAGE_RE.finditer(s):
            u = str(m.group(1) or "").strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
                if len(out) >= max_items:
                    return out
        for m in _HTML_IMG_RE.finditer(s):
            u = str(m.group(1) or "").strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
                if len(out) >= max_items:
                    return out
    return out


class AstrBookSubAgent:
    """AstrBook 论坛子 Agent：抓取最新帖子并生成 Markdown 小节。"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        meta = source.meta if isinstance(source.meta, dict) else {}
        token = str(meta.get("token") or "").strip()
        category = _valid_category(meta.get("category"))
        limit = max(1, min(int(source.max_articles or 10), 50))

        client = AstrBookClient(token=token)
        if not client.enabled:
            astrbot_logger.warning(
                "[dailynews] astrbook source missing token; skip fetch (source=%s)",
                source.name,
            )
            return source.name, []

        data = await client.list_threads(page=1, page_size=limit, category=category)
        if data.get("error"):
            astrbot_logger.warning(
                "[dailynews] astrbook list_threads failed: %s", data.get("error")
            )
            return source.name, []

        items = data.get("items")
        if not isinstance(items, list):
            # Some APIs may wrap list under `data` or return list directly.
            if isinstance(data.get("data"), list):
                items = data.get("data")
            else:
                items = []

        out: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tid = it.get("id")
            try:
                tid_int = int(tid)
            except Exception:
                continue
            title = str(it.get("title") or "").strip()
            if len(title) > 80:
                title = title[:77] + "..."
            out.append(
                {
                    "id": tid_int,
                    "title": title,
                    "category": str(it.get("category") or "").strip(),
                    "url": f"{ASTRBOOK_API_BASE.rstrip('/')}/api/threads/{tid_int}",
                    "content_preview": str(it.get("content_preview") or "").strip(),
                    "reply_count": int(it.get("reply_count") or 0),
                }
            )

        return source.name, out

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]], llm: LLMRunner
    ) -> dict[str, Any]:
        meta = source.meta if isinstance(source.meta, dict) else {}
        category = _valid_category(meta.get("category"))

        system_prompt = (
            "你是子 Agent（信息侦察）。你会收到某个论坛（AstrBook）最新帖子列表（标题、预览、回复数、链接）。\n"
            "请判断今天值得关注的议题，给出写作角度建议。只输出 JSON。\n"
        )
        prompt = {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": ASTRBOOK_API_BASE,
            "category_filter": category or "",
            "latest_threads": articles[:20],
            "output_schema": {
                "source_name": source.name,
                "topics": ["topic"],
                "quality_score": 0,
                "today_angle": "string",
            },
        }
        raw = await llm.ask(
            system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False)
        )
        data = _json_from_text(raw) or {}
        data["source_name"] = source.name
        data["source_type"] = source.type
        data["source_url"] = ASTRBOOK_API_BASE
        data["priority"] = source.priority
        data.setdefault("article_count", len(articles))
        return data

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: list[dict[str, Any]],
        llm: LLMRunner,
        user_config: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        meta = source.meta if isinstance(source.meta, dict) else {}
        token = str(meta.get("token") or "").strip()
        category = _valid_category(meta.get("category"))

        client = AstrBookClient(token=token)
        if not client.enabled:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                images=None,
                error="AstrBook token 未配置（请在 news_sources -> AstrBook 模板中填写 token）",
            )

        limit = max(1, int(source.max_articles or 10))
        chosen = [a for a in (articles or []) if isinstance(a, dict)][:limit]

        thread_details: list[dict[str, Any]] = []
        images: list[str] = []
        for a in chosen:
            tid = a.get("id")
            try:
                tid_int = int(tid)
            except Exception:
                continue
            data = await client.read_thread(tid_int, page=1, page_size=20)
            if data.get("error"):
                continue
            # Keep the raw payload (structure may change).
            thread_details.append({"id": tid_int, "thread": data})

            # Try extract images from common keys
            pieces: list[str] = []
            if isinstance(data.get("content"), str):
                pieces.append(data.get("content"))
            if isinstance(data.get("thread"), dict) and isinstance(
                data["thread"].get("content"), str
            ):
                pieces.append(data["thread"].get("content"))
            replies = []
            if isinstance(data.get("replies"), list):
                replies = data.get("replies")
            elif isinstance(data.get("items"), list):
                replies = data.get("items")
            for r in replies[:30]:
                if isinstance(r, dict) and isinstance(r.get("content"), str):
                    pieces.append(r.get("content"))
            images.extend(_extract_image_urls(*pieces, max_items=60))

        # de-dup images
        uniq_images: list[str] = []
        seen: set[str] = set()
        for u in images:
            if u and u not in seen:
                seen.add(u)
                uniq_images.append(u)

        system_prompt = (
            "你是子 Agent（写作）。你会收到 AstrBook 论坛的多个帖子内容（可能含楼层回复）。\n"
            "请根据写作指令，写出该来源在今日日报中的一个 Markdown 小节：包含小标题、要点、并附上帖子链接。\n"
            "只输出 JSON，不要输出其它文本。\n\n"
            "CRITICAL OUTPUT RULES (must follow):\n"
            "1) Never output raw URLs as plain text. All links must be Markdown links like [查看](URL).\n"
            "2) If you cite a thread, include its link.\n"
        )
        prompt = {
            "source_name": source.name,
            "category_filter": category or "",
            "instruction": instruction,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threads": thread_details[:limit],
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
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] astrbook subagent write failed, fallback: %s",
                e,
                exc_info=True,
            )
            lines = [f"## {source.name}", ""]
            for a in chosen:
                title = str(a.get("title") or "").strip()
                url = str(a.get("url") or "").strip()
                if title and url:
                    lines.append(f"- {title} ([查看帖子]({url}))")
                elif url:
                    lines.append(f"- [查看帖子]({url})")
            return SubAgentResult(
                source_name=source.name,
                content="\n".join(lines).strip(),
                summary="",
                key_points=[],
                images=uniq_images or None,
                error=None,
            )

        data = _json_from_text(raw)
        if not isinstance(data, dict):
            return SubAgentResult(
                source_name=source.name,
                content=str(raw),
                summary="",
                key_points=[],
                images=uniq_images or None,
                error=None,
            )

        summary = str(data.get("summary") or "").strip()
        key_points = data.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        section = str(data.get("section_markdown") or "").strip()

        # Ensure the selected thread URLs are present in output at least once.
        section = ensure_section_links(
            section,
            [
                {"title": str(a.get("title") or ""), "url": str(a.get("url") or "")}
                for a in chosen
                if isinstance(a, dict)
            ],
        )

        return SubAgentResult(
            source_name=source.name,
            content=section,
            summary=summary,
            key_points=[str(x) for x in key_points[:10]],
            images=uniq_images or None,
            error=None,
        )

