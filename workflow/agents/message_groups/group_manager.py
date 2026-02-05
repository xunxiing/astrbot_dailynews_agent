from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.llm import LLMRunner
from ...core.models import SubAgentResult
from ...pipeline.rendering import load_template
from .classifier_agent import ClassifiedItem, TagClassifierAgent, extract_links
from .tag_store import TagStore


@dataclass(frozen=True)
class GroupedEntry:
    tag: str
    source: str
    topic: str
    summary: str
    links: list[str]
    suggested_new_tag: str = ""


class MessageGroupManager:
    """
    New mode:
    Map (classify) -> Shuffle (route) -> Reduce (write markdown) -> Feedback (learn tags).
    """

    TAG_ORDER: tuple[str, ...] = (
        "[国内政策]",
        "[国外政策]",
        "[科技新闻]",
        "[AI日报]",
        "[AstrBot]",
        "[GitHub项目]",
        "[游戏/二次元]",
        "[待定/新发现]",
    )

    def __init__(self) -> None:
        self._store = TagStore()

    async def build_report(
        self,
        *,
        sub_results: Sequence[Any],
        llm_classify: LLMRunner | None,
        llm_write: LLMRunner | None,
        promote_new_tags: bool,
        promote_min_count: int,
    ) -> str:
        ok: list[SubAgentResult] = []
        suggestions: dict[str, int] = {}

        for r in sub_results:
            if (
                isinstance(r, SubAgentResult)
                and (not r.error)
                and (r.content or "").strip()
            ):
                ok.append(r)

        if not ok:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return "\n".join(
                [
                    "# 每日资讯日报",
                    f"*生成时间: {now_str}*",
                    "",
                    "今日暂无可用资讯。",
                ]
            ).strip()

        tags = self._store.load_tags()

        items: list[dict[str, Any]] = []
        for idx, r in enumerate(ok, start=1):
            items.append(
                {
                    "id": str(idx),
                    "source": r.source_name,
                    "text": (r.summary or "").strip() or (r.content or ""),
                    "content": r.content or "",
                }
            )

        classifier = TagClassifierAgent()
        classified = await classifier.classify(items=items, tags=tags, llm=llm_classify)
        by_id: dict[str, ClassifiedItem] = {c.item_id: c for c in classified}

        grouped: dict[str, list[GroupedEntry]] = {}
        for it in items:
            cid = by_id.get(str(it.get("id") or ""))
            if cid is None:
                continue

            links = extract_links(str(it.get("content") or ""), max_links=3)
            entry = GroupedEntry(
                tag=cid.tag,
                source=str(it.get("source") or ""),
                topic=cid.topic,
                summary=cid.summary,
                links=links,
                suggested_new_tag=cid.suggested_new_tag,
            )
            grouped.setdefault(cid.tag, []).append(entry)

            if cid.suggested_new_tag:
                suggestions[cid.suggested_new_tag] = (
                    suggestions.get(cid.suggested_new_tag, 0) + 1
                )

        ordered_tags = self._ordered_tags(grouped)

        out = ""
        if llm_write is not None:
            out = await self._write_markdown_with_llm(
                llm=llm_write, ordered_tags=ordered_tags, grouped=grouped, items=items
            )

        if not out:
            out = self._render_programmatic(ordered_tags=ordered_tags, grouped=grouped)

        if promote_new_tags and suggestions:
            try:
                changed = self._store.promote_learned_tags(
                    suggestions=suggestions, min_count=promote_min_count
                )
                if changed:
                    astrbot_logger.info(
                        "[dailynews] tag store updated: %s", self._store.identifier
                    )
            except Exception:
                astrbot_logger.warning(
                    "[dailynews] tag store promotion failed", exc_info=True
                )

        return out

    async def _write_markdown_with_llm(
        self,
        *,
        llm: LLMRunner,
        ordered_tags: list[str],
        grouped: dict[str, list[GroupedEntry]],
        items: list[dict[str, Any]],
    ) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _strip_for_evidence(md: str, *, max_chars: int = 1200) -> str:
            s = (md or "").strip()
            if not s:
                return ""
            s = re.sub(r"```.*?```", " ", s, flags=re.S)
            s = re.sub(r"`[^`]+`", " ", s)
            s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)
            s = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", s)
            s = re.sub(r"https?://[^\s)>\"]+", " ", s)
            s = re.sub(r"<[^>]+>", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s[: max(0, int(max_chars))]

        payload_groups: list[dict[str, Any]] = []
        for tag in ordered_tags:
            entries = grouped.get(tag) or []
            if not entries:
                continue
            rows: list[dict[str, Any]] = []
            for e in entries[:30]:
                evidence = ""
                for it in items:
                    if str(it.get("source") or "") == e.source:
                        evidence = _strip_for_evidence(
                            str(it.get("content") or ""), max_chars=1200
                        )
                        break
                rows.append(
                    {
                        "tag": e.tag,
                        "source": e.source,
                        "topic": e.topic,
                        "summary": e.summary,
                        "evidence": evidence,
                        "links": e.links[:3],
                    }
                )
            payload_groups.append({"tag": tag, "entries": rows})

        if not payload_groups:
            return ""

        system_prompt = str(
            load_template("templates/prompts/message_group_writer_system.txt") or ""
        ).strip()

        prompt = {"now": now_str, "groups": payload_groups, "max_bullets_per_tag": 6}

        try:
            md = await llm.ask(
                system_prompt=system_prompt,
                prompt=json.dumps(prompt, ensure_ascii=False),
            )
        except Exception:
            astrbot_logger.warning(
                "[dailynews] group mode LLM write failed", exc_info=True
            )
            return ""

        md = (md or "").strip()
        if not md:
            return ""

        if not md.lstrip().startswith("#"):
            md = f"# 每日资讯日报\n*生成时间: {now_str}*\n\n{md}"
        return re.sub(r"\n{3,}", "\n\n", md).strip()

    def _render_programmatic(
        self, *, ordered_tags: list[str], grouped: dict[str, list[GroupedEntry]]
    ) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts: list[str] = ["# 每日资讯日报", f"*生成时间: {now_str}*", ""]
        for tag in ordered_tags:
            entries = grouped.get(tag) or []
            if not entries:
                continue
            parts.append(f"## {tag}")
            parts.append("")
            for e in entries[:60]:
                head = e.topic or e.source
                line = f"- **{head}**"
                if e.summary and e.summary != e.topic:
                    line += f"：{e.summary}"
                if e.source:
                    line += f"（{e.source}）"
                if e.links:
                    line += f" ( [查看来源]({e.links[0]}) )"
                parts.append(line)
            parts.append("")
        return re.sub(r"\n{3,}", "\n\n", "\n".join(parts).strip()).strip()

    def _ordered_tags(self, grouped: dict[str, list[GroupedEntry]]) -> list[str]:
        present = set(grouped.keys())
        out: list[str] = [t for t in self.TAG_ORDER if t in present]
        for t in sorted(present):
            if t not in out:
                out.append(t)
        return out
