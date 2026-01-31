from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from astrbot.core.agent.tool import ToolSet

from ....tools import MarkdownDocApplyEditsTool, MarkdownDocReadTool
from ...core.config_models import SingleAgentConfig
from ...core.internal_event import make_internal_event
from ...core.llm import LLMRunner
from ...core.markdown_sanitizer import sanitize_markdown_for_publish
from ...core.models import NewsSourceConfig
from ...core.utils import _json_from_text
from ...pipeline.rendering import load_template
from ...storage.md_doc_store import create_doc, read_doc, write_doc


def _norm_mode(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"single", "single_agent", "single-agent"}:
        return "single"
    return "multi"


@dataclass(frozen=True)
class SingleAgentInput:
    source_name: str
    source_type: str
    source_url: str
    priority: int
    items: List[Dict[str, Any]]


class SingleAgentNewsWriter:
    """
    Single-agent mode: send all collected materials to one model, give it a todo-list,
    and let it write the daily report via markdown doc tools.

    Notes:
    - This mode intentionally disables image insertion.
    - Single-agent mode does not hard-append any extra sections.
    """

    async def _collect_materials(
        self,
        *,
        sources: List[NewsSourceConfig],
        fetched: Dict[str, List[Dict[str, Any]]],
        user_config: Dict[str, Any],
    ) -> List[SingleAgentInput]:
        """Collect materials for the single-agent model."""
        materials: List[SingleAgentInput] = []

        for src in sources:
            items = fetched.get(src.name, []) or []
            if src.type == "plugin_registry":
                # Do not inject plugin list in single-agent mode.
                continue

            materials.append(
                SingleAgentInput(
                    source_name=str(src.name),
                    source_type=str(src.type),
                    source_url=str(src.url),
                    priority=int(src.priority or 1),
                    items=[it for it in items if isinstance(it, dict)][:50],
                )
            )

        return materials

    def _pick_provider_id(self, *, user_config: Dict[str, Any], single_cfg: SingleAgentConfig) -> str:
        provider_id = str(single_cfg.provider_id or "").strip()
        if provider_id:
            return provider_id
        # fallback: reuse main agent provider
        provider_id = str(user_config.get("main_agent_provider_id") or "").strip()
        if provider_id:
            return provider_id
        raw_list = user_config.get("main_agent_fallback_provider_ids") or []
        if isinstance(raw_list, list):
            for x in raw_list:
                if isinstance(x, str) and x.strip():
                    return x.strip()
        for k in (
            "main_agent_fallback_provider_id_1",
            "main_agent_fallback_provider_id_2",
            "main_agent_fallback_provider_id_3",
        ):
            v = user_config.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    async def write_report(
        self,
        *,
        sources: List[NewsSourceConfig],
        fetched: Dict[str, List[Dict[str, Any]]],
        user_config: Dict[str, Any],
        astrbot_context: Any,
    ) -> str:
        mode = _norm_mode(user_config.get("news_workflow_mode", "multi"))
        if mode != "single":
            raise RuntimeError("SingleAgentNewsWriter called while not in single mode")

        single_cfg = SingleAgentConfig.from_mapping(user_config)
        provider_id = self._pick_provider_id(user_config=user_config, single_cfg=single_cfg)
        if not provider_id:
            raise RuntimeError("single agent provider_id is empty; set single_agent_provider_id or main_agent_provider_id")

        materials = await self._collect_materials(
            sources=sources,
            fetched=fetched,
            user_config=user_config,
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a doc for the model to edit.
        placeholder = "<!-- DAILYNEWS_SINGLE_AGENT_PLACEHOLDER -->"
        draft = "\n".join(
            [
                "# 每日资讯日报",
                f"*生成时间: {now}*",
                "",
                placeholder,
                "",
            ]
        )
        doc_id, _ = create_doc(draft)

        system_prompt = str(load_template("templates/prompts/single_agent_system.txt") or "").strip()
        if not system_prompt:
            raise RuntimeError("single agent system prompt is empty: templates/prompts/single_agent_system.txt")

        # Dynamic constraints only (keep the main prompt in templates/).
        system_prompt = (
            system_prompt
            + "\n\n"
            + "## Dynamic constraints\n"
            + f"- 禁止插入图片（不要输出 `![]()` 或 `<img>`）。\n"
            + f"- 日报字数建议 {int(single_cfg.min_chars)}-{int(single_cfg.max_chars)} 字。\n"
        )

        tools = ToolSet([MarkdownDocReadTool(), MarkdownDocApplyEditsTool()])

        payload = {
            "mode": "single_agent",
            "now": now,
            "doc_id": doc_id,
            "placeholder": placeholder,
            "todolist": [
                {"id": 1, "task": "总结今日消息", "status": "todo"},
                {"id": 2, "task": "分析有价值的内容", "status": "todo"},
                {"id": 3, "task": "调用工具写文章", "status": "todo"},
            ],
            "constraints": {
                "min_chars": int(single_cfg.min_chars),
                "max_chars": int(single_cfg.max_chars),
                "no_images": True,
                "no_raw_urls": True,
                "language": "zh-CN",
            },
            "materials": [
                {
                    "source_name": m.source_name,
                    "source_type": m.source_type,
                    "source_url": m.source_url,
                    "priority": m.priority,
                    "items": m.items,
                }
                for m in materials
            ],
            "tools": {
                "read": "md_doc_read(doc_id, start=0, max_chars=2400)",
                "apply_edits": "md_doc_apply_edits(doc_id, edits=[...])",
            },
            "output_schema": {
                "done": "boolean",
                "todolist": [{"id": 1, "task": "string", "status": "todo|doing|done"}],
                "notes": "string(optional)",
                "patched_markdown": "string(optional)",
            },
        }

        # Ensure image features are not used even if user enabled image layout in config.
        if bool(user_config.get("image_layout_enabled", False)):
            astrbot_logger.info("[dailynews] [single_agent] ignore image_layout_enabled=true (single mode)")

        try:
            resp = await astrbot_context.tool_loop_agent(
                event=make_internal_event(session_id=f"single:{doc_id}"),
                chat_provider_id=provider_id,
                prompt=json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                tools=tools,
                image_urls=[],
                max_steps=max(5, int(single_cfg.max_steps)),
            )
            raw = getattr(resp, "completion_text", "") or ""
            data = _json_from_text(raw) or {}
            if not isinstance(data, dict):
                astrbot_logger.warning("[dailynews] [single_agent] model did not return JSON; using doc as-is")
            else:
                patched_whole = str(data.get("patched_markdown") or "").strip()
                if patched_whole:
                    write_doc(doc_id, patched_whole)
        except Exception as e:
            astrbot_logger.warning("[dailynews] [single_agent] tool_loop_agent failed: %s", e, exc_info=True)
            # Fallback: do one plain LLM call without tools.
            llm = LLMRunner(
                astrbot_context,
                timeout_s=max(60, int(user_config.get("llm_write_timeout_s", 360) or 360)),
                max_retries=max(0, int(user_config.get("llm_max_retries", 1) or 1)),
                provider_id=provider_id,
            )
            raw_md = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(payload, ensure_ascii=False))
            raw_md = (raw_md or "").strip()
            if raw_md:
                write_doc(doc_id, raw_md)

        md = read_doc(doc_id).strip()
        if placeholder in md:
            # If the model never edited the doc, remove placeholder to avoid leaking.
            md = md.replace(placeholder, "").strip()

        # Safety sanitize: no raw URLs / local file paths / debug leaks.
        md = sanitize_markdown_for_publish(md)
        return md
