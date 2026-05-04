from __future__ import annotations

import asyncio
import json
import random
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.config_models import ReactAgentConfig
from ...core.internal_event import make_internal_event
from ...pipeline.rendering import load_template
from .shared_memory import SharedMemory
from .tool_registry import ToolRegistry
from .writer_style import REACT_CHIEF_EDITOR_CONCISE_HINT


@dataclass(frozen=True)
class ToolTrace:
    step: int
    tool_name: str
    args: dict[str, Any]
    ok: bool
    content_preview: str
    tool_call_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReactRunResult:
    status: str
    final_markdown: str
    steps: int
    missing_parts: list[str] = field(default_factory=list)
    tool_trace: list[ToolTrace] = field(default_factory=list)
    termination_reason: str = ""
    layout_sub_results: list[Any] = field(default_factory=list)
    image_layout_guidance: str = ""


class ReActAgent:
    def __init__(
        self,
        *,
        astrbot_context: Any,
        registry: ToolRegistry,
        shared_memory: SharedMemory,
        config: ReactAgentConfig,
        provider_id: str = "",
        provider_ids: Iterable[str] | None = None,
        session_id: str | None = None,
    ):
        self._ctx = astrbot_context
        self._registry = registry
        self._memory = shared_memory
        self._config = config
        self._provider_id = str(provider_id or "").strip()
        ids: list[str] = []
        if provider_ids is not None:
            for x in provider_ids:
                if not isinstance(x, str):
                    continue
                s = x.strip()
                if not s or s in ids:
                    continue
                ids.append(s)
        if self._provider_id and self._provider_id not in ids:
            ids.insert(0, self._provider_id)
        self._provider_ids = ids
        self._session_id = session_id or f"react:{uuid.uuid4().hex[:10]}"

    def _resolve_provider_ids(self) -> list[str]:
        if self._provider_ids:
            return list(self._provider_ids)
        if self._provider_id:
            return [self._provider_id]
        return [""]

    def _is_transient_llm_error(self, exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        transient_markers = (
            "too many concurrent",
            "rate limit",
            "ratelimit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "temporarily",
            "try again",
            "server error",
            "internalservererror",
            "bad_response_status_code",
        )
        return isinstance(exc, TimeoutError | asyncio.TimeoutError) or any(
            marker in text for marker in transient_markers
        )

    def _retry_delay_s(self, attempt: int) -> float:
        base = float(getattr(self._config, "llm_retry_base_s", 2.0) or 2.0)
        max_s = float(getattr(self._config, "llm_retry_max_s", 20.0) or 20.0)
        delay = min(max_s, base * (2 ** max(0, attempt - 1)))
        return delay + random.uniform(0.0, min(1.0, delay * 0.25))

    def _format_memory_for_prompt(self, *, max_item_chars: int = 6000) -> str:
        memory = self._memory.read_all()
        if not memory:
            return "(empty)"
        lines: list[str] = []
        for aid, payload in memory.items():
            if isinstance(payload, str):
                txt = payload.strip()
            else:
                try:
                    txt = json.dumps(payload, ensure_ascii=False, indent=2)
                except Exception:
                    txt = str(payload)
            txt = txt.strip()
            if len(txt) > max_item_chars:
                txt = txt[: max_item_chars - 3] + "..."
            lines.append(f"[{aid}]\n{txt}")
        return "\n\n".join(lines).strip()

    def _tools_prompt_block(self) -> str:
        specs = self._registry.list_specs_for_prompt()
        if not specs:
            return "(none)"
        rows: list[str] = []
        for idx, spec in enumerate(specs, start=1):
            rows.append(
                f"{idx}. {spec.get('name')}: {str(spec.get('description') or '').strip()}"
            )
        return "\n".join(rows)

    def _search_tool_strategy_block(self) -> str:
        names = {
            str(spec.get("name") or "").strip()
            for spec in self._registry.list_specs_for_prompt()
        }
        availability: list[str] = []
        for name in (
            "web_search_tavily",
            "tavily_extract_web_page",
            "grok_web_search",
            "web_search",
            "tool_search_web",
        ):
            availability.append(
                f"- {name}: {'available' if name in names else 'unavailable'}"
            )
        template = str(
            load_template("templates/prompts/react_search_strategy.txt") or ""
        ).strip()
        return template.replace(
            "{{SEARCH_TOOL_AVAILABILITY}}", "\n".join(availability)
        ).strip()

    def _system_prompt(self) -> str:
        template = str(
            load_template("templates/prompts/react_chief_editor_system.txt") or ""
        ).strip()
        return template.replace(
            "{{STYLE_HINT}}", str(REACT_CHIEF_EDITOR_CONCISE_HINT or "")
        ).strip()

    def _user_prompt(self, *, user_goal: str, initial_context: str) -> str:
        template = str(
            load_template("templates/prompts/react_chief_editor_user.txt") or ""
        ).strip()
        replacements = {
            "{{USER_GOAL}}": str(user_goal or ""),
            "{{COLLECTED_INTELLIGENCE}}": self._format_memory_for_prompt(),
            "{{INITIAL_CONTEXT}}": str(initial_context or "(none)"),
            "{{AVAILABLE_TOOLS}}": self._tools_prompt_block(),
            "{{SEARCH_STRATEGY}}": self._search_tool_strategy_block(),
        }
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template.strip()

    async def _fallback_plain_llm(
        self, *, user_goal: str, initial_context: str, reason: str
    ) -> str:
        provider_ids = self._resolve_provider_ids()
        system_prompt = self._system_prompt()
        suffix = str(
            load_template("templates/prompts/react_fallback_plain_suffix.txt") or ""
        ).strip()
        suffix = suffix.replace("{{REASON}}", str(reason or "unknown"))
        prompt = "\n\n".join(
            [
                self._user_prompt(user_goal=user_goal, initial_context=initial_context),
                suffix,
            ]
        )
        max_retries = max(0, int(getattr(self._config, "llm_max_retries", 2) or 0))
        for pid in provider_ids:
            if not pid:
                continue
            for attempt in range(1, max_retries + 2):
                try:
                    resp = await self._ctx.llm_generate(
                        chat_provider_id=pid,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    )
                    text = (getattr(resp, "completion_text", "") or "").strip()
                    if text:
                        return text
                    break
                except Exception as e:
                    can_retry = attempt <= max_retries and self._is_transient_llm_error(e)
                    astrbot_logger.warning(
                        "[dailynews][react] fallback plain LLM failed provider=%s attempt=%s/%s retry=%s: %s",
                        pid,
                        attempt,
                        max_retries + 1,
                        can_retry,
                        e,
                        exc_info=not can_retry,
                    )
                    if not can_retry:
                        break
                    await asyncio.sleep(self._retry_delay_s(attempt))
        return ""

    async def run(self, *, user_goal: str, initial_context: str = "") -> ReactRunResult:
        toolset = self._registry.build_toolset()
        event = make_internal_event(session_id=self._session_id)
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(
            user_goal=user_goal, initial_context=initial_context
        )
        provider_ids = self._resolve_provider_ids()
        max_steps = int(self._config.max_steps)
        tool_call_timeout = int(self._config.tool_call_timeout_s)
        max_retries = max(0, int(getattr(self._config, "llm_max_retries", 2) or 0))

        last_error: BaseException | None = None
        for pid in provider_ids:
            if not pid:
                continue
            for attempt in range(1, max_retries + 2):
                try:
                    if attempt > 1:
                        astrbot_logger.info(
                            "[dailynews][react] retry tool_loop_agent provider=%s attempt=%s/%s",
                            pid,
                            attempt,
                            max_retries + 1,
                        )
                    resp = await self._ctx.tool_loop_agent(
                        event=event,
                        chat_provider_id=pid,
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        tools=toolset,
                        max_steps=max_steps,
                        tool_call_timeout=tool_call_timeout,
                    )
                    final_text = (resp.completion_text or "").strip()
                    if final_text:
                        return ReactRunResult(
                            status="success",
                            final_markdown=final_text,
                            steps=max_steps,
                            missing_parts=[],
                            tool_trace=[],
                            termination_reason="completed",
                        )
                    astrbot_logger.warning(
                        "[dailynews][react] provider=%s returned empty completion_text, try next",
                        pid,
                    )
                    break
                except Exception as e:
                    last_error = e
                    can_retry = attempt <= max_retries and self._is_transient_llm_error(e)
                    astrbot_logger.warning(
                        "[dailynews][react] provider=%s tool_loop_agent failed attempt=%s/%s retry=%s: %s",
                        pid,
                        attempt,
                        max_retries + 1,
                        can_retry,
                        e,
                        exc_info=not can_retry,
                    )
                    if not can_retry:
                        break
                    await asyncio.sleep(self._retry_delay_s(attempt))

        fallback_md = await self._fallback_plain_llm(
            user_goal=user_goal,
            initial_context=initial_context,
            reason="tool_loop_agent failed for all providers",
        )
        if fallback_md:
            return ReactRunResult(
                status="partial",
                final_markdown=fallback_md,
                steps=max_steps,
                missing_parts=[],
                tool_trace=[],
                termination_reason="fallback_plain_llm",
            )

        raise RuntimeError(
            f"react provider chain failed: {type(last_error).__name__}: {last_error}"
            if last_error
            else "react provider chain failed: no provider available"
        )
