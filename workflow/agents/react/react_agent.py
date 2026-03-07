from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from astrbot.core.agent.message import (
    AssistantMessageSegment,
    Message,
    ToolCallMessageSegment,
)
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.provider.provider import Provider

from ...core.config_models import ReactAgentConfig
from ...core.internal_event import make_internal_event
from .shared_memory import SharedMemory
from .tool_registry import ToolExecution, ToolRegistry


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


class ReActAgent:
    def __init__(
        self,
        *,
        astrbot_context: Any,
        registry: ToolRegistry,
        shared_memory: SharedMemory,
        config: ReactAgentConfig,
        provider_id: str = "",
        session_id: str | None = None,
    ):
        self._ctx = astrbot_context
        self._registry = registry
        self._memory = shared_memory
        self._config = config
        self._provider_id = str(provider_id or "").strip()
        self._session_id = session_id or f"react:{uuid.uuid4().hex[:10]}"

    async def _resolve_provider(self) -> Provider:
        if self._provider_id:
            prov = await self._ctx.provider_manager.get_provider_by_id(
                self._provider_id
            )
            if prov is not None:
                return prov
            astrbot_logger.warning(
                "[dailynews][react] provider_id=%s not found, fallback to current provider",
                self._provider_id,
            )
        prov = self._ctx.get_using_provider()
        if prov is None:
            raise RuntimeError("No chat provider available for react mode")
        return prov

    def _format_memory_for_prompt(self, *, max_item_chars: int = 2400) -> str:
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
        rows = [
            "Web investigation priority:",
            "1) Prefer Tavily-class search when available: `web_search_tavily` for retrieval and `tavily_extract_web_page` for extracting the target page.",
            "2) `grok_web_search` is the same priority tier as Tavily and can be used as a first-class search tool when available.",
            "3) Use `web_search` only as the normal fallback when Tavily/Grok tools are unavailable or unsuitable.",
            "4) If none of the above exist, use local `tool_search_web` only as the final safety fallback.",
            "5) After searching, prefer opening/extracting the most relevant page instead of relying only on result snippets.",
            "6) Avoid repeating near-identical searches once evidence is sufficient.",
            "",
            "Current search-tool availability:",
        ]
        for name in (
            "web_search_tavily",
            "tavily_extract_web_page",
            "grok_web_search",
            "web_search",
            "tool_search_web",
        ):
            rows.append(f"- {name}: {'available' if name in names else 'unavailable'}")
        return "\n".join(rows).strip()

    def _system_prompt(self) -> str:
        return (
            "You are a senior AI daily-news chief editor with strong planning and dispatch ability.\n"
            "You must use native function-calling tools whenever information is missing.\n"
            "For web investigation, treat Tavily-based tools and `grok_web_search` as first-class search options.\n"
            "Prefer Tavily when page extraction is needed, and use `web_search` only as the normal fallback.\n"
            "Do not output fake tool calls in plain text.\n"
            "When the collected evidence is enough, stop using tools and produce the final markdown report."
        )

    def _user_prompt(
        self,
        *,
        user_goal: str,
        initial_context: str,
        step: int,
        max_steps: int,
    ) -> str:
        return (
            "User Goal:\n"
            f"{user_goal}\n\n"
            "Collected Intelligence:\n"
            f"{self._format_memory_for_prompt()}\n\n"
            "Initial Context:\n"
            f"{initial_context or '(none)'}\n\n"
            "Available Tools:\n"
            f"{self._tools_prompt_block()}\n\n"
            "Search Strategy:\n"
            f"{self._search_tool_strategy_block()}\n\n"
            "Execution Rules:\n"
            "1) If information for the goal is missing, call the right tool.\n"
            "2) If coverage is sufficient, stop tools and return final markdown directly.\n"
            "3) If some parts are unavailable, clearly mention missing parts in the final report.\n"
            "4) Use web-search tools for recent facts, external background, and cross-source verification.\n"
            "5) If a good page-extraction tool is available, extract the target page before summarizing it.\n"
            f"Current step: {step}/{max_steps}."
        )

    async def _finalize_without_tools(
        self,
        *,
        provider: Provider,
        conversation: list[Message],
        user_goal: str,
        reason: str,
        initial_context: str,
    ) -> str:
        msgs: list[Message] = [
            Message(role="system", content=self._system_prompt()),
            Message(
                role="user",
                content=self._user_prompt(
                    user_goal=user_goal,
                    initial_context=initial_context,
                    step=self._config.max_steps,
                    max_steps=self._config.max_steps,
                ),
            ),
            *conversation,
            Message(
                role="user",
                content=(
                    f"Stop tool usage now due to: {reason}. "
                    "Produce the best possible final markdown report with explicit missing-parts notes if needed."
                ),
            ),
        ]
        resp = await provider.text_chat(contexts=msgs, func_tool=None)
        return (resp.completion_text or "").strip()

    async def run(self, *, user_goal: str, initial_context: str = "") -> ReactRunResult:
        provider = await self._resolve_provider()
        toolset = self._registry.build_toolset()
        event = make_internal_event(session_id=self._session_id)
        run_context = ContextWrapper(
            context=AstrAgentContext(context=self._ctx, event=event),
            tool_call_timeout=int(self._config.tool_call_timeout_s),
        )

        conversation: list[Message] = []
        traces: list[ToolTrace] = []
        failure_count = 0
        no_progress_rounds = 0
        repeat_count = 0
        last_action_fp = ""
        termination_reason = ""

        for step in range(1, int(self._config.max_steps) + 1):
            messages: list[Message] = [
                Message(role="system", content=self._system_prompt()),
                Message(
                    role="user",
                    content=self._user_prompt(
                        user_goal=user_goal,
                        initial_context=initial_context,
                        step=step,
                        max_steps=int(self._config.max_steps),
                    ),
                ),
                *conversation,
            ]

            llm_resp = await provider.text_chat(contexts=messages, func_tool=toolset)
            if llm_resp.role == "err":
                termination_reason = f"llm_error: {llm_resp.completion_text}"
                break

            tool_names = llm_resp.tools_call_name or []
            tool_args_list = llm_resp.tools_call_args or []
            tool_ids = llm_resp.tools_call_ids or []

            if not tool_names:
                final_text = (llm_resp.completion_text or "").strip()
                return ReactRunResult(
                    status="success" if final_text else "failed",
                    final_markdown=final_text,
                    steps=step,
                    missing_parts=[],
                    tool_trace=traces,
                    termination_reason="final_answer",
                )

            conversation.append(
                AssistantMessageSegment(
                    tool_calls=llm_resp.to_openai_to_calls_model(),
                    content=(llm_resp.completion_text or None),
                )
            )

            round_progress = False
            for idx, tool_name in enumerate(tool_names):
                raw_args = tool_args_list[idx] if idx < len(tool_args_list) else {}
                if isinstance(raw_args, dict):
                    args = raw_args
                elif isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                        args = parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        args = {}
                else:
                    args = {}
                tool_call_id = tool_ids[idx] if idx < len(tool_ids) else f"call_{idx}"
                action_fp = f"{tool_name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                if action_fp == last_action_fp:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_action_fp = action_fp

                if repeat_count > int(self._config.max_repeat_action):
                    termination_reason = f"repeat_action_limit_exceeded: {tool_name} repeated {repeat_count} times"
                    break

                exec_res: ToolExecution = await self._registry.execute(
                    name=tool_name,
                    args=args,
                    run_context=run_context,
                    tool_call_id=tool_call_id,
                )
                conversation.append(
                    ToolCallMessageSegment(
                        role="tool",
                        tool_call_id=tool_call_id,
                        content=exec_res.content,
                    )
                )
                preview = exec_res.content.strip()
                if len(preview) > 500:
                    preview = preview[:497] + "..."
                if bool(self._config.enable_trace):
                    traces.append(
                        ToolTrace(
                            step=step,
                            tool_name=tool_name,
                            args=args,
                            ok=exec_res.ok,
                            content_preview=preview,
                            tool_call_id=tool_call_id,
                            error=exec_res.error,
                        )
                    )

                if exec_res.ok and preview and not preview.lower().startswith("error:"):
                    round_progress = True
                else:
                    failure_count += 1
                    if failure_count >= int(self._config.max_tool_failures):
                        termination_reason = (
                            f"tool_failure_limit_exceeded: {failure_count}"
                        )
                        break

            if termination_reason:
                break

            if round_progress:
                no_progress_rounds = 0
            else:
                no_progress_rounds += 1
                if no_progress_rounds >= int(self._config.max_no_progress_rounds):
                    termination_reason = (
                        f"no_progress_limit_exceeded: {no_progress_rounds}"
                    )
                    break

        if not termination_reason:
            termination_reason = "max_steps_reached"

        final_md = await self._finalize_without_tools(
            provider=provider,
            conversation=conversation,
            user_goal=user_goal,
            reason=termination_reason,
            initial_context=initial_context,
        )
        return ReactRunResult(
            status="partial" if final_md else "failed",
            final_markdown=final_md,
            steps=int(self._config.max_steps),
            missing_parts=[],
            tool_trace=traces,
            termination_reason=termination_reason,
        )
