from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import mcp.types

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor


ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    tool_name: str
    content: str
    tool_call_id: str | None = None
    error: str | None = None


class InternalFunctionTool(FunctionTool[AstrAgentContext]):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            handler=None,
        )
        self._handler = handler

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs):
        return await self._handler(context=context, **kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, FunctionTool[AstrAgentContext]] = {}

    def register(self, tool: FunctionTool[AstrAgentContext]) -> None:
        self._tools[str(tool.name)] = tool

    def register_callable(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        self.register(
            InternalFunctionTool(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
            )
        )

    def get(self, name: str) -> FunctionTool[AstrAgentContext] | None:
        return self._tools.get(str(name or "").strip())

    def list_specs_for_prompt(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tool in self._tools.values():
            out.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return out

    def build_toolset(self) -> ToolSet:
        return ToolSet(list(self._tools.values()))

    def merge_global_tools(
        self,
        astrbot_context: Any,
        *,
        prefer_existing: bool = True,
        include_inactive: bool = False,
    ) -> int:
        """
        Merge globally-registered AstrBot tools into this local registry.
        Returns merged count.
        """
        tmgr = None
        try:
            if hasattr(astrbot_context, "get_llm_tool_manager"):
                tmgr = astrbot_context.get_llm_tool_manager()
            elif (
                hasattr(astrbot_context, "provider_manager")
                and hasattr(astrbot_context.provider_manager, "llm_tools")
            ):
                tmgr = astrbot_context.provider_manager.llm_tools
        except Exception:
            tmgr = None

        if tmgr is None:
            return 0

        candidates: list[FunctionTool[AstrAgentContext]] = []
        try:
            if hasattr(tmgr, "get_full_tool_set"):
                ts = tmgr.get_full_tool_set()
                if ts:
                    for tool in ts:
                        if isinstance(tool, FunctionTool):
                            candidates.append(tool)
        except Exception:
            pass

        if not candidates:
            raw = getattr(tmgr, "func_list", None)
            if isinstance(raw, list):
                for tool in raw:
                    if isinstance(tool, FunctionTool):
                        candidates.append(tool)

        merged = 0
        for tool in candidates:
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue
            if (
                (not include_inactive)
                and hasattr(tool, "active")
                and (not bool(getattr(tool, "active", True)))
            ):
                continue
            if prefer_existing and name in self._tools:
                continue
            self._tools[name] = tool
            merged += 1
        return merged

    def _filter_args(
        self, tool: FunctionTool[AstrAgentContext], args: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            return {}
        props = (tool.parameters or {}).get("properties")
        if not isinstance(props, dict) or not props:
            return args
        allowed = set(props.keys())
        return {k: v for k, v in args.items() if k in allowed}

    @staticmethod
    def _result_to_text(resp: mcp.types.CallToolResult) -> str:
        if not resp.content:
            return ""
        chunks: list[str] = []
        for item in resp.content:
            if isinstance(item, mcp.types.TextContent):
                if item.text:
                    chunks.append(item.text)
            elif isinstance(item, mcp.types.ImageContent):
                chunks.append(
                    "Tool returned an image. Use this as visual evidence in your report."
                )
            elif isinstance(item, mcp.types.EmbeddedResource):
                resource = item.resource
                if isinstance(resource, mcp.types.TextResourceContents):
                    chunks.append(resource.text or "")
                elif (
                    isinstance(resource, mcp.types.BlobResourceContents)
                    and resource.mimeType
                    and resource.mimeType.startswith("image/")
                ):
                    chunks.append(
                        "Tool returned an image resource. Use this as visual evidence in your report."
                    )
        return "\n".join([c for c in chunks if c]).strip()

    async def execute(
        self,
        *,
        name: str,
        args: dict[str, Any],
        run_context: ContextWrapper[AstrAgentContext],
        tool_call_id: str | None = None,
    ) -> ToolExecution:
        tool_name = str(name or "").strip()
        tool = self.get(tool_name)
        if tool is None:
            return ToolExecution(
                ok=False,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                content=f"error: Tool `{tool_name}` not found.",
                error="tool_not_found",
            )

        valid_args = self._filter_args(tool, args if isinstance(args, dict) else {})
        try:
            results: list[str] = []
            executor = FunctionToolExecutor.execute(
                tool=tool,
                run_context=run_context,
                **valid_args,
            )
            async for resp in executor:  # type: ignore
                if isinstance(resp, mcp.types.CallToolResult):
                    text = self._result_to_text(resp)
                    if text:
                        results.append(text)
                elif resp is None:
                    results.append(
                        "The tool sent output directly and returned no textual payload."
                    )
                else:
                    results.append(str(resp))

            content = "\n".join([x for x in results if x]).strip()
            if not content:
                content = "Tool execution succeeded with no textual output."
            return ToolExecution(
                ok=True,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                content=content,
                error=None,
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews][react] tool execution failed tool=%s args=%s error=%s",
                tool_name,
                json.dumps(valid_args, ensure_ascii=False),
                e,
                exc_info=True,
            )
            return ToolExecution(
                ok=False,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                content=f"error: {e}",
                error=str(e),
            )
