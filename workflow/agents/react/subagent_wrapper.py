from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .shared_memory import SharedMemory


SubAgentRunner = Callable[[str], Awaitable[Any]]


@dataclass(frozen=True)
class SubAgentExecutionResult:
    ok: bool
    agent_id: str
    content: Any
    injected_dependencies: dict[str, Any]
    composed_system_prompt: str
    error: str | None = None


class SubAgentWrapper:
    def __init__(
        self,
        *,
        agent_id: str,
        task_description: str,
        runner: SubAgentRunner,
        dependency_ids: list[str] | None = None,
    ):
        self.agent_id = str(agent_id or "").strip()
        self.task_description = str(task_description or "").strip()
        self.runner = runner
        self.dependency_ids = list(dependency_ids or [])
        if not self.agent_id:
            raise ValueError("agent_id is required")
        if not self.task_description:
            raise ValueError("task_description is required")

    @staticmethod
    def _to_text(value: Any, *, max_chars: int = 2400) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                text = str(value)
        text = text.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def _compose_system_prompt(self, dep_payload: dict[str, Any]) -> str:
        if not dep_payload:
            return self.task_description

        lines = [self.task_description, "", "Context from completed teammate agents:"]
        for aid, content in dep_payload.items():
            lines.append(f"[{aid}]")
            lines.append(self._to_text(content))
            lines.append("")
        lines.append(
            "Use the context above as background knowledge and complete your task."
        )
        return "\n".join(lines).strip()

    async def execute(self, *, shared_memory: SharedMemory) -> SubAgentExecutionResult:
        dep_payload = (
            shared_memory.read(self.dependency_ids) if self.dependency_ids else {}
        )
        composed_prompt = self._compose_system_prompt(dep_payload)
        try:
            result = await self.runner(composed_prompt)
            shared_memory.write(self.agent_id, result)
            return SubAgentExecutionResult(
                ok=True,
                agent_id=self.agent_id,
                content=result,
                injected_dependencies=dep_payload,
                composed_system_prompt=composed_prompt,
                error=None,
            )
        except Exception as e:
            shared_memory.write(
                self.agent_id,
                {"error": str(e), "agent_id": self.agent_id},
            )
            return SubAgentExecutionResult(
                ok=False,
                agent_id=self.agent_id,
                content="",
                injected_dependencies=dep_payload,
                composed_system_prompt=composed_prompt,
                error=str(e),
            )

