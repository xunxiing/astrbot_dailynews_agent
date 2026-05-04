from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from datetime import datetime
from typing import Any, Callable, TypeVar

from ...core.decorators import get_logger
from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.utils import _json_from_text

T = TypeVar("T", bound="BaseSourceAgent")

DEFAULT_ANALYZE_SYSTEM_PROMPT = (
    "你是子Agent（信息侦察）。"
    "你将收到某个来源的最新文章标题与链接。"
    "请快速判断今日主要看点/主题，并给出可写作的角度建议。"
    "只输出 JSON，不要输出其它文本。"
)

DEFAULT_ANALYZE_RULES = (
    "\n\nCRITICAL OUTPUT RULES (must follow):\n"
    "1) Never output raw URLs (no lines starting with http/https). "
    "All links must be Markdown links like [阅读原文](URL).\n"
    "2) Ban vague filler like '优化体验/修复部分bug'. "
    "Use concrete details from the provided excerpts: event name, version/patch, mechanics/changes, numbers.\n"
    "3) Prefer 3-6 bullets max. Each bullet:\n"
    "   - **标题**：一句话结论。 ( [阅读原文](url) )\n"
    "     - 细节：至少 1 条具体细节。\n"
    "4) If you cannot extract concrete details, output an empty section_markdown (do NOT make up content).\n"
)

DEFAULT_PROCESS_SYSTEM_PROMPT = (
    "你是子Agent（写作）。"
    "你会收到某个信息来源的最新文章摘要和写作指令。"
    "请写出今日日报中的一个 Markdown 小节：包含小标题、要点、并附上原文链接。"
    "风格轻松、像科技/游戏博主。只输出 JSON。"
    "请你务必使用中文进行总结。"
)


class BaseSourceAgent(ABC):
    """Base class for news source agents.

    Subclasses **must** implement :meth:`fetch_latest_articles`.

    Subclasses **may** override:
    - :meth:`analyze_source` — default uses :meth:`get_analyze_system_prompt`,
      :meth:`build_analyze_prompt`, and :meth:`parse_analyze_response`.
    - :meth:`process_source` — default uses :meth:`get_process_system_prompt`,
      :meth:`build_process_prompt`, and :meth:`parse_process_response`.

    Subclasses **may** also just override the individual hook methods
    (e.g. change only the system prompt) and keep the default flow.
    """

    source_type: str = ""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__module__)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_latest_articles(
        self,
        source: NewsSourceConfig,
        user_config: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Fetch raw article list for *source*.

        Returns ``(source_name, articles)`` where *articles* is a list of
        dicts with at least ``title`` and ``url`` keys.
        """

    # ------------------------------------------------------------------
    # Analyze (default implementation using hooks)
    # ------------------------------------------------------------------

    async def analyze_source(
        self,
        source: NewsSourceConfig,
        articles: list[dict[str, Any]],
        llm: LLMRunner,
    ) -> dict[str, Any]:
        prompt_data = self.build_analyze_prompt(source, articles)
        raw = await llm.ask(
            system_prompt=self.get_analyze_system_prompt(),
            prompt=json.dumps(prompt_data, ensure_ascii=False),
        )
        return self.parse_analyze_response(raw, source, articles)

    def get_analyze_system_prompt(self) -> str:
        return DEFAULT_ANALYZE_SYSTEM_PROMPT + DEFAULT_ANALYZE_RULES

    def build_analyze_prompt(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
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

    def parse_analyze_response(
        self,
        raw: str,
        source: NewsSourceConfig,
        articles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        data = _json_from_text(raw) or {}
        topics = data.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        quality_score = self._normalize_quality_score(
            data.get("quality_score"), articles, topics
        )
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

    # ------------------------------------------------------------------
    # Process (default implementation using hooks)
    # ------------------------------------------------------------------

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
                error="该来源未抓取到任何内容",
            )
        prompt_data = self.build_process_prompt(
            source, instruction, articles, user_config
        )
        raw = await llm.ask(
            system_prompt=self.get_process_system_prompt(),
            prompt=json.dumps(prompt_data, ensure_ascii=False),
        )
        return self.parse_process_response(raw, source, articles, user_config)

    def get_process_system_prompt(self) -> str:
        return DEFAULT_PROCESS_SYSTEM_PROMPT

    def build_process_prompt(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: list[dict[str, Any]],
        user_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "source_name": source.name,
            "instruction": instruction,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "articles": articles,
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

    def parse_process_response(
        self,
        raw: str,
        source: NewsSourceConfig,
        articles: list[dict[str, Any]],
        user_config: dict[str, Any] | None,
    ) -> SubAgentResult:
        data = _json_from_text(raw) or {}
        section = str(data.get("section_markdown") or "").strip()
        summary = str(data.get("summary") or "").strip()
        key_points = (
            data.get("key_points") if isinstance(data.get("key_points"), list) else []
        )
        key_points = [str(x) for x in key_points if isinstance(x, (str, int, float))][
            :10
        ]
        return SubAgentResult(
            source_name=source.name,
            content=section,
            summary=summary,
            key_points=key_points,
            error=None if section else "empty section",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_quality_score(
        quality: Any,
        articles: list[dict[str, Any]],
        topics: list[Any],
    ) -> int:
        try:
            if isinstance(quality, float) and 0 <= quality <= 1:
                return int(quality * 100)
            if isinstance(quality, str):
                q = quality.strip()
                qf = float(q[:-1]) if q.endswith("%") else float(q)
                return int(qf * 100) if 0 <= qf <= 1 else int(qf)
            return int(quality)
        except Exception:
            return len(articles) * 2 + len(topics)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_source_registry: dict[str, type[BaseSourceAgent]] = {}


def register_source(
    source_type: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator that registers a source agent class.

    Usage::

        @register_source("rss")
        class RssSubAgent(BaseSourceAgent):
            async def fetch_latest_articles(self, source, user_config):
                ...
    """

    def decorator(cls: type[T]) -> type[T]:
        st = (
            source_type
            or getattr(cls, "source_type", "")
            or cls.__name__.removesuffix("SubAgent").lower()
        )
        if not st:
            raise ValueError(
                f"Cannot infer source_type for {cls.__name__}; "
                "pass it explicitly to @register_source(...)"
            )
        cls.source_type = st
        _source_registry[st] = cls
        return cls

    return decorator


def get_source_registry() -> dict[str, type[BaseSourceAgent]]:
    """Return a shallow copy of the current registry."""
    return dict(_source_registry)


def unregister_source(source_type: str) -> type[BaseSourceAgent] | None:
    return _source_registry.pop(source_type, None)


def clear_source_registry() -> None:
    _source_registry.clear()
