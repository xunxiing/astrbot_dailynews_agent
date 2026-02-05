from dataclasses import dataclass
from typing import Any


@dataclass
class NewsSourceConfig:
    name: str
    url: str
    type: str = "wechat"  # wechat, miyoushe, etc.
    priority: int = 1
    max_articles: int = 3
    album_keyword: str | None = None
    meta: dict[str, Any] | None = None


@dataclass
class SubAgentResult:
    source_name: str
    content: str
    summary: str
    key_points: list[str]
    images: list[str] | None = None
    error: str | None = None
    no_llm_merge: bool = False


@dataclass
class MainAgentDecision:
    sources_to_process: list[str]
    processing_instructions: dict[str, str]
    final_format: str = "markdown"
