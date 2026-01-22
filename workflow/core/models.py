from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class NewsSourceConfig:
    name: str
    url: str
    type: str = "wechat"  # wechat, miyoushe, etc.
    priority: int = 1
    max_articles: int = 3
    album_keyword: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SubAgentResult:
    source_name: str
    content: str
    summary: str
    key_points: List[str]
    images: Optional[List[str]] = None
    error: Optional[str] = None
    no_llm_merge: bool = False


@dataclass
class MainAgentDecision:
    sources_to_process: List[str]
    processing_instructions: Dict[str, str]
    final_format: str = "markdown"
