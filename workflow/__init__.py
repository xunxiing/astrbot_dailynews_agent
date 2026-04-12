from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "DailyNewsScheduler",
    "load_template",
    "get_plugin_data_dir",
    "merge_images_vertical",
    "ImageLayoutAgent",
    "SubAgentResult",
    "NewsSourceConfig",
    "MainAgentDecision",
    "RenderImageStyleConfig",
    "RenderPipelineConfig",
    "ImageLayoutConfig",
    "LayoutRefineConfig",
    "render_daily_news_pages",
    "split_pages",
    "NewsWorkflowManager",
    "LLMRunner",
    "_run_sync",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DailyNewsScheduler": (".pipeline.scheduler", "DailyNewsScheduler"),
    "load_template": (".pipeline.rendering", "load_template"),
    "get_plugin_data_dir": (".core.image_utils", "get_plugin_data_dir"),
    "merge_images_vertical": (".core.image_utils", "merge_images_vertical"),
    "ImageLayoutAgent": (".agents.image_layout_agent", "ImageLayoutAgent"),
    "SubAgentResult": (".core.models", "SubAgentResult"),
    "NewsSourceConfig": (".core.models", "NewsSourceConfig"),
    "MainAgentDecision": (".core.models", "MainAgentDecision"),
    "RenderImageStyleConfig": (".core.config_models", "RenderImageStyleConfig"),
    "RenderPipelineConfig": (".core.config_models", "RenderPipelineConfig"),
    "ImageLayoutConfig": (".core.config_models", "ImageLayoutConfig"),
    "LayoutRefineConfig": (".core.config_models", "LayoutRefineConfig"),
    "render_daily_news_pages": (".pipeline.render_pipeline", "render_daily_news_pages"),
    "split_pages": (".pipeline.render_pipeline", "split_pages"),
    "NewsWorkflowManager": (".pipeline.workflow_manager", "NewsWorkflowManager"),
    "LLMRunner": (".core.llm", "LLMRunner"),
    "_run_sync": (".core.utils", "_run_sync"),
}

if TYPE_CHECKING:
    from .agents.image_layout_agent import ImageLayoutAgent
    from .core.config_models import (
        ImageLayoutConfig,
        LayoutRefineConfig,
        RenderImageStyleConfig,
        RenderPipelineConfig,
    )
    from .core.image_utils import get_plugin_data_dir, merge_images_vertical
    from .core.llm import LLMRunner
    from .core.models import MainAgentDecision, NewsSourceConfig, SubAgentResult
    from .core.utils import _run_sync
    from .pipeline.render_pipeline import render_daily_news_pages, split_pages
    from .pipeline.rendering import load_template
    from .pipeline.scheduler import DailyNewsScheduler
    from .pipeline.workflow_manager import NewsWorkflowManager


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
