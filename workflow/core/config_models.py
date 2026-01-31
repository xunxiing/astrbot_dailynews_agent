from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .models import NewsSourceConfig


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, bool):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return str(default)
    return str(value)

def _to_optional_str(value: Any) -> Optional[str]:
    s = _to_str(value, "").strip()
    return s if s else None


@dataclass(frozen=True)
class RenderImageStyleConfig:
    # NOTE: These values are intentionally hard-coded (see user request).
    # The dashboard config fields for these widths/threshold are disabled/removed.
    full_max_width: int = 400
    medium_max_width: int = 500
    narrow_max_width: int = 300
    float_threshold: int = 480
    float_enabled: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "RenderImageStyleConfig":
        return cls(
            # Hard-coded layout constants, only float_enabled remains configurable.
            full_max_width=400,
            medium_max_width=500,
            narrow_max_width=300,
            float_threshold=480,
            float_enabled=_to_bool(cfg.get("render_img_float_enabled"), True),
        )


@dataclass(frozen=True)
class RenderPipelineConfig:
    page_chars: int = 2600
    max_pages: int = 4
    retries: int = 2
    poll_timeout_s: float = 6.0
    poll_interval_ms: int = 200
    playwright_fallback: bool = True
    playwright_timeout_ms: int = 20000

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "RenderPipelineConfig":
        return cls(
            page_chars=max(0, _to_int(cfg.get("render_page_chars"), 2600)),
            max_pages=max(1, _to_int(cfg.get("render_max_pages"), 4)),
            retries=max(0, _to_int(cfg.get("render_retries"), 2)),
            poll_timeout_s=max(0.5, _to_float(cfg.get("render_poll_timeout_s"), 6.0)),
            poll_interval_ms=max(50, _to_int(cfg.get("render_poll_interval_ms"), 200)),
            playwright_fallback=_to_bool(cfg.get("render_playwright_fallback"), True),
            playwright_timeout_ms=max(1000, _to_int(cfg.get("render_playwright_timeout_ms"), 20000)),
        )


@dataclass(frozen=True)
class ImageLayoutConfig:
    enabled: bool = False
    provider_id: str = ""
    max_images_total: int = 6
    max_images_per_source: int = 3
    pass_images_to_model: bool = True
    max_images_to_model: int = 6
    preview_enabled: bool = False
    preview_max_images: int = 6
    preview_max_width: int = 1080
    preview_gap: int = 8
    request_max_requests: int = 1
    request_max_images: int = 6
    tool_enabled: bool = True
    tool_rounds: int = 2
    tool_max_steps: int = 25
    shuffle_candidates: bool = True
    shuffle_seed: str = ""

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "ImageLayoutConfig":
        return cls(
            enabled=_to_bool(cfg.get("image_layout_enabled"), False),
            provider_id=_to_str(cfg.get("image_layout_provider_id"), "").strip(),
            max_images_total=max(0, _to_int(cfg.get("image_layout_max_images_total"), 6)),
            max_images_per_source=max(1, _to_int(cfg.get("image_layout_max_images_per_source"), 3)),
            pass_images_to_model=_to_bool(cfg.get("image_layout_pass_images_to_model"), True),
            max_images_to_model=max(1, _to_int(cfg.get("image_layout_max_images_to_model"), 6)),
            preview_enabled=_to_bool(cfg.get("image_layout_preview_enabled"), False),
            preview_max_images=max(1, _to_int(cfg.get("image_layout_preview_max_images"), 6)),
            preview_max_width=max(200, _to_int(cfg.get("image_layout_preview_max_width"), 1080)),
            preview_gap=max(0, _to_int(cfg.get("image_layout_preview_gap"), 8)),
            request_max_requests=max(0, _to_int(cfg.get("image_layout_request_max_requests"), 1)),
            request_max_images=max(1, _to_int(cfg.get("image_layout_request_max_images"), 6)),
            tool_enabled=_to_bool(cfg.get("image_layout_tool_enabled"), True),
            tool_rounds=max(1, _to_int(cfg.get("image_layout_tool_rounds"), 2)),
            tool_max_steps=max(5, _to_int(cfg.get("image_layout_tool_max_steps"), 25)),
            shuffle_candidates=_to_bool(cfg.get("image_layout_shuffle_candidates"), True),
            shuffle_seed=_to_str(cfg.get("image_layout_shuffle_seed"), "").strip(),
        )


@dataclass(frozen=True)
class ImageLabelConfig:
    enabled: bool = False
    provider_id: str = ""
    max_images_total: int = 24
    batch_size: int = 2
    concurrency: int = 4
    force_refresh: bool = False
    llm_max_retries: int = 2
    llm_retry_base_s: float = 1.0
    llm_retry_max_s: float = 12.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "ImageLabelConfig":
        return cls(
            enabled=_to_bool(cfg.get("image_label_enabled"), False),
            provider_id=_to_str(cfg.get("image_label_provider_id"), "").strip(),
            max_images_total=max(0, _to_int(cfg.get("image_label_max_images_total"), 24)),
            batch_size=max(1, min(2, _to_int(cfg.get("image_label_batch_size"), 2))),
            concurrency=max(1, _to_int(cfg.get("image_label_concurrency"), 4)),
            force_refresh=_to_bool(cfg.get("image_label_force_refresh"), False),
            llm_max_retries=max(0, _to_int(cfg.get("image_label_llm_max_retries"), 2)),
            llm_retry_base_s=max(0.1, _to_float(cfg.get("image_label_llm_retry_base_s"), 1.0)),
            llm_retry_max_s=max(0.5, _to_float(cfg.get("image_label_llm_retry_max_s"), 12.0)),
        )


@dataclass(frozen=True)
class LayoutRefineConfig:
    enabled: bool = False
    rounds: int = 2
    max_requests: int = 2
    request_max_images: int = 6
    preview_page_chars: int = 2400
    preview_pages: int = 1
    preview_timeout_ms: int = 20000

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "LayoutRefineConfig":
        return cls(
            enabled=_to_bool(cfg.get("image_layout_refine_enabled"), False),
            rounds=max(1, _to_int(cfg.get("image_layout_refine_rounds"), 2)),
            max_requests=max(0, _to_int(cfg.get("image_layout_refine_max_requests"), 2)),
            request_max_images=max(1, _to_int(cfg.get("image_layout_refine_request_max_images"), 6)),
            preview_page_chars=max(400, _to_int(cfg.get("image_layout_refine_preview_page_chars"), 2400)),
            preview_pages=max(1, _to_int(cfg.get("image_layout_refine_preview_pages"), 1)),
            preview_timeout_ms=max(1000, _to_int(cfg.get("image_layout_refine_preview_timeout_ms"), 20000)),
        )


@dataclass(frozen=True)
class LayoutPrompts:
    image_layout_system: str
    image_layout_tool_system: str
    layout_refiner_system: str
    image_labeler_system: str

    @classmethod
    def from_files(
        cls,
        *,
        load_template: Any,
        image_layout_system_path: str = "templates/prompts/image_layout_agent_system.txt",
        image_layout_tool_system_path: str = "templates/prompts/image_layout_tool_agent_system.txt",
        layout_refiner_system_path: str = "templates/prompts/layout_refiner_system.txt",
        image_labeler_system_path: str = "templates/prompts/image_labeler_system.txt",
    ) -> "LayoutPrompts":
        return cls(
            image_layout_system=str(load_template(image_layout_system_path) or "").strip(),
            image_layout_tool_system=str(load_template(image_layout_tool_system_path) or "").strip(),
            layout_refiner_system=str(load_template(layout_refiner_system_path) or "").strip(),
            image_labeler_system=str(load_template(image_labeler_system_path) or "").strip(),
        )


@dataclass(frozen=True)
class NewsSourcesConfig:
    sources: list[NewsSourceConfig]

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "NewsSourcesConfig":
        raw = cfg.get("news_sources") or []
        if not isinstance(raw, list):
            return cls(sources=[])

        out: list[NewsSourceConfig] = []
        seen: set[tuple[str, str]] = set()

        for item in raw:
            if not isinstance(item, Mapping):
                continue
            tkey = _to_str(item.get("__template_key"), "").strip().lower()
            if not tkey:
                continue

            if tkey == "wechat":
                url = _to_str(item.get("url"), "").strip()
                if not url:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 3)
                album_keyword = _to_optional_str(item.get("album_keyword"))
                src = NewsSourceConfig(
                    name=name or f"公众号{len(out) + 1}",
                    url=url,
                    type="wechat",
                    priority=max(1, priority),
                    max_articles=max(1, max_articles),
                    album_keyword=album_keyword,
                )
            elif tkey == "miyoushe":
                url = _to_str(item.get("url"), "").strip()
                if not url:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 3)
                src = NewsSourceConfig(
                    name=name or f"米游社{len(out) + 1}",
                    url=url,
                    type="miyoushe",
                    priority=max(1, priority),
                    max_articles=max(1, max_articles),
                    album_keyword=None,
                )
            elif tkey == "github":
                repo = _to_str(item.get("repo") or item.get("url"), "").strip()
                if not repo:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                src = NewsSourceConfig(
                    name=name or f"GitHub {repo}",
                    url=repo,
                    type="github",
                    priority=max(1, priority),
                    max_articles=1,
                    album_keyword=None,
                )
            elif tkey == "twitter":
                url = _to_str(item.get("url"), "").strip()
                if not url:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 3)
                src = NewsSourceConfig(
                    name=name or f"X/{len(out) + 1}",
                    url=url,
                    type="twitter",
                    priority=max(1, priority),
                    max_articles=max(1, max_articles),
                    album_keyword=None,
                    meta=None,
                )
            elif tkey == "plugin_registry_official":
                since_hours = max(1, _to_int(item.get("since_hours"), 27))
                max_plugins = max(1, _to_int(item.get("max_plugins"), 20))
                name = _to_str(item.get("name"), "").strip()
                src = NewsSourceConfig(
                    name=name or "插件源（官方）",
                    url="official",
                    type="plugin_registry",
                    priority=1,
                    max_articles=max_plugins,
                    album_keyword=None,
                    meta={
                        "registry_kind": "official",
                        "name": name or "插件源（官方）",
                        "since_hours": since_hours,
                        "max_plugins": max_plugins,
                    },
                )
            elif tkey == "plugin_registry_custom":
                path = _to_str(item.get("path"), "").strip()
                if not path:
                    continue
                since_hours = max(1, _to_int(item.get("since_hours"), 27))
                max_plugins = max(1, _to_int(item.get("max_plugins"), 20))
                name = _to_str(item.get("name"), "").strip()
                src = NewsSourceConfig(
                    name=name or "插件源（第三方）",
                    url=path,
                    type="plugin_registry",
                    priority=1,
                    max_articles=max_plugins,
                    album_keyword=None,
                    meta={
                        "registry_kind": "custom",
                        "path": path,
                        "name": name or "插件源（第三方）",
                        "since_hours": since_hours,
                        "max_plugins": max_plugins,
                    },
                )
            elif tkey == "xiuxiu_ai":
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_items = _to_int(item.get("max_items"), 20)
                date = _to_str(item.get("date"), "").strip()
                days_ago = _to_int(item.get("days_ago"), 0)
                src = NewsSourceConfig(
                    name=name or "虎嗅AI产品日报",
                    url="https://xiuxiu.huxiu.com/",
                    type="xiuxiu_ai",
                    priority=max(1, priority),
                    max_articles=max(5, max_items),
                    album_keyword=None,
                    meta={
                        "date": date,
                        "days_ago": max(0, days_ago),
                    },
                )
            else:
                continue

            key = (src.type, src.url.strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(src)

        return cls(sources=out)


@dataclass(frozen=True)
class NewsWorkflowModeConfig:
    """
    Workflow mode switch.
    - multi: existing multi-agent pipeline (default)
    - single: single-agent mode (no image insertion)
    """

    mode: str = "multi"

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "NewsWorkflowModeConfig":
        raw = _to_str(cfg.get("news_workflow_mode"), "multi").strip().lower()
        if raw in {"single", "single_agent", "single-agent"}:
            return cls(mode="single")
        return cls(mode="multi")


@dataclass(frozen=True)
class SingleAgentConfig:
    provider_id: str = ""
    max_steps: int = 18
    min_chars: int = 700
    max_chars: int = 1500

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "SingleAgentConfig":
        min_chars = max(100, _to_int(cfg.get("single_agent_min_chars"), 700))
        max_chars = max(min_chars, _to_int(cfg.get("single_agent_max_chars"), 1500))
        return cls(
            provider_id=_to_str(cfg.get("single_agent_provider_id"), "").strip(),
            max_steps=max(5, _to_int(cfg.get("single_agent_max_steps"), 18)),
            min_chars=min_chars,
            max_chars=max_chars,
        )
