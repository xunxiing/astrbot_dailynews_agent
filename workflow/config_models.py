from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


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


@dataclass(frozen=True)
class RenderImageStyleConfig:
    full_max_width: int = 1000
    medium_max_width: int = 820
    narrow_max_width: int = 420
    float_threshold: int = 480
    float_enabled: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "RenderImageStyleConfig":
        return cls(
            full_max_width=max(200, _to_int(cfg.get("render_img_full_max_width"), 1000)),
            medium_max_width=max(200, _to_int(cfg.get("render_img_medium_max_width"), 820)),
            narrow_max_width=max(160, _to_int(cfg.get("render_img_narrow_max_width"), 420)),
            float_threshold=max(160, _to_int(cfg.get("render_img_float_threshold"), 480)),
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

    @classmethod
    def from_files(
        cls,
        *,
        load_template: Any,
        image_layout_system_path: str = "templates/prompts/image_layout_agent_system.txt",
        image_layout_tool_system_path: str = "templates/prompts/image_layout_tool_agent_system.txt",
        layout_refiner_system_path: str = "templates/prompts/layout_refiner_system.txt",
    ) -> "LayoutPrompts":
        return cls(
            image_layout_system=str(load_template(image_layout_system_path) or "").strip(),
            image_layout_tool_system=str(load_template(image_layout_tool_system_path) or "").strip(),
            layout_refiner_system=str(load_template(layout_refiner_system_path) or "").strip(),
        )
