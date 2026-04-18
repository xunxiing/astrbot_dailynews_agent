from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..agents.sources import GENERIC_SOURCE_TEMPLATE_KEYS
from .astrbook_client import ASTRBOOK_API_BASE
from .models import NewsSourceConfig

ASTRBOT_RELEASES_ATOM_URL = "https://github.com/AstrBotDevs/AstrBot/releases.atom"
AVAILABLE_SOURCE_PRESET_TAGS: tuple[str, ...] = ("astrbot",)

IMAGE_LABEL_ENABLED = True
IMAGE_LABEL_PROVIDER_ID = "modelscope_source/Qwen/Qwen3-VL-235B-A22B-Instruct"
IMAGE_LABEL_MAX_IMAGES_TOTAL = 24

IMAGE_LAYOUT_ENABLED = True
IMAGE_LAYOUT_PROVIDER_ID = "modelscope_source/Qwen/Qwen3-VL-235B-A22B-Instruct"
IMAGE_LAYOUT_MAX_IMAGES_TOTAL = 6
IMAGE_LAYOUT_MAX_IMAGES_PER_SOURCE = 3
IMAGE_LAYOUT_SOURCES: tuple[str, ...] = ()
IMAGE_LAYOUT_PASS_IMAGES_TO_MODEL = True
IMAGE_LAYOUT_MAX_IMAGES_TO_MODEL = 6
IMAGE_LAYOUT_REQUEST_MAX_REQUESTS = 1
IMAGE_LAYOUT_REQUEST_MAX_IMAGES = 6
IMAGE_LAYOUT_GENERATION_ENABLED = False
IMAGE_LAYOUT_GENERATION_RESOLUTION = "1K"
IMAGE_LAYOUT_GENERATION_ASPECT_RATIO = "4:3"

DAILY_NEWS_TEMPLATE_WIDTH = 1120
CHENYU_TEMPLATE_WIDTH = 1280


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
    if isinstance(value, int | float):
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


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def _to_optional_str(value: Any) -> str | None:
    s = _to_str(value, "").strip()
    return s if s else None


def normalize_source_preset_tags(value: Any) -> list[str]:
    out: list[str] = []
    for item in _to_str_list(value):
        tag = item.strip().lower()
        if tag and tag in AVAILABLE_SOURCE_PRESET_TAGS and tag not in out:
            out.append(tag)
    return out


def _build_source_preset_items(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    tags = normalize_source_preset_tags(cfg.get("source_preset_tags"))

    items: list[dict[str, Any]] = []
    if "astrbot" in tags:
        items.extend(
            [
                {
                    "__template_key": "plugin_registry_official",
                    "name": "AstrBot 插件市场",
                    "since_hours": 24,
                    "max_plugins": 20,
                },
                {
                    "__template_key": "github",
                    "name": "AstrBot Core",
                    "repo": "AstrBotDevs/AstrBot",
                    "since_hours": 24,
                    "max_releases": 3,
                    "max_commits": 10,
                    "max_prs": 10,
                },
                {
                    "__template_key": "rss",
                    "name": "AstrBot 版本更新",
                    "url": ASTRBOT_RELEASES_ATOM_URL,
                    "since_hours": 24,
                    "max_articles": 5,
                },
            ]
        )

    return items


def _to_meta_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:
            return {}
        if isinstance(data, Mapping):
            return dict(data)
    return {}


def _build_generic_source_config(
    item: Mapping[str, Any],
    *,
    default_type: str,
    default_index: int,
) -> NewsSourceConfig | None:
    source_type = _to_str(item.get("type"), default_type).strip().lower()
    url = _to_str(item.get("url"), "").strip()
    if not source_type or not url:
        return None

    meta = _to_meta_mapping(item.get("meta"))
    for key, value in item.items():
        if key in {
            "__template_key",
            "name",
            "url",
            "type",
            "priority",
            "max_articles",
            "album_keyword",
            "meta",
        }:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list | dict) and not value:
            continue
        meta.setdefault(key, value)

    return NewsSourceConfig(
        name=_to_str(item.get("name"), "").strip() or f"{source_type} {default_index}",
        url=url,
        type=source_type,
        priority=max(1, _to_int(item.get("priority"), 1)),
        max_articles=max(1, _to_int(item.get("max_articles"), 3)),
        album_keyword=_to_optional_str(item.get("album_keyword")),
        meta=meta or None,
    )


@dataclass(frozen=True)
class RenderImageStyleConfig:
    # NOTE: These values are intentionally hard-coded (see user request).
    # The dashboard config fields for these widths/threshold are disabled/removed.
    full_max_width: int = 500
    medium_max_width: int = 420
    narrow_max_width: int = 340
    float_threshold: int = 360
    float_enabled: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> RenderImageStyleConfig:
        return cls(
            # Hard-coded layout constants, only float_enabled remains configurable.
            full_max_width=500,
            medium_max_width=420,
            narrow_max_width=340,
            float_threshold=360,
            float_enabled=_to_bool(cfg.get("render_img_float_enabled"), True),
        )


@dataclass(frozen=True)
class LayoutModelContext:
    template_name: str = "daily_news"
    page_width_px: int = DAILY_NEWS_TEMPLATE_WIDTH
    content_width_px: int = 1020
    float_enabled: bool = True
    float_image_min_px: int = 300
    float_image_max_px: int = 360
    float_image_max_ratio: float = 0.46
    float_gap_px: int = 16
    min_text_column_px: int = 644
    full_max_width: int = 500
    medium_max_width: int = 420
    narrow_max_width: int = 340
    float_threshold: int = 360

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "LayoutModelContext":
        style = RenderImageStyleConfig.from_mapping(cfg)
        template_name = _to_str(cfg.get("render_template_name"), "daily_news").strip().lower()
        is_chenyu = "chenyu" in template_name
        page_width_px = CHENYU_TEMPLATE_WIDTH if is_chenyu else DAILY_NEWS_TEMPLATE_WIDTH
        # Approximate readable markdown width after container/paper paddings in the templates.
        content_width_px = 1120 if is_chenyu else 1020
        float_image_min_px = 300
        float_image_max_px = min(360, content_width_px)
        float_gap_px = 16
        min_text_column_px = max(
            320,
            int(content_width_px - float_image_max_px - float_gap_px),
        )
        return cls(
            template_name=template_name or "daily_news",
            page_width_px=page_width_px,
            content_width_px=content_width_px,
            float_enabled=style.float_enabled,
            float_image_min_px=float_image_min_px,
            float_image_max_px=float_image_max_px,
            float_image_max_ratio=0.46,
            float_gap_px=float_gap_px,
            min_text_column_px=min_text_column_px,
            full_max_width=int(style.full_max_width),
            medium_max_width=int(style.medium_max_width),
            narrow_max_width=int(style.narrow_max_width),
            float_threshold=int(style.float_threshold),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "template_name": self.template_name,
            "page_width_px": int(self.page_width_px),
            "content_width_px": int(self.content_width_px),
            "float_enabled": bool(self.float_enabled),
            "float_image_min_px": int(self.float_image_min_px),
            "float_image_max_px": int(self.float_image_max_px),
            "float_image_max_ratio": float(self.float_image_max_ratio),
            "float_gap_px": int(self.float_gap_px),
            "min_text_column_px": int(self.min_text_column_px),
            "full_image_width_px": int(self.full_max_width),
            "medium_image_width_px": int(self.medium_max_width),
            "narrow_image_width_px": int(self.narrow_max_width),
            "float_threshold_px": int(self.float_threshold),
            "rules": [
                "Prefer side float only when the image can fit inside the readable column without crushing text.",
                "Use center/full for hero posters, atmosphere art, or extra-wide panorama images.",
                "Use external when the image is an extra-tall long poster or long screenshot.",
            ],
        }


@dataclass(frozen=True)
class RenderPipelineConfig:
    page_chars: int = 2600
    max_pages: int = 4
    retries: int = 2
    poll_timeout_s: float = 6.0
    poll_interval_ms: int = 200

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> RenderPipelineConfig:
        return cls(
            page_chars=max(0, _to_int(cfg.get("render_page_chars"), 2600)),
            max_pages=max(1, _to_int(cfg.get("render_max_pages"), 4)),
            retries=max(0, _to_int(cfg.get("render_retries"), 2)),
            poll_timeout_s=max(0.5, _to_float(cfg.get("render_poll_timeout_s"), 6.0)),
            poll_interval_ms=max(50, _to_int(cfg.get("render_poll_interval_ms"), 200)),
        )


@dataclass(frozen=True)
class ImageLayoutConfig:
    enabled: bool = IMAGE_LAYOUT_ENABLED
    provider_id: str = IMAGE_LAYOUT_PROVIDER_ID
    max_images_total: int = IMAGE_LAYOUT_MAX_IMAGES_TOTAL
    max_images_per_source: int = IMAGE_LAYOUT_MAX_IMAGES_PER_SOURCE
    pass_images_to_model: bool = IMAGE_LAYOUT_PASS_IMAGES_TO_MODEL
    max_images_to_model: int = IMAGE_LAYOUT_MAX_IMAGES_TO_MODEL
    preview_enabled: bool = False
    preview_max_images: int = 6
    preview_max_width: int = 1080
    preview_gap: int = 8
    request_max_requests: int = IMAGE_LAYOUT_REQUEST_MAX_REQUESTS
    request_max_images: int = IMAGE_LAYOUT_REQUEST_MAX_IMAGES
    generation_enabled: bool = IMAGE_LAYOUT_GENERATION_ENABLED
    generation_resolution: str = IMAGE_LAYOUT_GENERATION_RESOLUTION
    generation_aspect_ratio: str = IMAGE_LAYOUT_GENERATION_ASPECT_RATIO
    tool_enabled: bool = True
    tool_rounds: int = 2
    tool_max_steps: int = 25
    shuffle_candidates: bool = True
    shuffle_seed: str = ""

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> ImageLayoutConfig:
        return cls(
            enabled=_to_bool(cfg.get("image_layout_enabled"), IMAGE_LAYOUT_ENABLED),
            provider_id=_to_str(
                cfg.get("image_layout_provider_id"), IMAGE_LAYOUT_PROVIDER_ID
            ).strip(),
            max_images_total=IMAGE_LAYOUT_MAX_IMAGES_TOTAL,
            max_images_per_source=IMAGE_LAYOUT_MAX_IMAGES_PER_SOURCE,
            pass_images_to_model=IMAGE_LAYOUT_PASS_IMAGES_TO_MODEL,
            max_images_to_model=IMAGE_LAYOUT_MAX_IMAGES_TO_MODEL,
            preview_enabled=_to_bool(cfg.get("image_layout_preview_enabled"), False),
            preview_max_images=max(
                1, _to_int(cfg.get("image_layout_preview_max_images"), 6)
            ),
            preview_max_width=max(
                200, _to_int(cfg.get("image_layout_preview_max_width"), 1080)
            ),
            preview_gap=max(0, _to_int(cfg.get("image_layout_preview_gap"), 8)),
            request_max_requests=IMAGE_LAYOUT_REQUEST_MAX_REQUESTS,
            request_max_images=IMAGE_LAYOUT_REQUEST_MAX_IMAGES,
            generation_enabled=_to_bool(
                cfg.get("image_layout_generation_enabled"),
                IMAGE_LAYOUT_GENERATION_ENABLED,
            ),
            generation_resolution=_to_str(
                cfg.get("image_layout_generation_resolution"),
                IMAGE_LAYOUT_GENERATION_RESOLUTION,
            ).strip()
            or IMAGE_LAYOUT_GENERATION_RESOLUTION,
            generation_aspect_ratio=_to_str(
                cfg.get("image_layout_generation_aspect_ratio"),
                IMAGE_LAYOUT_GENERATION_ASPECT_RATIO,
            ).strip()
            or IMAGE_LAYOUT_GENERATION_ASPECT_RATIO,
            tool_enabled=_to_bool(cfg.get("image_layout_tool_enabled"), True),
            tool_rounds=max(1, _to_int(cfg.get("image_layout_tool_rounds"), 2)),
            tool_max_steps=max(5, _to_int(cfg.get("image_layout_tool_max_steps"), 25)),
            shuffle_candidates=_to_bool(
                cfg.get("image_layout_shuffle_candidates"), True
            ),
            shuffle_seed=_to_str(cfg.get("image_layout_shuffle_seed"), "").strip(),
        )


@dataclass(frozen=True)
class ImageLabelConfig:
    enabled: bool = IMAGE_LABEL_ENABLED
    provider_id: str = IMAGE_LABEL_PROVIDER_ID
    max_images_total: int = IMAGE_LABEL_MAX_IMAGES_TOTAL
    batch_size: int = 2
    concurrency: int = 4
    force_refresh: bool = False
    llm_max_retries: int = 2
    llm_retry_base_s: float = 1.0
    llm_retry_max_s: float = 12.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> ImageLabelConfig:
        return cls(
            enabled=_to_bool(cfg.get("image_label_enabled"), IMAGE_LABEL_ENABLED),
            provider_id=_to_str(
                cfg.get("image_label_provider_id"), IMAGE_LABEL_PROVIDER_ID
            ).strip(),
            max_images_total=IMAGE_LABEL_MAX_IMAGES_TOTAL,
            batch_size=max(1, min(2, _to_int(cfg.get("image_label_batch_size"), 2))),
            concurrency=max(1, _to_int(cfg.get("image_label_concurrency"), 4)),
            force_refresh=_to_bool(cfg.get("image_label_force_refresh"), False),
            llm_max_retries=max(0, _to_int(cfg.get("image_label_llm_max_retries"), 2)),
            llm_retry_base_s=max(
                0.1, _to_float(cfg.get("image_label_llm_retry_base_s"), 1.0)
            ),
            llm_retry_max_s=max(
                0.5, _to_float(cfg.get("image_label_llm_retry_max_s"), 12.0)
            ),
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
    def from_mapping(cls, cfg: Mapping[str, Any]) -> LayoutRefineConfig:
        return cls(
            enabled=_to_bool(cfg.get("image_layout_refine_enabled"), False),
            rounds=max(1, _to_int(cfg.get("image_layout_refine_rounds"), 2)),
            max_requests=max(
                0, _to_int(cfg.get("image_layout_refine_max_requests"), 2)
            ),
            request_max_images=max(
                1, _to_int(cfg.get("image_layout_refine_request_max_images"), 6)
            ),
            preview_page_chars=max(
                400, _to_int(cfg.get("image_layout_refine_preview_page_chars"), 2400)
            ),
            preview_pages=max(
                1, _to_int(cfg.get("image_layout_refine_preview_pages"), 1)
            ),
            preview_timeout_ms=max(
                1000, _to_int(cfg.get("image_layout_refine_preview_timeout_ms"), 20000)
            ),
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
    ) -> LayoutPrompts:
        return cls(
            image_layout_system=str(
                load_template(image_layout_system_path) or ""
            ).strip(),
            image_layout_tool_system=str(
                load_template(image_layout_tool_system_path) or ""
            ).strip(),
            layout_refiner_system=str(
                load_template(layout_refiner_system_path) or ""
            ).strip(),
            image_labeler_system=str(
                load_template(image_labeler_system_path) or ""
            ).strip(),
        )


@dataclass(frozen=True)
class NewsSourcesConfig:
    sources: list[NewsSourceConfig]

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> NewsSourcesConfig:
        raw = cfg.get("news_sources") or []
        raw_items = list(raw) if isinstance(raw, list) else []
        raw_items.extend(_build_source_preset_items(cfg))

        out: list[NewsSourceConfig] = []
        seen: set[tuple[str, str]] = set()
        legacy_template_keys = {
            "wechat",
            "miyoushe",
            "github",
            "twitter",
            "rss",
            "skland_official",
            "astrbook",
            "plugin_registry_official",
            "plugin_registry_custom",
            "xiuxiu_ai",
        }

        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            tkey = _to_str(item.get("__template_key"), "").strip().lower()
            generic_type = ""
            if tkey in GENERIC_SOURCE_TEMPLATE_KEYS:
                generic_type = _to_str(item.get("type"), "").strip().lower()
            elif tkey and tkey not in legacy_template_keys:
                generic_type = _to_str(item.get("type"), tkey).strip().lower()
            elif not tkey:
                generic_type = _to_str(item.get("type"), "").strip().lower()

            if generic_type:
                src = _build_generic_source_config(
                    item,
                    default_type=generic_type,
                    default_index=len(out) + 1,
                )
                if src is None:
                    continue
            elif tkey == "wechat":
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
                    meta=None,
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
                meta: dict[str, Any] = {}
                token = _to_str(item.get("token"), "").strip()
                if token:
                    meta["token"] = token
                if "since_hours" in item and item.get("since_hours") is not None:
                    meta["since_hours"] = max(1, _to_int(item.get("since_hours"), 30))
                if "max_releases" in item and item.get("max_releases") is not None:
                    meta["max_releases"] = max(0, _to_int(item.get("max_releases"), 3))
                if "max_commits" in item and item.get("max_commits") is not None:
                    meta["max_commits"] = max(0, _to_int(item.get("max_commits"), 6))
                if "max_prs" in item and item.get("max_prs") is not None:
                    meta["max_prs"] = max(0, _to_int(item.get("max_prs"), 6))
                src = NewsSourceConfig(
                    name=name or f"GitHub {repo}",
                    url=repo,
                    type="github",
                    priority=max(1, priority),
                    max_articles=1,
                    album_keyword=None,
                    meta=meta or None,
                )
            elif tkey == "twitter":
                url = _to_str(item.get("url"), "").strip()
                if not url:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 3)
                meta: dict[str, Any] = {}
                proxy = _to_str(item.get("proxy"), "").strip()
                if proxy:
                    meta["proxy"] = proxy
                src = NewsSourceConfig(
                    name=name or f"X/{len(out) + 1}",
                    url=url,
                    type="twitter",
                    priority=max(1, priority),
                    max_articles=max(1, max_articles),
                    album_keyword=None,
                    meta=meta or None,
                )
            elif tkey == "rss":
                url = _to_str(item.get("url"), "").strip()
                if not url:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 5)
                timeout_s = _to_int(item.get("timeout_s"), 20)
                include_content = _to_bool(item.get("include_content"), True)
                src = NewsSourceConfig(
                    name=name or f"RSS {len(out) + 1}",
                    url=url,
                    type="rss",
                    priority=max(1, priority),
                    max_articles=max(1, min(max_articles, 30)),
                    album_keyword=None,
                    meta={
                        "timeout_s": max(5, min(timeout_s, 60)),
                        "include_content": include_content,
                        "since_hours": max(1, _to_int(item.get("since_hours"), 0))
                        if item.get("since_hours") is not None
                        else 0,
                    },
                )
            elif tkey == "skland_official":
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 20)
                meta: dict[str, Any] = {}
                for key in ("d_id", "thumbcache", "date", "md_dir"):
                    value = _to_str(item.get(key), "").strip()
                    if value:
                        meta[key] = value
                selected_games: list[str] = []
                for value in _to_str_list(item.get("games")):
                    if value not in selected_games:
                        selected_games.append(value)
                for value in _to_str_list(item.get("games_custom")):
                    if value not in selected_games:
                        selected_games.append(value)
                if selected_games:
                    meta["games"] = ",".join(selected_games)
                if "page_size" in item:
                    meta["page_size"] = max(1, min(_to_int(item.get("page_size"), 5), 5))
                if "max_pages" in item:
                    meta["max_pages"] = max(1, min(_to_int(item.get("max_pages"), 10), 50))
                src = NewsSourceConfig(
                    name=name or "森空岛官方",
                    url="skland://official",
                    type="skland_official",
                    priority=max(1, priority),
                    max_articles=max(1, min(max_articles, 100)),
                    album_keyword=None,
                    meta=meta or None,
                )
            elif tkey == "astrbook":
                token = _to_str(item.get("token"), "").strip()
                if not token:
                    continue
                name = _to_str(item.get("name"), "").strip()
                priority = _to_int(item.get("priority"), 1)
                max_articles = _to_int(item.get("max_articles"), 10)
                category = _to_str(item.get("category"), "").strip().lower()
                meta: dict[str, Any] = {"token": token}
                if category:
                    meta["category"] = category
                src = NewsSourceConfig(
                    name=name or "AstrBook 论坛",
                    url=ASTRBOOK_API_BASE,
                    type="astrbook",
                    priority=max(1, priority),
                    max_articles=max(1, min(max_articles, 50)),
                    album_keyword=None,
                    meta=meta,
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
    Multi-agent mode is removed. Legacy `multi` values are normalized to `react`.
    """

    mode: str = "react"

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> NewsWorkflowModeConfig:
        raw = _to_str(cfg.get("news_workflow_mode"), "react").strip().lower()
        if raw in {"single", "single_agent", "single-agent"}:
            return cls(mode="single")
        return cls(mode="react")


@dataclass(frozen=True)
class SingleAgentConfig:
    provider_id: str = ""
    max_steps: int = 18
    min_chars: int = 700
    max_chars: int = 1500

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> SingleAgentConfig:
        min_chars = max(100, _to_int(cfg.get("single_agent_min_chars"), 700))
        max_chars = max(min_chars, _to_int(cfg.get("single_agent_max_chars"), 1500))
        return cls(
            provider_id=_to_str(cfg.get("single_agent_provider_id"), "").strip(),
            max_steps=max(5, _to_int(cfg.get("single_agent_max_steps"), 18)),
            min_chars=min_chars,
            max_chars=max_chars,
        )


@dataclass(frozen=True)
class ReactAgentConfig:
    provider_id: str = ""
    max_steps: int = 18
    max_tool_failures: int = 6
    max_no_progress_rounds: int = 3
    max_repeat_action: int = 2
    tool_call_timeout_s: int = 90
    enable_trace: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> ReactAgentConfig:
        return cls(
            provider_id=_to_str(cfg.get("react_agent_provider_id"), "").strip(),
            max_steps=max(3, _to_int(cfg.get("react_agent_max_steps"), 18)),
            max_tool_failures=max(
                1, _to_int(cfg.get("react_agent_max_tool_failures"), 6)
            ),
            max_no_progress_rounds=max(
                1, _to_int(cfg.get("react_agent_max_no_progress_rounds"), 3)
            ),
            max_repeat_action=max(
                1, _to_int(cfg.get("react_agent_max_repeat_action"), 2)
            ),
            tool_call_timeout_s=max(
                5, _to_int(cfg.get("react_agent_tool_call_timeout_s"), 90)
            ),
            enable_trace=_to_bool(cfg.get("react_agent_enable_trace"), True),
        )
