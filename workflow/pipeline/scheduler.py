import asyncio
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from astrbot.api.event import MessageChain
except Exception:  # pragma: no cover
    MessageChain = None  # type: ignore

from ..agents.sources.astrbook_agent import AstrBookSubAgent
from ..agents.sources.github_agent import GitHubSubAgent
from ..agents.sources.github_source import build_github_sources_from_config
from ..agents.sources.miyoushe_agent import MiyousheSubAgent
from ..agents.sources.plugin_registry_agent import PluginRegistrySubAgent
from ..agents.sources.rss_agent import RssSubAgent
from ..agents.sources.skland_official_agent import SklandOfficialSubAgent
from ..agents.sources.twitter_agent import TwitterSubAgent
from ..agents.sources.wechat_agent import WechatSubAgent
from ..agents.sources.xiuxiu_ai_agent import XiuxiuAISubAgent
from ..core.astrbook_client import ASTRBOOK_API_BASE, AstrBookClient
from ..core.config_models import (
    ImageLabelConfig,
    ImageLayoutConfig,
    LayoutRefineConfig,
    NewsSourcesConfig,
    NewsWorkflowModeConfig,
    RenderImageStyleConfig,
    RenderPipelineConfig,
    SingleAgentConfig,
)
from ..core.image_utils import get_plugin_data_dir
from ..core.models import NewsSourceConfig
from .render_pipeline import render_daily_news_pages, split_pages
from .rendering import load_template
from .workflow_manager import NewsWorkflowManager

try:
    from astrbot.core.message.components import Image as _ImageComponent
except Exception:  # pragma: no cover
    _ImageComponent = None  # type: ignore

try:
    from astrbot.core.message.components import Node as _NodeComponent
    from astrbot.core.message.components import Nodes as _NodesComponent
    from astrbot.core.message.components import Plain as _PlainComponent
except Exception:  # pragma: no cover
    _NodeComponent = None  # type: ignore
    _NodesComponent = None  # type: ignore
    _PlainComponent = None  # type: ignore

try:
    from astrbot.core import html_renderer as _astrbot_html_renderer
except Exception:  # pragma: no cover
    _astrbot_html_renderer = None  # type: ignore


_UMO_PATTERN = re.compile(
    r"(?P<umo>[A-Za-z0-9_-]+:[A-Za-z]+Message:[^\s\"'“”‘’「」<>]+)"
)


UMO_DEFAULT_PLATFORM_ID = "napcat"
UMO_DEFAULT_MESSAGE_TYPE = "GroupMessage"

WECHAT_SEED_PERSIST = True
WECHAT_CHASE_MAX_HOPS = 6

MIYOUSHE_HEADLESS = True
MIYOUSHE_SLEEP_BETWEEN_S = 0.6

LAYOUT_REFINE_PREVIEW_TIMEOUT_MS = 20000


_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\((https?://[^)]+)\)")
_H2_SECTION_RE = re.compile(r"^##\s*(.+?)\s*$")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\((https?://[^)\s]+)\)", flags=re.I)
_HTML_IMG_RE = re.compile(r"<img[^>]+src=[\"'](https?://[^\"'>\s]+)", flags=re.I)


def _strip_md_inline(s: str) -> str:
    t = str(s or "")
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_report_links(
    md: str, *, max_links: int = 80
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract markdown links from the report and group them by H2 section.
    Returns: {section_title: [(label, url), ...]}
    """
    out: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()

    current_section = "来源链接"
    for raw in (md or "").splitlines():
        line = (raw or "").strip()
        if not line:
            continue
        m_sec = _H2_SECTION_RE.match(line)
        if m_sec:
            current_section = m_sec.group(1).strip() or current_section
            continue

        matches = list(_MD_LINK_RE.finditer(line))
        if not matches:
            continue

        label = _strip_md_inline(line)
        label = re.sub(r"\(\s*查看来源\s*\)$", "", label).strip()
        if len(label) > 80:
            label = label[:77] + "..."

        for m in matches:
            url = str(m.group(2) or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.setdefault(current_section, []).append((label, url))
            if len(seen) >= max(1, int(max_links)):
                return out

    return out


def _build_forward_nodes_from_links(
    links_by_section: dict[str, list[tuple[str, str]]],
    *,
    title: str = "日报来源链接",
    max_node_chars: int = 650,
    max_nodes: int = 12,
) -> list[str]:
    """
    Build text chunks for merged-forward nodes.
    Each chunk is a plain text message. URLs are kept as-is for clickability.
    """
    lines: list[str] = [title, ""]
    for sec, rows in links_by_section.items():
        if not rows:
            continue
        lines.append(f"【{sec}】")
        for label, url in rows:
            if label and label != url:
                lines.append(f"- {label}")
                lines.append(f"  {url}")
            else:
                lines.append(f"- {url}")
        lines.append("")

    all_text = "\n".join(lines).strip()
    if not all_text:
        return []

    chunks: list[str] = []
    buf = ""
    for line in all_text.splitlines():
        cand = (buf + "\n" + line).strip("\n") if buf else line
        if len(cand) > max_node_chars and buf:
            chunks.append(buf.strip())
            buf = line
            if len(chunks) >= max_nodes:
                break
        else:
            buf = cand
    if buf and len(chunks) < max_nodes:
        chunks.append(buf.strip())
    return [c for c in chunks if c.strip()]


def _normalize_umo(
    raw: Any,
    *,
    default_platform_id: str = UMO_DEFAULT_PLATFORM_ID,
    default_message_type: str = UMO_DEFAULT_MESSAGE_TYPE,
) -> str | None:
    """
    Normalize an AstrBot session string to `platform:MessageType:session_id`.

    Accepts:
    - full UMO: `napcat:GroupMessage:1030223077`
    - shorthand: `1030223077` (expanded using defaults)
    - wrapped text: `UMO: 「napcat:GroupMessage:1030223077」`
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    s2 = s.strip(" \t\r\n\"'“”‘’「」[]()<>")

    parts = [p.strip() for p in s2.split(":")]
    if len(parts) == 3 and all(parts):
        return ":".join(parts)

    if s2.isdigit():
        if not default_platform_id or not default_message_type:
            return None
        return f"{default_platform_id}:{default_message_type}:{s2}"

    m = _UMO_PATTERN.search(s)
    if not m:
        return None

    candidate = m.group("umo").strip()
    parts = [p.strip() for p in candidate.split(":")]
    if len(parts) == 3 and all(parts):
        return ":".join(parts)
    return None


def _normalize_umo_list(
    raw_list: Any,
    *,
    default_platform_id: str = UMO_DEFAULT_PLATFORM_ID,
    default_message_type: str = UMO_DEFAULT_MESSAGE_TYPE,
) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    if not isinstance(raw_list, list):
        return normalized, invalid

    for raw in raw_list:
        umo = _normalize_umo(
            raw,
            default_platform_id=default_platform_id,
            default_message_type=default_message_type,
        )
        if not umo:
            if raw is not None and str(raw).strip():
                invalid.append(str(raw))
            continue
        if umo in seen:
            continue
        seen.add(umo)
        normalized.append(umo)

    return normalized, invalid


def _is_valid_image_file(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        if path.stat().st_size < 128:
            return False
        head = path.read_bytes()[:16]
        if head.startswith(b"\xff\xd8\xff"):  # JPEG
            return True
        if head.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return True
        if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
            return True
        return False
    except Exception:
        return False


def _select_render_template(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("render_template_name") or "daily_news").strip().lower()
    if name in {"chenyu", "chenyu_style", "chenyu-style"}:
        return load_template("templates/chenyu-style.html").strip()
    return load_template("templates/daily_news.html").strip()


class DailyNewsScheduler:
    """每日自动新闻：定时拉取 -> 多 Agent -> 汇总 -> 推送（配置来自 AstrBotConfig）"""

    def __init__(self, astrbot_context: Any, config: Any):
        self.context = astrbot_context
        self.config = config
        self.workflow_manager = NewsWorkflowManager()
        self.running = False
        self.task: asyncio.Task | None = None
        self._workflow_task: asyncio.Task | None = None
        self._did_migrate_news_sources = False
        self._report_cache_lock = asyncio.Lock()

        self._init_workflow_manager()

    def _init_workflow_manager(self):
        self.workflow_manager.register_sub_agent("wechat", WechatSubAgent)
        self.workflow_manager.register_sub_agent("miyoushe", MiyousheSubAgent)
        self.workflow_manager.register_sub_agent("github", GitHubSubAgent)
        self.workflow_manager.register_sub_agent("twitter", TwitterSubAgent)
        self.workflow_manager.register_sub_agent("rss", RssSubAgent)
        self.workflow_manager.register_sub_agent(
            "skland_official", SklandOfficialSubAgent
        )
        self.workflow_manager.register_sub_agent("astrbook", AstrBookSubAgent)
        self.workflow_manager.register_sub_agent(
            "plugin_registry", PluginRegistrySubAgent
        )
        self.workflow_manager.register_sub_agent("xiuxiu_ai", XiuxiuAISubAgent)

    def _save_config(self):
        if hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
            except Exception:
                astrbot_logger.error(
                    "[dailynews] config.save_config failed", exc_info=True
                )

    def _report_cache_path(self) -> Path:
        return get_plugin_data_dir("daily_report_cache") / "latest_report.json"

    def _report_cache_images_dir(self) -> Path:
        return get_plugin_data_dir("daily_report_cache") / "images"

    def _report_cache_ttl_minutes(self, config: dict[str, Any]) -> int:
        try:
            ttl = int(config.get("report_cache_ttl_minutes", 0) or 0)
        except Exception:
            ttl = 0
        return max(0, min(ttl, 24 * 60 * 7))

    def _report_cache_enabled(self, config: dict[str, Any]) -> bool:
        return bool(config.get("report_cache_enabled", False)) and bool(
            self._report_cache_ttl_minutes(config) > 0
        )

    def _raw_delivery_mode(self, config: dict[str, Any]) -> str:
        return (
            str(config.get("delivery_mode", "html_image") or "html_image")
            .strip()
            .lower()
        )

    def _send_links_forward(self, config: dict[str, Any]) -> bool:
        return self._raw_delivery_mode(config) == "html_image_with_links"

    def _actual_delivery_mode(self, config: dict[str, Any]) -> str:
        raw = self._raw_delivery_mode(config)
        if raw == "html_image_with_links":
            return "html_image"
        return raw

    def _parse_cached_datetime(self, value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def _valid_cached_img_paths(self, paths: Any) -> list[str]:
        if not isinstance(paths, list):
            return []
        out: list[str] = []
        for item in paths:
            try:
                resolved = Path(str(item or "").strip()).resolve()
            except Exception:
                continue
            if _is_valid_image_file(resolved):
                out.append(resolved.as_posix())
        return out

    def _load_report_cache_sync(self) -> dict[str, Any] | None:
        path = self._report_cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    async def _load_report_cache(self) -> dict[str, Any] | None:
        async with self._report_cache_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._load_report_cache_sync)

    def _store_report_cache_sync(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        cache_dir = get_plugin_data_dir("daily_report_cache")
        images_dir = self._report_cache_images_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        stored_img_paths: list[str] = []
        for idx, src in enumerate(payload.get("img_paths") or [], start=1):
            try:
                resolved = Path(str(src or "").strip()).resolve()
            except Exception:
                continue
            if not _is_valid_image_file(resolved):
                continue
            suffix = resolved.suffix if resolved.suffix else ".jpg"
            dst = images_dir / f"page_{idx}{suffix}"
            shutil.copy2(resolved, dst)
            stored_img_paths.append(dst.resolve().as_posix())

        ttl_minutes = self._report_cache_ttl_minutes(config)
        now = datetime.now()
        entry = {
            "content": str(payload.get("content") or ""),
            "img_paths": stored_img_paths,
            "link_node_chunks": [
                str(x)
                for x in (payload.get("link_node_chunks") or [])
                if str(x).strip()
            ],
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=ttl_minutes)).isoformat(),
        }
        path = self._report_cache_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)
        return entry

    async def _store_report_cache(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any] | None:
        async with self._report_cache_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._store_report_cache_sync(payload, config)
            )

    def _can_store_report_cache(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> bool:
        content = str(payload.get("content") or "").strip()
        if not content or content.startswith("\u751f\u6210\u5931\u8d25"):
            return False
        if self._actual_delivery_mode(config) == "html_image":
            return bool(self._valid_cached_img_paths(payload.get("img_paths") or []))
        return True

    def _cache_entry_usable(
        self, entry: dict[str, Any] | None, config: dict[str, Any]
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        content = str(entry.get("content") or "").strip()
        if not content:
            return False
        expires_at = self._parse_cached_datetime(entry.get("expires_at"))
        if expires_at is None or expires_at <= datetime.now():
            return False
        if self._actual_delivery_mode(config) == "html_image":
            return bool(self._valid_cached_img_paths(entry.get("img_paths") or []))
        return True

    def _build_link_node_chunks(self, content: str) -> list[str]:
        try:
            links_by_section = _extract_report_links(content, max_links=120)
            return _build_forward_nodes_from_links(
                links_by_section,
                title="\u65e5\u62a5\u6765\u6e90\u94fe\u63a5\uff08\u53ef\u70b9\u51fb\uff09",
                max_node_chars=650,
                max_nodes=14,
            )
        except Exception:
            return []

    async def _build_report_payload(
        self,
        content: str,
        config: dict[str, Any],
        *,
        cache_hit: bool,
    ) -> dict[str, Any]:
        payload = {
            "content": str(content or ""),
            "img_paths": [],
            "link_node_chunks": [],
            "cache_hit": bool(cache_hit),
        }
        if self._send_links_forward(config):
            payload["link_node_chunks"] = self._build_link_node_chunks(content)
        if self._actual_delivery_mode(config) == "html_image":
            payload["img_paths"] = self._valid_cached_img_paths(
                await self._render_content_images(content, config=config)
            )
        return payload

    async def prepare_report(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        source: str = "manual",
        prefer_cache: bool = True,
    ) -> dict[str, Any]:
        config = cfg or self._normalized_config()
        if prefer_cache and self._report_cache_enabled(config):
            cached = await self._load_report_cache()
            if self._cache_entry_usable(cached, config):
                content = str(cached.get("content") or "")
                payload = {
                    "content": content,
                    "img_paths": self._valid_cached_img_paths(
                        cached.get("img_paths") or []
                    ),
                    "link_node_chunks": [
                        str(x)
                        for x in (cached.get("link_node_chunks") or [])
                        if str(x).strip()
                    ],
                    "cache_hit": True,
                    "created_at": str(cached.get("created_at") or ""),
                    "expires_at": str(cached.get("expires_at") or ""),
                }
                if self._send_links_forward(config) and not payload["link_node_chunks"]:
                    payload["link_node_chunks"] = self._build_link_node_chunks(content)
                astrbot_logger.info(
                    "[dailynews] report cache hit source=%s expires_at=%s",
                    source,
                    payload.get("expires_at") or "",
                )
                return payload

        content = await self.generate_once(config, source=source)
        payload = await self._build_report_payload(content, config, cache_hit=False)
        if self._report_cache_enabled(config) and self._can_store_report_cache(
            payload, config
        ):
            stored = await self._store_report_cache(payload, config)
            if isinstance(stored, dict):
                payload["img_paths"] = self._valid_cached_img_paths(
                    stored.get("img_paths") or []
                )
                payload["created_at"] = str(stored.get("created_at") or "")
                payload["expires_at"] = str(stored.get("expires_at") or "")
        return payload

    def _split_sources(self, raw: Any) -> list[str]:
        """
        Config UI uses list items, but users often paste multiple URLs into one item.
        Split by whitespace/newlines and return a de-duplicated list.
        """
        urls: list[str] = []
        seen = set()

        if isinstance(raw, str):
            raw = [raw]

        if not isinstance(raw, list):
            return []

        for item in raw:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            for part in re.split(r"[\s\r\n]+", s):
                u = part.strip()
                if not u:
                    continue
                if u in seen:
                    continue
                seen.add(u)
                urls.append(u)
        return urls

    def get_config_snapshot(self) -> dict[str, Any]:
        return self._normalized_config()

    def _normalized_config(self) -> dict[str, Any]:
        self._maybe_migrate_legacy_news_sources()
        cfg: dict[str, Any] = dict(self.config)
        cfg.setdefault("enabled", False)
        cfg.setdefault("schedule_time", "09:00")
        cfg.setdefault("output_format", "markdown")
        cfg.setdefault("delivery_mode", "html_image")
        cfg.setdefault("report_cache_enabled", False)
        cfg.setdefault("report_cache_ttl_minutes", 60)
        cfg.setdefault("render_template_name", "daily_news")

        mode_cfg = NewsWorkflowModeConfig.from_mapping(cfg)
        cfg.setdefault("news_workflow_mode", mode_cfg.mode)
        single_cfg = SingleAgentConfig.from_mapping(cfg)
        cfg.setdefault("single_agent_provider_id", single_cfg.provider_id)
        cfg.setdefault("single_agent_max_steps", single_cfg.max_steps)
        cfg.setdefault("single_agent_min_chars", single_cfg.min_chars)
        cfg.setdefault("single_agent_max_chars", single_cfg.max_chars)
        cfg.setdefault("react_user_goal", "")
        cfg.setdefault("react_agent_provider_id", "")
        cfg.setdefault("react_agent_max_steps", 18)
        cfg.setdefault("react_agent_max_tool_failures", 6)
        cfg.setdefault("react_agent_max_no_progress_rounds", 3)
        cfg.setdefault("react_agent_max_repeat_action", 2)
        cfg.setdefault("react_agent_tool_call_timeout_s", 90)
        cfg.setdefault("react_agent_enable_trace", True)
        cfg.setdefault("react_vertical_llm_timeout_s", 45)
        cfg.setdefault("react_vertical_llm_max_retries", 0)
        cfg.setdefault("react_vertical_analyze_timeout_s", 60)
        cfg.setdefault("react_vertical_process_timeout_s", 120)
        cfg.setdefault("react_vertical_tool_timeout_floor_s", 200)
        cfg.setdefault("max_sources_per_day", 3)
        cfg.setdefault("news_group_mode", "source")
        cfg.setdefault("news_group_router_provider_id", "")
        cfg.setdefault("news_group_writeback_tags", True)
        cfg.setdefault("news_group_promote_min_count", 2)

        pipeline = RenderPipelineConfig.from_mapping(cfg)
        style = RenderImageStyleConfig.from_mapping(cfg)
        cfg.setdefault("render_page_chars", pipeline.page_chars)
        cfg.setdefault("render_max_pages", pipeline.max_pages)
        cfg.setdefault("render_retries", pipeline.retries)
        cfg.setdefault("render_poll_timeout_s", pipeline.poll_timeout_s)
        cfg.setdefault("render_poll_interval_ms", pipeline.poll_interval_ms)
        cfg.setdefault("render_img_float_enabled", style.float_enabled)
        cfg.setdefault("render_img_float_threshold", style.float_threshold)
        cfg.setdefault("render_img_full_max_width", style.full_max_width)
        cfg.setdefault("render_img_medium_max_width", style.medium_max_width)
        cfg.setdefault("render_img_narrow_max_width", style.narrow_max_width)

        cfg.setdefault("preferred_source_types", ["wechat"])
        cfg.setdefault("github_enabled", False)
        cfg.setdefault("github_repos", [])
        cfg.setdefault("github_token", "")
        cfg.setdefault("github_since_hours", 30)
        cfg.setdefault("github_max_releases", 3)
        cfg.setdefault("github_max_commits", 6)
        cfg.setdefault("github_max_prs", 6)
        cfg.setdefault("llm_timeout_s", 180)
        cfg.setdefault("llm_write_timeout_s", 360)
        cfg.setdefault("llm_merge_timeout_s", 240)
        cfg.setdefault("llm_max_retries", 1)
        cfg.setdefault("main_agent_provider_id", "")
        cfg.setdefault("main_agent_fallback_provider_ids", [])
        cfg.setdefault("main_agent_fallback_provider_id_1", "")
        cfg.setdefault("main_agent_fallback_provider_id_2", "")
        cfg.setdefault("main_agent_fallback_provider_id_3", "")
        cfg.setdefault("image_plan_enabled", True)

        cfg.setdefault("twitter_enabled", False)
        cfg.setdefault("twitter_targets", [])
        cfg.setdefault("twitter_proxy", "")
        cfg.setdefault("twitter_priority", 1)
        cfg.setdefault("twitter_max_tweets", 3)

        cfg.setdefault("news_sources", [])

        cfg.setdefault("astrbook_publish_enabled", False)
        cfg.setdefault("astrbook_token", "")
        cfg.setdefault("astrbook_publish_category", "tech")
        cfg.setdefault("astrbook_publish_title_prefix", "每日资讯日报")
        cfg.setdefault("astrbook_publish_append_image_links", True)
        cfg.setdefault("astrbook_publish_max_image_links", 40)

        layout = ImageLayoutConfig.from_mapping(cfg)
        cfg.setdefault("image_layout_enabled", layout.enabled)
        cfg.setdefault("image_layout_provider_id", layout.provider_id)
        cfg.setdefault("image_layout_max_images_total", layout.max_images_total)
        cfg.setdefault(
            "image_layout_max_images_per_source", layout.max_images_per_source
        )
        cfg.setdefault("image_layout_sources", [])
        cfg.setdefault("image_layout_pass_images_to_model", layout.pass_images_to_model)
        cfg.setdefault("image_layout_max_images_to_model", layout.max_images_to_model)
        cfg.setdefault("image_layout_preview_enabled", layout.preview_enabled)
        cfg.setdefault("image_layout_preview_max_images", layout.preview_max_images)
        cfg.setdefault("image_layout_preview_max_width", layout.preview_max_width)
        cfg.setdefault("image_layout_preview_gap", layout.preview_gap)
        cfg.setdefault("image_layout_shuffle_candidates", layout.shuffle_candidates)
        cfg.setdefault("image_layout_shuffle_seed", layout.shuffle_seed)

        label = ImageLabelConfig.from_mapping(cfg)
        cfg.setdefault("image_label_enabled", label.enabled)
        cfg.setdefault("image_label_provider_id", label.provider_id)

        refine = LayoutRefineConfig.from_mapping(cfg)
        cfg.setdefault("image_layout_refine_enabled", refine.enabled)
        cfg.setdefault("image_layout_refine_rounds", refine.rounds)
        cfg.setdefault("image_layout_refine_max_requests", refine.max_requests)
        cfg.setdefault(
            "image_layout_refine_request_max_images", refine.request_max_images
        )
        cfg.setdefault(
            "image_layout_refine_preview_page_chars", refine.preview_page_chars
        )
        cfg.setdefault("image_layout_refine_preview_pages", refine.preview_pages)
        # Hardcode these to keep UI/config simple.
        cfg["image_layout_refine_preview_timeout_ms"] = LAYOUT_REFINE_PREVIEW_TIMEOUT_MS

        # Hardcode these to keep UI/config simple.
        cfg["wechat_seed_persist"] = WECHAT_SEED_PERSIST
        cfg["wechat_chase_max_hops"] = WECHAT_CHASE_MAX_HOPS
        cfg["miyoushe_headless"] = MIYOUSHE_HEADLESS
        cfg["miyoushe_sleep_between_s"] = MIYOUSHE_SLEEP_BETWEEN_S
        cfg.setdefault("target_sessions", [])
        cfg.setdefault("admin_sessions", [])
        cfg.setdefault("last_run_date", "")
        cfg.setdefault("last_run_schedule_time", "")

        cfg.setdefault("wechat_sources", [])
        cfg.setdefault("miyoushe_sources", [])
        if not isinstance(cfg.get("news_sources"), list):
            cfg["news_sources"] = []
        if not isinstance(cfg.get("wechat_sources"), list):
            cfg["wechat_sources"] = []
        if not isinstance(cfg.get("miyoushe_sources"), list):
            cfg["miyoushe_sources"] = []
        if not isinstance(cfg.get("target_sessions"), list):
            cfg["target_sessions"] = []
        if not isinstance(cfg.get("admin_sessions"), list):
            cfg["admin_sessions"] = []
        if not isinstance(cfg.get("preferred_source_types"), list):
            cfg["preferred_source_types"] = ["wechat"]
        if not isinstance(cfg.get("image_layout_sources"), list):
            cfg["image_layout_sources"] = []
        if not isinstance(cfg.get("github_repos"), list):
            cfg["github_repos"] = []
        if not isinstance(cfg.get("twitter_targets"), list):
            cfg["twitter_targets"] = []
        if not isinstance(cfg.get("main_agent_fallback_provider_ids"), list):
            cfg["main_agent_fallback_provider_ids"] = []
        if str(cfg.get("news_group_mode") or "").strip().lower() not in {
            "source",
            "group",
        }:
            cfg["news_group_mode"] = "source"
        if not isinstance(cfg.get("news_group_router_provider_id"), str):
            cfg["news_group_router_provider_id"] = ""
        if not isinstance(cfg.get("news_group_writeback_tags"), bool):
            cfg["news_group_writeback_tags"] = bool(
                cfg.get("news_group_writeback_tags", True)
            )
        try:
            cfg["news_group_promote_min_count"] = int(
                cfg.get("news_group_promote_min_count") or 2
            )
        except Exception:
            cfg["news_group_promote_min_count"] = 2

        return cfg

    def _maybe_migrate_legacy_news_sources(self) -> None:
        if self._did_migrate_news_sources:
            return

        try:
            raw = self.config.get("news_sources", None)
        except Exception:
            self._did_migrate_news_sources = True
            return

        if isinstance(raw, list) and raw:
            changed = False
            normalized_items: list[Any] = []
            for item in raw:
                if not isinstance(item, dict):
                    normalized_items.append(item)
                    continue
                entry = dict(item)
                if entry.get("__template_key") == "skland_official":
                    games_value = entry.get("games", None)
                    if isinstance(games_value, str):
                        entry["games"] = [
                            part.strip()
                            for part in games_value.replace("\n", ",").split(",")
                            if part.strip()
                        ]
                        changed = True
                    elif games_value is None:
                        entry["games"] = []
                        changed = True
                normalized_items.append(entry)

            if changed:
                try:
                    self.config["news_sources"] = normalized_items
                    self._save_config()
                    astrbot_logger.info(
                        "[dailynews] normalized template_list source shapes (count=%s)",
                        len(normalized_items),
                    )
                except Exception:
                    pass

            self._did_migrate_news_sources = True
            return

        migrated: list[dict[str, Any]] = []

        try:
            wechat = self._split_sources(self.config.get("wechat_sources", []) or [])
            miyoushe = self._split_sources(
                self.config.get("miyoushe_sources", []) or []
            )
            github_repos = self.config.get("github_repos", []) or []
            twitter_targets = self.config.get("twitter_targets", []) or []
        except Exception:
            self._did_migrate_news_sources = True
            return

        for idx, url in enumerate(wechat, start=1):
            migrated.append(
                {
                    "__template_key": "wechat",
                    "name": f"公众号{idx}",
                    "url": url,
                    "priority": 1,
                    "max_articles": 3,
                    "album_keyword": "",
                }
            )

        for idx, url in enumerate(miyoushe, start=1):
            migrated.append(
                {
                    "__template_key": "miyoushe",
                    "name": f"米游社{idx}",
                    "url": url,
                    "priority": 1,
                    "max_articles": 3,
                }
            )

        if isinstance(github_repos, list):
            for r in github_repos:
                s = str(r or "").strip()
                if not s:
                    continue
                migrated.append(
                    {
                        "__template_key": "github",
                        "name": f"GitHub {s}",
                        "repo": s,
                        "priority": 1,
                    }
                )

        if isinstance(twitter_targets, list):
            try:
                prio = int(self.config.get("twitter_priority") or 1)
            except Exception:
                prio = 1
            try:
                max_tw = int(self.config.get("twitter_max_tweets") or 3)
            except Exception:
                max_tw = 3
            for idx, r in enumerate(twitter_targets, start=1):
                s = str(r or "").strip()
                if not s:
                    continue
                migrated.append(
                    {
                        "__template_key": "twitter",
                        "name": f"X/{idx}",
                        "url": s,
                        "priority": prio,
                        "max_articles": max_tw,
                    }
                )

        if migrated:
            try:
                self.config["news_sources"] = migrated
                self._save_config()
                astrbot_logger.info(
                    "[dailynews] migrated legacy sources -> news_sources (count=%s)",
                    len(migrated),
                )
            except Exception:
                pass

        self._did_migrate_news_sources = True

    async def start(self):
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())
        try:
            cfg = self._normalized_config()
            astrbot_logger.info(
                "[dailynews] scheduler started (enabled=%s, schedule_time=%s, last_run_date=%s)",
                bool(cfg.get("enabled", False)),
                cfg.get("schedule_time", ""),
                cfg.get("last_run_date", ""),
            )
        except Exception:
            astrbot_logger.info("[dailynews] scheduler started")

    async def stop(self):
        self.running = False
        await self.cancel_running_workflow()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
        astrbot_logger.info("[dailynews] scheduler stopped")

    async def cancel_running_workflow(self):
        t = self._workflow_task
        if t is None or t.done():
            self._workflow_task = None
            return
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._workflow_task = None

    async def _loop(self):
        while self.running:
            try:
                cfg = self._normalized_config()
                if not bool(cfg.get("enabled", False)):
                    await asyncio.sleep(10)
                    continue

                schedule_time_str = str(cfg.get("schedule_time", "09:00"))
                now = datetime.now()
                try:
                    schedule_time = datetime.strptime(schedule_time_str, "%H:%M").time()
                except Exception:
                    schedule_time = datetime.strptime("09:00", "%H:%M").time()

                today = now.strftime("%Y-%m-%d")
                last_run_date = str(self.config.get("last_run_date", "") or "")
                last_run_schedule_time = str(
                    self.config.get("last_run_schedule_time", "") or ""
                )

                # Run once per day per configured schedule_time.
                already_ran = (
                    last_run_date == today
                    and last_run_schedule_time == schedule_time_str
                )
                should_run = (not already_ran) and (now.time() >= schedule_time)

                if should_run:
                    publish_enabled = bool(cfg.get("astrbook_publish_enabled", False))
                    astrbook_token = str(cfg.get("astrbook_token") or "").strip()
                    if publish_enabled and not astrbook_token:
                        publish_enabled = False
                        astrbot_logger.warning(
                            "[dailynews] astrbook_publish_enabled=true but astrbook_token is empty; skip publishing"
                        )

                    raw_targets = list(cfg.get("target_sessions", []) or [])
                    targets, invalid = _normalize_umo_list(
                        raw_targets,
                        default_platform_id=UMO_DEFAULT_PLATFORM_ID,
                        default_message_type=UMO_DEFAULT_MESSAGE_TYPE,
                    )

                    if raw_targets and not targets and not publish_enabled:
                        astrbot_logger.warning(
                            "[dailynews] scheduled run skipped: all target_sessions invalid (example=%s). Expected `napcat:GroupMessage:1030223077` or shorthand `1030223077`.",
                            invalid[:3],
                        )
                        await asyncio.sleep(120)
                        continue

                    if not raw_targets and not publish_enabled:
                        astrbot_logger.info(
                            "[dailynews] scheduled run skipped: no target_sessions configured"
                        )
                        await asyncio.sleep(120)
                        continue

                    prepared = await self.prepare_report(
                        cfg, source="scheduled", prefer_cache=True
                    )
                    content = str(prepared.get("content") or "")
                    cache_hit = bool(prepared.get("cache_hit", False))
                    sent = 0
                    if targets:
                        sent = await self._send_to_targets(
                            content,
                            targets,
                            config=cfg,
                            prepared=prepared,
                        )

                    published = False
                    if cache_hit and publish_enabled:
                        astrbot_logger.info(
                            "[dailynews] skip astrbook publish on cached report"
                        )
                    elif publish_enabled and astrbook_token:
                        try:
                            res = await self.publish_report_to_astrbook(
                                content, config=cfg
                            )
                            published = bool(res.get("ok", False))
                        except Exception as e:
                            astrbot_logger.error(
                                "[dailynews] astrbook publish failed: %s",
                                e,
                                exc_info=True,
                            )
                            published = False

                    if sent > 0 or published:
                        self.config["last_run_date"] = today
                        self.config["last_run_schedule_time"] = schedule_time_str
                        self._save_config()
                    else:
                        astrbot_logger.warning(
                            "[dailynews] scheduled run generated content but delivered=0 (sent=%s published=%s); will retry later",
                            sent,
                            published,
                        )
                        await asyncio.sleep(180)
                        continue

                    await asyncio.sleep(65)
                    continue

                await asyncio.sleep(5)
            except Exception as e:
                astrbot_logger.error(
                    "[dailynews] scheduler loop error: %s", e, exc_info=True
                )
                await asyncio.sleep(30)

    async def update_workflow_sources_from_config(self, cfg: dict[str, Any]):
        self.workflow_manager.news_sources.clear()

        # Preferred (v4.10.4+): template_list-based sources.
        try:
            templated = NewsSourcesConfig.from_mapping(cfg).sources
        except Exception:
            templated = []
        if templated:
            validated: list[NewsSourceConfig] = []
            for s in templated:
                low = str(s.url or "").lower()
                if s.type == "wechat" and "mp.weixin.qq.com" not in low:
                    astrbot_logger.warning(
                        "[dailynews] news_sources(wechat) doesn't look like a wechat article url; skipping: %s",
                        s.url,
                    )
                    continue
                if s.type == "miyoushe" and "miyoushe.com" not in low:
                    astrbot_logger.warning(
                        "[dailynews] news_sources(miyoushe) doesn't look like a miyoushe post list url; skipping: %s",
                        s.url,
                    )
                    continue
                if s.type == "twitter" and (
                    not low.startswith(("http://", "https://"))
                    or ("x.com" not in low and "twitter.com" not in low)
                ):
                    astrbot_logger.warning(
                        "[dailynews] news_sources(twitter) invalid (%s). Expected https://x.com/<user> or https://twitter.com/<user>.",
                        s.url,
                    )
                    continue
                if s.type == "rss" and not low.startswith(("http://", "https://")):
                    astrbot_logger.warning(
                        "[dailynews] news_sources(rss) invalid (%s). Expected an http/https RSS or Atom feed URL.",
                        s.url,
                    )
                    continue
                if (
                    s.type == "skland_official"
                    and str(s.url or "").strip() != "skland://official"
                ):
                    astrbot_logger.warning(
                        "[dailynews] news_sources(skland_official) invalid (%s). Expected template-managed skland://official.",
                        s.url,
                    )
                    continue
                validated.append(s)

            for s in validated:
                self.workflow_manager.add_source(s)
            astrbot_logger.info(
                "[dailynews] loaded %s news_sources via template_list: %s",
                len(self.workflow_manager.news_sources),
                [s.name for s in self.workflow_manager.news_sources],
            )
            return

        # WeChat sources
        for idx, url in enumerate(
            self._split_sources(cfg.get("wechat_sources", [])), start=1
        ):
            if "mp.weixin.qq.com" not in url.lower():
                astrbot_logger.warning(
                    "[dailynews] wechat_sources[%s] doesn't look like a wechat article url; skipping: %s",
                    idx,
                    url,
                )
                continue
            self.workflow_manager.add_source(
                NewsSourceConfig(
                    name=f"公众号{idx}",
                    url=url,
                    type="wechat",
                    priority=1,
                    max_articles=3,
                    album_keyword=None,
                )
            )

        # MiYoShe sources
        for idx, url in enumerate(
            self._split_sources(cfg.get("miyoushe_sources", [])), start=1
        ):
            if "miyoushe.com" not in url.lower():
                astrbot_logger.warning(
                    "[dailynews] miyoushe_sources[%s] doesn't look like a miyoushe post list url; skipping: %s",
                    idx,
                    url,
                )
                continue
            self.workflow_manager.add_source(
                NewsSourceConfig(
                    name=f"米游社{idx}",
                    url=url,
                    type="miyoushe",
                    priority=1,
                    max_articles=3,
                    album_keyword=None,
                )
            )

        # GitHub repos live in a dedicated list, but we map each repo into a source for the workflow.
        for src in build_github_sources_from_config(cfg):
            self.workflow_manager.add_source(src)

        # Twitter/X targets live in a dedicated list, but we map each into a source for the workflow.
        if bool(cfg.get("twitter_enabled", False)):
            raw_targets = cfg.get("twitter_targets", []) or []
            if isinstance(raw_targets, list):
                for idx, u in enumerate(raw_targets, start=1):
                    if not isinstance(u, str):
                        continue
                    url = u.strip()
                    if not url:
                        continue
                    if url.lower().startswith(("socks5://", "socks5h://")):
                        astrbot_logger.warning(
                            "[dailynews] twitter_targets[%s] looks like a proxy (%s). You may have swapped twitter_targets and twitter_proxy.",
                            idx,
                            url,
                        )
                        continue
                    if not url.lower().startswith(("http://", "https://")) or (
                        "x.com" not in url.lower() and "twitter.com" not in url.lower()
                    ):
                        astrbot_logger.warning(
                            "[dailynews] twitter_targets[%s] invalid (%s). Expected https://x.com/<user> or https://twitter.com/<user>.",
                            idx,
                            url,
                        )
                        continue
                    self.workflow_manager.add_source(
                        NewsSourceConfig(
                            name=f"X/{idx}",
                            url=url,
                            type="twitter",
                            priority=int(cfg.get("twitter_priority") or 1),
                            max_articles=int(cfg.get("twitter_max_tweets") or 3),
                        )
                    )

        astrbot_logger.info(
            "[dailynews] loaded %s news_sources: %s",
            len(self.workflow_manager.news_sources),
            [s.name for s in self.workflow_manager.news_sources],
        )

    async def generate_once(
        self, cfg: dict[str, Any] | None = None, source: str = "manual"
    ) -> str:
        config = cfg or self._normalized_config()

        await self.update_workflow_sources_from_config(config)
        self._workflow_task = asyncio.create_task(
            self.workflow_manager.run_workflow(
                config, astrbot_context=self.context, source=source
            )
        )
        try:
            result = await self._workflow_task
        finally:
            self._workflow_task = None
        if result.get("status") == "success":
            return str(result.get("final_summary") or "")
        return f"生成失败：{result.get('error') or '未知错误'}"

    async def generate_and_send(self, cfg: dict[str, Any] | None = None) -> str:
        config = cfg or self._normalized_config()
        prepared = await self.prepare_report(
            config, source="scheduled", prefer_cache=True
        )
        content = str(prepared.get("content") or "")
        await self._send_to_targets(
            content,
            list(config.get("target_sessions", []) or []),
            config=config,
            prepared=prepared,
        )
        return content

    async def _render_content_images(
        self, content: str, config: dict[str, Any]
    ) -> list[str]:
        """
        用 AstrBot 的 html_render（底层 t2i 渲染服务）把日报渲染为本地图片文件路径列表。
        返回空列表表示渲染失败或内容为空。
        """
        s = (content or "").strip()
        if not s:
            return []

        try:
            from astrbot.core import html_renderer
        except Exception:
            astrbot_logger.error(
                "[dailynews] astrbot.core.html_renderer unavailable", exc_info=True
            )
            return []

        pipeline_cfg = RenderPipelineConfig.from_mapping(config)
        style_cfg = RenderImageStyleConfig.from_mapping(config)

        pages = split_pages(
            s,
            page_chars=pipeline_cfg.page_chars,
            max_pages=pipeline_cfg.max_pages,
        )
        if not pages:
            return []

        template_name = config.get("t2i_active_template") or getattr(
            self.context, "_config", {}
        ).get("t2i_active_template")

        async def _render_html(ctx: dict) -> Path | None:
            try:
                p = await html_renderer.render_custom_template(
                    _select_render_template(config),
                    ctx,
                    return_url=False,
                )
                return Path(str(p)).resolve()
            except Exception:
                astrbot_logger.error(
                    "[dailynews] render_custom_template failed", exc_info=True
                )
                return None

        async def _render_t2i(text: str) -> Path | None:
            try:
                if _astrbot_html_renderer is not None:
                    p = await _astrbot_html_renderer.render_t2i(
                        text,
                        use_network=False,
                        return_url=False,
                        template_name=template_name,
                    )
                else:
                    p = await html_renderer.render_t2i(
                        text,
                        use_network=False,
                        return_url=False,
                        template_name=template_name,
                    )
                return Path(str(p)).resolve()
            except Exception:
                return None

        rendered = await render_daily_news_pages(
            pages=pages,
            template_str=_select_render_template(config),
            render_html=_render_html,
            render_t2i=_render_t2i,
            pipeline=pipeline_cfg,
            style=style_cfg,
            chenyu_font_files=config.get("chenyu_font_files", []),
            title="每日资讯日报",
            subtitle_fmt="第 {idx}/{total} 页",
        )

        out: list[str] = []
        for r in rendered:
            if r.image_path is None or not _is_valid_image_file(
                Path(r.image_path).resolve()
            ):
                return []
            out.append(Path(r.image_path).resolve().as_posix())
        return out

    async def _send_to_targets(
        self,
        content: str,
        target_sessions: list[str],
        config: dict[str, Any],
        prepared: dict[str, Any] | None = None,
    ) -> int:
        normalized_targets, invalid_targets = _normalize_umo_list(
            target_sessions,
            default_platform_id=UMO_DEFAULT_PLATFORM_ID,
            default_message_type=UMO_DEFAULT_MESSAGE_TYPE,
        )
        if invalid_targets:
            astrbot_logger.warning(
                "[dailynews] invalid target_sessions entries skipped (example=%s). Expected `napcat:GroupMessage:1030223077` or shorthand `1030223077`.",
                invalid_targets[:3],
            )
        target_sessions = normalized_targets

        if not target_sessions:
            astrbot_logger.info(
                "[dailynews] no target_sessions configured; skip sending"
            )
            return 0

        if MessageChain is None:
            astrbot_logger.warning("[dailynews] MessageChain unavailable; skip sending")
            return 0

        delivery_mode = self._actual_delivery_mode(config)
        send_links_forward = self._send_links_forward(config)

        link_node_chunks: list[str] = []
        if isinstance(prepared, dict):
            link_node_chunks = [
                str(x)
                for x in (prepared.get("link_node_chunks") or [])
                if str(x).strip()
            ]
        if send_links_forward and not link_node_chunks:
            link_node_chunks = self._build_link_node_chunks(content)

        img_paths: list[str] = []
        if delivery_mode == "html_image":
            if isinstance(prepared, dict):
                img_paths = self._valid_cached_img_paths(
                    prepared.get("img_paths") or []
                )
                if img_paths:
                    astrbot_logger.info(
                        "[dailynews] reuse cached report images: count=%s",
                        len(img_paths),
                    )
            if not img_paths:
                img_paths = await self._render_content_images(content, config=config)
            if not img_paths:
                astrbot_logger.warning(
                    "[dailynews] html_image enabled but render returned empty; fallback to text"
                )
                delivery_mode = "plain"

        sent_sessions = 0
        for umo in target_sessions:
            sent_this = False
            try:
                if delivery_mode == "html_image":
                    for img_path in img_paths:
                        # 通过本地文件路径发送，避免变成“链接卡片”（并兼容 Napcat 对路径格式的要求）
                        p = Path(str(img_path)).resolve().as_posix()
                        if _ImageComponent is not None:
                            chain = MessageChain()
                            chain.chain.append(
                                _ImageComponent(file=f"file:///{p}", path=p)
                            )
                        else:
                            chain = MessageChain().file_image(p)
                        try:
                            await self.context.send_message(umo, chain)
                            sent_this = True
                        except Exception:
                            continue
                    if sent_this and send_links_forward and link_node_chunks:
                        try:
                            if (
                                _NodesComponent is not None
                                and _NodeComponent is not None
                                and _PlainComponent is not None
                            ):
                                nodes = [
                                    _NodeComponent(
                                        uin="0",
                                        name="每日资讯日报",
                                        content=[_PlainComponent(chunk)],
                                    )
                                    for chunk in link_node_chunks
                                ]
                                chain = MessageChain()
                                chain.chain.append(_NodesComponent(nodes))
                                await self.context.send_message(umo, chain)
                            else:
                                chain = MessageChain().message(
                                    "\n\n".join(link_node_chunks)
                                )
                                await self.context.send_message(umo, chain)
                        except Exception:
                            # Fallback to plain text if merged-forward fails on this platform.
                            try:
                                chain = MessageChain().message(
                                    "\n\n".join(link_node_chunks)
                                )
                                await self.context.send_message(umo, chain)
                            except Exception:
                                pass
                    if not sent_this:
                        # Fallback: if sending images fails (platform timeout / upload issues), try plain text.
                        try:
                            chain = MessageChain().message(content)
                            await self.context.send_message(umo, chain)
                            sent_this = True
                        except Exception:
                            sent_this = False
                else:
                    chain = MessageChain().message(content)
                    await self.context.send_message(umo, chain)
                    sent_this = True
            except Exception as e:
                astrbot_logger.error(
                    "[dailynews] send_message failed: %s", e, exc_info=True
                )

            if sent_this:
                sent_sessions += 1

        return sent_sessions

    def _extract_image_urls_from_markdown(
        self, md: str, *, max_items: int = 60
    ) -> list[str]:
        s = str(md or "")
        out: list[str] = []
        seen: set[str] = set()
        for m in _MD_IMAGE_RE.finditer(s):
            u = str(m.group(1) or "").strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
                if len(out) >= max_items:
                    return out
        for m in _HTML_IMG_RE.finditer(s):
            u = str(m.group(1) or "").strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
                if len(out) >= max_items:
                    return out
        return out

    async def publish_report_to_astrbook(
        self, content: str, *, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cfg = config or self._normalized_config()
        if not bool(cfg.get("astrbook_publish_enabled", False)):
            astrbot_logger.info("[dailynews] astrbook publish skipped: disabled")
            return {"ok": False, "skipped": True, "reason": "disabled"}

        s = (content or "").strip()
        if not s:
            astrbot_logger.warning(
                "[dailynews] astrbook publish skipped: empty content"
            )
            return {"ok": False, "error": "empty_content"}
        if s.startswith("生成失败"):
            astrbot_logger.warning(
                "[dailynews] astrbook publish skipped: generate_failed"
            )
            return {"ok": False, "error": "generate_failed"}

        token = str(cfg.get("astrbook_token") or "").strip()
        if not token:
            astrbot_logger.warning(
                "[dailynews] astrbook publish skipped: token missing"
            )
            return {"ok": False, "error": "token_missing"}

        category = str(cfg.get("astrbook_publish_category") or "tech").strip().lower()
        if category not in {"chat", "deals", "misc", "tech", "help", "intro", "acg"}:
            category = "tech"

        title_prefix = str(
            cfg.get("astrbook_publish_title_prefix") or "每日资讯日报"
        ).strip()
        today = datetime.now().strftime("%Y-%m-%d")
        title = f"{title_prefix} {today}".strip()

        post_md = s
        if bool(cfg.get("astrbook_publish_append_image_links", True)):
            max_links = int(cfg.get("astrbook_publish_max_image_links") or 40)
            urls = self._extract_image_urls_from_markdown(
                post_md, max_items=max(1, max_links)
            )
            if urls:
                lines = ["", "", "---", "", "## 图片直链", ""]
                lines.extend([f"- {u}" for u in urls[:max_links]])
                post_md = (post_md.rstrip() + "\n" + "\n".join(lines)).strip()

        client = AstrBookClient(token=token)
        resp = await client.create_thread(
            title=title, content=post_md, category=category
        )
        if resp.get("error"):
            astrbot_logger.warning(
                "[dailynews] astrbook create_thread failed: %s (%s)",
                resp.get("error"),
                resp.get("text") or resp.get("detail") or "",
            )
            return {
                "ok": False,
                "error": resp.get("error"),
                "detail": resp.get("text") or resp.get("detail") or "",
            }

        tid = resp.get("id")
        astrbot_logger.info(
            "[dailynews] published report to astrbook: id=%s base=%s category=%s",
            tid,
            ASTRBOOK_API_BASE,
            category,
        )
        return {
            "ok": True,
            "id": tid,
            "api_url": f"{ASTRBOOK_API_BASE.rstrip('/')}/api/threads/{tid}"
            if tid
            else "",
        }

    async def notify_admin(self, text: str):
        cfg = self._normalized_config()
        admins = list(cfg.get("admin_sessions", []) or [])
        await self._send_to_targets(f"dailynews error\n\n{text}", admins, config=cfg)
