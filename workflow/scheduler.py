import asyncio
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from astrbot.api.event import MessageChain
except Exception:  # pragma: no cover
    MessageChain = None  # type: ignore

from .agents import MiyousheSubAgent, NewsSourceConfig, NewsWorkflowManager, WechatSubAgent
from .github_agent import GitHubSubAgent
from .github_source import build_github_sources_from_config
from .plugin_registry_agent import PluginRegistrySubAgent
from .twitter_agent import TwitterSubAgent
from .rendering import load_template
from .config_models import (
    ImageLayoutConfig,
    LayoutRefineConfig,
    NewsSourcesConfig,
    RenderImageStyleConfig,
    RenderPipelineConfig,
)
from .render_pipeline import render_daily_news_pages, split_pages

try:
    from astrbot.core.message.components import Image as _ImageComponent
except Exception:  # pragma: no cover
    _ImageComponent = None  # type: ignore

try:
    from astrbot.core import html_renderer as _astrbot_html_renderer
except Exception:  # pragma: no cover
    _astrbot_html_renderer = None  # type: ignore


_UMO_PATTERN = re.compile(r"(?P<umo>[A-Za-z0-9_-]+:[A-Za-z]+Message:[^\s\"'“”‘’「」<>]+)")


UMO_DEFAULT_PLATFORM_ID = "napcat"
UMO_DEFAULT_MESSAGE_TYPE = "GroupMessage"

WECHAT_SEED_PERSIST = True
WECHAT_CHASE_MAX_HOPS = 6

MIYOUSHE_HEADLESS = True
MIYOUSHE_SLEEP_BETWEEN_S = 0.6

LAYOUT_REFINE_PREVIEW_TIMEOUT_MS = 20000


def _normalize_umo(
    raw: Any,
    *,
    default_platform_id: str = UMO_DEFAULT_PLATFORM_ID,
    default_message_type: str = UMO_DEFAULT_MESSAGE_TYPE,
) -> Optional[str]:
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
        if head.startswith(b"\xFF\xD8\xFF"):  # JPEG
            return True
        if head.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return True
        if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
            return True
        return False
    except Exception:
        return False


def _select_render_template(cfg: Dict[str, Any]) -> str:
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
        self.task: Optional[asyncio.Task] = None
        self._workflow_task: Optional[asyncio.Task] = None
        self._did_migrate_news_sources = False

        self._init_workflow_manager()

    def _init_workflow_manager(self):
        self.workflow_manager.register_sub_agent("wechat", WechatSubAgent)
        self.workflow_manager.register_sub_agent("miyoushe", MiyousheSubAgent)
        self.workflow_manager.register_sub_agent("github", GitHubSubAgent)
        self.workflow_manager.register_sub_agent("twitter", TwitterSubAgent)
        self.workflow_manager.register_sub_agent("plugin_registry", PluginRegistrySubAgent)

    def _save_config(self):
        if hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
            except Exception:
                astrbot_logger.error("[dailynews] config.save_config failed", exc_info=True)

    def _split_sources(self, raw: Any) -> List[str]:
        """
        Config UI uses list items, but users often paste multiple URLs into one item.
        Split by whitespace/newlines and return a de-duplicated list.
        """
        urls: List[str] = []
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

    def get_config_snapshot(self) -> Dict[str, Any]:
        return self._normalized_config()

    def _normalized_config(self) -> Dict[str, Any]:
        self._maybe_migrate_legacy_news_sources()
        cfg: Dict[str, Any] = dict(self.config)
        cfg.setdefault("enabled", False)
        cfg.setdefault("schedule_time", "09:00")
        cfg.setdefault("output_format", "markdown")
        cfg.setdefault("delivery_mode", "html_image")
        cfg.setdefault("render_template_name", "daily_news")
        cfg.setdefault("max_sources_per_day", 3)

        pipeline = RenderPipelineConfig.from_mapping(cfg)
        style = RenderImageStyleConfig.from_mapping(cfg)
        cfg.setdefault("render_page_chars", pipeline.page_chars)
        cfg.setdefault("render_max_pages", pipeline.max_pages)
        cfg.setdefault("render_retries", pipeline.retries)
        cfg.setdefault("render_poll_timeout_s", pipeline.poll_timeout_s)
        cfg.setdefault("render_poll_interval_ms", pipeline.poll_interval_ms)
        cfg.setdefault("render_playwright_fallback", pipeline.playwright_fallback)
        cfg.setdefault("render_playwright_timeout_ms", pipeline.playwright_timeout_ms)
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

        layout = ImageLayoutConfig.from_mapping(cfg)
        cfg.setdefault("image_layout_enabled", layout.enabled)
        cfg.setdefault("image_layout_provider_id", layout.provider_id)
        cfg.setdefault("image_layout_max_images_total", layout.max_images_total)
        cfg.setdefault("image_layout_max_images_per_source", layout.max_images_per_source)
        cfg.setdefault("image_layout_sources", [])
        cfg.setdefault("image_layout_pass_images_to_model", layout.pass_images_to_model)
        cfg.setdefault("image_layout_max_images_to_model", layout.max_images_to_model)
        cfg.setdefault("image_layout_preview_enabled", layout.preview_enabled)
        cfg.setdefault("image_layout_preview_max_images", layout.preview_max_images)
        cfg.setdefault("image_layout_preview_max_width", layout.preview_max_width)
        cfg.setdefault("image_layout_preview_gap", layout.preview_gap)
        cfg.setdefault("image_layout_shuffle_candidates", layout.shuffle_candidates)
        cfg.setdefault("image_layout_shuffle_seed", layout.shuffle_seed)

        refine = LayoutRefineConfig.from_mapping(cfg)
        cfg.setdefault("image_layout_refine_enabled", refine.enabled)
        cfg.setdefault("image_layout_refine_rounds", refine.rounds)
        cfg.setdefault("image_layout_refine_max_requests", refine.max_requests)
        cfg.setdefault("image_layout_refine_request_max_images", refine.request_max_images)
        cfg.setdefault("image_layout_refine_preview_page_chars", refine.preview_page_chars)
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
            self._did_migrate_news_sources = True
            return

        migrated: list[dict[str, Any]] = []

        try:
            wechat = self._split_sources(self.config.get("wechat_sources", []) or [])
            miyoushe = self._split_sources(self.config.get("miyoushe_sources", []) or [])
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
                astrbot_logger.info("[dailynews] migrated legacy sources -> news_sources (count=%s)", len(migrated))
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
                last_run_schedule_time = str(self.config.get("last_run_schedule_time", "") or "")

                # Run once per day per configured schedule_time.
                already_ran = last_run_date == today and last_run_schedule_time == schedule_time_str
                should_run = (not already_ran) and (now.time() >= schedule_time)

                if should_run:
                    raw_targets = list(cfg.get("target_sessions", []) or [])
                    targets, invalid = _normalize_umo_list(
                        raw_targets,
                        default_platform_id=UMO_DEFAULT_PLATFORM_ID,
                        default_message_type=UMO_DEFAULT_MESSAGE_TYPE,
                    )

                    if raw_targets and not targets:
                        astrbot_logger.warning(
                            "[dailynews] scheduled run skipped: all target_sessions invalid (example=%s). Expected `napcat:GroupMessage:1030223077` or shorthand `1030223077`.",
                            invalid[:3],
                        )
                        await asyncio.sleep(120)
                        continue

                    if not raw_targets:
                        astrbot_logger.info("[dailynews] scheduled run skipped: no target_sessions configured")
                        await asyncio.sleep(120)
                        continue

                    content = await self.generate_once(cfg, source="scheduled")
                    sent = await self._send_to_targets(content, targets, config=cfg)

                    if sent > 0:
                        self.config["last_run_date"] = today
                        self.config["last_run_schedule_time"] = schedule_time_str
                        self._save_config()
                    else:
                        astrbot_logger.warning(
                            "[dailynews] scheduled run generated content but sent=0; will retry later (check platform_id/message_type/targets)"
                        )
                        await asyncio.sleep(180)
                        continue

                    await asyncio.sleep(65)
                    continue

                await asyncio.sleep(5)
            except Exception as e:
                astrbot_logger.error("[dailynews] scheduler loop error: %s", e, exc_info=True)
                await asyncio.sleep(30)

    async def update_workflow_sources_from_config(self, cfg: Dict[str, Any]):
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
        for idx, url in enumerate(self._split_sources(cfg.get("wechat_sources", [])), start=1):
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
        for idx, url in enumerate(self._split_sources(cfg.get("miyoushe_sources", [])), start=1):
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

    async def generate_once(self, cfg: Optional[Dict[str, Any]] = None, source: str = "manual") -> str:
        config = cfg or self._normalized_config()
        await self.update_workflow_sources_from_config(config)
        self._workflow_task = asyncio.create_task(
            self.workflow_manager.run_workflow(config, astrbot_context=self.context, source=source)
        )
        try:
            result = await self._workflow_task
        finally:
            self._workflow_task = None
        if result.get("status") == "success":
            return str(result.get("final_summary") or "")
        return f"生成失败：{result.get('error') or '未知错误'}"

    async def generate_and_send(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        config = cfg or self._normalized_config()
        content = await self.generate_once(config)
        await self._send_to_targets(content, list(config.get("target_sessions", []) or []), config=config)
        return content

    async def _render_content_images(self, content: str, config: Dict[str, Any]) -> List[str]:
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
            astrbot_logger.error("[dailynews] astrbot.core.html_renderer unavailable", exc_info=True)
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

        template_name = (
            config.get("t2i_active_template")
            or getattr(self.context, "_config", {}).get("t2i_active_template")
        )

        async def _render_html(ctx: dict) -> Path | None:
            try:
                p = await html_renderer.render_custom_template(
                    _select_render_template(config),
                    ctx,
                    return_url=False,
                )
                return Path(str(p)).resolve()
            except Exception:
                astrbot_logger.error("[dailynews] render_custom_template failed", exc_info=True)
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
            title="每日资讯日报",
            subtitle_fmt="第 {idx}/{total} 页",
        )

        out: List[str] = []
        for r in rendered:
            if r.image_path is None or not _is_valid_image_file(Path(r.image_path).resolve()):
                return []
            out.append(Path(r.image_path).resolve().as_posix())
        return out

    async def _send_to_targets(self, content: str, target_sessions: List[str], config: Dict[str, Any]) -> int:
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
            astrbot_logger.info("[dailynews] no target_sessions configured; skip sending")
            return 0

        if MessageChain is None:
            astrbot_logger.warning("[dailynews] MessageChain unavailable; skip sending")
            return 0

        delivery_mode = str(config.get("delivery_mode", "html_image") or "html_image")
        img_paths: List[str] = []
        if delivery_mode == "html_image":
            img_paths = await self._render_content_images(content, config=config)
            if not img_paths:
                astrbot_logger.warning("[dailynews] html_image enabled but render returned empty; fallback to text")
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
                            chain.chain.append(_ImageComponent(file=f"file:///{p}", path=p))
                        else:
                            chain = MessageChain().file_image(p)
                        await self.context.send_message(umo, chain)
                        sent_this = True
                else:
                    chain = MessageChain().message(content)
                    await self.context.send_message(umo, chain)
                    sent_this = True
            except Exception as e:
                astrbot_logger.error("[dailynews] send_message failed: %s", e, exc_info=True)

            if sent_this:
                sent_sessions += 1

        return sent_sessions

    async def notify_admin(self, text: str):
        cfg = self._normalized_config()
        admins = list(cfg.get("admin_sessions", []) or [])
        await self._send_to_targets(f"dailynews error\n\n{text}", admins, config=cfg)
