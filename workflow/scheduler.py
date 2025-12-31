import asyncio
import base64
import json
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
from .rendering import load_template
from .config_models import (
    ImageLayoutConfig,
    LayoutRefineConfig,
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


DAILY_NEWS_HTML_TMPL = load_template("templates/daily_news.html").strip()


class DailyNewsScheduler:
    """每日自动新闻：定时拉取 -> 多 Agent -> 汇总 -> 推送（配置来自 AstrBotConfig）"""

    def __init__(self, astrbot_context: Any, config: Any):
        self.context = astrbot_context
        self.config = config
        self.workflow_manager = NewsWorkflowManager()
        self.running = False
        self.task: Optional[asyncio.Task] = None

        self._init_workflow_manager()

    def _init_workflow_manager(self):
        self.workflow_manager.register_sub_agent("wechat", WechatSubAgent)
        self.workflow_manager.register_sub_agent("miyoushe", MiyousheSubAgent)
        self.workflow_manager.register_sub_agent("github", GitHubSubAgent)

    def _save_config(self):
        if hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
            except Exception:
                astrbot_logger.error("[dailynews] config.save_config failed", exc_info=True)

    def get_news_sources(self) -> List[Dict[str, Any]]:
        raw = self.config.get("news_sources", [])
        if isinstance(raw, list):
            out: List[Dict[str, Any]] = []
            for x in raw:
                if isinstance(x, dict):
                    out.append(x)
                    continue
                if isinstance(x, str):
                    s = x.strip()
                    if not s:
                        continue
                    # 兼容：列表元素也允许填 JSON 对象字符串
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            obj = json.loads(s)
                            if isinstance(obj, dict):
                                out.append(obj)
                                continue
                        except Exception:
                            pass
                    out.append({"url": s})
            return out
        if not isinstance(raw, str):
            return []
        try:
            data = json.loads(raw or "[]")
            if isinstance(data, list):
                out: List[Dict[str, Any]] = []
                for x in data:
                    if isinstance(x, dict):
                        out.append(x)
                    elif isinstance(x, str) and x.strip():
                        out.append({"url": x.strip()})
                return out
        except Exception:
            pass
        return []

    def get_config_snapshot(self) -> Dict[str, Any]:
        return self._normalized_config()

    def _normalized_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = dict(self.config)
        cfg.setdefault("enabled", False)
        cfg.setdefault("schedule_time", "09:00")
        cfg.setdefault("output_format", "markdown")
        cfg.setdefault("delivery_mode", "html_image")
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
        cfg.setdefault("image_plan_enabled", True)

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
        cfg.setdefault("image_layout_refine_preview_timeout_ms", refine.preview_timeout_ms)

        cfg.setdefault("wechat_seed_persist", True)
        cfg.setdefault("wechat_chase_max_hops", 6)
        cfg.setdefault("miyoushe_headless", True)
        cfg.setdefault("miyoushe_sleep_between_s", 0.6)
        cfg.setdefault("target_sessions", [])
        cfg.setdefault("admin_sessions", [])
        cfg.setdefault("last_run_date", "")

        cfg["news_sources"] = self.get_news_sources()
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

        return cfg

    async def start(self):
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())
        astrbot_logger.info("[dailynews] scheduler started")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
        astrbot_logger.info("[dailynews] scheduler stopped")

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

                last_run = str(self.config.get("last_run_date", "") or "")
                due = now.time().hour == schedule_time.hour and now.time().minute == schedule_time.minute

                if due and last_run != now.strftime("%Y-%m-%d"):
                    await self.generate_and_send(cfg)
                    self.config["last_run_date"] = now.strftime("%Y-%m-%d")
                    self._save_config()
                    await asyncio.sleep(65)
                    continue

                await asyncio.sleep(5)
            except Exception as e:
                astrbot_logger.error("[dailynews] scheduler loop error: %s", e, exc_info=True)
                await asyncio.sleep(30)

    async def update_workflow_sources_from_config(self, cfg: Dict[str, Any]):
        self.workflow_manager.news_sources.clear()
        for idx, source_data in enumerate(cfg.get("news_sources", []), start=1):
            # 支持两种写法：
            # 1) list[str]：每项是公众号文章 URL
            # 2) list[dict]：带 name/type/priority/max_articles/album_keyword 等字段
            if isinstance(source_data, str):
                source_data = {"url": source_data}
            if not isinstance(source_data, dict):
                continue

            url = str(source_data.get("url") or "").strip()
            if not url:
                continue
            source_type = str(source_data.get("type") or "").strip()
            if not source_type:
                u = url.lower()
                if "miyoushe.com" in u:
                    source_type = "miyoushe"
                elif "mp.weixin.qq.com" in u:
                    source_type = "wechat"
                else:
                    source_type = "wechat"
            source = NewsSourceConfig(
                name=str(source_data.get("name") or f"来源{idx}"),
                url=url,
                type=source_type,
                priority=int(source_data.get("priority") or 1),
                max_articles=int(source_data.get("max_articles") or 3),
                album_keyword=(
                    str(source_data.get("album_keyword")).strip()
                    if source_data.get("album_keyword")
                    else None
                ),
            )
            self.workflow_manager.add_source(source)

        # GitHub repos live in a dedicated list, but we map each repo into a source for the workflow.
        for src in build_github_sources_from_config(cfg):
            self.workflow_manager.add_source(src)

        astrbot_logger.info(
            "[dailynews] loaded %s news_sources: %s",
            len(self.workflow_manager.news_sources),
            [s.name for s in self.workflow_manager.news_sources],
        )

    async def generate_once(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        config = cfg or self._normalized_config()
        await self.update_workflow_sources_from_config(config)
        result = await self.workflow_manager.run_workflow(config, astrbot_context=self.context)
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
                    DAILY_NEWS_HTML_TMPL,
                    ctx,
                    return_url=False,
                )
                return Path(str(p)).resolve()
            except Exception:
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
            template_str=DAILY_NEWS_HTML_TMPL,
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

    async def _send_to_targets(self, content: str, target_sessions: List[str], config: Dict[str, Any]):
        if not target_sessions:
            astrbot_logger.info("[dailynews] no target_sessions configured; skip sending")
            return

        if MessageChain is None:
            astrbot_logger.warning("[dailynews] MessageChain unavailable; skip sending")
            return

        delivery_mode = str(config.get("delivery_mode", "html_image") or "html_image")
        img_paths: List[str] = []
        if delivery_mode == "html_image":
            img_paths = await self._render_content_images(content, config=config)
            if not img_paths:
                astrbot_logger.warning("[dailynews] html_image enabled but render returned empty; fallback to text")
                delivery_mode = "plain"

        for umo in target_sessions:
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
                else:
                    chain = MessageChain().message(content)
                    await self.context.send_message(umo, chain)
            except Exception as e:
                astrbot_logger.error("[dailynews] send_message failed: %s", e, exc_info=True)

    async def notify_admin(self, text: str):
        cfg = self._normalized_config()
        admins = list(cfg.get("admin_sessions", []) or [])
        await self._send_to_targets(f"dailynews error\n\n{text}", admins)
