import asyncio
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api import logger as astrbot_logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .tools import (
    ImageUrlDownloadTool,
    ImageUrlsPreviewTool,
    MarkdownDocApplyEditsTool,
    MarkdownDocCreateTool,
    MarkdownDocMatchInsertImageTool,
    MarkdownDocReadTool,
    WechatAlbumLatestArticlesTool,
    WechatArticleMarkdownTool,
)
from .workflow import (
    DailyNewsScheduler,
    ImageLayoutAgent,
    RenderImageStyleConfig,
    RenderPipelineConfig,
    SubAgentResult,
    ensure_playwright_chromium_installed,
    get_plugin_data_dir,
    load_template,
    merge_images_vertical,
    render_daily_news_pages,
    split_pages,
)
from .workflow.pipeline.playwright_bootstrap import (
    check_playwright_chromium_ready,
    config_needs_playwright,
    detect_windows_bootstrap_download_root,
)

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
        if head.startswith(b"\xff\xd8\xff"):  # JPEG
            return True
        if head.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return True
        if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
            return True
        return False
    except Exception:
        return False


def _select_render_template(cfg) -> str:
    name = str((cfg or {}).get("render_template_name") or "daily_news").strip().lower()
    if name in {"chenyu", "chenyu_style", "chenyu-style"}:
        return load_template("templates/chenyu-style.html").strip()
    return load_template("templates/daily_news.html").strip()


@register(
    "astrbot_dailynews_agent",
    "your_name",
    "AI 日报插件：定时抓取公众号最新内容，多 Agent 总结并自动推送",
    "0.3.0",
    "https://github.com/your/repo",
)
class DailyNewsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._sent_playwright_setup_hint = False
        self._sent_playwright_preflight_hint = False

        tools = [
            WechatArticleMarkdownTool(),
            WechatAlbumLatestArticlesTool(),
            ImageUrlsPreviewTool(),
            ImageUrlDownloadTool(),
            MarkdownDocCreateTool(),
            MarkdownDocReadTool(),
            MarkdownDocApplyEditsTool(),
            MarkdownDocMatchInsertImageTool(),
        ]

        if hasattr(self.context, "add_llm_tools"):
            self.context.add_llm_tools(*tools)
        else:
            tool_mgr = self.context.provider_manager.llm_tools
            tool_mgr.func_list.extend(tools)

        self.scheduler = DailyNewsScheduler(self.context, self.config)
        self._scheduler_task = None
        self._playwright_bootstrap_task = None
        try:
            self._scheduler_task = self.context.loop.create_task(self.scheduler.start())
        except Exception:
            try:
                import asyncio

                self._scheduler_task = asyncio.create_task(self.scheduler.start())
            except Exception:
                astrbot_logger.error(
                    "[dailynews] failed to start scheduler", exc_info=True
                )

        # Warn users if they still have the Windows-only bootstrap browser in plugin data.
        # This commonly happens when migrating AstrBot data from Windows -> Linux.
        try:
            bootstrap_root = detect_windows_bootstrap_download_root()
            if bootstrap_root is not None:
                if sys.platform.startswith("win"):
                    astrbot_logger.warning(
                        "[dailynews] Detected plugin bootstrap Chromium: %s (Windows-only). "
                        "If you run AstrBot on Linux/macOS, delete this folder and use: playwright install --with-deps chromium",
                        bootstrap_root,
                    )
                else:
                    astrbot_logger.warning(
                        "[dailynews] Detected Windows-only bootstrap Chromium under non-Windows system: %s. "
                        "Please delete it and use: playwright install --with-deps chromium",
                        bootstrap_root,
                    )
        except Exception:
            pass

        # Non-Windows: do NOT auto-download browsers. Guide users to official Playwright install.
        try:
            pipeline_cfg = RenderPipelineConfig.from_mapping(self.config)
            if (
                not sys.platform.startswith("win")
            ) and pipeline_cfg.playwright_fallback:
                pw_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
                if pw_path and "playwright_browsers" in pw_path.replace("\\", "/"):
                    astrbot_logger.warning(
                        "[dailynews] PLAYWRIGHT_BROWSERS_PATH=%s seems to point to plugin directory; "
                        "this may break Linux rendering. Consider unsetting it and restart AstrBot.",
                        pw_path,
                    )
                astrbot_logger.warning(
                    "[dailynews] Linux/macOS detected: this plugin will NOT auto-install Playwright browsers. "
                    "Run: playwright install --with-deps chromium",
                )
        except Exception:
            pass

        # Warm up Playwright browser install in background (best-effort, non-blocking).
        # Windows only: we bootstrap a local headless-shell zip for better out-of-box behavior.
        if sys.platform.startswith("win"):
            try:
                self._playwright_bootstrap_task = self.context.loop.create_task(
                    ensure_playwright_chromium_installed()
                )
            except Exception:
                try:
                    import asyncio

                    self._playwright_bootstrap_task = asyncio.create_task(
                        ensure_playwright_chromium_installed()
                    )
                except Exception:
                    self._playwright_bootstrap_task = None

    async def terminate(self):
        # Best-effort: cancel any in-flight workflow & background tasks to avoid leaking across reloads.
        if getattr(self, "scheduler", None) is not None:
            try:
                await self.scheduler.stop()
            except Exception:
                pass

        t = getattr(self, "_playwright_bootstrap_task", None)
        if t is not None and hasattr(t, "cancel"):
            try:
                t.cancel()
            except Exception:
                pass

        t = getattr(self, "_scheduler_task", None)
        if t is not None and hasattr(t, "cancel"):
            try:
                t.cancel()
            except Exception:
                pass

    async def _maybe_send_playwright_setup_hint(
        self, event: AstrMessageEvent, pipeline_cfg: RenderPipelineConfig
    ):
        """
        Linux/macOS: explicitly tell users to run official Playwright install when Chromium is missing.

        We intentionally rely on Playwright's default browser cache path (e.g. ~/.cache/ms-playwright),
        instead of downloading Windows-only zips into plugin directories.
        """
        if self._sent_playwright_setup_hint:
            return
        if sys.platform.startswith("win"):
            return
        if not pipeline_cfg.playwright_fallback:
            return

        custom_path = (pipeline_cfg.custom_browser_path or "").strip()
        if custom_path:
            if Path(custom_path).expanduser().exists():
                # Still warn once: not recommended to modify unless you know what you're doing.
                self._sent_playwright_setup_hint = True
                yield event.plain_result(
                    f"提示：检测到已配置 custom_browser_path={custom_path}（不建议修改；一般留空即可）。"
                )
                return
            self._sent_playwright_setup_hint = True
            yield event.plain_result(
                "Playwright 配置问题：custom_browser_path 指向的文件不存在。\n"
                f"当前：{custom_path}\n"
                "建议留空，并在 AstrBot 安装依赖后执行：playwright install --with-deps chromium"
            )
            return

        # Auto-detect: rely on official Playwright browser cache (e.g. ~/.cache/ms-playwright).
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except Exception:
            self._sent_playwright_setup_hint = True
            yield event.plain_result(
                "Playwright 未安装或不可用。请先安装依赖后再执行：playwright install --with-deps chromium"
            )
            return

        ready = False
        try:
            async with async_playwright() as p:
                exe = Path(str(p.chromium.executable_path)).expanduser()
                ready = exe.exists()
        except Exception:
            ready = False

        if not ready:
            self._sent_playwright_setup_hint = True
            yield event.plain_result(
                "Linux/macOS 需要先安装 Playwright Chromium 浏览器（本插件不再下载 zip）：\n"
                "playwright install --with-deps chromium\n"
                "（如命令不可用，可用：python -m playwright install --with-deps chromium）\n"
                "安装完成后重启 AstrBot/插件。"
            )

    async def _playwright_preflight_error(self, cfg: dict) -> str | None:
        if sys.platform.startswith("win"):
            return None
        if not config_needs_playwright(cfg):
            return None
        pipeline_cfg = RenderPipelineConfig.from_mapping(cfg)
        ok, msg = await check_playwright_chromium_ready(
            custom_browser_path=pipeline_cfg.custom_browser_path
        )
        if ok:
            return None
        # Don't spam the same session repeatedly, but always stop execution.
        if self._sent_playwright_preflight_hint:
            return "Playwright Chromium 未安装/不可用，已停止执行；请运行：playwright install --with-deps chromium"
        self._sent_playwright_preflight_hint = True
        return msg

    # ====== commands ======

    async def _send_as_html_images(self, event: AstrMessageEvent, content: str):
        pipeline_cfg = RenderPipelineConfig.from_mapping(self.config)
        style_cfg = RenderImageStyleConfig.from_mapping(self.config)

        async for rr in self._maybe_send_playwright_setup_hint(event, pipeline_cfg):
            yield rr

        pages = split_pages(
            content,
            page_chars=pipeline_cfg.page_chars,
            max_pages=pipeline_cfg.max_pages,
        )
        if not pages:
            yield event.plain_result("生成失败：日报内容为空，请查看 AstrBot 日志。")
            return

        template_name = getattr(self.context, "_config", {}).get("t2i_active_template")
        renderer = _astrbot_html_renderer
        if renderer is None:
            try:
                from astrbot.core import html_renderer as renderer  # type: ignore
            except Exception:
                renderer = None

        async def _send_image(path: Path):
            p = Path(path).resolve().as_posix()
            if _ImageComponent is not None:
                yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
            else:
                yield event.image_result(p)

        async def _render_html(ctx: dict) -> Path | None:
            try:
                p = await self.html_render(
                    _select_render_template(self.config), ctx, return_url=False
                )
                return Path(str(p)).resolve()
            except Exception:
                astrbot_logger.error("[dailynews] html_render failed", exc_info=True)
                return None

        async def _render_t2i(text: str) -> Path | None:
            try:
                if renderer is None:
                    p = await self.text_to_image(text, return_url=False)
                else:
                    p = await renderer.render_t2i(
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
            template_str=_select_render_template(self.config),
            render_html=_render_html,
            render_t2i=_render_t2i,
            pipeline=pipeline_cfg,
            style=style_cfg,
            title="每日资讯日报",
            subtitle_fmt="第 {idx}/{total} 页",
        )

        for r in rendered:
            if r.image_path is None or not _is_valid_image_file(
                Path(r.image_path).resolve()
            ):
                yield event.plain_result(r.markdown)
                continue

            astrbot_logger.info(
                "[dailynews] rendered page %s/%s via %s -> %s",
                r.index,
                r.total,
                r.method or "unknown",
                r.image_path,
            )
            async for rr in _send_image(Path(r.image_path)):
                yield rr

    @filter.command("daily_news")
    async def daily_news(self, event: AstrMessageEvent):
        """手动生成一次日报（并回发到当前会话）"""
        err = await self._playwright_preflight_error(dict(self.config or {}))
        if err:
            yield event.plain_result(err)
            return
        yield event.plain_result("正在生成日报，请稍候...")
        content = await self.scheduler.generate_once()
        if not (content or "").strip():
            astrbot_logger.warning(
                "[dailynews] /daily_news got empty content; sending fallback text"
            )
            content = "生成失败：日报内容为空（可能是抓取/LLM 超时或被其它插件过滤），请查看 AstrBot 日志。"
        # Manual trigger should also publish to AstrBook when enabled.
        try:
            res = await self.scheduler.publish_report_to_astrbook(content)
            if not bool(res.get("ok", False)) and not bool(res.get("skipped", False)):
                astrbot_logger.warning(
                    "[dailynews] manual publish to astrbook failed: %s %s",
                    res.get("error"),
                    (res.get("detail") or ""),
                )
        except Exception:
            astrbot_logger.error(
                "[dailynews] manual publish to astrbook failed", exc_info=True
            )

        # Send to current session directly (with fallback inside scheduler) to avoid
        # "prepared but failed to send" cases when platform times out on images.
        try:
            cfg = self.scheduler.get_config_snapshot()
            sent = await self.scheduler._send_to_targets(
                content, [event.unified_msg_origin], config=cfg
            )
            if sent > 0:
                return
        except Exception:
            astrbot_logger.error("[dailynews] manual send failed", exc_info=True)

        # Last resort: let pipeline try to send plain text.
        yield event.plain_result(content)

    @filter.command("news_toggle")
    async def news_toggle(self, event: AstrMessageEvent):
        """切换自动日报开关"""
        self.config["enabled"] = not bool(self.config.get("enabled", False))
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result(
            f"自动日报已{'开启' if self.config['enabled'] else '关闭'}"
        )

    @filter.command("news_config")
    async def news_config(self, event: AstrMessageEvent):
        """查看当前配置（JSON）"""
        snapshot = self.scheduler.get_config_snapshot()
        yield event.plain_result(json.dumps(snapshot, ensure_ascii=False, indent=2))

    @filter.command("news_test_md")
    async def news_test_md(self, event: AstrMessageEvent, args: str = ""):
        """
        测试用 Markdown（用于检查渲染/分页/发送链路与日志排版）
        用法：/news_test_md [plain|html] [long]
        """
        parts = (args or "").strip().split()
        force_mode = parts[0].strip().lower() if parts else ""
        long_mode = any(p.strip().lower() == "long" for p in parts[1:]) or (
            parts and parts[0].strip().lower() == "long"
        )

        test_md = textwrap.dedent(
            """
            # 每日资讯日报（测试稿）

            *生成时间：{now}*

            ## 1. 标题/列表/链接
            - 要点 1：带链接 https://example.com
            - 要点 2：带括号与中文标点（测试）
            - 要点 3：长行测试：{long_line}

            ## 2. 代码块
            ```python
            def hello(name: str) -> str:
                return f"hello, {{name}}"
            ```

            ## 3. 引用与分隔
            > 这是一段引用（测试换行与缩进）。
            >
            > - 引用内列表 A
            > - 引用内列表 B

            ---

            ## 4. 结尾
            - 支持分页/多图渲染：`render_page_chars`、`render_max_pages`
            """
        ).strip()

        if long_mode:
            filler = "\n".join(
                [f"- filler line {i}: {('内容' * 40)}" for i in range(1, 120)]
            )
            test_md = f"{test_md}\n\n## 5. 长内容（分页测试）\n{filler}\n"

        content = test_md.format(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            long_line=("A" * 180),
        )

        astrbot_logger.info(
            "[dailynews] /news_test_md mode=%s long=%s chars=%s",
            force_mode or "(config)",
            long_mode,
            len(content),
        )
        astrbot_logger.debug(
            "[dailynews] /news_test_md preview=%s",
            (
                content[:260].replace("\n", "\\n")
                + ("..." if len(content) > 260 else "")
            ),
        )

        delivery_mode = str(
            self.config.get("delivery_mode", "html_image") or "html_image"
        )
        if force_mode in {"plain", "text"}:
            delivery_mode = "plain"
        elif force_mode in {"html", "img", "image"}:
            delivery_mode = "html_image"

        if delivery_mode == "html_image":
            async for r in self._send_as_html_images(event, content):
                yield r
            return

        yield event.plain_result(content)

    @filter.command("news_image_preview")
    async def news_image_preview(self, event: AstrMessageEvent, args: str = ""):
        """
        合并多张图片 URL 为纵向预览图。
        用法：/news_image_preview url1 url2 url3 ...
        """
        from .workflow.image_utils import (
            get_plugin_data_dir,
            merge_images_vertical,
            parse_image_urls,
        )

        urls = parse_image_urls(args)
        if not urls:
            yield event.plain_result("用法：/news_image_preview url1 url2 url3 ...")
            return

        out_dir = get_plugin_data_dir("image_previews")
        out_path = out_dir / f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            merged = await merge_images_vertical(urls, out_path=out_path)
        except Exception as e:
            astrbot_logger.error(
                "[dailynews] news_image_preview failed: %s", e, exc_info=True
            )
            yield event.plain_result(f"合并失败：{e}")
            return

        img_file = Path(str(merged)).resolve()
        if not _is_valid_image_file(img_file):
            yield event.plain_result("合并失败：生成的图片无效")
            return

        p = img_file.as_posix()
        if _ImageComponent is not None:
            yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
        else:
            yield event.image_result(p)

    @filter.command("news_image_debug")
    async def news_image_debug(self, event: AstrMessageEvent, args: str = ""):
        """
        调试：列出每个来源抓到的图片直链，并可选拼接预览图。
        用法：/news_image_debug [preview] [max_urls]
        """
        parts = (args or "").strip().split()
        want_preview = any(p.lower() == "preview" for p in parts)
        max_urls = 12
        for p in parts:
            if p.isdigit():
                max_urls = max(1, min(50, int(p)))
                break

        cfg = self.scheduler.get_config_snapshot()
        err = await self._playwright_preflight_error(cfg)
        if err:
            yield event.plain_result(err)
            return
        await self.scheduler.update_workflow_sources_from_config(cfg)
        sources = list(self.scheduler.workflow_manager.news_sources)
        if not sources:
            yield event.plain_result("未配置 news_sources")
            return

        # 1) 先抓每个来源的最新文章列表
        fetch_tasks = []
        fetch_sources = []
        for s in sources:
            agent_cls = self.scheduler.workflow_manager.sub_agents.get(s.type)
            if not agent_cls:
                continue
            fetch_sources.append(s)
            fetch_tasks.append(agent_cls().fetch_latest_articles(s, cfg))

        fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        source_articles = {}
        for idx, r in enumerate(fetched):
            src = fetch_sources[idx]
            if isinstance(r, Exception):
                astrbot_logger.warning(
                    "[dailynews] news_image_debug fetch list failed: %s",
                    r,
                    exc_info=True,
                )
                source_articles[src.name] = []
                continue
            name, articles = r
            source_articles[name] = articles or []

        # 2) 抓正文并提取 image_urls（不走 LLM）
        from .analysis.miyousheanalysis.analysis import fetch_miyoushe_post
        from .analysis.wechatanalysis.analysis import fetch_wechat_article
        from .workflow import _run_sync

        async def _collect_images(
            source_name: str, source_type: str, arts, max_articles: int
        ):
            urls = []
            for a in (arts or [])[: max(1, int(max_articles or 1))]:
                u = (a.get("url") or "").strip() if isinstance(a, dict) else ""
                if not u:
                    continue
                try:
                    if source_type == "miyoushe":
                        d = await _run_sync(fetch_miyoushe_post, u)
                    else:
                        d = await _run_sync(fetch_wechat_article, u)
                    imgs = d.get("image_urls") or []
                    if isinstance(imgs, list):
                        for x in imgs:
                            xs = str(x).strip()
                            if xs and xs not in urls:
                                urls.append(xs)
                                if len(urls) >= max_urls:
                                    return urls
                except Exception:
                    continue
            return urls

        tasks = []
        meta = []
        for s in sources:
            tasks.append(
                _collect_images(
                    s.name,
                    s.type,
                    source_articles.get(s.name, []),
                    int(getattr(s, "max_articles", 2) or 2),
                )
            )
            meta.append(s)
        results = await asyncio.gather(*tasks, return_exceptions=False)

        lines = ["# image_debug", ""]
        all_for_preview = []
        for s, imgs in zip(meta, results):
            lines.append(f"## {s.name} ({s.type})")
            lines.append(f"- images: {len(imgs)}")
            for u in imgs[: min(8, len(imgs))]:
                lines.append(f"  - {u}")
            lines.append("")
            all_for_preview.extend(imgs)

        if not want_preview:
            yield event.plain_result("\n".join(lines).strip())
            return

        if not all_for_preview:
            yield event.plain_result("\n".join(lines + ["无可预览图片"]).strip())
            return

        out_path = (
            get_plugin_data_dir("image_previews")
            / f"debug_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        try:
            merged = await merge_images_vertical(
                all_for_preview[:max_urls], out_path=out_path
            )
            img_file = Path(str(merged)).resolve()
            if not _is_valid_image_file(img_file):
                yield event.plain_result(
                    "\n".join(lines + ["预览图生成失败：图片无效"]).strip()
                )
                return
            p = img_file.as_posix()
            if _ImageComponent is not None:
                yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
            else:
                yield event.image_result(p)
        except Exception as e:
            astrbot_logger.error(
                "[dailynews] news_image_debug preview failed: %s", e, exc_info=True
            )
            yield event.plain_result(
                "\n".join(lines + [f"预览图生成失败：{e}"]).strip()
            )

    @filter.command("news_layout_test")
    async def news_layout_test(self, event: AstrMessageEvent, args: str = ""):
        """
        仅测试“图片排版 Agent”的插图能力，不跑写作/汇总。
        用法：/news_layout_test [plain|html] [preview] [max_urls]
        """
        parts = (args or "").strip().split()
        mode = "html" if any(p.lower() == "html" for p in parts) else "plain"
        force_preview = any(p.lower() == "preview" for p in parts)
        max_urls = 12
        for p in parts:
            if p.isdigit():
                max_urls = max(1, min(50, int(p)))
                break

        cfg = self.scheduler.get_config_snapshot()
        cfg["image_layout_enabled"] = True
        if force_preview:
            cfg["image_layout_preview_enabled"] = True

        err = await self._playwright_preflight_error(cfg)
        if err:
            yield event.plain_result(err)
            return
        await self.scheduler.update_workflow_sources_from_config(cfg)
        sources = list(self.scheduler.workflow_manager.news_sources)
        if not sources:
            yield event.plain_result("未配置 news_sources")
            return

        # 1) 抓取每个来源最新文章列表
        fetch_tasks = []
        fetch_sources = []
        for s in sources:
            agent_cls = self.scheduler.workflow_manager.sub_agents.get(s.type)
            if not agent_cls:
                continue
            fetch_sources.append(s)
            fetch_tasks.append(agent_cls().fetch_latest_articles(s, cfg))

        fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        source_articles = {}
        for idx, r in enumerate(fetched):
            src = fetch_sources[idx]
            if isinstance(r, Exception):
                astrbot_logger.warning(
                    "[dailynews] news_layout_test fetch list failed: %s",
                    r,
                    exc_info=True,
                )
                source_articles[src.name] = []
                continue
            name, articles = r
            source_articles[name] = articles or []

        # 2) 抓正文并提取 image_urls（不走 LLM 写作）
        from .analysis.miyousheanalysis.analysis import fetch_miyoushe_post
        from .analysis.wechatanalysis.analysis import fetch_wechat_article
        from .workflow import _run_sync

        async def _collect_images(
            source_name: str, source_type: str, arts, max_articles: int
        ):
            urls = []
            for a in (arts or [])[: max(1, int(max_articles or 1))]:
                u = (a.get("url") or "").strip() if isinstance(a, dict) else ""
                if not u:
                    continue
                try:
                    if source_type == "miyoushe":
                        d = await _run_sync(fetch_miyoushe_post, u)
                    else:
                        d = await _run_sync(fetch_wechat_article, u)
                    imgs = d.get("image_urls") or []
                    if isinstance(imgs, list):
                        for x in imgs:
                            xs = str(x).strip()
                            if xs and xs not in urls:
                                urls.append(xs)
                                if len(urls) >= max_urls:
                                    return urls
                except Exception:
                    continue
            return urls

        tasks = []
        meta = []
        for s in sources:
            tasks.append(
                _collect_images(
                    s.name,
                    s.type,
                    source_articles.get(s.name, []),
                    int(getattr(s, "max_articles", 2) or 2),
                )
            )
            meta.append(s)
        results = await asyncio.gather(*tasks, return_exceptions=False)

        sub_results = []
        for s, imgs in zip(meta, results):
            if not imgs:
                continue
            sub_results.append(
                SubAgentResult(
                    source_name=s.name,
                    summary="",
                    key_points=[],
                    content="",
                    images=imgs,
                )
            )

        test_md = textwrap.dedent(
            """
            # 每日资讯日报（排版测试）

            *生成时间：{now}*

            ## 1. 标题/列表/链接
            - 要点 1：带链接 https://example.com
            - 要点 2：带括号与中文标点（测试）
            - 要点 3：长行测试：{long_line}

            ## 2. 代码块
            ```python
            def hello(name: str) -> str:
                return f"hello, {{name}}"
            ```

            ## 3. 引用与分隔
            > 这是一段引用（测试换行与缩进）。
            >
            > - 引用内列表 A
            > - 引用内列表 B

            ---

            ## 4. 结尾
            - 支持分页/多图渲染：`render_page_chars`、`render_max_pages`
            """
        ).strip()
        test_md = test_md.format(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            long_line="A" * 200,
        )

        patched = await ImageLayoutAgent().enhance_markdown(
            draft_markdown=test_md,
            sub_results=sub_results,
            user_config=cfg,
            astrbot_context=self.context,
            image_plan=None,
        )

        if mode == "html":
            async for r in self._send_as_html_images(event, patched):
                yield r
        else:
            yield event.plain_result(patched)

    @filter.command("news_subscribe")
    async def news_subscribe(self, event: AstrMessageEvent):
        """订阅：把当前会话加入推送列表"""
        targets: list[str] = list(self.config.get("target_sessions", []) or [])
        if event.unified_msg_origin not in targets:
            targets.append(event.unified_msg_origin)
        self.config["target_sessions"] = targets
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已订阅本会话的每日推送")

    @filter.command("news_unsubscribe")
    async def news_unsubscribe(self, event: AstrMessageEvent):
        """退订：把当前会话移出推送列表"""
        targets: list[str] = list(self.config.get("target_sessions", []) or [])
        self.config["target_sessions"] = [
            x for x in targets if x != event.unified_msg_origin
        ]
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已退订本会话的每日推送")

    @filter.command("news_add_source")
    async def news_add_source(self, event: AstrMessageEvent, args: str):
        """
        添加新闻源:
        /news_add_source URL
        /news_add_source 名称 URL
        """
        parts = (args or "").strip().split()
        if len(parts) < 1:
            yield event.plain_result(
                "用法：/news_add_source URL（或 /news_add_source 名称 URL）"
            )
            return

        display_name = ""
        url = parts[0].strip() if len(parts) == 1 else parts[1].strip()
        if len(parts) >= 2:
            display_name = parts[0].strip()
        if not url:
            yield event.plain_result("URL 不能为空")
            return

        # Preferred (v4.10.4+): template_list sources.
        key = "wechat"
        low = url.lower()
        if "miyoushe.com" in low:
            key = "miyoushe"
        elif "github.com" in low or (
            "/" in url and " " not in url and "http" not in low and ":" not in url
        ):
            key = "github"
        elif "x.com" in low or "twitter.com" in low:
            key = "twitter"

        items = self.config.get("news_sources", []) or []
        if not isinstance(items, list):
            items = []

        def _exists(tk: str, value: str) -> bool:
            for it in items:
                if not isinstance(it, dict):
                    continue
                if str(it.get("__template_key") or "").strip().lower() != tk:
                    continue
                if tk == "github":
                    if str(it.get("repo") or "").strip() == value:
                        return True
                else:
                    if str(it.get("url") or "").strip() == value:
                        return True
            return False

        if key == "github":
            if not _exists("github", url):
                items.append(
                    {
                        "__template_key": "github",
                        "name": display_name,
                        "repo": url,
                        "priority": 1,
                    }
                )
        elif key == "twitter":
            if not _exists("twitter", url):
                items.append(
                    {
                        "__template_key": "twitter",
                        "name": display_name,
                        "url": url,
                        "priority": int(self.config.get("twitter_priority", 1) or 1),
                        "max_articles": int(
                            self.config.get("twitter_max_tweets", 3) or 3
                        ),
                    }
                )
        elif key == "miyoushe":
            if not _exists("miyoushe", url):
                items.append(
                    {
                        "__template_key": "miyoushe",
                        "name": display_name,
                        "url": url,
                        "priority": 1,
                        "max_articles": 3,
                    }
                )
        else:
            if not _exists("wechat", url):
                items.append(
                    {
                        "__template_key": "wechat",
                        "name": display_name,
                        "url": url,
                        "priority": 1,
                        "max_articles": 3,
                        "album_keyword": "",
                    }
                )

        self.config["news_sources"] = items
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已添加来源（已加入 news_sources）")

    @filter.command("news_remove_source")
    async def news_remove_source(self, event: AstrMessageEvent, name: str):
        """删除新闻源: /news_remove_source URL"""
        url = (name or "").strip()
        if not url:
            yield event.plain_result("用法：/news_remove_source URL")
            return

        items = self.config.get("news_sources", []) or []
        if isinstance(items, list) and items:
            kept = []
            removed = 0
            for it in items:
                if not isinstance(it, dict):
                    kept.append(it)
                    continue
                tk = str(it.get("__template_key") or "").strip().lower()
                if tk == "github":
                    if str(it.get("repo") or "").strip() == url:
                        removed += 1
                        continue
                else:
                    if str(it.get("url") or "").strip() == url:
                        removed += 1
                        continue
                kept.append(it)
            self.config["news_sources"] = kept
            if hasattr(self.config, "save_config"):
                self.config.save_config()
            yield event.plain_result(f"已删除来源（news_sources: removed={removed}）")
            return

        # Legacy fallback (older configs)
        sources: list[str] = list(self.config.get("wechat_sources", []) or [])
        self.config["wechat_sources"] = [x for x in sources if x != url]
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已删除来源（兼容模式：仅从 wechat_sources 移除）")
