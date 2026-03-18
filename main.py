import asyncio
import json
import textwrap
from datetime import datetime
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api import logger as astrbot_logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .tools import (
    ImageUrlDownloadTool,
    ImageUrlsDownloadBatchTool,
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
    NewsSourceConfig,
    RenderImageStyleConfig,
    RenderPipelineConfig,
    SubAgentResult,
    get_plugin_data_dir,
    load_template,
    merge_images_vertical,
    render_daily_news_pages,
    split_pages,
)

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
        self._plugin_module_prefix = str(__package__ or "").strip()

        tools = [
            WechatArticleMarkdownTool(),
            WechatAlbumLatestArticlesTool(),
            ImageUrlsPreviewTool(),
            ImageUrlDownloadTool(),
            ImageUrlsDownloadBatchTool(),
            MarkdownDocCreateTool(),
            MarkdownDocReadTool(),
            MarkdownDocApplyEditsTool(),
            MarkdownDocMatchInsertImageTool(),
        ]
        self._registered_llm_tool_names = {tool.name for tool in tools}

        # Hot reload in AstrBot does not always clean stale tool registrations reliably.
        # Purge any old dailynews-owned tool objects before registering the current set.
        self._purge_dailynews_llm_tools(include_internal_only=True)

        if hasattr(self.context, "add_llm_tools"):
            self.context.add_llm_tools(*tools)
        else:
            tool_mgr = self.context.provider_manager.llm_tools
            tool_mgr.func_list.extend(tools)

        self.scheduler = DailyNewsScheduler(self.context, self.config)
        self._scheduler_task = None
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

    def _tool_modules(self, tool) -> set[str]:
        mods: set[str] = set()
        for value in (
            getattr(tool, "handler_module_path", None),
            getattr(tool, "__module__", None),
            getattr(getattr(tool, "__class__", None), "__module__", None),
        ):
            text = str(value or "").strip()
            if text:
                mods.add(text)
        return mods

    def _is_dailynews_owned_tool(self, tool) -> bool:
        prefix = self._plugin_module_prefix
        if not prefix:
            return False
        return any(mod.startswith(prefix) for mod in self._tool_modules(tool))

    def _is_dailynews_internal_layout_tool(self, tool) -> bool:
        name = str(getattr(tool, "name", "") or "").strip()
        if name == "gemini_layout_generate_image":
            return True
        if name != "gemini_image_generation":
            return False
        modules = self._tool_modules(tool)
        return any(
            "astrbot_dailynews_agent" in mod and "image_tools" in mod for mod in modules
        )

    def _purge_dailynews_llm_tools(self, *, include_internal_only: bool) -> None:
        tool_mgr = getattr(getattr(self.context, "provider_manager", None), "llm_tools", None)
        if tool_mgr is None:
            return
        func_list = getattr(tool_mgr, "func_list", None)
        if not isinstance(func_list, list):
            return

        removed: list[str] = []
        for tool in list(func_list):
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue
            should_remove = False
            if self._is_dailynews_owned_tool(tool) and (
                name in self._registered_llm_tool_names or include_internal_only
            ):
                should_remove = True
            if include_internal_only and self._is_dailynews_internal_layout_tool(tool):
                should_remove = True
            if not should_remove:
                continue
            try:
                func_list.remove(tool)
                removed.append(name)
            except ValueError:
                continue

        if removed:
            astrbot_logger.info(
                "[dailynews] purged stale llm tools: %s",
                ", ".join(sorted(set(removed))),
            )

    async def terminate(self):
        # Best-effort: cancel any in-flight workflow & background tasks to avoid leaking across reloads.
        if getattr(self, "scheduler", None) is not None:
            try:
                await self.scheduler.stop()
            except Exception:
                pass

        t = getattr(self, "_scheduler_task", None)
        if t is not None and hasattr(t, "cancel"):
            try:
                t.cancel()
            except Exception:
                pass

        try:
            self._purge_dailynews_llm_tools(include_internal_only=True)
        except Exception:
            astrbot_logger.warning(
                "[dailynews] failed to purge llm tools during terminate",
                exc_info=True,
            )

    async def _get_configured_sources(self, cfg: dict) -> list:
        await self.scheduler.update_workflow_sources_from_config(cfg)
        return list(self.scheduler.workflow_manager.news_sources)

    def _is_likely_url(self, value: str) -> bool:
        text = str(value or "").strip().lower()
        return text.startswith("http://") or text.startswith("https://")

    def _match_sources(self, sources: list, query: str) -> list:
        key = str(query or "").strip().lower()
        if not key:
            return []
        exact = []
        fuzzy = []
        for src in sources:
            name = str(getattr(src, "name", "") or "")
            low = name.lower()
            if low == key:
                exact.append(src)
            elif key in low:
                fuzzy.append(src)
        return exact or fuzzy

    async def _fetch_articles_for_sources(self, sources: list, cfg: dict) -> dict[str, list]:
        fetch_tasks = []
        fetch_sources = []
        for src in sources:
            agent_cls = self.scheduler.workflow_manager.sub_agents.get(src.type)
            if not agent_cls:
                continue
            fetch_sources.append(src)
            fetch_tasks.append(agent_cls().fetch_latest_articles(src, cfg))

        fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        source_articles: dict[str, list] = {}
        for idx, result in enumerate(fetched):
            src = fetch_sources[idx]
            if isinstance(result, Exception):
                astrbot_logger.warning(
                    "[dailynews] fetch source articles failed: %s",
                    result,
                    exc_info=True,
                )
                source_articles[src.name] = []
                continue
            name, articles = result
            source_articles[name] = articles or []
        return source_articles

    async def _collect_images_for_source(
        self,
        *,
        source,
        articles: list,
        max_urls: int,
    ) -> list[str]:
        from .analysis.miyousheanalysis.analysis import fetch_miyoushe_post
        from .analysis.wechatanalysis.analysis import fetch_wechat_article
        from .workflow import _run_sync

        urls: list[str] = []
        for article in (articles or [])[: max(1, int(getattr(source, "max_articles", 2) or 2))]:
            article_url = (
                (article.get("url") or "").strip() if isinstance(article, dict) else ""
            )
            if not article_url:
                continue
            try:
                if str(getattr(source, "type", "") or "") == "miyoushe":
                    data = await _run_sync(fetch_miyoushe_post, article_url)
                else:
                    data = await _run_sync(fetch_wechat_article, article_url)
                images = data.get("image_urls") or []
                if not isinstance(images, list):
                    continue
                for item in images:
                    image_url = str(item or "").strip()
                    if image_url and image_url not in urls:
                        urls.append(image_url)
                        if len(urls) >= max_urls:
                            return urls
            except Exception:
                continue
        return urls

    # ====== commands ======

    async def _send_as_html_images(
        self,
        event: AstrMessageEvent,
        content: str,
        *,
        force_pillow: bool = False,
    ):
        pipeline_cfg = RenderPipelineConfig.from_mapping(self.config)
        style_cfg = RenderImageStyleConfig.from_mapping(self.config)

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
            if force_pillow:
                return None
            try:
                p = await self.html_render(
                    _select_render_template(self.config), ctx, return_url=False
                )
                return Path(str(p)).resolve()
            except Exception:
                astrbot_logger.error("[dailynews] html_render failed", exc_info=True)
                return None

        async def _render_t2i(text: str) -> Path | None:
            if force_pillow:
                return None
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
            chenyu_font_files=self.config.get("chenyu_font_files", []),
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

    async def _send_prepared_report_to_event(
        self,
        event: AstrMessageEvent,
        *,
        content: str,
        prepared: dict,
    ):
        img_paths = self.scheduler._valid_cached_img_paths(prepared.get("img_paths") or [])
        for img_path in img_paths:
            p = Path(str(img_path)).resolve().as_posix()
            if _ImageComponent is not None:
                yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
            else:
                yield event.image_result(p)

        link_node_chunks = [
            str(x) for x in (prepared.get("link_node_chunks") or []) if str(x).strip()
        ]
        appendix_images = self.scheduler._normalize_appendix_images(
            prepared.get("appendix_images") or []
        )
        if not link_node_chunks and not appendix_images:
            return

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
            nodes.extend(self.scheduler._build_appendix_image_nodes(appendix_images))
            if nodes:
                yield event.chain_result([_NodesComponent(nodes)])
                return

        fallback_parts = list(link_node_chunks)
        fallback_parts.extend(
            [
                str(item.get("url") or "").strip()
                for item in appendix_images
                if str(item.get("url") or "").strip()
            ]
        )
        if fallback_parts:
            yield event.plain_result("\n\n".join(fallback_parts))

    @filter.command("daily_news")
    async def daily_news(self, event: AstrMessageEvent, args: str = ""):
        """手动生成一次日报（并回发到当前会话）
        用法：/daily_news [force]
        """
        parts = [p.strip().lower() for p in (args or "").strip().split() if p.strip()]
        force_refresh = "force" in parts
        yield event.plain_result(
            "正在强制刷新日报，请稍候..." if force_refresh else "正在准备日报，请稍候..."
        )
        cfg = self.scheduler.get_config_snapshot()
        prepared = await self.scheduler.prepare_report(
            cfg, source="manual", prefer_cache=not force_refresh
        )
        content = str(prepared.get("content") or "")
        if not (content or "").strip():
            astrbot_logger.warning(
                "[dailynews] /daily_news got empty content; sending fallback text"
            )
            content = "生成失败：日报内容为空（可能是抓取、LLM 超时或被其它插件过滤），请查看 AstrBot 日志。"
        # Manual trigger should also publish to AstrBook when enabled.
        try:
            if not bool(prepared.get("cache_hit", False)):
                res = await self.scheduler.publish_report_to_astrbook(
                    content, config=cfg
                )
            else:
                res = {"ok": False, "skipped": True, "reason": "cache_hit"}
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
            sent = await self.scheduler._send_to_targets(
                content,
                [event.unified_msg_origin],
                config=cfg,
                prepared=prepared,
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
        用法：/news_test_md [plain|html|pillow] [long]
        """
        parts = [p.strip().lower() for p in (args or "").strip().split() if p.strip()]
        selected_modes: list[str] = []
        long_mode = False
        unknown_parts: list[str] = []

        for p in parts:
            if p in {"plain", "text"}:
                selected_modes.append("plain")
            elif p in {"html", "img", "image"}:
                selected_modes.append("html_image")
            elif p in {"pillow", "pil"}:
                selected_modes.append("pillow_image")
            elif p == "long":
                long_mode = True
            else:
                unknown_parts.append(p)

        if len(set(selected_modes)) > 1 or unknown_parts:
            yield event.plain_result(
                "用法：/news_test_md [plain|html|pillow] [long]\n示例：/news_test_md pillow long"
            )
            return

        force_mode = selected_modes[-1] if selected_modes else ""

        test_md = textwrap.dedent(
            """
            # 每日资讯日报（Markdown 测试）

            > 这个命令用于检查 Markdown 渲染、分页、发送链路与日志排版。

            - 生成时间：{now}
            - 输出模式：{mode}
            - 长文模式：{long_mode}

            ## 1. 标题 / 列表 / 链接
            - 要点 1：普通链接 https://example.com
            - 要点 2：[带标题链接](https://example.com/docs)
            - 要点 3：超长行（用于测试换行）：
              {long_line}

            ## 2. 有序列表
            1. 第一步：抓取内容
            2. 第二步：生成摘要
            3. 第三步：渲染并发送

            ## 3. 引用块
            > 引用段落 A
            >
            > - 引用内列表 1
            > - 引用内列表 2

            ## 4. 代码块
            ```python
            def hello(name: str) -> str:
                return f"hello, {{name}}"
            ```

            ```json
            {{"ok": true, "source": "news_test_md"}}
            ```

            ## 5. 表格
            | 字段 | 值 |
            | --- | --- |
            | page_chars | `render_page_chars` |
            | max_pages | `render_max_pages` |

            ## 6. 任务列表
            - [x] 标题渲染
            - [x] 列表渲染
            - [x] 代码块渲染
            - [x] 表格渲染
            - [ ] 跨客户端一致性核验

            ---

            ## 7. 结尾
            如果你看到本段，说明基础 Markdown 结构已完整输出。
            """
        ).strip()

        if long_mode:
            filler = "\n".join(
                [f"- filler line {i}: {('内容' * 36)}" for i in range(1, 180)]
            )
            test_md = f"{test_md}\n\n## 8. 长文分页测试\n{filler}\n"

        content = test_md.format(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode=force_mode or "(config)",
            long_mode=long_mode,
            long_line=("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" * 5),
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
        if force_mode:
            delivery_mode = force_mode

        if delivery_mode in {"html_image", "pillow_image"}:
            async for r in self._send_as_html_images(
                event,
                content,
                force_pillow=(delivery_mode == "pillow_image"),
            ):
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
            html_cfg = dict(cfg)
            html_cfg["delivery_mode"] = "html_image"
            prepared = await self.scheduler._build_report_payload(
                patched,
                html_cfg,
                cache_hit=False,
            )
            async for r in self._send_prepared_report_to_event(
                event,
                content=patched,
                prepared=prepared,
            ):
                yield r
        else:
            yield event.plain_result(patched)

    @filter.command("news_source_test")
    async def news_source_test(self, event: AstrMessageEvent, args: str = ""):
        """
        只测试单个信息源的抓图或排版。
        用法：/news_source_test <source_name|wechat_article_url> [images|layout] [plain|html] [preview] [max_urls]
        """
        parts = [p.strip() for p in (args or "").strip().split() if p.strip()]
        if not parts:
            yield event.plain_result(
                "用法：news_source_test <source_name|wechat_article_url> [images|layout] [plain|html] [preview] [max_urls]"
            )
            return

        source_query = parts[0]
        flags = [p.lower() for p in parts[1:]]
        test_kind = "images"
        if "layout" in flags:
            test_kind = "layout"
        elif "images" in flags:
            test_kind = "images"
        output_mode = "html" if "html" in flags else "plain"
        force_preview = "preview" in flags
        max_urls = 12
        for p in parts[1:]:
            if p.isdigit():
                max_urls = max(1, min(50, int(p)))
                break

        cfg = self.scheduler.get_config_snapshot()
        cfg["image_layout_enabled"] = True
        if force_preview:
            cfg["image_layout_preview_enabled"] = True

        sources = await self._get_configured_sources(cfg)
        if not sources:
            yield event.plain_result("未配置 news_sources")
            return

        source = None
        if self._is_likely_url(source_query):
            source = NewsSourceConfig(
                name="临时公众号测试",
                url=source_query,
                type="wechat",
                priority=1,
                max_articles=3,
            )
        else:
            matched = self._match_sources(sources, source_query)
            if not matched:
                names = ", ".join(str(getattr(s, "name", "") or "") for s in sources[:20])
                yield event.plain_result(
                    f"未找到信息源：{source_query}\n可用信息源：{names}\n也可以直接传一篇公众号文章 URL 作为临时测试源。"
                )
                return
            if len(matched) > 1:
                names = ", ".join(str(getattr(s, "name", "") or "") for s in matched)
                yield event.plain_result(f"匹配到多个信息源，请写更精确一些：{names}")
                return
            source = matched[0]

        source_articles = await self._fetch_articles_for_sources([source], cfg)
        articles = source_articles.get(source.name, []) or []
        image_urls = await self._collect_images_for_source(
            source=source,
            articles=articles,
            max_urls=max_urls,
        )

        if test_kind == "images":
            lines = [
                f"# source_test: {source.name}",
                f"- type: {source.type}",
                f"- articles: {len(articles)}",
                f"- images: {len(image_urls)}",
                "",
            ]
            if self._is_likely_url(source_query):
                lines.insert(3, f"- seed_url: {source.url}")
            for article in articles[: min(5, len(articles))]:
                if not isinstance(article, dict):
                    continue
                title = str(article.get("title") or "").strip()
                article_url = str(article.get("url") or "").strip()
                if title:
                    lines.append(f"- article: {title}")
                if article_url:
                    lines.append(f"  {article_url}")
            if image_urls:
                lines.append("")
                lines.append("## image_urls")
                for u in image_urls:
                    lines.append(f"- {u}")

            if force_preview and image_urls:
                out_path = (
                    get_plugin_data_dir("image_previews")
                    / f"source_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                try:
                    merged = await merge_images_vertical(
                        image_urls[:max_urls], out_path=out_path
                    )
                    img_file = Path(str(merged)).resolve()
                    if _is_valid_image_file(img_file):
                        p = img_file.as_posix()
                        if _ImageComponent is not None:
                            yield event.chain_result(
                                [_ImageComponent(file=f"file:///{p}", path=p)]
                            )
                        else:
                            yield event.image_result(p)
                except Exception as e:
                    astrbot_logger.warning(
                        "[dailynews] news_source_test preview failed: %s",
                        e,
                        exc_info=True,
                    )
                    lines.append("")
                    lines.append(f"preview_failed: {e}")

            yield event.plain_result("\n".join(lines).strip())
            return

        article_lines = []
        for idx, article in enumerate(articles[: min(6, len(articles))], start=1):
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "").strip() or f"条目 {idx}"
            article_url = str(article.get("url") or "").strip()
            if article_url:
                article_lines.append(f"- [{title}]({article_url})")
            else:
                article_lines.append(f"- {title}")
        if not article_lines:
            article_lines.append("- 本次未抓到文章标题，仅测试图片插入。")

        test_md = "\n".join(
            [
                f"# 单源排版测试：{source.name}",
                "",
                f"*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                "",
                "## 最新内容",
                *article_lines,
                "",
                "## 编辑说明",
                "- 仅允许使用这个信息源的候选图。",
                "- 优先让图片贴合对应条目，不要无意义插图。",
                "- 如果候选图不合适，可以在配置开启时调用文生图工具生成辅助氛围图。",
            ]
        ).strip()

        sub_results = []
        if image_urls:
            sub_results.append(
                SubAgentResult(
                    source_name=source.name,
                    summary="",
                    key_points=[],
                    content="\n".join(article_lines),
                    images=image_urls,
                )
            )

        patched = await ImageLayoutAgent().enhance_markdown(
            draft_markdown=test_md,
            sub_results=sub_results,
            user_config=cfg,
            astrbot_context=self.context,
            image_plan={source.name: min(len(image_urls), 3)} if image_urls else None,
        )

        if output_mode == "html":
            html_cfg = dict(cfg)
            html_cfg["delivery_mode"] = "html_image"
            prepared = await self.scheduler._build_report_payload(
                patched,
                html_cfg,
                cache_hit=False,
            )
            async for r in self._send_prepared_report_to_event(
                event,
                content=patched,
                prepared=prepared,
            ):
                yield r
            return

        yield event.plain_result(patched)

    @filter.command("news_url_test")
    async def news_url_test(self, event: AstrMessageEvent, args: str = ""):
        """
        使用一篇微信公众号文章 URL 作为临时单源测试入口。
        用法：/news_url_test <wechat_article_url> [images|layout] [plain|html] [preview] [max_urls]
        """
        parts = [p.strip() for p in (args or "").strip().split() if p.strip()]
        if not parts:
            yield event.plain_result(
                "用法：news_url_test <wechat_article_url> [images|layout] [plain|html] [preview] [max_urls]"
            )
            return
        if not self._is_likely_url(parts[0]):
            yield event.plain_result("news_url_test 只接受公众号文章 URL。")
            return
        async for item in self.news_source_test(event, args=args):
            yield item

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
        is_rss_like = (
            low.endswith((".xml", ".rss", ".atom"))
            or "/feed" in low
            or "rss.xml" in low
            or "atom.xml" in low
            or "format=rss" in low
            or "feed.xml" in low
        )
        if "miyoushe.com" in low:
            key = "miyoushe"
        elif is_rss_like:
            key = "rss"
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
        elif key == "rss":
            if not _exists("rss", url):
                items.append(
                    {
                        "__template_key": "rss",
                        "name": display_name,
                        "url": url,
                        "priority": 1,
                        "max_articles": 5,
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
