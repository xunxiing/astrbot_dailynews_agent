import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
import textwrap
from typing import List

from astrbot.api import logger as astrbot_logger
from astrbot.api import AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register

from .tools import (
    ImageUrlDownloadTool,
    ImageUrlsPreviewTool,
    WechatAlbumLatestArticlesTool,
    WechatArticleMarkdownTool,
)
from .workflow.scheduler import DailyNewsScheduler
from .workflow.rendering import load_template
from .workflow.image_utils import get_plugin_data_dir, merge_images_vertical
from .workflow.image_layout_agent import ImageLayoutAgent
from .workflow.models import SubAgentResult
from .workflow.config_models import RenderImageStyleConfig, RenderPipelineConfig
from .workflow.render_pipeline import render_daily_news_pages, split_pages

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


@register(
    "astrbot_dailynews_agent",
    "your_name",
    "AI 日报插件：定时抓取公众号最新内容，多 Agent 总结并自动推送",
    "0.2.0",
    "https://github.com/your/repo",
)
class DailyNewsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        tools = [
            WechatArticleMarkdownTool(),
            WechatAlbumLatestArticlesTool(),
            ImageUrlsPreviewTool(),
            ImageUrlDownloadTool(),
        ]

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
                astrbot_logger.error("[dailynews] failed to start scheduler", exc_info=True)

    async def terminate(self):
        if getattr(self, "scheduler", None) is not None:
            await self.scheduler.stop()

    # ====== commands ======

    async def _send_as_html_images(self, event: AstrMessageEvent, content: str):
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
            try:
                p = await self.html_render(DAILY_NEWS_HTML_TMPL, ctx, return_url=False)
                return Path(str(p)).resolve()
            except Exception:
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
            template_str=DAILY_NEWS_HTML_TMPL,
            render_html=_render_html,
            render_t2i=_render_t2i,
            pipeline=pipeline_cfg,
            style=style_cfg,
            title="每日资讯日报",
            subtitle_fmt="第 {idx}/{total} 页",
        )

        for r in rendered:
            if r.image_path is None or not _is_valid_image_file(Path(r.image_path).resolve()):
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
        content = await self.scheduler.generate_once()
        if not (content or "").strip():
            astrbot_logger.warning("[dailynews] /daily_news got empty content; sending fallback text")
            content = "生成失败：日报内容为空（可能是抓取/LLM 超时或被其它插件过滤），请查看 AstrBot 日志。"
        delivery_mode = str(self.config.get("delivery_mode", "html_image") or "html_image")
        if delivery_mode == "html_image":
            async for r in self._send_as_html_images(event, content):
                yield r
            return

        yield event.plain_result(content)

    @filter.command("news_toggle")
    async def news_toggle(self, event: AstrMessageEvent):
        """切换自动日报开关"""
        self.config["enabled"] = not bool(self.config.get("enabled", False))
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result(f"自动日报已{'开启' if self.config['enabled'] else '关闭'}")

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
            filler = "\n".join([f"- filler line {i}: {('内容' * 40)}" for i in range(1, 120)])
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
            (content[:260].replace("\n", "\\n") + ("..." if len(content) > 260 else "")),
        )

        delivery_mode = str(self.config.get("delivery_mode", "html_image") or "html_image")
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
        from .workflow.image_utils import get_plugin_data_dir, merge_images_vertical, parse_image_urls

        urls = parse_image_urls(args)
        if not urls:
            yield event.plain_result("用法：/news_image_preview url1 url2 url3 ...")
            return

        out_dir = get_plugin_data_dir("image_previews")
        out_path = out_dir / f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            merged = await merge_images_vertical(urls, out_path=out_path)
        except Exception as e:
            astrbot_logger.error("[dailynews] news_image_preview failed: %s", e, exc_info=True)
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
                astrbot_logger.warning("[dailynews] news_image_debug fetch list failed: %s", r, exc_info=True)
                source_articles[src.name] = []
                continue
            name, articles = r
            source_articles[name] = articles or []

        # 2) 抓正文并提取 image_urls（不走 LLM）
        from .workflow.utils import _run_sync
        from .analysis.wechatanalysis.analysis import fetch_wechat_article
        from .analysis.miyousheanalysis.analysis import fetch_miyoushe_post

        async def _collect_images(source_name: str, source_type: str, arts, max_articles: int):
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

        out_path = get_plugin_data_dir("image_previews") / f"debug_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            merged = await merge_images_vertical(all_for_preview[: max_urls], out_path=out_path)
            img_file = Path(str(merged)).resolve()
            if not _is_valid_image_file(img_file):
                yield event.plain_result("\n".join(lines + ["预览图生成失败：图片无效"]).strip())
                return
            p = img_file.as_posix()
            if _ImageComponent is not None:
                yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
            else:
                yield event.image_result(p)
        except Exception as e:
            astrbot_logger.error("[dailynews] news_image_debug preview failed: %s", e, exc_info=True)
            yield event.plain_result("\n".join(lines + [f"预览图生成失败：{e}"]).strip())

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
                astrbot_logger.warning("[dailynews] news_layout_test fetch list failed: %s", r, exc_info=True)
                source_articles[src.name] = []
                continue
            name, articles = r
            source_articles[name] = articles or []

        # 2) 抓正文并提取 image_urls（不走 LLM 写作）
        from .workflow.utils import _run_sync
        from .analysis.wechatanalysis.analysis import fetch_wechat_article
        from .analysis.miyousheanalysis.analysis import fetch_miyoushe_post

        async def _collect_images(source_name: str, source_type: str, arts, max_articles: int):
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
        targets: List[str] = list(self.config.get("target_sessions", []) or [])
        if event.unified_msg_origin not in targets:
            targets.append(event.unified_msg_origin)
        self.config["target_sessions"] = targets
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已订阅本会话的每日推送")

    @filter.command("news_unsubscribe")
    async def news_unsubscribe(self, event: AstrMessageEvent):
        """退订：把当前会话移出推送列表"""
        targets: List[str] = list(self.config.get("target_sessions", []) or [])
        self.config["target_sessions"] = [x for x in targets if x != event.unified_msg_origin]
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
            yield event.plain_result("用法：/news_add_source URL（或 /news_add_source 名称 URL）")
            return

        url = parts[0].strip() if len(parts) == 1 else parts[1].strip()
        if not url:
            yield event.plain_result("URL 不能为空")
            return

        sources: List[str] = list(self.config.get("news_sources", []) or [])
        if url not in sources:
            sources.append(url)
        self.config["news_sources"] = sources
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已添加来源")

    @filter.command("news_remove_source")
    async def news_remove_source(self, event: AstrMessageEvent, name: str):
        """删除新闻源: /news_remove_source URL"""
        url = (name or "").strip()
        if not url:
            yield event.plain_result("用法：/news_remove_source URL")
            return

        sources: List[str] = list(self.config.get("news_sources", []) or [])
        self.config["news_sources"] = [x for x in sources if x != url]
        if hasattr(self.config, "save_config"):
            self.config.save_config()
        yield event.plain_result("已删除来源")
