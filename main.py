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

from .tools import WechatAlbumLatestArticlesTool, WechatArticleMarkdownTool
from .workflow.scheduler import DailyNewsScheduler
from .workflow.rendering import load_template, markdown_to_html, safe_text

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
    "astrbot_plugin_dailynews",
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

    def _get_image_b64(self, filename: str) -> str:
        # 尝试定位 image 目录
        # 1. 相对于当前文件 main.py -> ./image
        base_dir = Path(__file__).parent
        img_path = base_dir / "image" / filename
        
        if not img_path.exists():
            # 2. 相对于 cwd
            img_path = Path.cwd() / "image" / filename

        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
        return ""

    def _split_pages(self, text: str, page_chars: int, max_pages: int) -> List[str]:
        s = (text or "").strip()
        if not s:
            return []
        if page_chars <= 0:
            return [s]

        pages: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for line in s.splitlines():
            piece = line + "\n"
            if buf_len + len(piece) > page_chars and buf:
                pages.append("".join(buf).rstrip())
                if len(pages) >= max_pages:
                    return pages
                buf, buf_len = [], 0
            buf.append(piece)
            buf_len += len(piece)

        if buf and len(pages) < max_pages:
            pages.append("".join(buf).rstrip())
        return pages

    async def _send_as_html_images(self, event: AstrMessageEvent, content: str):
        page_chars = int(self.config.get("render_page_chars", 2600) or 2600)
        max_pages = int(self.config.get("render_max_pages", 4) or 4)
        pages = self._split_pages(content, page_chars=page_chars, max_pages=max_pages)
        if not pages:
            yield event.plain_result("生成失败：日报内容为空，请查看 AstrBot 日志。")
            return

        bg_img = self._get_image_b64("sunsetbackground.jpg")
        char_img = self._get_image_b64("transparent_output.png")

        for idx, page in enumerate(pages, start=1):
            try:
                # 用 return_url=False 拿到本地图片路径，避免平台把 URL 当成“链接卡片”导致空白预览
                img_path = await self.html_render(
                    DAILY_NEWS_HTML_TMPL,
                    {
                        "title": safe_text("每日资讯日报"),
                        "subtitle": safe_text(f"第 {idx}/{len(pages)} 页"),
                        "body_html": markdown_to_html(page),
                        "bg_img": bg_img,
                        "char_img": char_img,
                    },
                    return_url=False,
                )
                astrbot_logger.info("[dailynews] rendered page %s/%s -> %s", idx, len(pages), img_path)
                # 兼容 Napcat/aiocqhttp：避免生成 file:///C:\\... 这种不规范 file URI
                img_file = Path(str(img_path)).resolve()
                if not _is_valid_image_file(img_file):
                    astrbot_logger.error(
                        "[dailynews] html_render returned invalid image file: %s (size=%s)",
                        img_file,
                        img_file.stat().st_size if img_file.exists() else -1,
                    )
                    raise RuntimeError("html_render invalid image")

                p = img_file.as_posix()
                if _ImageComponent is not None:
                    yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
                else:
                    yield event.image_result(p)
            except Exception:
                astrbot_logger.error("[dailynews] html_render failed; fallback to text_to_image", exc_info=True)
                try:
                    template_name = self.context._config.get("t2i_active_template")

                    # 优先走 AstrBot 内置渲染：先尝试默认（可能是网络渲染），若返回的“图片”文件不合法则强制本地渲染。
                    renderer = _astrbot_html_renderer
                    if renderer is None:
                        try:
                            from astrbot.core import html_renderer as renderer  # type: ignore
                        except Exception:
                            renderer = None

                    if renderer is not None:
                        img_path = await renderer.render_t2i(
                            page,
                            return_url=False,
                            template_name=template_name,
                        )
                        img_file = Path(str(img_path)).resolve()
                        if not _is_valid_image_file(img_file):
                            astrbot_logger.error(
                                "[dailynews] render_t2i returned invalid image file: %s (size=%s); retry local",
                                img_file,
                                img_file.stat().st_size if img_file.exists() else -1,
                            )
                            img_path = await renderer.render_t2i(
                                page,
                                use_network=False,
                                return_url=False,
                                template_name=template_name,
                            )
                    else:
                        img_path = await self.text_to_image(page, return_url=False)

                    astrbot_logger.info(
                        "[dailynews] fallback render_t2i page %s/%s -> %s",
                        idx,
                        len(pages),
                        img_path,
                    )
                    img_file = Path(str(img_path)).resolve()
                    if not _is_valid_image_file(img_file):
                        astrbot_logger.error(
                            "[dailynews] fallback render returned invalid image file: %s (size=%s)",
                            img_file,
                            img_file.stat().st_size if img_file.exists() else -1,
                        )
                        yield event.plain_result(page)
                        continue

                    p = img_file.as_posix()
                    if _ImageComponent is not None:
                        yield event.chain_result([_ImageComponent(file=f"file:///{p}", path=p)])
                    else:
                        yield event.image_result(p)
                except Exception:
                    astrbot_logger.error("[dailynews] text_to_image fallback failed", exc_info=True)
                    yield event.plain_result("生成失败：网页渲染失败，请查看 AstrBot 日志。")
                    return

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
