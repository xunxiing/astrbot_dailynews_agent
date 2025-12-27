import json
from pathlib import Path
from typing import List

from astrbot.api import logger as astrbot_logger
from astrbot.api import AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register

from .tools import WechatAlbumLatestArticlesTool, WechatArticleMarkdownTool
from .workflow.scheduler import DailyNewsScheduler


DAILY_NEWS_HTML_TMPL = """
<div style="width: 980px; padding: 36px 44px; background: #ffffff; color: #111; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;">
  <div style="font-size: 30px; font-weight: 800; margin-bottom: 10px;">{{ title | e }}</div>
  <div style="font-size: 14px; color: #666; margin-bottom: 18px;">{{ subtitle | e }}</div>
  <div style="border-top: 1px solid #eee; margin: 14px 0 18px;"></div>
  <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 15px; line-height: 1.55; margin: 0;">{{ body | e }}</pre>
</div>
""".strip()


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

        for idx, page in enumerate(pages, start=1):
            try:
                # 用 return_url=False 拿到本地图片路径，避免平台把 URL 当成“链接卡片”导致空白预览
                img_path = await self.html_render(
                    DAILY_NEWS_HTML_TMPL,
                    {"title": "每日资讯日报", "subtitle": f"第 {idx}/{len(pages)} 页", "body": page},
                    return_url=False,
                )
                astrbot_logger.info("[dailynews] rendered page %s/%s -> %s", idx, len(pages), img_path)
                # 兼容 Napcat/aiocqhttp：避免生成 file:///C:\\... 这种不规范 file URI
                yield event.image_result(Path(str(img_path)).resolve().as_posix())
            except Exception:
                astrbot_logger.error("[dailynews] html_render failed; fallback to text_to_image", exc_info=True)
                try:
                    img_path = await self.text_to_image(page, return_url=False)
                    astrbot_logger.info(
                        "[dailynews] fallback text_to_image page %s/%s -> %s",
                        idx,
                        len(pages),
                        img_path,
                    )
                    yield event.image_result(Path(str(img_path)).resolve().as_posix())
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
