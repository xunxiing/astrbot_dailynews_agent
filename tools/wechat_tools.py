import asyncio
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

# 直接复用你写好的脚本逻辑
from ..analysis.wechatanalysis.analysis import wechat_to_markdown
from ..analysis.wechatanalysis.latest_articles import get_album_articles

try:
    from astrbot.api.star import StarTools
except Exception:  # pragma: no cover
    StarTools = None  # type: ignore


# 插件根目录：.../astrbot_plugin_dailynews
PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# 默认把 Markdown 输出到 analysis/wechatanalysis/output
def _default_output_dir() -> Path:
    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir()) / "wechat_markdown"
        except Exception:
            pass
    return PLUGIN_ROOT / "analysis" / "wechatanalysis" / "output"


async def _run_in_thread(func, *args, **kwargs):
    """
    通用的同步 -> 异步封装：
    把阻塞的函数放到线程池里执行，避免阻塞事件循环。
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ========== Tool 1: 解析单篇公众号为 Markdown ==========

@dataclass
class WechatArticleMarkdownTool(FunctionTool[AstrAgentContext]):
    """
    将单篇微信公众号文章解析为 Markdown，并保存到插件目录下。
    """
    name: str = "wechat_article_to_markdown"
    description: str = (
        "解析微信公众号文章，将其保存为本地 Markdown 文件，并返回文件路径。"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "微信公众号文章链接。",
                },
                "max_concurrency": {
                    "type": "integer",
                    "description": "下载图片时的最大并发数，默认 10。",
                },
            },
            "required": ["url"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        # 兼容一下 url / article_url 两种字段名
        url = kwargs.get("url") or kwargs.get("article_url")
        if not url:
            return "参数错误：必须提供 url 字段（微信公众号文章链接）。"

        max_concurrency = kwargs.get("max_concurrency", 10)
        try:
            max_concurrency = int(max_concurrency)
        except (TypeError, ValueError):
            max_concurrency = 10

        output_dir = _default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 在线程池里跑你原来的同步逻辑，避免阻塞事件循环
        md_path = await _run_in_thread(
            wechat_to_markdown,
            url,
            str(output_dir),
            max_concurrency,
        )

        # ToolExecResult 可以是字符串，这里就简单返回路径，
        # LLM 拿到后可以继续读文件 / 做后续处理
        return f"已将公众号文章解析为 Markdown 文件：{md_path}"


# ========== Tool 2: 获取『AI 早报 · 目录』最近几篇文章 ==========

@dataclass
class WechatAlbumLatestArticlesTool(FunctionTool[AstrAgentContext]):
    """
    从『AI 早报』专辑目录中抓取最近的若干篇文章信息。
    """
    name: str = "wechat_album_latest_articles"
    description: str = (
        "从微信公众号的『AI 早报』专辑目录中抓取最近的若干篇文章标题和链接。"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "专辑中任意一篇文章的链接（从这个页面能点开『AI 早报 · 目录』）。",
                },
                "album_keyword": {
                    "type": "string",
                    "description": "可选：目录入口名称匹配关键字，不填则默认点击第一个目录入口",
                },
                "limit": {
                    "type": "integer",
                    "description": "要获取的文章数量，默认 5。",
                },
            },
            "required": ["url"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        url = kwargs.get("url")
        if not url:
            return "参数错误：必须提供 url 字段（任意一篇所属专辑的文章链接）。"

        limit = kwargs.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5

        # 这里同样在线程池中运行同步的 Playwright 逻辑
        try:
            articles: List[Dict[str, Any]] = await _run_in_thread(
                get_album_articles,
                url,
                limit,
                album_keyword=kwargs.get("album_keyword"),
            )
        except Exception as e:
            return f"获取专辑文章列表失败：{e}"

        if not articles:
            return "没有从目录弹窗里解析到任何文章，请确认该链接属于『AI 早报』专辑。"

        lines = [f"共获取到最近 {len(articles)} 篇文章：", ""]
        for i, a in enumerate(articles, start=1):
            title = a.get("title", "")
            link = a.get("url", "")
            lines.append(f"{i}. {title}\n   {link}")

        return "\n".join(lines)
