import asyncio
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

try:
    from ..analysis.wechatanalysis.analysis import wechat_to_markdown
    from ..analysis.wechatanalysis.latest_articles import get_album_articles
except (ImportError, ValueError):
    import sys

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.append(root)
    from analysis.wechatanalysis.analysis import wechat_to_markdown
    from analysis.wechatanalysis.latest_articles import get_album_articles

try:
    from astrbot.api.star import StarTools
except Exception:  # pragma: no cover
    StarTools = None  # type: ignore


PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _default_output_dir() -> Path:
    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir()) / "wechat_markdown"
        except Exception:
            pass
    return PLUGIN_ROOT / "analysis" / "wechatanalysis" / "output"


async def _run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


@dataclass
class WechatArticleMarkdownTool(FunctionTool[AstrAgentContext]):
    """Parse a single WeChat article and save as local markdown."""

    name: str = "wechat_article_to_markdown"
    description: str = "解析微信公众号文章并保存为本地 Markdown 文件。"
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "微信公众号文章链接。",
                },
            },
            "required": ["url"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        _ = context
        url = kwargs.get("url") or kwargs.get("article_url")
        if not url:
            return "参数错误：必须提供 url 字段（微信公众号文章链接）。"

        output_dir = _default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = await _run_in_thread(
            wechat_to_markdown,
            str(url),
            str(output_dir),
        )
        return f"已将公众号文章解析为 Markdown 文件：{md_path}"


@dataclass
class WechatAlbumLatestArticlesTool(FunctionTool[AstrAgentContext]):
    """Fetch latest wechat articles by account/album API from a seed article URL."""

    name: str = "wechat_album_latest_articles"
    description: str = "获取公众号最新文章列表（支持 auto/account/album）。"
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "任意一篇公众号文章链接。",
                },
                "album_keyword": {
                    "type": "string",
                    "description": "保留参数（当前 HTTP API 模式下不使用）。",
                },
                "limit": {
                    "type": "integer",
                    "description": "要获取的文章数量，默认 5。",
                },
                "latest_scope": {
                    "type": "string",
                    "description": "latest source scope: auto/account/album",
                },
            },
            "required": ["url"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        _ = context
        url = kwargs.get("url")
        if not url:
            return "参数错误：必须提供 url 字段（公众号文章链接）。"

        limit = kwargs.get("limit", 5)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5

        try:
            articles: list[dict[str, Any]] = await _run_in_thread(
                get_album_articles,
                str(url),
                int(limit),
                album_keyword=kwargs.get("album_keyword"),
                latest_scope=str(kwargs.get("latest_scope") or "auto"),
            )
        except Exception as e:
            return f"获取公众号文章列表失败：{e}"

        if not articles:
            return "未获取到文章，请确认链接可访问且 latest_scope 配置正确（auto/account/album）。"

        lines = [f"共获取到最近 {len(articles)} 篇文章：", ""]
        for i, a in enumerate(articles, start=1):
            title = a.get("title", "")
            link = a.get("url", "")
            lines.append(f"{i}. {title}\n   {link}")

        return "\n".join(lines)
