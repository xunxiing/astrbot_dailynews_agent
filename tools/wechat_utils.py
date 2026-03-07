import asyncio
from pathlib import Path

# 这里用相对导入，你需要保证 astrbot_plugin_dailynews、analysis、
# analysis/wechatanalysis、tools 这些目录都有 __init__.py
from ..analysis.wechatanalysis.analysis import wechat_to_markdown as _wechat_to_markdown
from ..analysis.wechatanalysis.latest_articles import (
    get_album_articles as _get_album_articles,
)

try:
    from astrbot.api.star import StarTools
except Exception:  # pragma: no cover
    StarTools = None  # type: ignore


# 插件根目录：.../astrbot_plugin_dailynews
PLUGIN_ROOT = Path(__file__).resolve().parent.parent


# 统一把公众号的 md 输出到 analysis/wechatanalysis/output 目录
def _wechat_output_dir() -> Path:
    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir()) / "wechat_markdown"
        except Exception:
            pass
    return PLUGIN_ROOT / "analysis" / "wechatanalysis" / "output"


async def async_wechat_to_markdown(url: str) -> str:
    """
    在异步环境里安全地调用 wechat_to_markdown()，
    避免在已有 event loop 里直接 asyncio.run() 报错。
    """
    out_dir = _wechat_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    def _run():
        # 复用公众号 HTTP 解析逻辑，并把输出目录固定到 WECHAT_OUTPUT_DIR。
        return _wechat_to_markdown(
            url,
            output_dir=str(out_dir),
        )

    md_path = await loop.run_in_executor(None, _run)
    return md_path


def get_latest_wechat_articles(
    article_url: str,
    limit: int = 5,
    album_keyword: str | None = None,
    latest_scope: str = "auto",
) -> list[dict[str, str]]:
    """
    使用 HTTP API 获取公众号最近文章（账号/合集，取决于 latest_scope）。
    """
    return _get_album_articles(
        article_url,
        limit=limit,
        album_keyword=album_keyword,
        latest_scope=latest_scope,
    )
