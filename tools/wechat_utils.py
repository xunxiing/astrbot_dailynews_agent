import asyncio
from pathlib import Path
from typing import List, Dict

# 这里用相对导入，你需要保证 astrbot_plugin_dailynews、analysis、
# analysis/wechatanalysis、tools 这些目录都有 __init__.py
from ..analysis.wechatanalysis.analysis import wechat_to_markdown as _wechat_to_markdown
from ..analysis.wechatanalysis.latest_articles import get_album_articles as _get_album_articles


# 插件根目录：.../astrbot_plugin_dailynews
PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# 统一把公众号的 md 输出到 analysis/wechatanalysis/output 目录
WECHAT_OUTPUT_DIR = PLUGIN_ROOT / "analysis" / "wechatanalysis" / "output"


async def async_wechat_to_markdown(url: str) -> str:
    """
    在异步环境里安全地调用你原来的 wechat_to_markdown()，
    避免在已有 event loop 里直接 asyncio.run() 报错。
    """
    WECHAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    def _run():
        # 这里复用你原始的逻辑，只是把输出目录指定为 WECHAT_OUTPUT_DIR
        return _wechat_to_markdown(
            url,
            output_dir=str(WECHAT_OUTPUT_DIR),
            max_concurrency=10,
        )

    md_path = await loop.run_in_executor(None, _run)
    return md_path


def get_latest_wechat_articles(article_url: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    使用你写的 get_album_articles() 从『AI 早报』专辑弹窗里拿最近几篇文章。
    """
    return _get_album_articles(article_url, limit=limit)