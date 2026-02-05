import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    import sys
    from pathlib import Path
    # workflow/agents/sources/twitter_agent.py -> parents[3] is plugin root
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.append(root)
    from analysis.twitteranalysis.latest_tweets import fetch_latest_tweets
except Exception:  # pragma: no cover
    from analysis.twitteranalysis.latest_tweets import fetch_latest_tweets  # type: ignore

from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...pipeline.playwright_bootstrap import get_chromium_executable_path
from ...core.utils import _json_from_text


class TwitterSubAgent:
    """X/Twitter 子 Agent：抓取主页最新推文（含图片）并生成 Markdown 小节。"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, str]]]:
        limit = int(user_config.get("twitter_max_tweets", source.max_articles or 3) or 3)
        meta = source.meta if isinstance(source.meta, dict) else {}
        proxy = str(meta.get("proxy") or user_config.get("twitter_proxy", "") or "").strip()
        exe = get_chromium_executable_path()

        tweets = await fetch_latest_tweets(
            source.url,
            limit=max(1, min(limit, 6)),
            proxy=proxy,
            executable_path=str(exe) if exe else None,
        )

        articles: List[Dict[str, str]] = []
        for t in tweets:
            if not isinstance(t, dict):
                continue
            text = str(t.get("text") or "").strip()
            url = str(t.get("url") or source.url).strip()
            title = text.replace("\n", " ").strip()
            if len(title) > 60:
                title = title[:60] + "…"
            articles.append({"title": title, "url": url})

        if not articles:
            articles = [{"title": "", "url": source.url}]

        return source.name, articles

    async def analyze_source(
        self, source: NewsSourceConfig, articles: List[Dict[str, str]], llm: LLMRunner
    ) -> Dict[str, Any]:
        system_prompt = (
            "你是子Agent（信息侦察）。你会收到某个 X/Twitter 主页最近推文的标题与链接。"
            "请判断今日值得关注的话题，并给出可写作角度建议。只输出 JSON。"
        )
        prompt = {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "latest_posts": articles[:10],
            "output_schema": {
                "source_name": source.name,
                "topics": ["topic"],
                "quality_score": 0,
                "today_angle": "string",
            },
        }
        raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        data = _json_from_text(raw) or {}
        data["source_name"] = source.name
        data["source_type"] = source.type
        return data

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: List[Dict[str, str]],
        llm: LLMRunner,
        user_config: Dict[str, Any] | None = None,
    ) -> SubAgentResult:
        limit = max(1, int(source.max_articles or 3))
        meta = source.meta if isinstance(source.meta, dict) else {}
        proxy = str(meta.get("proxy") or (user_config or {}).get("twitter_proxy", "") or "").strip()

        exe = get_chromium_executable_path()

        tweets = []
        try:
            tweets = await fetch_latest_tweets(
                source.url,
                limit=min(limit, 6),
                proxy=proxy,
                executable_path=str(exe) if exe else None,
            )
        except Exception as e:
            msg = str(e) or type(e).__name__
            if "swap `twitter_targets` and `twitter_proxy`" in msg:
                astrbot_logger.warning(
                    "[dailynews] twitter fetch failed: %s. Fix config: set `twitter_targets` to https://x.com/<user> and `twitter_proxy` to socks5/http proxy.",
                    msg,
                )
            else:
                astrbot_logger.warning("[dailynews] twitter fetch failed: %s", msg, exc_info=True)
            tweets = []

        images: List[str] = []
        seen = set()
        for t in tweets:
            for u in (t.get("image_urls") or []) if isinstance(t, dict) else []:
                if isinstance(u, str) and u and u not in seen:
                    seen.add(u)
                    images.append(u)

        system_prompt = (
            "你是子Agent（写作）。你会收到 X/Twitter 主页最近推文的内容摘要。"
            "请写出今日日报中的一个 Markdown 小节：包含小标题、要点、并附上推文链接。"
            "风格轻松、像科技/游戏博主。只输出 JSON。"
             "请你务必使用中文进行总结"

        )
        prompt = {
            "source_name": source.name,
            "instruction": instruction,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tweets": tweets[:limit],
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

        raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        data = _json_from_text(raw) or {}
        section = str(data.get("section_markdown") or "").strip()
        summary = str(data.get("summary") or "").strip()
        key_points = data.get("key_points") if isinstance(data.get("key_points"), list) else []
        key_points = [str(x) for x in key_points if isinstance(x, (str, int, float))][:10]

        if not section:
            # fallback: minimal section without LLM
            lines = [f"## {source.name}", ""]
            for t in tweets[:limit]:
                if not isinstance(t, dict):
                    continue
                text = str(t.get("text") or "").strip() or "_无文本_"
                url = str(t.get("url") or "").strip()
                if url:
                    lines.append(f"- {text}（{url}）")
                else:
                    lines.append(f"- {text}")
            section = "\n".join(lines).strip()

        return SubAgentResult(
            source_name=source.name,
            content=section,
            summary=summary,
            key_points=key_points,
            images=images or None,
            error=None,
        )
