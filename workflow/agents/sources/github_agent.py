from __future__ import annotations

import json
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.utils import _json_from_text
from .github_source import fetch_github_snapshot_for_source, updates_from_snapshot


class GitHubSubAgent:
    """GitHub 子 Agent：通过 github_source 抓取更新快照，然后写入日报小节。"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        snap = await fetch_github_snapshot_for_source(
            source=source, user_config=user_config
        )
        if not isinstance(snap, dict):
            return source.name, []
        updates = updates_from_snapshot(snap)
        return source.name, [{"snapshot": snap, "updates": updates}]

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]], llm: LLMRunner
    ) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        updates: list[dict[str, Any]] = []
        if articles and isinstance(articles[0], dict):
            snap = articles[0].get("snapshot") or {}
            updates = articles[0].get("updates") or []

        commits = len(
            (snap.get("commits_recent") or []) if isinstance(snap, dict) else []
        )
        prs = len((snap.get("prs_recent") or []) if isinstance(snap, dict) else [])
        rels = len(
            (snap.get("releases_recent") or []) if isinstance(snap, dict) else []
        )
        hours = 30
        try:
            window = snap.get("window") if isinstance(snap.get("window"), dict) else {}
            if isinstance(window, dict):
                hours = int(window.get("hours") or 30)
        except Exception:
            hours = 30
        angle = f"近{hours}小时：{rels}个Release/{commits}个Commit/{prs}个PR更新"

        topics = []
        if rels:
            topics.append("release")
        if commits:
            topics.append("commit")
        if prs:
            topics.append("pr")

        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": len(updates),
            "topics": topics,
            "quality_score": rels * 8 + prs * 2 + commits,
            "today_angle": angle,
            "sample_articles": updates[:3],
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: list[dict[str, Any]],
        llm: LLMRunner,
        user_config: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        if not articles or not isinstance(articles[0], dict):
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                error="GitHub 来源未抓取到任何数据（检查 github_enabled/github_repos/github_token）",
            )
        updates = articles[0].get("updates") or []
        if not isinstance(updates, list):
            updates = []
        # No recent updates -> omit this section (do not output "no updates" noise).
        if not updates:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                images=None,
                error=None,
            )
        snap = articles[0].get("snapshot") or {}
        if not isinstance(snap, dict):
            snap = {}
        hours = 30
        try:
            window = snap.get("window") if isinstance(snap.get("window"), dict) else {}
            if isinstance(window, dict):
                hours = int(window.get("hours") or 30)
        except Exception:
            hours = 30

        system_prompt = (
            "你是子Agent（写作）。"
            f"你会收到一个 GitHub 仓库近 {hours} 小时的更新快照（release/commit/pr）以及仓库当前版本信息。"
            "请写出该来源在今日日报中的一段 Markdown 小节（含小标题、要点、链接）。"
            "需要覆盖：当前版本/最新 release（如有）/最新 commit/近窗口内 PR 与 commit 摘要。"
            "只输出 JSON，不要输出其它文本。"
        )
        system_prompt += (
            "\n\nCRITICAL OUTPUT RULES (must follow):\n"
            "1) Never output raw URLs (no lines starting with http/https). All links must be Markdown links like [View](URL).\n"
            "2) Ban vague filler like “optimized experience / fixed some bugs”. Every bullet must state concrete changes: feature name, behavior change, bug symptom + fix, and reference (#PR/#issue/commit sha) when available.\n"
            "3) Structure per item:\n"
            "   - **Title**: 1-line summary. ( [View](url) )\n"
            "     - Details: include PR/issue number or short commit hash; include parameters/metrics if present.\n"
            "4) If the snapshot lacks concrete details, output an empty section_markdown (do NOT make up content).\n"
        )
        prompt = {
            "source_name": source.name,
            "instruction": instruction,
            "github": snap,
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

        try:
            raw = await llm.ask(
                system_prompt=system_prompt,
                prompt=json.dumps(prompt, ensure_ascii=False),
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] github subagent write failed, fallback: %s",
                e,
                exc_info=True,
            )
            repo = snap.get("repo") if isinstance(snap.get("repo"), dict) else {}
            lines = [f"## {source.name}", ""]
            if isinstance(repo, dict):
                full = str(repo.get("full_name") or "")
                url = str(repo.get("html_url") or source.url)
                ver = str(snap.get("version") or "")
                if full:
                    lines.append(f"- 仓库：[{full}]({url})")
                if ver:
                    lines.append(f"- 当前版本：`{ver}`")
            for k, key in [
                ("Release", "releases_recent"),
                ("Commit", "commits_recent"),
                ("PR", "prs_recent"),
            ]:
                items = snap.get(key) if isinstance(snap.get(key), list) else []
                if items:
                    lines.append(f"- 最近更新（{k}）：{len(items)} 条")
            return SubAgentResult(
                source_name=source.name,
                content="\n".join(lines).strip(),
                summary="",
                key_points=[],
                images=None,
                error=None,
            )

        data = _json_from_text(raw)
        if not isinstance(data, dict):
            return SubAgentResult(
                source_name=source.name,
                content=str(raw),
                summary="",
                key_points=[],
                images=None,
                error=None,
            )

        summary = str(data.get("summary") or "")
        key_points = data.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        section = str(data.get("section_markdown") or "")
        return SubAgentResult(
            source_name=source.name,
            content=section,
            summary=summary,
            key_points=[str(x) for x in key_points[:10]],
            images=None,
            error=None,
        )
