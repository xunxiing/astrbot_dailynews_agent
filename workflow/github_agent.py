from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .llm import LLMRunner
from .models import NewsSourceConfig, SubAgentResult
from .utils import _json_from_text


_REPO_RE = re.compile(
    r"""
    (?:
        https?://github\.com/
    )?
    (?P<owner>[A-Za-z0-9_.-]+)/
    (?P<repo>[A-Za-z0-9_.-]+)
    (?:\.git)?
    (?:/.*)?$
    """,
    re.X,
)


def _parse_repo(s: str) -> Optional[Tuple[str, str]]:
    m = _REPO_RE.match((s or "").strip())
    if not m:
        return None
    owner = (m.group("owner") or "").strip()
    repo = (m.group("repo") or "").strip()
    if not owner or not repo:
        return None
    return owner, repo


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _first_line(s: str) -> str:
    return (s or "").strip().splitlines()[0].strip() if (s or "").strip() else ""


@dataclass(frozen=True)
class GitHubConfig:
    enabled: bool
    token: str
    since_hours: int
    max_commits: int
    max_prs: int
    max_releases: int

    @classmethod
    def from_user_config(cls, cfg: Dict[str, Any]) -> "GitHubConfig":
        def _to_bool(v: Any, default: bool) -> bool:
            if v is None:
                return bool(default)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v).strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
            return bool(default)

        def _to_int(v: Any, default: int) -> int:
            try:
                if v is None or isinstance(v, bool):
                    return int(default)
                return int(v)
            except Exception:
                return int(default)

        token = str(cfg.get("github_token") or "").strip()
        return cls(
            enabled=_to_bool(cfg.get("github_enabled"), False),
            token=token,
            since_hours=max(1, _to_int(cfg.get("github_since_hours"), 30)),
            max_commits=max(0, _to_int(cfg.get("github_max_commits"), 6)),
            max_prs=max(0, _to_int(cfg.get("github_max_prs"), 6)),
            max_releases=max(0, _to_int(cfg.get("github_max_releases"), 3)),
        )


class GitHubClient:
    def __init__(self, *, token: str):
        self._token = (token or "").strip()

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "astrbot-dailynews-agent/1.0",
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _get_json(self, session: aiohttp.ClientSession, url: str, *, attempts: int = 3) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(1, max(1, int(attempts)) + 1):
            try:
                async with session.get(url, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    # common transient / rate-limit statuses
                    if resp.status in {429, 500, 502, 503, 504}:
                        await asyncio.sleep(min(1.5 * attempt, 6.0))
                        continue
                    if resp.status == 403:
                        remaining = (resp.headers.get("X-RateLimit-Remaining") or "").strip()
                        reset = (resp.headers.get("X-RateLimit-Reset") or "").strip()
                        if remaining == "0" and reset.isdigit():
                            now = int(datetime.now(tz=timezone.utc).timestamp())
                            wait_s = max(1, min(int(reset) - now + 1, 10))
                            await asyncio.sleep(wait_s)
                            continue
                    txt = await resp.text()
                    raise RuntimeError(f"GET {url} -> {resp.status}: {txt[:180]}")
            except Exception as e:
                last_err = e if isinstance(e, Exception) else Exception(str(e))
                if attempt < max(1, int(attempts)):
                    await asyncio.sleep(min(1.5 * attempt, 6.0))
                    continue
        raise RuntimeError(str(last_err) if last_err else "github request failed")

    async def fetch_repo_snapshot(
        self,
        *,
        owner: str,
        repo: str,
        since: datetime,
        max_commits: int,
        max_prs: int,
        max_releases: int,
    ) -> Dict[str, Any]:
        base = f"https://api.github.com/repos/{owner}/{repo}"
        since_iso = _iso_utc(since)
        async with aiohttp.ClientSession() as session:
            repo_info = await self._get_json(session, base)
            html_url = str(repo_info.get("html_url") or f"https://github.com/{owner}/{repo}")
            default_branch = str(repo_info.get("default_branch") or "main")

            releases_url = f"{base}/releases?per_page=10"
            tags_url = f"{base}/tags?per_page=1"
            commits_url = f"{base}/commits?per_page=30&sha={default_branch}"
            pulls_url = f"{base}/pulls?state=all&sort=updated&direction=desc&per_page=30"

            rels, tags, commits, pulls = await asyncio.gather(
                self._get_json(session, releases_url),
                self._get_json(session, tags_url),
                self._get_json(session, commits_url),
                self._get_json(session, pulls_url),
            )

        releases_all = rels if isinstance(rels, list) else []
        tags_all = tags if isinstance(tags, list) else []
        commits_all = commits if isinstance(commits, list) else []
        pulls_all = pulls if isinstance(pulls, list) else []

        def _dt(s: str) -> Optional[datetime]:
            if not s:
                return None
            try:
                # github uses Z
                if s.endswith("Z"):
                    s = s.replace("Z", "+00:00")
                return datetime.fromisoformat(s).astimezone(timezone.utc)
            except Exception:
                return None

        releases_recent: List[Dict[str, Any]] = []
        for r in releases_all:
            if not isinstance(r, dict):
                continue
            published_at = _dt(str(r.get("published_at") or "")) or _dt(str(r.get("created_at") or ""))
            if published_at is None or published_at < since:
                continue
            releases_recent.append(
                {
                    "name": str(r.get("name") or "") or str(r.get("tag_name") or ""),
                    "tag": str(r.get("tag_name") or ""),
                    "url": str(r.get("html_url") or ""),
                    "published_at": _iso_utc(published_at),
                    "prerelease": bool(r.get("prerelease") or False),
                }
            )
            if max_releases > 0 and len(releases_recent) >= max_releases:
                break

        latest_release = None
        if releases_all:
            r0 = releases_all[0] if isinstance(releases_all[0], dict) else None
            if r0:
                latest_release = {
                    "name": str(r0.get("name") or "") or str(r0.get("tag_name") or ""),
                    "tag": str(r0.get("tag_name") or ""),
                    "url": str(r0.get("html_url") or ""),
                    "published_at": str(r0.get("published_at") or ""),
                    "prerelease": bool(r0.get("prerelease") or False),
                }

        latest_tag = ""
        if tags_all and isinstance(tags_all[0], dict):
            latest_tag = str(tags_all[0].get("name") or "").strip()

        commits_recent: List[Dict[str, Any]] = []
        latest_commit = None
        for c in commits_all:
            if not isinstance(c, dict):
                continue
            commit = c.get("commit") if isinstance(c.get("commit"), dict) else {}
            cdate = _dt(str((commit or {}).get("committer", {}).get("date") or ""))  # type: ignore
            msg = _first_line(str((commit or {}).get("message") or ""))
            url = str(c.get("html_url") or "")
            sha = str(c.get("sha") or "")
            if latest_commit is None:
                latest_commit = {
                    "sha": sha[:7],
                    "message": msg,
                    "url": url,
                    "date": _iso_utc(cdate) if cdate else "",
                }
            if cdate is not None and cdate >= since and msg:
                commits_recent.append(
                    {
                        "sha": sha[:7],
                        "message": msg,
                        "url": url,
                        "date": _iso_utc(cdate),
                    }
                )
                if max_commits > 0 and len(commits_recent) >= max_commits:
                    break

        prs_recent: List[Dict[str, Any]] = []
        for pr in pulls_all:
            if not isinstance(pr, dict):
                continue
            updated = _dt(str(pr.get("updated_at") or ""))
            if updated is None or updated < since:
                continue
            title = str(pr.get("title") or "").strip()
            url = str(pr.get("html_url") or "")
            prs_recent.append(
                {
                    "title": title,
                    "url": url,
                    "state": str(pr.get("state") or ""),
                    "updated_at": _iso_utc(updated),
                }
            )
            if max_prs > 0 and len(prs_recent) >= max_prs:
                break

        version = ""
        if isinstance(latest_release, dict) and str(latest_release.get("tag") or "").strip():
            version = str(latest_release.get("tag") or "").strip()
        elif latest_tag:
            version = latest_tag
        elif isinstance(latest_commit, dict) and str(latest_commit.get("sha") or "").strip():
            version = str(latest_commit.get("sha") or "").strip()

        return {
            "repo": {
                "full_name": str(repo_info.get("full_name") or f"{owner}/{repo}"),
                "html_url": html_url,
                "description": str(repo_info.get("description") or ""),
                "default_branch": default_branch,
                "stars": int(repo_info.get("stargazers_count") or 0),
                "forks": int(repo_info.get("forks_count") or 0),
                "language": str(repo_info.get("language") or ""),
            },
            "window": {
                "since": since_iso,
                "hours": int((datetime.now(tz=timezone.utc) - since).total_seconds() // 3600),
            },
            "version": version,
            "latest_release": latest_release,
            "releases_recent": releases_recent,
            "latest_commit": latest_commit,
            "commits_recent": commits_recent,
            "prs_recent": prs_recent,
        }


class GitHubSubAgent:
    """GitHub 子 Agent：读取近 N 小时 release/commit/pr，并写入日报小节。"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        cfg = GitHubConfig.from_user_config(user_config)
        if not cfg.enabled:
            return source.name, []
        repo = _parse_repo(source.url)
        if not repo:
            astrbot_logger.warning("[dailynews] github invalid repo url: %s", source.url)
            return source.name, []
        owner, name = repo
        since = datetime.now(tz=timezone.utc) - timedelta(hours=int(cfg.since_hours))
        client = GitHubClient(token=cfg.token)
        snap = await client.fetch_repo_snapshot(
            owner=owner,
            repo=name,
            since=since,
            max_commits=int(cfg.max_commits),
            max_prs=int(cfg.max_prs),
            max_releases=int(cfg.max_releases),
        )
        # "articles" here is a lightweight update list for the scout/main agent.
        updates: List[Dict[str, Any]] = []
        for r in snap.get("releases_recent") or []:
            if isinstance(r, dict):
                updates.append({"kind": "release", "title": r.get("name") or r.get("tag"), "url": r.get("url")})
        for c in snap.get("commits_recent") or []:
            if isinstance(c, dict):
                updates.append({"kind": "commit", "title": c.get("message"), "url": c.get("url")})
        for pr in snap.get("prs_recent") or []:
            if isinstance(pr, dict):
                updates.append({"kind": "pr", "title": pr.get("title"), "url": pr.get("url")})
        return source.name, [{"snapshot": snap, "updates": updates}]

    async def analyze_source(
        self, source: NewsSourceConfig, articles: List[Dict[str, Any]], llm: LLMRunner
    ) -> Dict[str, Any]:
        snap = {}
        updates: List[Dict[str, Any]] = []
        if articles and isinstance(articles[0], dict):
            snap = articles[0].get("snapshot") or {}
            updates = articles[0].get("updates") or []
        commits = len((snap.get("commits_recent") or []) if isinstance(snap, dict) else [])
        prs = len((snap.get("prs_recent") or []) if isinstance(snap, dict) else [])
        rels = len((snap.get("releases_recent") or []) if isinstance(snap, dict) else [])
        hours = 30
        try:
            window = snap.get("window") if isinstance(snap, dict) else {}
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
        repo_name = ""
        if isinstance(snap, dict):
            repo = snap.get("repo") if isinstance(snap.get("repo"), dict) else {}
            repo_name = str((repo or {}).get("full_name") or "")
        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": len(updates),
            "topics": topics,
            "quality_score": rels * 8 + prs * 2 + commits,
            "today_angle": angle if repo_name else angle,
            "sample_articles": updates[:3],
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: List[Dict[str, Any]],
        llm: LLMRunner,
    ) -> SubAgentResult:
        if not articles or not isinstance(articles[0], dict):
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                error="GitHub 来源未抓取到任何数据（检查 github_enabled/github_repos/github_token）",
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
            raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        except Exception as e:
            astrbot_logger.warning("[dailynews] github subagent write failed, fallback: %s", e, exc_info=True)
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
            for k, key in [("Release", "releases_recent"), ("Commit", "commits_recent"), ("PR", "prs_recent")]:
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
