from __future__ import annotations

import asyncio
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

from .models import NewsSourceConfig


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


def parse_repo(value: str) -> Optional[Tuple[str, str]]:
    m = _REPO_RE.match((value or "").strip())
    if not m:
        return None
    owner = (m.group("owner") or "").strip()
    repo = (m.group("repo") or "").strip()
    if not owner or not repo:
        return None
    return owner, repo


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def first_line(s: str) -> str:
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
                async with session.get(
                    url,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
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
        since_iso = iso_utc(since)
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
                    "published_at": iso_utc(published_at),
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
            msg = first_line(str((commit or {}).get("message") or ""))
            url = str(c.get("html_url") or "")
            sha = str(c.get("sha") or "")
            if latest_commit is None:
                latest_commit = {
                    "sha": sha[:7],
                    "message": msg,
                    "url": url,
                    "date": iso_utc(cdate) if cdate else "",
                }
            if cdate is not None and cdate >= since and msg:
                commits_recent.append(
                    {
                        "sha": sha[:7],
                        "message": msg,
                        "url": url,
                        "date": iso_utc(cdate),
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
                    "updated_at": iso_utc(updated),
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


def updates_from_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for r in snapshot.get("releases_recent") or []:
        if isinstance(r, dict):
            updates.append({"kind": "release", "title": r.get("name") or r.get("tag"), "url": r.get("url")})
    for c in snapshot.get("commits_recent") or []:
        if isinstance(c, dict):
            updates.append({"kind": "commit", "title": c.get("message"), "url": c.get("url")})
    for pr in snapshot.get("prs_recent") or []:
        if isinstance(pr, dict):
            updates.append({"kind": "pr", "title": pr.get("title"), "url": pr.get("url")})
    return updates


def build_github_sources_from_config(cfg: Dict[str, Any]) -> List[NewsSourceConfig]:
    if not bool(cfg.get("github_enabled", False)):
        return []
    repos = cfg.get("github_repos") or []
    if not isinstance(repos, list):
        return []
    out: List[NewsSourceConfig] = []
    for r in repos:
        s = str(r or "").strip()
        if not s:
            continue
        name = s
        if "github.com/" in s:
            try:
                name = s.split("github.com/", 1)[1].strip().strip("/")
            except Exception:
                name = s
        out.append(
            NewsSourceConfig(
                name=f"GitHub {name}",
                url=s,
                type="github",
                priority=1,
                max_articles=1,
            )
        )
    return out


async def fetch_github_snapshot_for_source(
    *,
    source: NewsSourceConfig,
    user_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    cfg = GitHubConfig.from_user_config(user_config)
    if not cfg.enabled:
        return None
    repo = parse_repo(source.url)
    if not repo:
        astrbot_logger.warning("[dailynews] github invalid repo url: %s", source.url)
        return None
    owner, name = repo
    since = datetime.now(tz=timezone.utc) - timedelta(hours=int(cfg.since_hours))
    client = GitHubClient(token=cfg.token)
    return await client.fetch_repo_snapshot(
        owner=owner,
        repo=name,
        since=since,
        max_commits=int(cfg.max_commits),
        max_prs=int(cfg.max_prs),
        max_releases=int(cfg.max_releases),
    )

