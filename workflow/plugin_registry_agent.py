from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncio

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .github_source import parse_repo
from .image_utils import get_plugin_data_dir
from .llm import LLMRunner
from .models import NewsSourceConfig, SubAgentResult


def _parse_iso_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _first_line(s: str) -> str:
    return (s or "").strip().splitlines()[0].strip() if (s or "").strip() else ""


def _default_official_registry_path() -> Path:
    # plugin path: .../AstrBot/data/plugins/<plugin>/workflow/plugin_registry_agent.py
    # official file: .../AstrBot/data/plugins.json
    try:
        return Path(__file__).resolve().parents[3] / "plugins.json"
    except Exception:
        return Path("plugins.json")

def _registry_cache_id(registry_path: str) -> str:
    s = str(registry_path or "").strip().replace("\\", "/").lower()
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:16]


@dataclass(frozen=True)
class RegistrySnapshot:
    snapshot_at: Optional[datetime]
    keys: set[str]
    snapshot_path: str
    prev_snapshot_path: str
    meta_path: str


def _load_snapshot(cache_dir: Path, registry_path: str) -> RegistrySnapshot:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cid = _registry_cache_id(registry_path)
    meta_path = cache_dir / f"{cid}.meta.json"
    snap_path = cache_dir / f"{cid}.snapshot.json"
    prev_path = cache_dir / f"{cid}.snapshot.prev.json"

    snapshot_at: Optional[datetime] = None
    keys: set[str] = set()
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            snapshot_at = _parse_iso_dt(str(meta.get("snapshot_at") or ""))
    except Exception:
        snapshot_at = None

    try:
        if snap_path.exists():
            data = json.loads(snap_path.read_text(encoding="utf-8"))
            root = data.get("data") if isinstance(data, dict) else None
            if isinstance(root, dict):
                keys = {str(k) for k in root.keys()}
    except Exception:
        keys = set()

    return RegistrySnapshot(
        snapshot_at=snapshot_at,
        keys=keys,
        snapshot_path=str(snap_path),
        prev_snapshot_path=str(prev_path),
        meta_path=str(meta_path),
    )


def _write_snapshot(
    snapshot: RegistrySnapshot,
    *,
    registry_path: str,
    now: datetime,
    raw_text: str,
) -> None:
    snap_path = Path(snapshot.snapshot_path)
    prev_path = Path(snapshot.prev_snapshot_path)
    meta_path = Path(snapshot.meta_path)
    snap_path.parent.mkdir(parents=True, exist_ok=True)

    # keep previous snapshot for comparison/debugging
    try:
        if snap_path.exists():
            try:
                if prev_path.exists():
                    prev_path.unlink()
            except Exception:
                pass
            try:
                snap_path.replace(prev_path)
            except Exception:
                prev_path.write_text(snap_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    try:
        snap_path.write_text(raw_text, encoding="utf-8")
    except Exception:
        # fallback to canonical json if raw contains non-utf8 or other issues
        try:
            data = json.loads(raw_text)
            snap_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            snap_path.write_text(raw_text.encode("utf-8", "ignore").decode("utf-8"), encoding="utf-8")

    meta = {
        "snapshot_at": now.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "registry_path": str(registry_path),
        "snapshot_file": str(snap_path),
        "prev_snapshot_file": str(prev_path),
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


@dataclass(frozen=True)
class RegistryPlugin:
    key: str
    display_name: str
    repo: str
    version: str
    registry_updated_at: Optional[datetime]
    stars: int
    desc: str


@dataclass(frozen=True)
class RepoMeta:
    owner: str
    repo: str
    html_url: str
    created_at: Optional[datetime]
    pushed_at: Optional[datetime]
    updated_at: Optional[datetime]
    stars: int
    forks: int
    description: str


class PluginRegistrySubAgent:
    """插件源子 Agent：读取 plugins.json / 第三方 registry，并用 GitHub API 校验最近窗口内的新仓库/活跃仓库。"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        meta = source.meta or {}
        since_hours = int(meta.get("since_hours") or 27)
        max_plugins = int(meta.get("max_plugins") or source.max_articles or 20)
        since_hours = max(1, min(since_hours, 24 * 14))
        max_plugins = max(1, min(max_plugins, 200))

        registry_path = self._resolve_registry_path(source)
        raw_text = ""
        try:
            raw_text = Path(registry_path).read_text(encoding="utf-8")
        except Exception as e:
            astrbot_logger.warning("[dailynews] plugin_registry read failed (%s): %s", registry_path, e)
            return source.name, []

        plugins = self._load_registry_from_text(raw_text)
        if not plugins:
            return source.name, []

        since_dt = datetime.now(tz=timezone.utc) - timedelta(hours=since_hours)

        # 24h snapshot cache/diff (fallback signal for "newly listed" plugins).
        cache_dir = get_plugin_data_dir("plugin_registry_cache")
        snapshot = _load_snapshot(cache_dir, registry_path)
        now = datetime.now(tz=timezone.utc)
        baseline_at = snapshot.snapshot_at
        baseline_keys = set(snapshot.keys)
        curr_keys = {p.key for p in plugins}

        rotated = False
        first_init = baseline_at is None
        if first_init:
            _write_snapshot(snapshot, registry_path=registry_path, now=now, raw_text=raw_text)
            rotated = True
            baseline_at = now
            baseline_keys = set(curr_keys)
        else:
            age = now - (baseline_at or now)
            if age >= timedelta(hours=24):
                # Diff against previous snapshot, then rotate snapshot to current.
                rotated = True
                _write_snapshot(snapshot, registry_path=registry_path, now=now, raw_text=raw_text)

        added_keys = set()
        if baseline_keys:
            added_keys = curr_keys - baseline_keys

        baseline_path = snapshot.snapshot_path
        if rotated and not first_init:
            baseline_path = snapshot.prev_snapshot_path

        window_keys = {
            p.key for p in plugins if p.registry_updated_at is not None and p.registry_updated_at >= since_dt
        }

        # Candidates: union of (within updated_at window) + (newly listed vs snapshot).
        candidates: List[RegistryPlugin] = []
        key_to_plugin = {p.key: p for p in plugins}
        for k in list(window_keys) + list(added_keys):
            p = key_to_plugin.get(k)
            if p:
                candidates.append(p)

        # Sort by registry update time desc (fallback: keep stable by key).
        candidates.sort(
            key=lambda x: (
                x.registry_updated_at.timestamp() if x.registry_updated_at else 0.0,
                x.key,
            ),
            reverse=True,
        )
        candidates = candidates[: max_plugins * 3]  # allow some to be filtered out after GitHub check

        token = str(user_config.get("github_token") or "").strip()
        repo_metas = await self._fetch_repo_metas(candidates, token=token)

        rows: List[Dict[str, Any]] = []
        for p in candidates:
            repo = parse_repo(p.repo)
            rmeta = None
            if repo:
                owner, name = repo
                rmeta = repo_metas.get((owner, name))

            is_new_repo = bool(rmeta and rmeta.created_at and rmeta.created_at >= since_dt)
            is_active_repo = bool(rmeta and rmeta.pushed_at and rmeta.pushed_at >= since_dt)
            in_window = p.key in window_keys
            is_new_in_registry = p.key in added_keys

            rows.append(
                {
                    "key": p.key,
                    "display_name": p.display_name,
                    "repo": p.repo,
                    "version": p.version,
                    "registry_updated_at": p.registry_updated_at.isoformat().replace("+00:00", "Z")
                    if p.registry_updated_at
                    else "",
                    "plugin_stars": int(p.stars),
                    "desc": p.desc,
                    "repo_html_url": rmeta.html_url if rmeta else "",
                    "repo_created_at": rmeta.created_at.isoformat().replace("+00:00", "Z")
                    if (rmeta and rmeta.created_at)
                    else "",
                    "repo_pushed_at": rmeta.pushed_at.isoformat().replace("+00:00", "Z") if (rmeta and rmeta.pushed_at) else "",
                    "repo_updated_at": rmeta.updated_at.isoformat().replace("+00:00", "Z")
                    if (rmeta and rmeta.updated_at)
                    else "",
                    "repo_stars": int(rmeta.stars) if rmeta else 0,
                    "repo_forks": int(rmeta.forks) if rmeta else 0,
                    "is_new_repo": bool(is_new_repo),
                    "is_active_repo": bool(is_active_repo),
                    "is_in_registry_window": bool(in_window),
                    "is_new_in_registry_snapshot": bool(is_new_in_registry),
                    "registry_snapshot_path": snapshot.snapshot_path,
                    "registry_snapshot_prev_path": snapshot.prev_snapshot_path,
                    "registry_snapshot_baseline_path": baseline_path,
                    "registry_snapshot_baseline_at": baseline_at.isoformat().replace("+00:00", "Z")
                    if baseline_at
                    else "",
                    "registry_snapshot_rotated": bool(rotated),
                    "registry_snapshot_first_init": bool(first_init),
                }
            )

        # Keep:
        # 1) newly listed vs snapshot
        # 2) within window AND (repo new/active)
        kept: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        def _push(row: Dict[str, Any]) -> None:
            k = str(row.get("key") or "")
            if not k or k in seen_keys:
                return
            seen_keys.add(k)
            kept.append(row)

        for r in rows:
            if r.get("is_new_in_registry_snapshot"):
                _push(r)
        for r in rows:
            if r.get("is_in_registry_window") and (r.get("is_new_repo") or r.get("is_active_repo")):
                _push(r)
        for r in rows:
            if r.get("is_in_registry_window") and not (r.get("is_new_repo") or r.get("is_active_repo")):
                _push(r)

        kept = kept[:max_plugins]

        return source.name, kept

    async def analyze_source(
        self, source: NewsSourceConfig, articles: List[Dict[str, Any]], llm: LLMRunner
    ) -> Dict[str, Any]:
        meta = source.meta or {}
        since_hours = int(meta.get("since_hours") or 27)
        new_cnt = 0
        active_cnt = 0
        for a in articles or []:
            if isinstance(a, dict) and a.get("is_new_repo"):
                new_cnt += 1
            if isinstance(a, dict) and a.get("is_active_repo"):
                active_cnt += 1

        angle = f"近 {since_hours} 小时：{new_cnt} 个新仓库插件 / {active_cnt} 个活跃插件"
        sample: List[Dict[str, Any]] = []
        for a in (articles or [])[:3]:
            if not isinstance(a, dict):
                continue
            sample.append(
                {
                    "display_name": a.get("display_name") or a.get("key"),
                    "repo": a.get("repo_html_url") or a.get("repo"),
                    "is_new_in_registry_snapshot": bool(a.get("is_new_in_registry_snapshot") or False),
                    "is_new_repo": bool(a.get("is_new_repo") or False),
                    "is_active_repo": bool(a.get("is_active_repo") or False),
                }
            )
        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": int(source.priority or 1),
            "article_count": len(articles or []),
            "topics": ["plugins", "registry", "github"],
            "quality_score": int(new_cnt * 8 + active_cnt * 2),
            "today_angle": angle,
            "sample_articles": sample,
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: List[Dict[str, Any]],
        llm: LLMRunner,
        user_config: Dict[str, Any] | None = None,
    ) -> SubAgentResult:
        meta = source.meta or {}
        since_hours = int(meta.get("since_hours") or 27)
        registry_path = self._resolve_registry_path(source)

        if not articles:
            # Still show snapshot info (it may have been initialized/refreshed in fetch_latest_articles).
            cache_dir = get_plugin_data_dir("plugin_registry_cache")
            snap = _load_snapshot(cache_dir, registry_path)
            snap_at = snap.snapshot_at.isoformat().replace("+00:00", "Z") if snap.snapshot_at else ""
            lines = [
                f"## {source.name}",
                "",
                f"- 索引文件：`{registry_path}`",
                f"- 窗口：近 {since_hours} 小时（GitHub `created_at/pushed_at` 校验）",
            ]
            if snap.snapshot_path:
                lines.append(f"- 当前快照：`{snap.snapshot_path}`" + (f"（`{snap_at}`）" if snap_at else ""))
            if snap.prev_snapshot_path and Path(snap.prev_snapshot_path).exists():
                lines.append(f"- 上一次快照：`{snap.prev_snapshot_path}`")
            lines.extend(["", f"未抓取到符合条件的插件（近 {since_hours} 小时内新仓库/活跃仓库；或今日无新增上架）。"])
            return SubAgentResult(
                source_name=source.name,
                content="\n".join(lines).strip(),
                summary="",
                key_points=[],
                error=None,
                no_llm_merge=True,
            )

        snapshot_at = ""
        snapshot_path = ""
        prev_snapshot_path = ""
        baseline_snapshot_path = ""
        rotated = False
        for a in articles:
            if not isinstance(a, dict):
                continue
            snapshot_at = snapshot_at or str(a.get("registry_snapshot_baseline_at") or "")
            snapshot_path = snapshot_path or str(a.get("registry_snapshot_path") or "")
            prev_snapshot_path = prev_snapshot_path or str(a.get("registry_snapshot_prev_path") or "")
            baseline_snapshot_path = baseline_snapshot_path or str(a.get("registry_snapshot_baseline_path") or "")
            rotated = rotated or bool(a.get("registry_snapshot_rotated") or False)

        newly_listed: List[Dict[str, Any]] = []
        window_new_repo: List[Dict[str, Any]] = []
        window_active_repo: List[Dict[str, Any]] = []
        window_not_active: List[Dict[str, Any]] = []
        for a in articles:
            if not isinstance(a, dict):
                continue
            if a.get("is_new_in_registry_snapshot"):
                newly_listed.append(a)
            in_window = bool(a.get("is_in_registry_window") or False)
            if not in_window:
                continue
            if a.get("is_new_repo"):
                window_new_repo.append(a)
            elif a.get("is_active_repo"):
                window_active_repo.append(a)
            else:
                window_not_active.append(a)

        def _line(p: Dict[str, Any]) -> str:
            name = str(p.get("display_name") or p.get("key") or "").strip() or "plugin"
            repo_url = str(p.get("repo_html_url") or p.get("repo") or "").strip()
            version = str(p.get("version") or "").strip()
            reg_u = str(p.get("registry_updated_at") or "").strip()
            created = str(p.get("repo_created_at") or "").strip()
            pushed = str(p.get("repo_pushed_at") or "").strip()
            stars = int(p.get("repo_stars") or 0)
            desc = _first_line(str(p.get("desc") or ""))

            pieces = []
            title = f"[{name}]({repo_url})" if repo_url else name
            if version:
                v = version
                if not v.lower().startswith("v"):
                    v = f"v{v}"
                pieces.append(f"{title} `{v}`")
            else:
                pieces.append(title)
            if stars:
                pieces.append(f"★{stars}")
            if created:
                pieces.append(f"created `{created}`")
            if pushed:
                pieces.append(f"pushed `{pushed}`")
            if reg_u:
                pieces.append(f"registry `{reg_u}`")
            suffix = " · ".join(pieces[1:]) if len(pieces) > 1 else ""
            head = pieces[0] + (f"（{suffix}）" if suffix else "")
            return f"- {head}" + (f"\n  - {desc}" if desc else "")

        lines: List[str] = []
        lines.append(f"## {source.name}")
        lines.append("")
        lines.append(f"- 索引文件：`{registry_path}`")
        lines.append(f"- 窗口：近 {since_hours} 小时（GitHub `created_at/pushed_at` 校验）")
        if baseline_snapshot_path and Path(baseline_snapshot_path).exists():
            lines.append(f"- 对比基线快照：`{baseline_snapshot_path}`（`{snapshot_at or 'unknown'}`）")
        if snapshot_path and snapshot_path != baseline_snapshot_path and Path(snapshot_path).exists():
            lines.append(f"- 当前快照：`{snapshot_path}`")
        if prev_snapshot_path and prev_snapshot_path != baseline_snapshot_path and Path(prev_snapshot_path).exists():
            lines.append(f"- 上一次快照：`{prev_snapshot_path}`")
        if rotated:
            lines.append("- 本次运行已刷新 24h 快照（用于明日对比）")
        lines.append("")
        if newly_listed:
            lines.append("### 新增上架（相对 24h 快照新增的插件）")
            for p in newly_listed:
                lines.append(_line(p))
            lines.append("")
        if window_new_repo:
            lines.append("### 新仓库（created_at 在窗口内）")
            for p in window_new_repo:
                lines.append(_line(p))
            lines.append("")
        if window_active_repo:
            lines.append("### 最近活跃（pushed_at 在窗口内）")
            for p in window_active_repo:
                lines.append(_line(p))
            lines.append("")
        if window_not_active:
            lines.append("### 索引更新但仓库无新 push（仅 updated_at 在窗口内）")
            for p in window_not_active:
                lines.append(_line(p))
            lines.append("")

        return SubAgentResult(
            source_name=source.name,
            content="\n".join(lines).strip(),
            summary="",
            key_points=[],
            images=[],
            error=None,
            no_llm_merge=True,
        )

    def _resolve_registry_path(self, source: NewsSourceConfig) -> str:
        meta = source.meta or {}
        kind = str(meta.get("registry_kind") or "").strip().lower()
        if kind == "custom":
            raw = str(meta.get("path") or source.url or "").strip()
            raw = os.path.expandvars(raw)
            raw = os.path.expanduser(raw)
            try:
                p = Path(raw)
            except Exception:
                p = Path(str(raw))

            # If it's not an absolute path, treat it as a filename under the official registry directory.
            if not p.is_absolute():
                base_dir = _default_official_registry_path().resolve().parent
                p = base_dir / p

            # Allow omitting ".json" suffix for convenience.
            if p.suffix.lower() != ".json":
                p2 = Path(str(p) + ".json")
                if p2.exists():
                    p = p2
            return str(p.resolve())
        return str(_default_official_registry_path().resolve())

    def _load_registry(self, registry_path: str) -> List[RegistryPlugin]:
        p = Path(str(registry_path))
        if not p.exists() or not p.is_file():
            astrbot_logger.warning("[dailynews] plugin_registry file not found: %s", p)
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            astrbot_logger.warning("[dailynews] plugin_registry invalid json (%s): %s", p, e, exc_info=True)
            return []
        return self._plugins_from_loaded_json(data)

    def _load_registry_from_text(self, text: str) -> List[RegistryPlugin]:
        try:
            data = json.loads(text or "")
        except Exception:
            return []
        return self._plugins_from_loaded_json(data)

    def _plugins_from_loaded_json(self, data: Any) -> List[RegistryPlugin]:
        root = data.get("data") if isinstance(data, dict) else None
        if not isinstance(root, dict):
            return []
        out: List[RegistryPlugin] = []
        for key, v in root.items():
            if not isinstance(v, dict):
                continue
            repo = str(v.get("repo") or "").strip()
            if not repo:
                continue
            display = str(v.get("display_name") or v.get("name") or key).strip() or key
            version = str(v.get("version") or "").strip()
            updated_at = _parse_iso_dt(str(v.get("updated_at") or ""))
            stars = 0
            try:
                stars = int(v.get("stars") or 0)
            except Exception:
                stars = 0
            desc = str(v.get("desc") or "").strip()
            out.append(
                RegistryPlugin(
                    key=str(key),
                    display_name=display,
                    repo=repo,
                    version=version,
                    registry_updated_at=updated_at,
                    stars=stars,
                    desc=desc,
                )
            )
        return out

    async def _fetch_repo_metas(self, plugins: List[RegistryPlugin], *, token: str) -> Dict[Tuple[str, str], RepoMeta]:
        repos: List[Tuple[str, str]] = []
        for p in plugins:
            parsed = parse_repo(p.repo)
            if not parsed:
                continue
            repos.append(parsed)

        uniq: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for r in repos:
            if r in seen:
                continue
            seen.add(r)
            uniq.append(r)

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "astrbot-dailynews-agent/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        sem = asyncio.Semaphore(6)
        timeout = aiohttp.ClientTimeout(total=20)

        async def _one(session: aiohttp.ClientSession, owner: str, repo: str) -> Optional[RepoMeta]:
            url = f"https://api.github.com/repos/{owner}/{repo}"
            async with sem:
                try:
                    async with session.get(url, headers=headers, timeout=timeout) as resp:
                        if resp.status != 200:
                            txt = await resp.text()
                            astrbot_logger.debug(
                                "[dailynews] github repo meta failed %s/%s: %s %s",
                                owner,
                                repo,
                                resp.status,
                                txt[:120],
                            )
                            return None
                        j = await resp.json()
                except Exception:
                    return None

            return RepoMeta(
                owner=owner,
                repo=repo,
                html_url=str(j.get("html_url") or f"https://github.com/{owner}/{repo}"),
                created_at=_parse_iso_dt(str(j.get("created_at") or "")),
                pushed_at=_parse_iso_dt(str(j.get("pushed_at") or "")),
                updated_at=_parse_iso_dt(str(j.get("updated_at") or "")),
                stars=int(j.get("stargazers_count") or 0),
                forks=int(j.get("forks_count") or 0),
                description=str(j.get("description") or ""),
            )

        out: Dict[Tuple[str, str], RepoMeta] = {}
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(_one(session, o, r)) for (o, r) in uniq]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res in enumerate(results):
            if isinstance(res, RepoMeta):
                out[(res.owner, res.repo)] = res
            elif isinstance(res, Exception):
                o, r = uniq[idx]
                astrbot_logger.debug("[dailynews] github repo meta exception %s/%s: %s", o, r, res)
        return out
