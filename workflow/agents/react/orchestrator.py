from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

import aiohttp

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.config_models import ReactAgentConfig
from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ..sources.github_source import GitHubClient, GitHubConfig, parse_repo
from .react_agent import ReActAgent, ReactRunResult
from .shared_memory import SharedMemory
from .subagent_wrapper import SubAgentWrapper
from .tool_registry import ToolRegistry


def _to_brief_text(payload: Any, *, max_chars: int = 2400) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            text = str(payload)
    text = text.strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def _safe_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = int(default)
    return max(minimum, min(maximum, n))


async def _search_web(query: str, *, max_results: int = 6) -> list[dict[str, str]]:
    q = str(query or "").strip()
    if not q:
        return []
    url = f"https://api.duckduckgo.com/?q={quote_plus(q)}&format=json&no_redirect=1&no_html=1"
    timeout = aiohttp.ClientTimeout(total=20)
    headers = {"User-Agent": "astrbot-dailynews-agent/1.0"}
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            return []
    if not isinstance(data, dict):
        return []
    abstract = str(data.get("AbstractText") or "").strip()
    abstract_url = str(data.get("AbstractURL") or "").strip()
    if abstract and abstract_url:
        seen.add(abstract_url)
        out.append({"title": abstract, "url": abstract_url})
    def _walk(items: list[Any]):
        for it in items:
            if len(out) >= max(1, int(max_results)):
                return
            if isinstance(it, dict) and isinstance(it.get("Topics"), list):
                _walk(it.get("Topics") or [])
                continue
            if not isinstance(it, dict):
                continue
            text = str(it.get("Text") or "").strip()
            first = str(it.get("FirstURL") or "").strip()
            if text and first and first not in seen:
                seen.add(first)
                out.append({"title": text, "url": first})
    related = data.get("RelatedTopics")
    if isinstance(related, list):
        _walk(related)
    return out[: max(1, int(max_results))]


def _format_web_results(query: str, rows: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    lines = [f"Query: {query}", "Top Results:"]
    for r in rows:
        t = str(r.get("title") or "").strip()
        u = str(r.get("url") or "").strip()
        if t and u:
            lines.append(f"- {t} ({u})")
    if len(lines) == 2:
        lines.append("- No search result found.")
    return "\n".join(lines).strip(), {"query": query, "results": rows}


def _focus_tokens(focus: str) -> list[str]:
    toks = [
        x.strip().lower()
        for x in re.split(r"[\s,;/|]+", str(focus or "").strip())
        if x and x.strip()
    ]
    return [t for t in toks if len(t) >= 2][:12]


def _format_github_snapshot(
    *, repo_name: str, focus: str, snapshot: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    repo = snapshot.get("repo") if isinstance(snapshot.get("repo"), dict) else {}
    full_name = str(repo.get("full_name") or repo_name).strip()
    repo_url = str(repo.get("html_url") or f"https://github.com/{full_name}").strip()
    version = str(snapshot.get("version") or "").strip()
    focus_tokens = _focus_tokens(focus)
    updates: list[dict[str, str]] = []
    for section, kind, tk in (
        ("releases_recent", "release", "name"),
        ("commits_recent", "commit", "message"),
        ("prs_recent", "pr", "title"),
    ):
        for item in snapshot.get(section) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get(tk) or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            if focus_tokens and not any(tok in title.lower() for tok in focus_tokens):
                continue
            updates.append({"kind": kind, "title": title, "url": url})
    if not updates:
        for section, kind, tk in (
            ("releases_recent", "release", "name"),
            ("commits_recent", "commit", "message"),
            ("prs_recent", "pr", "title"),
        ):
            for item in snapshot.get(section) or []:
                if not isinstance(item, dict):
                    continue
                title = str(item.get(tk) or "").strip()
                url = str(item.get("url") or "").strip()
                if title and url:
                    updates.append({"kind": kind, "title": title, "url": url})
    updates = updates[:10]
    lines = [
        f"Repository: {full_name}",
        f"Repo URL: {repo_url}",
        f"Focus: {focus or '(none)'}",
        f"Version Hint: {version or '(unknown)'}",
        "Updates:",
    ]
    if updates:
        for u in updates:
            lines.append(f"- [{u['kind']}] {u['title']} ({u['url']})")
    else:
        lines.append("- No recent updates found in current window.")
    return "\n".join(lines).strip(), {
        "repo": full_name,
        "repo_url": repo_url,
        "focus": focus,
        "version": version,
        "updates": updates,
        "raw_snapshot": snapshot,
    }


class ReActDailyNewsOrchestrator:
    def __init__(self, *, sub_agent_classes: dict[str, Any]):
        self._sub_agent_classes = dict(sub_agent_classes or {})

    def _pick_provider_id(
        self, *, user_config: dict[str, Any], react_cfg: ReactAgentConfig
    ) -> str:
        if react_cfg.provider_id:
            return react_cfg.provider_id
        provider_id = str(user_config.get("main_agent_provider_id") or "").strip()
        if provider_id:
            return provider_id
        raw_list = user_config.get("main_agent_fallback_provider_ids") or []
        if isinstance(raw_list, list):
            for x in raw_list:
                if isinstance(x, str) and x.strip():
                    return x.strip()
        for k in (
            "main_agent_fallback_provider_id_1",
            "main_agent_fallback_provider_id_2",
            "main_agent_fallback_provider_id_3",
        ):
            v = user_config.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    async def _vertical_web_fallback(
        self,
        *,
        platform: str,
        keyword: str,
        memory: SharedMemory,
        site_filter: str,
    ) -> str:
        query = f"site:{site_filter} {keyword}".strip()
        rows = await _search_web(query, max_results=6)
        rendered, packed = _format_web_results(query, rows)
        qhash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
        memory.write(f"{platform}_fallback::{qhash}", packed)
        return rendered

    async def _run_vertical_subagent(
        self,
        *,
        agent_cls: Any,
        source: NewsSourceConfig,
        instruction: str,
        llm_write: LLMRunner,
        user_config: dict[str, Any],
        memory: SharedMemory,
        agent_id: str,
        dependency_ids: list[str] | None = None,
        keyword: str = "",
    ) -> str:
        dep_ids = dependency_ids if isinstance(dependency_ids, list) else []
        kw = str(keyword or "").strip().lower()

        async def _runner(injected_prompt: str) -> Any:
            agent = agent_cls()
            _, articles = await agent.fetch_latest_articles(source, user_config)
            if kw:
                filtered: list[dict[str, Any]] = []
                for a in articles or []:
                    if not isinstance(a, dict):
                        continue
                    t = str(a.get("title") or a.get("name") or "").lower()
                    u = str(a.get("url") or a.get("link") or "").lower()
                    if kw in t or kw in u:
                        filtered.append(a)
                if filtered:
                    articles = filtered

            report = None
            try:
                report = await agent.analyze_source(source, articles, llm_write)
            except Exception:
                report = None

            full_instruction = instruction
            if injected_prompt:
                full_instruction = (
                    f"{instruction}\n\nDependency context:\n{injected_prompt}"
                )
            try:
                result = await agent.process_source(
                    source,
                    full_instruction,
                    articles,
                    llm_write,
                    user_config=user_config,
                )
            except TypeError:
                result = await agent.process_source(
                    source,
                    full_instruction,
                    articles,
                    llm_write,
                )

            if isinstance(result, SubAgentResult):
                return {
                    "source_name": result.source_name,
                    "summary": result.summary,
                    "key_points": result.key_points,
                    "content": result.content,
                    "images": result.images or [],
                    "error": result.error,
                    "analysis_report": report,
                }
            return {"analysis_report": report, "content": str(result or "")}

        wrapped = SubAgentWrapper(
            agent_id=agent_id,
            task_description=f"You are vertical analyzer for {source.type}.",
            dependency_ids=dep_ids,
            runner=_runner,
        )
        out = await wrapped.execute(shared_memory=memory)
        if not out.ok:
            return f"error: {out.error or 'vertical analyzer failed'}"
        return _to_brief_text(out.content, max_chars=4600)

    def _register_vertical_capability_tools(
        self,
        *,
        registry: ToolRegistry,
        llm_write: LLMRunner,
        user_config: dict[str, Any],
        memory: SharedMemory,
        user_goal: str,
    ) -> None:
        # Dynamic registration from sub_agent_classes
        for source_type, agent_cls in self._sub_agent_classes.items():
            st = str(source_type or "").strip().lower()

            if st == "miyoushe":
                async def _tool_analyze_miyoushe(
                    *,
                    context,
                    keyword: str = "",
                    uid: str = "",
                    profile_url: str = "",
                    max_articles: int = 5,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    uid_s = str(uid or "").strip()
                    url = str(profile_url or "").strip()
                    if not url and uid_s:
                        url = f"https://www.miyoushe.com/ys/accountCenter/postList?id={uid_s}"
                    if not url:
                        if not kw:
                            return "error: provide uid/profile_url or keyword."
                        return await self._vertical_web_fallback(
                            platform="miyoushe",
                            keyword=kw,
                            memory=memory,
                            site_filter="miyoushe.com",
                        )
                    source = NewsSourceConfig(
                        name=f"Miyoushe {uid_s or 'dynamic'}",
                        url=url,
                        type="miyoushe",
                        priority=1,
                        max_articles=_safe_int(max_articles, 5, minimum=1, maximum=12),
                    )
                    instruction = (
                        f"User goal: {user_goal}\nFocus keyword: {kw or '(none)'}\n"
                        "Extract relevant updates from Miyoushe posts."
                    )
                    return await self._run_vertical_subagent(
                        agent_cls=_agent_cls,
                        source=source,
                        instruction=instruction,
                        llm_write=llm_write,
                        user_config=user_config,
                        memory=memory,
                        agent_id=f"vertical::miyoushe::{uid_s or 'dynamic'}",
                        dependency_ids=dependency_ids,
                        keyword=kw,
                    )

                registry.register_callable(
                    name="tool_analyze_miyoushe",
                    description="Analyze Miyoushe by keyword/uid/profile_url.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "uid": {"type": "string"},
                            "profile_url": {"type": "string"},
                            "max_articles": {"type": "integer"},
                            "dependency_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    handler=_tool_analyze_miyoushe,
                )

            elif st == "twitter":
                async def _tool_analyze_twitter(
                    *,
                    context,
                    keyword: str,
                    username: str = "",
                    profile_url: str = "",
                    max_tweets: int = 4,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    uname = str(username or "").strip().lstrip("@")
                    url = str(profile_url or "").strip()
                    if not url and uname:
                        url = f"https://x.com/{uname}"
                    if not url and kw.startswith("@") and len(kw) > 1:
                        url = f"https://x.com/{kw.lstrip('@')}"
                    if not url:
                        return await self._vertical_web_fallback(
                            platform="twitter",
                            keyword=kw,
                            memory=memory,
                            site_filter="x.com",
                        )
                    source = NewsSourceConfig(
                        name=f"Twitter {uname or 'dynamic'}",
                        url=url,
                        type="twitter",
                        priority=1,
                        max_articles=_safe_int(max_tweets, 4, minimum=1, maximum=10),
                    )
                    instruction = (
                        f"User goal: {user_goal}\nFocus keyword: {kw}\n"
                        "Extract high-signal updates from recent tweets."
                    )
                    return await self._run_vertical_subagent(
                        agent_cls=_agent_cls,
                        source=source,
                        instruction=instruction,
                        llm_write=llm_write,
                        user_config=user_config,
                        memory=memory,
                        agent_id=f"vertical::twitter::{uname or 'dynamic'}",
                        dependency_ids=dependency_ids,
                        keyword=kw,
                    )

                registry.register_callable(
                    name="tool_analyze_twitter",
                    description="Analyze Twitter/X updates by keyword.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "username": {"type": "string"},
                            "profile_url": {"type": "string"},
                            "max_tweets": {"type": "integer"},
                            "dependency_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["keyword"],
                    },
                    handler=_tool_analyze_twitter,
                )

            elif st == "wechat":
                async def _tool_analyze_wechat(
                    *,
                    context,
                    keyword: str,
                    album_url: str = "",
                    album_keyword: str = "",
                    max_articles: int = 5,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    url = str(album_url or "").strip()
                    if not url:
                        return await self._vertical_web_fallback(
                            platform="wechat",
                            keyword=kw,
                            memory=memory,
                            site_filter="mp.weixin.qq.com",
                        )
                    source = NewsSourceConfig(
                        name="WeChat dynamic",
                        url=url,
                        type="wechat",
                        priority=1,
                        max_articles=_safe_int(max_articles, 5, minimum=1, maximum=12),
                        album_keyword=str(album_keyword or "").strip() or None,
                    )
                    instruction = (
                        f"User goal: {user_goal}\nFocus keyword: {kw}\n"
                        "Extract key updates from WeChat album articles."
                    )
                    return await self._run_vertical_subagent(
                        agent_cls=_agent_cls,
                        source=source,
                        instruction=instruction,
                        llm_write=llm_write,
                        user_config=user_config,
                        memory=memory,
                        agent_id="vertical::wechat::dynamic",
                        dependency_ids=dependency_ids,
                        keyword=kw,
                    )

                registry.register_callable(
                    name="tool_analyze_wechat",
                    description="Analyze WeChat updates by keyword.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "album_url": {"type": "string"},
                            "album_keyword": {"type": "string"},
                            "max_articles": {"type": "integer"},
                            "dependency_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["keyword"],
                    },
                    handler=_tool_analyze_wechat,
                )

            elif st in {"xiuxiu_ai", "xiuxiu-ai"}:
                async def _tool_analyze_xiuxiu(
                    *,
                    context,
                    keyword: str,
                    date: str = "",
                    days_ago: int = 0,
                    max_items: int = 20,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    source = NewsSourceConfig(
                        name="Xiuxiu AI Daily",
                        url="https://xiuxiu.huxiu.com/",
                        type="xiuxiu_ai",
                        priority=1,
                        max_articles=_safe_int(max_items, 20, minimum=5, maximum=50),
                        meta={
                            "date": str(date or "").strip(),
                            "days_ago": max(0, int(days_ago or 0)),
                        },
                    )
                    instruction = (
                        f"User goal: {user_goal}\nFocus keyword: {kw}\n"
                        "Analyze Xiuxiu AI daily events and produce concise evidence."
                    )
                    return await self._run_vertical_subagent(
                        agent_cls=_agent_cls,
                        source=source,
                        instruction=instruction,
                        llm_write=llm_write,
                        user_config=user_config,
                        memory=memory,
                        agent_id="vertical::xiuxiu_ai::dynamic",
                        dependency_ids=dependency_ids,
                        keyword=kw,
                    )

                registry.register_callable(
                    name="tool_analyze_xiuxiu",
                    description="Analyze Xiuxiu AI feed by keyword/date window.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "date": {"type": "string"},
                            "days_ago": {"type": "integer"},
                            "max_items": {"type": "integer"},
                            "dependency_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["keyword"],
                    },
                    handler=_tool_analyze_xiuxiu,
                )

    async def run(
        self,
        *,
        user_goal: str,
        user_config: dict[str, Any],
        astrbot_context: Any,
        sources: list[NewsSourceConfig],
        fetched: dict[str, list[dict[str, Any]]],
    ) -> ReactRunResult:
        _ = sources, fetched
        react_cfg = ReactAgentConfig.from_mapping(user_config)
        provider_id = self._pick_provider_id(user_config=user_config, react_cfg=react_cfg)
        memory = SharedMemory.instance()
        memory.reset()
        registry = ToolRegistry()
        initial_context = (
            f"North Star Goal: {user_goal}\n"
            "Plan autonomously and decide which capability tools to call."
        )

        llm_write = LLMRunner(
            astrbot_context,
            timeout_s=max(60, int(user_config.get("llm_write_timeout_s", 360) or 360)),
            max_retries=max(0, int(user_config.get("llm_max_retries", 1) or 1)),
            provider_id=provider_id or None,
        )

        async def _tool_search_github(*, context, repo_name: str, focus: str):
            _ = context
            repo_input = str(repo_name or "").strip()
            focus_text = str(focus or "").strip()
            if not repo_input or not focus_text:
                return "error: repo_name and focus are required."
            repo = parse_repo(repo_input) or parse_repo(f"https://github.com/{repo_input}")
            if not repo:
                return f"error: invalid repo_name: `{repo_input}`"
            owner, name = repo
            gh_cfg = GitHubConfig.from_user_config(user_config)
            since = datetime.now(tz=timezone.utc) - timedelta(
                hours=max(1, int(gh_cfg.since_hours))
            )
            client = GitHubClient(token=str(gh_cfg.token or "").strip())
            try:
                snapshot = await client.fetch_repo_snapshot(
                    owner=owner,
                    repo=name,
                    since=since,
                    max_commits=max(1, int(gh_cfg.max_commits)),
                    max_prs=max(1, int(gh_cfg.max_prs)),
                    max_releases=max(1, int(gh_cfg.max_releases)),
                )
            except Exception as e:
                return f"error: github search failed for `{owner}/{name}`: {e}"
            rendered, packed = _format_github_snapshot(
                repo_name=f"{owner}/{name}",
                focus=focus_text,
                snapshot=snapshot,
            )
            memory.write(f"github_search::{owner}_{name}", packed)
            return rendered

        registry.register_callable(
            name="tool_search_github",
            description="Search and summarize a GitHub repository by repo_name and focus.",
            parameters={
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["repo_name", "focus"],
            },
            handler=_tool_search_github,
        )

        async def _tool_search_web(*, context, query: str):
            _ = context
            q = str(query or "").strip()
            if not q:
                return "error: query is required."
            rows = await _search_web(q, max_results=6)
            rendered, packed = _format_web_results(q, rows)
            qhash = hashlib.sha1(q.encode("utf-8")).hexdigest()[:10]
            memory.write(f"web_search::{qhash}", packed)
            return rendered

        registry.register_callable(
            name="tool_search_web",
            description="Search the web by query and return top result links.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_tool_search_web,
        )

        self._register_vertical_capability_tools(
            registry=registry,
            llm_write=llm_write,
            user_config=user_config,
            memory=memory,
            user_goal=user_goal,
        )

        async def _tool_write_report(
            *,
            context,
            dependency_ids: list[str] | None = None,
            style_hint: str = "",
        ):
            _ = context
            dep_ids = dependency_ids if isinstance(dependency_ids, list) else []
            style = str(style_hint or "").strip()

            async def _runner(injected_prompt: str) -> Any:
                snapshot = memory.read(dep_ids) if dep_ids else memory.read_all()
                materials = _to_brief_text(snapshot, max_chars=22000)
                system_prompt = (
                    "You are a senior Chinese daily-news editor. "
                    "Write concise markdown with concrete facts and links."
                )
                prompt = (
                    f"User goal:\n{user_goal}\n\n"
                    f"Collected materials:\n{materials}\n\n"
                    f"Dependency-injected context:\n{injected_prompt or '(none)'}\n\n"
                    "Write the final markdown report. If key parts are missing, state them."
                )
                if style:
                    prompt += f"\nStyle hint: {style}"
                text = await llm_write.ask(system_prompt=system_prompt, prompt=prompt)
                return {"markdown": text}

            wrapper = SubAgentWrapper(
                agent_id="writer_agent",
                task_description="Compose final daily report from collected intelligence.",
                dependency_ids=dep_ids,
                runner=_runner,
            )
            out = await wrapper.execute(shared_memory=memory)
            if not out.ok:
                return f"error: writer failed: {out.error or 'unknown'}"
            payload = out.content
            if isinstance(payload, dict):
                md = str(payload.get("markdown") or "").strip()
                if md:
                    return md
            return _to_brief_text(payload, max_chars=4200)

        registry.register_callable(
            name="tool_write_report",
            description="Compose final markdown report from shared memory.",
            parameters={
                "type": "object",
                "properties": {
                    "dependency_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "style_hint": {"type": "string"},
                },
            },
            handler=_tool_write_report,
        )

        merged_global = registry.merge_global_tools(
            astrbot_context,
            prefer_existing=True,
            include_inactive=False,
        )
        astrbot_logger.info(
            "[dailynews][react] merged global AstrBot tools: %s", merged_global
        )

        agent = ReActAgent(
            astrbot_context=astrbot_context,
            registry=registry,
            shared_memory=memory,
            config=react_cfg,
            provider_id=provider_id,
        )
        result = await agent.run(user_goal=user_goal, initial_context=initial_context)
        astrbot_logger.info(
            "[dailynews][react] done status=%s steps=%s reason=%s",
            result.status,
            result.steps,
            result.termination_reason,
        )
        return result

