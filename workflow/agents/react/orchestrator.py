from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus, urlparse

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


def _pick_first_url_arg(*values: Any) -> str:
    for v in values:
        s = str(v or "").strip()
        if s.startswith("http://") or s.startswith("https://"):
            return s
    return ""


def _extract_image_urls(payload: Any, *, max_count: int = 20) -> list[str]:
    if not isinstance(payload, (dict, list)):
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _looks_like_image_url(url: str) -> bool:
        s = str(url or "").strip().lower()
        if not s.startswith(("http://", "https://")):
            return False
        if re.search(r"\.(png|jpe?g|webp|gif|bmp|svg)(?:\?|$)", s):
            return True
        return any(k in s for k in ("/image", "/images/", "/img/", "image.miyoushe"))

    def _walk(node: Any, depth: int = 0):
        if len(out) >= max_count or depth > 8:
            return
        if isinstance(node, dict):
            for key in ("images", "image_urls", "image_list", "img_list", "covers"):
                if key in node:
                    _walk(node.get(key), depth + 1)
            for key in ("image", "img", "src", "image_url"):
                val = node.get(key)
                if isinstance(val, str):
                    s = val.strip()
                    if s.startswith(("http://", "https://")) and s not in seen:
                        seen.add(s)
                        out.append(s)
                        if len(out) >= max_count:
                            return
            url_val = node.get("url")
            if isinstance(url_val, str):
                s = url_val.strip()
                if _looks_like_image_url(s) and s not in seen:
                    seen.add(s)
                    out.append(s)
                    if len(out) >= max_count:
                        return
            for val in node.values():
                if isinstance(val, (dict, list)):
                    _walk(val, depth + 1)
        elif isinstance(node, list):
            for item in node:
                _walk(item, depth + 1)
                if len(out) >= max_count:
                    return
        elif isinstance(node, str):
            s = node.strip()
            if s.startswith(("http://", "https://")) and s not in seen:
                seen.add(s)
                out.append(s)

    _walk(payload)
    return out[:max_count]


def _format_vertical_tool_output(payload: Any, *, max_chars: int = 5200) -> str:
    if not isinstance(payload, dict):
        return _to_brief_text(payload, max_chars=max_chars)

    source_name = str(payload.get("source_name") or "").strip() or "(unknown)"
    summary = str(payload.get("summary") or "").strip()
    section = str(payload.get("content") or "").strip()
    error = str(payload.get("error") or "").strip()

    key_points_raw = payload.get("key_points")
    key_points = (
        [str(x).strip() for x in key_points_raw if str(x).strip()]
        if isinstance(key_points_raw, list)
        else []
    )
    images = _extract_image_urls(payload, max_count=20)

    lines: list[str] = [f"Source: {source_name}"]
    if summary:
        lines.extend(["Summary:", summary, ""])
    if key_points:
        lines.append("Key Points:")
        for item in key_points[:8]:
            lines.append(f"- {item}")
        lines.append("")
    if section:
        lines.extend(["Section Markdown:", section, ""])
    if images:
        lines.append(f"Image URLs ({len(images)}):")
        for url in images[:12]:
            lines.append(f"- {url}")
        lines.append("")
        lines.append("Image Markdown Samples:")
        for idx, url in enumerate(images[:4], start=1):
            lines.append(f"![{source_name}-image-{idx}]({url})")
        lines.append("")
    if error:
        lines.append(f"Error: {error}")

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


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


def _extract_item_url(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in (
        "url",
        "link",
        "share_url",
        "first_url",
        "source_url",
        "article_url",
        "post_url",
        "profile_url",
    ):
        raw = str(item.get(key) or "").strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
    return ""


def _host_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        host = str(urlparse(raw).netloc or "").strip().lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _build_target_source_snapshot(
    *,
    sources: list[NewsSourceConfig],
    fetched: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    max_sources = 20
    max_urls_per_source = 6
    max_total_urls = 60
    max_titles_per_source = 4

    rows: list[dict[str, Any]] = []
    all_urls: list[str] = []
    all_hosts: list[str] = []

    for src in (sources or [])[:max_sources]:
        if not isinstance(src, NewsSourceConfig):
            continue
        source_urls: list[str] = []
        src_url = str(src.url or "").strip()
        if src_url and src_url not in source_urls:
            source_urls.append(src_url)

        items = fetched.get(src.name, []) or []
        sample_titles: list[str] = []
        for it in items:
            u = _extract_item_url(it)
            if u and u not in source_urls:
                source_urls.append(u)
            if len(source_urls) >= max_urls_per_source:
                break
        for it in items[:15]:
            if not isinstance(it, dict):
                continue
            t = str(it.get("title") or it.get("name") or "").strip()
            if t and t not in sample_titles:
                sample_titles.append(t)
            if len(sample_titles) >= max_titles_per_source:
                break

        source_urls = source_urls[:max_urls_per_source]
        rows.append(
            {
                "name": str(src.name or "").strip(),
                "type": str(src.type or "").strip(),
                "seed_url": src_url,
                "candidate_urls": source_urls,
                "article_count": len(items),
                "sample_titles": sample_titles,
            }
        )
        for u in source_urls:
            if u not in all_urls:
                all_urls.append(u)
            h = _host_from_url(u)
            if h and h not in all_hosts:
                all_hosts.append(h)
            if len(all_urls) >= max_total_urls:
                break
        if len(all_urls) >= max_total_urls:
            break

    return {
        "sources": rows,
        "target_urls": all_urls[:max_total_urls],
        "target_hosts": all_hosts[:20],
    }


def _format_target_source_context(snapshot: dict[str, Any], *, user_goal: str) -> str:
    urls = snapshot.get("target_urls") if isinstance(snapshot, dict) else []
    hosts = snapshot.get("target_hosts") if isinstance(snapshot, dict) else []
    rows = snapshot.get("sources") if isinstance(snapshot, dict) else []
    if not isinstance(urls, list):
        urls = []
    if not isinstance(hosts, list):
        hosts = []
    if not isinstance(rows, list):
        rows = []

    url_lines = [f"- {str(u).strip()}" for u in urls if str(u).strip()][:40]
    host_lines = [f"- {str(h).strip()}" for h in hosts if str(h).strip()][:12]
    source_lines: list[str] = []
    for row in rows[:18]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip() or "(unknown)"
        st = str(row.get("type") or "").strip() or "(unknown)"
        seed_url = str(row.get("seed_url") or "").strip() or "(none)"
        article_count = int(row.get("article_count") or 0)
        source_lines.append(
            f"- {name} [{st}] seed={seed_url} fetched_items={article_count}"
        )

    return (
        f"North Star Goal: {user_goal}\n\n"
        "Core Mission & Source Boundary:\n"
        "You must complete the report strictly based on the target sources below.\n"
        "Do not perform broad internet search outside these sources unless explicitly necessary.\n\n"
        "Target URLs:\n"
        f"{chr(10).join(url_lines) if url_lines else '(none)'}\n\n"
        "Target Host Whitelist:\n"
        f"{chr(10).join(host_lines) if host_lines else '(none)'}\n\n"
        "Configured Source Summary:\n"
        f"{chr(10).join(source_lines) if source_lines else '(none)'}\n\n"
        "Tool Parameter Rules:\n"
        "1) For vertical analyzer tools, pass concrete URL parameters from Target URLs whenever possible.\n"
        "   Prefer explicit argument name `url` (legacy aliases may still exist per tool).\n"
        "2) Avoid generic args like 'daily news updates'. Use source-specific keywords, account IDs, or URLs.\n"
        "3) If using web search, query must include at least one target host/domain to keep search bounded.\n"
        "4) If some target URLs are unreachable, report the gap explicitly in final output."
    )


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
        return _format_vertical_tool_output(out.content, max_chars=5200)

    def _register_vertical_capability_tools(
        self,
        *,
        registry: ToolRegistry,
        llm_write: LLMRunner,
        user_config: dict[str, Any],
        memory: SharedMemory,
        user_goal: str,
        target_urls: list[str] | None = None,
    ) -> None:
        target_url_pool = [
            str(x).strip()
            for x in (target_urls or [])
            if isinstance(x, str) and str(x).strip()
        ]

        def _pick_target_url(*domain_hints: str) -> str:
            hints = [str(x).strip().lower() for x in domain_hints if str(x).strip()]
            if not hints:
                return target_url_pool[0] if target_url_pool else ""
            for u in target_url_pool:
                lu = u.lower()
                if any(h in lu for h in hints):
                    return u
            return ""

        # Dynamic registration from sub_agent_classes
        for source_type, agent_cls in self._sub_agent_classes.items():
            st = str(source_type or "").strip().lower()

            if st == "miyoushe":
                async def _tool_analyze_miyoushe(
                    *,
                    context,
                    keyword: str = "",
                    uid: str = "",
                    url: str = "",
                    profile_url: str = "",
                    max_articles: int = 5,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    uid_s = str(uid or "").strip()
                    chosen_url = _pick_first_url_arg(url, profile_url)
                    if not chosen_url and uid_s:
                        chosen_url = (
                            f"https://www.miyoushe.com/ys/accountCenter/postList?id={uid_s}"
                        )
                    if not chosen_url:
                        chosen_url = _pick_target_url("miyoushe.com")
                    if not chosen_url:
                        if not kw:
                            return "error: provide uid/url/profile_url or keyword."
                        return await self._vertical_web_fallback(
                            platform="miyoushe",
                            keyword=kw,
                            memory=memory,
                            site_filter="miyoushe.com",
                        )
                    source = NewsSourceConfig(
                        name=f"Miyoushe {uid_s or 'dynamic'}",
                        url=chosen_url,
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
                    description="Analyze Miyoushe by keyword/uid/url (profile_url alias supported).",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "uid": {"type": "string"},
                            "url": {"type": "string"},
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
                    url: str = "",
                    profile_url: str = "",
                    max_tweets: int = 4,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    uname = str(username or "").strip().lstrip("@")
                    chosen_url = _pick_first_url_arg(url, profile_url)
                    if not chosen_url and uname:
                        chosen_url = f"https://x.com/{uname}"
                    if not chosen_url and kw.startswith("@") and len(kw) > 1:
                        chosen_url = f"https://x.com/{kw.lstrip('@')}"
                    if not chosen_url:
                        chosen_url = _pick_target_url("x.com", "twitter.com")
                    if not chosen_url:
                        return await self._vertical_web_fallback(
                            platform="twitter",
                            keyword=kw,
                            memory=memory,
                            site_filter="x.com",
                        )
                    source = NewsSourceConfig(
                        name=f"Twitter {uname or 'dynamic'}",
                        url=chosen_url,
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
                    description="Analyze Twitter/X updates by keyword/url.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "username": {"type": "string"},
                            "url": {"type": "string"},
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
                    url: str = "",
                    album_url: str = "",
                    album_keyword: str = "",
                    max_articles: int = 5,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    chosen_url = _pick_first_url_arg(url, album_url)
                    if not chosen_url:
                        chosen_url = _pick_target_url("mp.weixin.qq.com")
                    if not chosen_url:
                        return await self._vertical_web_fallback(
                            platform="wechat",
                            keyword=kw,
                            memory=memory,
                            site_filter="mp.weixin.qq.com",
                        )
                    source = NewsSourceConfig(
                        name="WeChat dynamic",
                        url=chosen_url,
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
                    description="Analyze WeChat updates by keyword/url.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "url": {"type": "string"},
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
                    keyword: str = "",
                    date: str = "",
                    days_ago: int = 0,
                    max_items: int = 20,
                    url: str = "",
                    source_url: str = "",
                    target_urls: list[str] | None = None,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    chosen_url = _pick_first_url_arg(url, source_url)
                    if not chosen_url:
                        chosen_url = _pick_target_url("xiuxiu.huxiu.com", "huxiu.com")
                    injected_urls = [
                        str(x).strip()
                        for x in (target_urls or [])
                        if isinstance(x, str) and str(x).strip()
                    ][:20]
                    source = NewsSourceConfig(
                        name="Xiuxiu AI Daily",
                        url=chosen_url or "https://xiuxiu.huxiu.com/",
                        type="xiuxiu_ai",
                        priority=1,
                        max_articles=_safe_int(max_items, 20, minimum=5, maximum=50),
                        meta={
                            "date": str(date or "").strip(),
                            "days_ago": max(0, int(days_ago or 0)),
                            "target_urls": injected_urls,
                        },
                    )
                    instruction = (
                        f"User goal: {user_goal}\nFocus keyword: {kw}\n"
                        f"Target URL: {chosen_url or '(default)'}\n"
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
                    description="Analyze Xiuxiu AI feed by keyword/date/url window.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "date": {"type": "string"},
                            "days_ago": {"type": "integer"},
                            "max_items": {"type": "integer"},
                            "url": {"type": "string"},
                            "source_url": {"type": "string"},
                            "target_urls": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "dependency_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
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
        react_cfg = ReactAgentConfig.from_mapping(user_config)
        provider_id = self._pick_provider_id(user_config=user_config, react_cfg=react_cfg)
        memory = SharedMemory.instance()
        memory.reset()
        registry = ToolRegistry()
        source_snapshot = _build_target_source_snapshot(sources=sources, fetched=fetched)
        target_hosts = source_snapshot.get("target_hosts") or []
        if not isinstance(target_hosts, list):
            target_hosts = []
        target_urls = source_snapshot.get("target_urls") or []
        if not isinstance(target_urls, list):
            target_urls = []
        initial_context = _format_target_source_context(
            source_snapshot, user_goal=user_goal
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
            ql = q.lower()
            if target_hosts:
                has_target_host = any(str(h).lower() in ql for h in target_hosts)
                if not has_target_host:
                    allowed = ", ".join([str(h) for h in target_hosts[:8]])
                    return (
                        "error: query is out of source boundary. "
                        f"include at least one target host/domain: {allowed}"
                    )
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
            target_urls=target_urls,
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
                    f"Target source boundary:\n{initial_context}\n\n"
                    f"Collected materials:\n{materials}\n\n"
                    f"Dependency-injected context:\n{injected_prompt or '(none)'}\n\n"
                    "Write the final markdown report. If key parts are missing, state them.\n"
                    "If reliable image URLs are present in materials, include key ones with markdown image syntax: ![caption](url)."
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
            whitelist=[],
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
