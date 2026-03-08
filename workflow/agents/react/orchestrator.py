from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from ...core.rss import fetch_rss_feed, format_rss_feed_for_tool
from ...core.skland_official import (
    fetch_skland_official_grouped,
    flatten_skland_grouped,
    format_skland_posts_for_tool,
)
from ...core.utils import _run_sync
from ...pipeline.rendering import load_template
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


def _safe_unix_ts(value: Any) -> int:
    try:
        n = int(str(value or "").strip())
    except Exception:
        return 0
    return n // 1000 if n > 10_000_000_000 else n


def _exception_text(exc: BaseException) -> str:
    msg = str(exc).strip()
    if msg:
        return msg
    rep = repr(exc).strip()
    if rep:
        return rep
    return type(exc).__name__


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
    analysis_report = payload.get("analysis_report")
    analysis_warning = ""
    if isinstance(analysis_report, dict):
        ar_phase = str(analysis_report.get("phase") or "").strip()
        ar_status = str(analysis_report.get("status") or "").strip()
        ar_error = str(analysis_report.get("error") or "").strip()
        if ar_error or (ar_status and ar_status not in {"ok", "success"}):
            parts = [
                f"phase={ar_phase or 'analyze_source'}",
                f"status={ar_status or 'error'}",
            ]
            if ar_error:
                parts.append(f"error={ar_error}")
            tmo = analysis_report.get("timeout_s")
            if tmo not in (None, ""):
                parts.append(f"timeout_s={tmo}")
            analysis_warning = " | ".join(parts)

    lines: list[str] = [f"Source: {source_name}"]
    if analysis_warning:
        lines.extend(["Tool Warning:", analysis_warning, ""])
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


def _build_vertical_fallback_payload(
    *,
    source: NewsSourceConfig,
    articles: list[dict[str, Any]],
    keyword: str,
    reason: str,
    analysis_report: Any = None,
) -> dict[str, Any]:
    chosen = articles[: max(1, int(getattr(source, "max_articles", 3) or 3))]
    lines = [f"## {source.name}", "", f"（子分析超时，已回退为链接摘要：{reason}）", ""]
    key_points: list[str] = []
    images: list[str] = []
    seen_img: set[str] = set()
    for a in chosen:
        if not isinstance(a, dict):
            continue
        t = str(a.get("title") or a.get("name") or "").strip()
        u = str(a.get("url") or a.get("link") or "").strip()
        if t and u:
            lines.append(f"- {t} ( [阅读原文]({u}) )")
            key_points.append(t)
        elif u:
            lines.append(f"- [阅读原文]({u})")
        elif t:
            lines.append(f"- {t}")
            key_points.append(t)

        for k in ("image_urls", "images", "image_list"):
            vals = a.get(k)
            if isinstance(vals, list):
                for x in vals:
                    s = str(x or "").strip()
                    if s.startswith(("http://", "https://")) and s not in seen_img:
                        seen_img.add(s)
                        images.append(s)
                        if len(images) >= 20:
                            break
            if len(images) >= 20:
                break

    summary = f"Fetched {len(articles)} items from source `{source.name}`."
    if keyword:
        summary += f" Keyword: {keyword}."
    return {
        "source_name": source.name,
        "summary": summary,
        "key_points": key_points[:8],
        "content": "\n".join(lines).strip(),
        "images": images,
        "error": reason,
        "analysis_report": analysis_report,
    }


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


def _format_web_results(
    query: str, rows: list[dict[str, str]]
) -> tuple[str, dict[str, Any]]:
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
    all_seed_urls: list[str] = []
    all_candidate_urls: list[str] = []
    all_hosts: list[str] = []

    for src in (sources or [])[:max_sources]:
        if not isinstance(src, NewsSourceConfig):
            continue
        source_urls: list[str] = []
        src_url = str(src.url or "").strip()
        if src_url and src_url not in source_urls:
            source_urls.append(src_url)
            if src_url not in all_seed_urls:
                all_seed_urls.append(src_url)
            h0 = _host_from_url(src_url)
            if h0 and h0 not in all_hosts:
                all_hosts.append(h0)

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
            if u not in all_candidate_urls:
                all_candidate_urls.append(u)
            h = _host_from_url(u)
            if h and h not in all_hosts:
                all_hosts.append(h)
            if len(all_candidate_urls) >= max_total_urls:
                break
        if len(all_candidate_urls) >= max_total_urls:
            break

    return {
        "sources": rows,
        # Keep tool-call boundary stable: pass only configured seed URLs.
        "target_urls": all_seed_urls[:max_total_urls],
        # Candidate URLs are still available for diagnostics.
        "candidate_urls": all_candidate_urls[:max_total_urls],
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
        "4) If some target URLs are unreachable, report the gap explicitly in final output.\n"
        "5) If image URLs are numerous, do not inline all of them in prompts. "
        "Use tool `image_url_download` (or `image_urls_download_batch`) to fetch only key images first."
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
        llm_timeout_s = _safe_int(
            getattr(llm_write, "_timeout_s", 45),
            45,
            minimum=5,
            maximum=300,
        )
        analyze_timeout_default = max(10, llm_timeout_s + 8)
        analyze_timeout_s = _safe_int(
            user_config.get(
                "react_vertical_analyze_timeout_s", analyze_timeout_default
            ),
            analyze_timeout_default,
            minimum=5,
            maximum=300,
        )
        process_timeout_s = _safe_int(
            user_config.get("react_vertical_process_timeout_s", 120),
            120,
            minimum=10,
            maximum=600,
        )
        min_analyze_timeout_s = max(10, llm_timeout_s + 5)
        if analyze_timeout_s < min_analyze_timeout_s:
            astrbot_logger.warning(
                "[dailynews][react] analyze timeout auto-raised source=%s %s->%s (llm_timeout_s=%s)",
                source.name,
                analyze_timeout_s,
                min_analyze_timeout_s,
                llm_timeout_s,
            )
            analyze_timeout_s = min_analyze_timeout_s

        min_process_timeout_s = max(15, llm_timeout_s + 20)
        if process_timeout_s < min_process_timeout_s:
            astrbot_logger.warning(
                "[dailynews][react] process timeout auto-raised source=%s %s->%s (llm_timeout_s=%s)",
                source.name,
                process_timeout_s,
                min_process_timeout_s,
                llm_timeout_s,
            )
            process_timeout_s = min_process_timeout_s

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
                report = await asyncio.wait_for(
                    agent.analyze_source(source, articles, llm_write),
                    timeout=float(analyze_timeout_s),
                )
            except asyncio.TimeoutError as e:
                err_text = _exception_text(e)
                astrbot_logger.warning(
                    "[dailynews][react] analyze_source timeout source=%s timeout_s=%s llm_timeout_s=%s article_count=%s keyword=%s err=%s",
                    source.name,
                    analyze_timeout_s,
                    llm_timeout_s,
                    len(articles or []),
                    kw or "",
                    err_text,
                    exc_info=True,
                )
                report = {
                    "source_name": source.name,
                    "phase": "analyze_source",
                    "status": "timeout",
                    "error": f"analyze_timeout>{analyze_timeout_s}s",
                    "exception": err_text,
                    "timeout_s": analyze_timeout_s,
                    "llm_timeout_s": llm_timeout_s,
                    "article_count": len(articles or []),
                    "recoverable": True,
                    "next_action": "continue_process_source",
                }
            except Exception as e:
                err_text = _exception_text(e)
                astrbot_logger.warning(
                    "[dailynews][react] analyze_source fallback source=%s type=%s timeout_s=%s llm_timeout_s=%s article_count=%s keyword=%s err=%s",
                    source.name,
                    type(e).__name__,
                    analyze_timeout_s,
                    llm_timeout_s,
                    len(articles or []),
                    kw or "",
                    err_text,
                    exc_info=True,
                )
                report = {
                    "source_name": source.name,
                    "phase": "analyze_source",
                    "status": "error",
                    "error": f"analyze_error:{type(e).__name__}",
                    "exception": err_text,
                    "timeout_s": analyze_timeout_s,
                    "llm_timeout_s": llm_timeout_s,
                    "article_count": len(articles or []),
                    "recoverable": True,
                    "next_action": "continue_process_source",
                }

            full_instruction = instruction
            if injected_prompt:
                full_instruction = (
                    f"{instruction}\n\nDependency context:\n{injected_prompt}"
                )

            async def _call_process() -> Any:
                try:
                    return await agent.process_source(
                        source,
                        full_instruction,
                        articles,
                        llm_write,
                        user_config=user_config,
                    )
                except TypeError:
                    return await agent.process_source(
                        source,
                        full_instruction,
                        articles,
                        llm_write,
                    )

            try:
                result = await asyncio.wait_for(
                    _call_process(),
                    timeout=float(process_timeout_s),
                )
            except asyncio.TimeoutError:
                astrbot_logger.warning(
                    "[dailynews][react] process_source timeout source=%s timeout_s=%s llm_timeout_s=%s article_count=%s keyword=%s",
                    source.name,
                    process_timeout_s,
                    llm_timeout_s,
                    len(articles or []),
                    kw or "",
                    exc_info=True,
                )
                return _build_vertical_fallback_payload(
                    source=source,
                    articles=articles,
                    keyword=kw,
                    reason=f"process_timeout>{process_timeout_s}s",
                    analysis_report=report,
                )
            except Exception as e:
                err_text = _exception_text(e)
                astrbot_logger.warning(
                    "[dailynews][react] process_source fallback source=%s type=%s timeout_s=%s llm_timeout_s=%s article_count=%s keyword=%s err=%s",
                    source.name,
                    type(e).__name__,
                    process_timeout_s,
                    llm_timeout_s,
                    len(articles or []),
                    kw or "",
                    err_text,
                    exc_info=True,
                )
                return _build_vertical_fallback_payload(
                    source=source,
                    articles=articles,
                    keyword=kw,
                    reason=f"process_error:{type(e).__name__}",
                    analysis_report=report,
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

        def _pick_miyoushe_post_list_url() -> str:
            for u in target_url_pool:
                lu = u.lower()
                if (
                    "miyoushe.com" in lu
                    and "accountcenter/postlist" in lu
                    and ("id=" in lu or "uid=" in lu)
                ):
                    return u
            for u in target_url_pool:
                lu = u.lower()
                if "miyoushe.com" in lu and "accountcenter/postlist" in lu:
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
                    if chosen_url and "/article/" in chosen_url.lower():
                        list_url = _pick_miyoushe_post_list_url()
                        if list_url:
                            chosen_url = list_url
                    elif (
                        chosen_url
                        and "miyoushe.com" in chosen_url.lower()
                        and "accountcenter/postlist" not in chosen_url.lower()
                    ):
                        list_url = _pick_miyoushe_post_list_url()
                        if list_url:
                            chosen_url = list_url
                    if not chosen_url and uid_s:
                        chosen_url = f"https://www.miyoushe.com/ys/accountCenter/postList?id={uid_s}"
                    if not chosen_url:
                        chosen_url = _pick_miyoushe_post_list_url() or _pick_target_url(
                            "miyoushe.com"
                        )
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
                    keyword: str = "",
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
                        if not kw:
                            return "error: provide username/url/profile_url or keyword."
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
                    },
                    handler=_tool_analyze_twitter,
                )

            elif st == "wechat":

                async def _tool_analyze_wechat(
                    *,
                    context,
                    keyword: str = "",
                    url: str = "",
                    album_url: str = "",
                    album_keyword: str = "",
                    latest_scope: str = "",
                    max_articles: int = 5,
                    dependency_ids: list[str] | None = None,
                    _agent_cls=agent_cls,
                ):
                    _ = context
                    kw = str(keyword or "").strip()
                    scope = str(latest_scope or "").strip().lower()
                    if scope not in {"auto", "account", "album"}:
                        scope = ""
                    chosen_url = _pick_first_url_arg(url, album_url)
                    wechat_target_urls = [
                        u
                        for u in target_url_pool
                        if isinstance(u, str) and "mp.weixin.qq.com" in u.lower()
                    ]
                    if not chosen_url and wechat_target_urls:
                        chosen_urls = wechat_target_urls[:6]
                    elif chosen_url:
                        chosen_urls = [chosen_url]
                    else:
                        chosen_urls = []
                    if not chosen_urls:
                        if not kw:
                            return "error: provide url/album_url or keyword."
                        return await self._vertical_web_fallback(
                            platform="wechat",
                            keyword=kw,
                            memory=memory,
                            site_filter="mp.weixin.qq.com",
                        )

                    # React-phase WeChat tool: preprocess URL to a usable article seed,
                    # then parse directly into markdown for downstream agent use.
                    try:
                        analysis_mod = importlib.import_module(
                            "analysis.wechatanalysis.analysis"
                        )
                    except Exception as e:
                        return f"error: import wechat analysis failed: {_exception_text(e)}"

                    fetch_latest = getattr(analysis_mod, "fetch_latest_articles", None)
                    to_markdown = getattr(analysis_mod, "wechat_to_markdown", None)
                    if not callable(to_markdown):
                        return "error: wechat_to_markdown is not available."

                    parse_limit = _safe_int(max_articles, 5, minimum=1, maximum=12)
                    max_age_hours = 36
                    cutoff_ts = (
                        int(datetime.now(timezone.utc).timestamp())
                        - int(max_age_hours) * 3600
                        if max_age_hours > 0
                        else 0
                    )
                    preferred_scope = scope or "auto"
                    pre_scope = (
                        "account" if preferred_scope in {"auto", "account"} else "album"
                    )
                    output_dir = (
                        Path(__file__).resolve().parents[3]
                        / "analysis"
                        / "wechatanalysis"
                        / "output"
                    )
                    fetch_article = getattr(analysis_mod, "fetch_wechat_article", None)

                    async def _analyze_one(input_url: str) -> tuple[bool, str]:
                        resolved_url = input_url
                        resolved_scope = pre_scope
                        meta: dict[str, Any] = {}
                        stale_only = False

                        if callable(fetch_latest):
                            try:
                                rows, meta = await _run_sync(
                                    fetch_latest,
                                    input_url,
                                    parse_limit,
                                    latest_scope=pre_scope,
                                )
                                if max_age_hours > 0 and isinstance(rows, list) and rows:
                                    filtered_rows: list[dict[str, Any]] = []
                                    stale_count = 0
                                    for row in rows:
                                        if not isinstance(row, dict):
                                            continue
                                        row_ts = _safe_unix_ts(row.get("create_time"))
                                        if row_ts > 0 and row_ts < cutoff_ts:
                                            stale_count += 1
                                            continue
                                        filtered_rows.append(row)
                                    if stale_count > 0:
                                        astrbot_logger.info(
                                            "[dailynews][react] wechat stale-filter dropped=%s kept=%s max_age_hours=%s url=%s",
                                            stale_count,
                                            len(filtered_rows),
                                            max_age_hours,
                                            input_url,
                                        )
                                    stale_only = bool(rows) and not bool(filtered_rows)
                                    rows = filtered_rows
                                if (
                                    isinstance(rows, list)
                                    and rows
                                    and isinstance(rows[0], dict)
                                    and str(rows[0].get("url") or "").strip()
                                ):
                                    resolved_url = str(rows[0].get("url")).strip()
                                resolved_scope = str((meta or {}).get("scope") or pre_scope)
                            except Exception as e:
                                astrbot_logger.warning(
                                    "[dailynews][react] wechat preprocess latest-list failed url=%s scope=%s err=%s",
                                    input_url,
                                    pre_scope,
                                    _exception_text(e),
                                    exc_info=True,
                                )
                        if stale_only:
                            return (
                                False,
                                "latest wechat articles exceed freshness window "
                                f"({max_age_hours}h)",
                            )

                        if cutoff_ts > 0 and callable(fetch_article):
                            try:
                                detail = await _run_sync(fetch_article, resolved_url)
                                article_ts = _safe_unix_ts(
                                    (detail or {}).get("ct")
                                    or (detail or {}).get("create_time")
                                )
                                if article_ts > 0 and article_ts < cutoff_ts:
                                    return (
                                        False,
                                        "wechat article is older than freshness window "
                                        f"({max_age_hours}h)",
                                    )
                            except Exception:
                                pass

                        markdown_scope = (
                            "account" if pre_scope == "account" else preferred_scope
                        )
                        try:
                            md_path = await _run_sync(
                                to_markdown,
                                resolved_url,
                                str(output_dir),
                                limit=parse_limit,
                                latest_scope=markdown_scope,
                                download_images=False,
                            )
                        except Exception as e:
                            return (
                                False,
                                "wechat_to_markdown failed "
                                f"(url={resolved_url}, scope={markdown_scope}): {_exception_text(e)}",
                            )

                        try:
                            with open(str(md_path), encoding="utf-8") as f:
                                md_text = f.read().strip()
                        except Exception as e:
                            return False, f"read markdown failed ({md_path}): {_exception_text(e)}"

                        if len(md_text) > 10000:
                            md_text = md_text[:10000] + "\n\n...(truncated)"
                        header = [
                            f"- input_url: {input_url}",
                            f"- resolved_url: {resolved_url}",
                            f"- preprocess_scope: {pre_scope}",
                            f"- resolved_scope: {resolved_scope or '-'}",
                            f"- latest_meta_error: {str((meta or {}).get('error') or '-')}",
                            f"- max_age_hours: {max_age_hours}",
                            f"- markdown_path: {md_path}",
                            "",
                        ]
                        return True, "\n".join(header) + md_text

                    success_blocks: list[str] = []
                    failure_blocks: list[str] = []
                    for current_url in chosen_urls:
                        try:
                            ok, block = await asyncio.wait_for(
                                _analyze_one(current_url), timeout=75.0
                            )
                        except asyncio.TimeoutError:
                            ok = False
                            block = f"wechat analyze timeout>75s ({current_url})"
                        if ok:
                            success_blocks.append(block)
                        else:
                            failure_blocks.append(f"- {current_url}: {block}")

                    if not success_blocks:
                        if failure_blocks:
                            return "error: all wechat targets failed\n" + "\n".join(
                                failure_blocks[:8]
                            )
                        return "error: no usable wechat targets found."

                    if len(success_blocks) == 1 and not failure_blocks:
                        return success_blocks[0]

                    merged: list[str] = [
                        f"# WeChat Multi-Source Analysis ({len(success_blocks)}/{len(chosen_urls)} succeeded)",
                        "",
                    ]
                    if failure_blocks:
                        merged.append("## Failed Targets")
                        merged.extend(failure_blocks[:8])
                        merged.append("")
                    for idx, block in enumerate(success_blocks, start=1):
                        merged.append(f"## Source {idx}")
                        merged.append("")
                        merged.append(block)
                        merged.append("")
                    merged_text = "\n".join(merged).strip()
                    if len(merged_text) > 18000:
                        merged_text = merged_text[:18000] + "\n\n...(truncated)"
                    return merged_text

                registry.register_callable(
                    name="tool_analyze_wechat",
                    description="Parse a WeChat link into markdown text (account-first preprocess).",
                    parameters={
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "url": {"type": "string"},
                            "album_url": {"type": "string"},
                            "album_keyword": {"type": "string"},
                            "latest_scope": {
                                "type": "string",
                                "description": "auto/account/album",
                            },
                            "max_articles": {"type": "integer"},
                        },
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
        provider_id = self._pick_provider_id(
            user_config=user_config, react_cfg=react_cfg
        )
        memory = SharedMemory.instance()
        memory.reset()
        registry = ToolRegistry()
        source_snapshot = _build_target_source_snapshot(
            sources=sources, fetched=fetched
        )
        target_hosts = source_snapshot.get("target_hosts") or []
        if not isinstance(target_hosts, list):
            target_hosts = []
        target_urls = source_snapshot.get("target_urls") or []
        if not isinstance(target_urls, list):
            target_urls = []
        initial_context = _format_target_source_context(
            source_snapshot, user_goal=user_goal
        )

        vertical_llm_timeout_s = _safe_int(
            user_config.get("react_vertical_llm_timeout_s", 45),
            45,
            minimum=5,
            maximum=300,
        )
        vertical_llm_max_retries = _safe_int(
            user_config.get("react_vertical_llm_max_retries", 0),
            0,
            minimum=0,
            maximum=3,
        )
        vertical_analyze_timeout_s = _safe_int(
            user_config.get(
                "react_vertical_analyze_timeout_s", max(10, vertical_llm_timeout_s + 8)
            ),
            max(10, vertical_llm_timeout_s + 8),
            minimum=5,
            maximum=300,
        )
        vertical_process_timeout_s = _safe_int(
            user_config.get("react_vertical_process_timeout_s", 120),
            120,
            minimum=10,
            maximum=600,
        )
        vertical_tool_timeout_floor_s = _safe_int(
            user_config.get(
                "react_vertical_tool_timeout_floor_s",
                vertical_analyze_timeout_s + vertical_process_timeout_s + 20,
            ),
            vertical_analyze_timeout_s + vertical_process_timeout_s + 20,
            minimum=30,
            maximum=1200,
        )
        if int(react_cfg.tool_call_timeout_s) < int(vertical_tool_timeout_floor_s):
            old_timeout = int(react_cfg.tool_call_timeout_s)
            react_cfg = replace(
                react_cfg, tool_call_timeout_s=vertical_tool_timeout_floor_s
            )
            astrbot_logger.warning(
                "[dailynews][react] tool timeout auto-raised: react_agent_tool_call_timeout_s %s -> %s (vertical_analyze=%s, vertical_process=%s, vertical_llm=%s)",
                old_timeout,
                int(react_cfg.tool_call_timeout_s),
                vertical_analyze_timeout_s,
                vertical_process_timeout_s,
                vertical_llm_timeout_s,
            )
        astrbot_logger.info(
            "[dailynews][react] timeout budgets tool_call=%s vertical_llm=%s vertical_analyze=%s vertical_process=%s",
            int(react_cfg.tool_call_timeout_s),
            vertical_llm_timeout_s,
            vertical_analyze_timeout_s,
            vertical_process_timeout_s,
        )
        llm_vertical = LLMRunner(
            astrbot_context,
            timeout_s=vertical_llm_timeout_s,
            max_retries=vertical_llm_max_retries,
            provider_id=provider_id or None,
        )

        llm_writer = LLMRunner(
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
            repo = parse_repo(repo_input) or parse_repo(
                f"https://github.com/{repo_input}"
            )
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

        async def _tool_read_rss_feed(
            *,
            context,
            url: str,
            limit: int = 8,
            focus: str = "",
            include_content: bool = False,
        ):
            _ = context
            link = _pick_first_url_arg(url)
            if not link:
                return "error: url is required."
            max_items = _safe_int(limit, 8, minimum=1, maximum=20)
            try:
                feed = await fetch_rss_feed(link, limit=max_items, timeout_s=20)
            except Exception as e:
                return f"error: rss fetch failed for `{link}`: {_exception_text(e)}"
            packed = {
                "url": link,
                "focus": str(focus or "").strip(),
                "feed_title": str(feed.get("feed_title") or "RSS Feed").strip(),
                "items": (
                    feed.get("items") if isinstance(feed.get("items"), list) else []
                )[:max_items],
            }
            feed_hash = hashlib.sha1(link.encode("utf-8")).hexdigest()[:10]
            memory.write(f"rss_feed::{feed_hash}", packed)
            return format_rss_feed_for_tool(
                feed,
                limit=max_items,
                focus=str(focus or "").strip(),
                include_content=bool(include_content),
            )

        registry.register_callable(
            name="tool_read_rss_feed",
            description="Read and summarize an RSS/Atom feed by subscription URL.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "limit": {"type": "integer"},
                    "focus": {"type": "string"},
                    "include_content": {"type": "boolean"},
                },
                "required": ["url"],
            },
            handler=_tool_read_rss_feed,
        )

        async def _tool_read_skland_official(
            *,
            context,
            official_name: str,
            limit: int = 8,
            date: str = "",
            focus: str = "",
        ):
            _ = context
            target_name = str(official_name or "").strip()
            if not target_name:
                return "error: official_name is required."

            target_name = re.sub(r"^森空岛", "", target_name).strip()
            target_name = re.sub(r"官方$", "", target_name).strip()
            if not target_name:
                return "error: official_name is invalid."

            max_items = _safe_int(limit, 8, minimum=1, maximum=20)
            date_text = str(date or "").strip() or None
            focus_text = str(focus or "").strip()
            try:
                grouped = await fetch_skland_official_grouped(
                    games_text=target_name,
                    date_text=date_text,
                    page_size=5,
                    max_pages=10,
                )
            except Exception as e:
                return (
                    f"error: skland official fetch failed for `{target_name}`: "
                    f"{_exception_text(e)}"
                )

            rows = flatten_skland_grouped(grouped)
            packed = {
                "official_name": target_name,
                "date": str(grouped.get("date") or "").strip(),
                "focus": focus_text,
                "items": rows[:max_items],
            }
            sk_hash = hashlib.sha1(
                f"{target_name}|{packed['date']}".encode("utf-8")
            ).hexdigest()[:10]
            memory.write(f"skland_official::{sk_hash}", packed)
            return format_skland_posts_for_tool(
                grouped,
                limit=max_items,
                focus=focus_text,
            )

        registry.register_callable(
            name="tool_read_skland_official",
            description=(
                "Read official posts from Skland for a given game/official name, "
                "such as 明日方舟、明日方舟：终末地、来自星尘。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "official_name": {"type": "string"},
                    "limit": {"type": "integer"},
                    "date": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["official_name"],
            },
            handler=_tool_read_skland_official,
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
            llm_write=llm_vertical,
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
                system_core = str(
                    load_template("templates/prompts/react_writer_system.txt") or ""
                ).strip()
                editorial_rules = str(
                    load_template("templates/prompts/daily_report_editorial_style.txt")
                    or ""
                ).strip()
                system_prompt = "\n\n".join(
                    [x for x in (system_core, editorial_rules) if x]
                ).strip()
                prompt_template = str(
                    load_template("templates/prompts/react_writer_user.txt") or ""
                ).strip()
                prompt = (
                    prompt_template.replace("{{USER_GOAL}}", str(user_goal or ""))
                    .replace(
                        "{{TARGET_SOURCE_BOUNDARY}}",
                        str(initial_context or "(none)"),
                    )
                    .replace("{{COLLECTED_MATERIALS}}", str(materials or "(none)"))
                    .replace("{{DEPENDENCY_CONTEXT}}", str(injected_prompt or "(none)"))
                    .replace("{{STYLE_HINT}}", style or "(none)")
                )
                text = await llm_writer.ask(system_prompt=system_prompt, prompt=prompt)
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

        preferred_global_tools = [
            "web_search",
            "web_search_tavily",
            "tavily_extract_web_page",
            "grok_web_search",
        ]
        merged_global = registry.merge_global_tools(
            astrbot_context,
            prefer_existing=True,
            include_inactive=False,
            whitelist=preferred_global_tools,
        )
        available_global_tools = [
            name for name in preferred_global_tools if registry.get(name) is not None
        ]
        astrbot_logger.info(
            "[dailynews][react] merged global AstrBot tools: count=%s names=%s",
            merged_global,
            available_global_tools,
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
