import asyncio
import re
import time
from datetime import datetime
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..agents.image_layout_agent import ImageLayoutAgent
from ..agents.main_agent import MainNewsAgent
from ..agents.message_groups.group_manager import MessageGroupManager
from ..agents.single_agent.single_agent_writer import SingleAgentNewsWriter
from ..core.llm import LLMRunner
from ..core.markdown_sanitizer import sanitize_markdown_for_publish
from ..core.models import NewsSourceConfig, SubAgentResult


class NewsWorkflowManager:
    """新闻多 Agent 工作流：抓取 -> 汇报 -> 分工 -> 写作 -> 汇总"""

    def __init__(self):
        self.sub_agents: dict[str, Any] = {}
        self.news_sources: list[NewsSourceConfig] = []
        self._lock = asyncio.Lock()

    def add_source(self, config: NewsSourceConfig):
        self.news_sources.append(config)

    def register_sub_agent(self, source_type: str, agent_class):
        self.sub_agents[source_type] = agent_class

    async def run_workflow(
        self,
        user_config: dict[str, Any],
        astrbot_context: Any,
        source: str = "unknown",
    ) -> dict[str, Any]:
        if self._lock.locked():
            astrbot_logger.warning(
                "[dailynews] [workflow] run_workflow rejected: another workflow is already running. Source: %s",
                source,
            )
            return {
                "status": "error",
                "error": "已有日报任务正在运行中，请稍后再试。",
                "timestamp": datetime.now().isoformat(),
            }

        async with self._lock:
            astrbot_logger.info(
                "[dailynews] [workflow] run_workflow started from source: %s", source
            )

            llm_timeout_s = int(user_config.get("llm_timeout_s", 180))
            llm_write_timeout_s = int(
                user_config.get("llm_write_timeout_s", max(llm_timeout_s, 360))
            )
            llm_merge_timeout_s = int(
                user_config.get("llm_merge_timeout_s", max(llm_timeout_s, 240))
            )
            llm_max_retries = int(user_config.get("llm_max_retries", 1))

            # 主 Agent provider 轮询链（主 + 备用），用于决策/汇总/图片计划等关键步骤
            main_provider_id = str(
                user_config.get("main_agent_provider_id") or ""
            ).strip()
            fallback_providers: list[str] = []
            raw_list = user_config.get("main_agent_fallback_provider_ids") or []
            if isinstance(raw_list, list):
                fallback_providers.extend(
                    [
                        str(x).strip()
                        for x in raw_list
                        if isinstance(x, str) and str(x).strip()
                    ]
                )
            for k in (
                "main_agent_fallback_provider_id_1",
                "main_agent_fallback_provider_id_2",
                "main_agent_fallback_provider_id_3",
            ):
                v = user_config.get(k)
                if isinstance(v, str) and v.strip():
                    fallback_providers.append(v.strip())

            uniq: list[str] = []
            for p in [main_provider_id] + fallback_providers:
                if not p or p in uniq:
                    continue
                uniq.append(p)
            main_provider_chain = uniq

            llm_scout = LLMRunner(
                astrbot_context,
                timeout_s=llm_timeout_s,
                max_retries=llm_max_retries,
                provider_ids=main_provider_chain or None,
            )
            llm_write = LLMRunner(
                astrbot_context,
                timeout_s=llm_write_timeout_s,
                max_retries=llm_max_retries,
                provider_ids=main_provider_chain or None,
            )
            llm_main_decision = LLMRunner(
                astrbot_context,
                timeout_s=llm_timeout_s,
                max_retries=llm_max_retries,
                provider_ids=main_provider_chain or None,
            )
            llm_main_merge = LLMRunner(
                astrbot_context,
                timeout_s=llm_merge_timeout_s,
                max_retries=llm_max_retries,
                provider_ids=main_provider_chain or None,
            )
            llm_group_router = None
            try:
                group_provider = str(
                    user_config.get("news_group_router_provider_id") or ""
                ).strip()
                group_provider_ids = (
                    [group_provider]
                    if group_provider
                    else (main_provider_chain or None)
                )
                llm_group_router = LLMRunner(
                    astrbot_context,
                    timeout_s=max(
                        30, min(120, int(user_config.get("llm_timeout_s", 180)) // 2)
                    ),
                    max_retries=max(
                        0, min(2, int(user_config.get("llm_max_retries", 1)))
                    ),
                    provider_ids=group_provider_ids,
                )
            except Exception:
                llm_group_router = None

            async def _wait_with_heartbeat(
                tasks: list[asyncio.Task], *, label: str, interval_s: float = 30.0
            ) -> list[Any]:
                pending = set(tasks)
                results: list[Any] = [None] * len(tasks)
                index_map = {t: i for i, t in enumerate(tasks)}

                while pending:
                    done, pending = await asyncio.wait(
                        pending, timeout=float(interval_s)
                    )
                    for t in done:
                        idx = index_map.get(t)
                        if idx is None:
                            continue
                        try:
                            results[idx] = t.result()
                        except Exception as e:
                            results[idx] = e

                    if pending:
                        names: list[str] = []
                        for t in list(pending)[:6]:
                            try:
                                names.append(t.get_name())
                            except Exception:
                                names.append("task")
                        astrbot_logger.info(
                            "[dailynews] [%s] heartbeat pending=%d examples=%s",
                            label,
                            len(pending),
                            names,
                        )

                return results

            async def _cancel_tasks(tasks: list[asyncio.Task], *, label: str):
                alive = [t for t in tasks if t is not None and not t.done()]
                if not alive:
                    return
                for t in alive:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*alive, return_exceptions=True),
                        timeout=5.0,
                    )
                except Exception:
                    astrbot_logger.warning(
                        "[dailynews] cancel %s tasks timeout", label, exc_info=True
                    )

            fetch_tasks: list[asyncio.Task] = []
            report_tasks: list[asyncio.Task] = []
            write_tasks: list[asyncio.Task] = []

            try:
                sources = list(self.news_sources)
                if not sources:
                    return {
                        "status": "error",
                        "error": "未配置任何 news_sources",
                        "timestamp": datetime.now().isoformat(),
                    }

                # 1) 每个来源抓一次「最新文章列表」
                fetched: dict[str, list[dict[str, Any]]] = {}
                fetch_task_sources: list[NewsSourceConfig] = []
                fetch_concurrency = max(
                    1, min(6, int(user_config.get("max_sources_per_day", 3)) * 2)
                )
                fetch_sem = asyncio.Semaphore(fetch_concurrency)
                fetch_list_timeout_s = 180.0

                async def _timed_fetch(src: NewsSourceConfig, agent) -> Any:
                    start = time.monotonic()
                    astrbot_logger.info(
                        "[dailynews] [fetch_list] start source=%s type=%s",
                        src.name,
                        src.type,
                    )
                    try:
                        async with fetch_sem:
                            return await asyncio.wait_for(
                                agent.fetch_latest_articles(src, user_config),
                                timeout=float(fetch_list_timeout_s),
                            )
                    finally:
                        astrbot_logger.info(
                            "[dailynews] [fetch_list] done source=%s type=%s cost_ms=%s",
                            src.name,
                            src.type,
                            int((time.monotonic() - start) * 1000),
                        )

                for src in sources:
                    agent_cls = self.sub_agents.get(src.type)
                    if not agent_cls:
                        continue
                    t = asyncio.create_task(_timed_fetch(src, agent_cls()))
                    try:
                        t.set_name(f"fetch_list:{src.name}")
                    except Exception:
                        pass
                    fetch_tasks.append(t)
                    fetch_task_sources.append(src)

                astrbot_logger.info(
                    "[dailynews] [workflow] stage 1: fetching article lists for %d sources (concurrency=%d timeout_s=%s)",
                    len(fetch_tasks),
                    fetch_concurrency,
                    int(fetch_list_timeout_s),
                )
                fetch_results = await _wait_with_heartbeat(
                    fetch_tasks, label="stage1/fetch_list", interval_s=20.0
                )
                astrbot_logger.info(
                    "[dailynews] [workflow] stage 1: finished fetching article lists"
                )
                for idx, r in enumerate(fetch_results):
                    if isinstance(r, Exception):
                        src = (
                            fetch_task_sources[idx]
                            if idx < len(fetch_task_sources)
                            else None
                        )
                        astrbot_logger.warning(
                            "[dailynews] fetch_latest_articles failed for %s: %s",
                            getattr(src, "name", "unknown"),
                            r,
                            exc_info=True,
                        )
                        continue
                    name, articles = r
                    fetched[name] = articles

                workflow_mode = (
                    str(user_config.get("news_workflow_mode", "multi") or "multi")
                    .strip()
                    .lower()
                )
                if workflow_mode in {"single", "single_agent", "single-agent"}:
                    astrbot_logger.info(
                        "[dailynews] [workflow] single-agent mode enabled"
                    )

                    t = asyncio.create_task(
                        SingleAgentNewsWriter().write_report(
                            sources=sources,
                            fetched=fetched,
                            user_config=user_config,
                            astrbot_context=astrbot_context,
                        )
                    )
                    try:
                        t.set_name("single_agent:write_report")
                    except Exception:
                        pass
                    write_tasks.append(t)

                    res = await _wait_with_heartbeat(
                        [t], label="single_agent/write", interval_s=20.0
                    )
                    final_md = res[0] if res else ""
                    if isinstance(final_md, Exception):
                        raise final_md

                    final_md = sanitize_markdown_for_publish(str(final_md or ""))
                    astrbot_logger.info(
                        "[dailynews] [workflow] single-agent workflow completed"
                    )
                    return {
                        "status": "success",
                        "decision": None,
                        "sub_reports": [],
                        "sub_results": [],
                        "image_plan": None,
                        "final_summary": final_md,
                        "timestamp": datetime.now().isoformat(),
                    }

                # 2) 子 Agent 汇报：快速判断“今日看点”
                reports: list[dict[str, Any]] = []
                report_task_sources: list[NewsSourceConfig] = []
                report_concurrency = max(1, min(6, len(sources)))
                report_sem = asyncio.Semaphore(report_concurrency)

                async def _timed_analyze(src: NewsSourceConfig, agent) -> Any:
                    start = time.monotonic()
                    astrbot_logger.info(
                        "[dailynews] [analyze] start source=%s type=%s",
                        src.name,
                        src.type,
                    )
                    try:
                        async with report_sem:
                            return await asyncio.wait_for(
                                agent.analyze_source(
                                    src, fetched.get(src.name, []), llm_scout
                                ),
                                timeout=float(llm_timeout_s + 30),
                            )
                    finally:
                        astrbot_logger.info(
                            "[dailynews] [analyze] done source=%s type=%s cost_ms=%s",
                            src.name,
                            src.type,
                            int((time.monotonic() - start) * 1000),
                        )

                for src in sources:
                    agent_cls = self.sub_agents.get(src.type)
                    if not agent_cls:
                        continue
                    t = asyncio.create_task(_timed_analyze(src, agent_cls()))
                    try:
                        t.set_name(f"analyze:{src.name}")
                    except Exception:
                        pass
                    report_tasks.append(t)
                    report_task_sources.append(src)

                astrbot_logger.info(
                    "[dailynews] [workflow] stage 2: sub-agent analyzing sources (%d tasks, concurrency=%d)",
                    len(report_tasks),
                    report_concurrency,
                )
                report_results = await _wait_with_heartbeat(
                    report_tasks, label="stage2/analyze", interval_s=20.0
                )
                astrbot_logger.info(
                    "[dailynews] [workflow] stage 2: finished sub-agent analysis"
                )
                for idx, r in enumerate(report_results):
                    if isinstance(r, Exception):
                        src = (
                            report_task_sources[idx]
                            if idx < len(report_task_sources)
                            else None
                        )
                        astrbot_logger.warning(
                            "[dailynews] analyze_source failed for %s: %s",
                            getattr(src, "name", "unknown"),
                            r,
                            exc_info=True,
                        )
                        if src is not None:
                            reports.append(
                                {
                                    "source_name": src.name,
                                    "source_type": src.type,
                                    "source_url": src.url,
                                    "priority": src.priority,
                                    "article_count": len(
                                        fetched.get(src.name, []) or []
                                    ),
                                    "topics": [],
                                    "quality_score": 0,
                                    "today_angle": "",
                                    "sample_articles": (
                                        fetched.get(src.name, []) or []
                                    )[:3],
                                    "error": str(r) or type(r).__name__,
                                }
                            )
                        continue
                    reports.append(r)

                # 3) 主 Agent 听汇报 -> 做分工决策
                main_agent = MainNewsAgent()
                decision = await main_agent.analyze_sub_agent_reports(
                    reports, user_config, llm_main_decision
                )

                # 兜底：当 LLM 只选择部分来源时，按“优先类型 + 其它类型”补齐到 max_sources_per_day
                preferred_types = user_config.get("preferred_source_types", ["wechat"])
                if not isinstance(preferred_types, list):
                    preferred_types = ["wechat"]
                preferred_set = {
                    str(x).strip() for x in preferred_types if str(x).strip()
                }
                max_sources = int(user_config.get("max_sources_per_day", 3))
                preferred_sources = [
                    s for s in sources if (not preferred_set) or s.type in preferred_set
                ]
                other_sources = [
                    s for s in sources if preferred_set and s.type not in preferred_set
                ]
                candidate_sources = preferred_sources + other_sources
                wanted = min(len(candidate_sources), max_sources)

                selected: list[str] = []
                for name in decision.sources_to_process or []:
                    if not isinstance(name, str):
                        continue
                    n = name.strip()
                    if not n or n in selected:
                        continue
                    if any(s.name == n for s in candidate_sources):
                        selected.append(n)
                    if len(selected) >= wanted:
                        break

                if len(selected) < wanted:
                    for s in candidate_sources:
                        if s.name in selected:
                            continue
                        selected.append(s.name)
                        if len(selected) >= wanted:
                            break

                decision.sources_to_process = selected

                # 填充 processing_instructions（优先使用 LLM 给的；缺失则使用 today_angle）
                report_map: dict[str, dict[str, Any]] = {
                    r.get("source_name"): r for r in reports if isinstance(r, dict)
                }
                decision.processing_instructions = (
                    decision.processing_instructions or {}
                )
                for name in decision.sources_to_process:
                    instr = str(
                        decision.processing_instructions.get(name, "") or ""
                    ).strip()
                    if instr:
                        continue
                    rep = report_map.get(name) or {}
                    angle = str(rep.get("today_angle") or "").strip()
                    if angle:
                        decision.processing_instructions[name] = angle
                        continue
                    topics = (
                        rep.get("topics", [])
                        if isinstance(rep.get("topics"), list)
                        else []
                    )
                    topic_str = (
                        " / ".join([str(t) for t in topics[:3]]) if topics else "热点"
                    )
                    decision.processing_instructions[name] = (
                        f"请聚焦{topic_str}，输出精炼要点并附上文章链接。"
                    )

                astrbot_logger.info(
                    "[dailynews] sources_to_process=%s (max=%s, candidates=%s)",
                    decision.sources_to_process,
                    max_sources,
                    [s.name for s in candidate_sources],
                )

                # 4) 子 Agent 按分工写各自部分（会再次抓取正文内容）
                write_concurrency = max(
                    1, min(4, len(decision.sources_to_process or []))
                )
                write_sem = asyncio.Semaphore(write_concurrency)

                async def _timed_write(
                    src: NewsSourceConfig, agent, instruction: str
                ) -> Any:
                    start = time.monotonic()
                    astrbot_logger.info(
                        "[dailynews] [write] start source=%s type=%s",
                        src.name,
                        src.type,
                    )
                    try:
                        async with write_sem:
                            return await asyncio.wait_for(
                                agent.process_source(
                                    src,
                                    instruction,
                                    fetched.get(src.name, []),
                                    llm_write,
                                    user_config=user_config,
                                ),
                                timeout=float(llm_write_timeout_s + 60),
                            )
                    finally:
                        astrbot_logger.info(
                            "[dailynews] [write] done source=%s type=%s cost_ms=%s",
                            src.name,
                            src.type,
                            int((time.monotonic() - start) * 1000),
                        )

                for source_name in decision.sources_to_process:
                    src = next((s for s in sources if s.name == source_name), None)
                    if not src:
                        continue
                    agent_cls = self.sub_agents.get(src.type)
                    if not agent_cls:
                        continue
                    instruction = str(
                        decision.processing_instructions.get(source_name, "") or ""
                    )
                    t = asyncio.create_task(_timed_write(src, agent_cls(), instruction))
                    try:
                        t.set_name(f"write:{src.name}")
                    except Exception:
                        pass
                    write_tasks.append(t)

                astrbot_logger.info(
                    "[dailynews] [workflow] stage 4: sub-agents writing content (%d tasks, concurrency=%d)",
                    len(write_tasks),
                    write_concurrency,
                )
                sub_results = await _wait_with_heartbeat(
                    write_tasks, label="stage4/write", interval_s=30.0
                )
                astrbot_logger.info(
                    "[dailynews] [workflow] stage 4: finished writing content"
                )

                image_plan = None
                if bool(user_config.get("image_layout_enabled", False)):
                    try:
                        img_counts = {
                            r.source_name: len(r.images or [])
                            for r in sub_results
                            if isinstance(r, SubAgentResult)
                        }
                        astrbot_logger.info(
                            "[dailynews] sub_results image counts: %s", img_counts
                        )
                    except Exception:
                        pass
                    # Image planning is now handled within ImageLayoutAgent or simplified.
                    # We skip the independent ImagePlanAgent call to reduce overhead.
                    image_plan = None

                group_mode = (
                    str(user_config.get("news_group_mode", "source") or "source")
                    .strip()
                    .lower()
                )
                if group_mode == "group":
                    promote = bool(user_config.get("news_group_writeback_tags", True))
                    min_cnt = int(
                        user_config.get("news_group_promote_min_count", 2) or 2
                    )
                    # Group mode only uses LLM-mergeable sources. no_llm_merge results are passthrough-only and
                    # should not appear unless explicitly included in the markdown.
                    group_sub_results = []
                    for r in sub_results:
                        if isinstance(r, SubAgentResult) and bool(
                            getattr(r, "no_llm_merge", False)
                        ):
                            continue
                        group_sub_results.append(r)
                    final_summary = await MessageGroupManager().build_report(
                        sub_results=group_sub_results,
                        llm_classify=llm_group_router,
                        llm_write=llm_write,
                        promote_new_tags=promote,
                        promote_min_count=max(2, min(10, min_cnt)),
                    )
                else:
                    final_summary = await main_agent.summarize_all_results(
                        sub_results, decision.final_format, llm_main_merge
                    )

                if not (final_summary or "").strip():
                    astrbot_logger.warning(
                        "[dailynews] final_summary is empty; fallback to concat sections"
                    )
                    parts: list[str] = [
                        "# 每日资讯日报",
                        f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                        "",
                    ]
                    failed_msgs: list[str] = []
                    for r in sub_results:
                        if isinstance(r, Exception):
                            failed_msgs.append(str(r) or type(r).__name__)
                            continue
                        if isinstance(r, SubAgentResult):
                            if r.content and r.content.strip():
                                parts.append(r.content.strip())
                                parts.append("")
                            elif r.error:
                                failed_msgs.append(f"{r.source_name}: {r.error}")
                    if False and failed_msgs:
                        parts.append("## 错误信息")
                        for msg in failed_msgs:
                            parts.append(f"- {msg}")
                    final_summary = "\n".join(parts).strip()

                # Production hard-sanitize: no raw URLs / local paths / debug leaks.
                final_summary = sanitize_markdown_for_publish(final_summary)

                # 5) 图片排版 Agent（可选）
                if (
                    bool(user_config.get("image_layout_enabled", False))
                    and (final_summary or "").strip()
                ):
                    try:
                        # IMPORTANT: avoid sending large passthrough sections (e.g. plugin registry "recently active")
                        # into any LLM-based layout/refine steps. Those sections are appended back verbatim after layout.
                        passthrough_blocks: list[str] = []
                        layout_sub_results: list[Any] = []
                        for r in sub_results:
                            if (
                                isinstance(r, SubAgentResult)
                                and (not r.error)
                                and bool(getattr(r, "no_llm_merge", False))
                            ):
                                # Only treat as passthrough when it is already part of final_summary.
                                # This prevents no_llm_merge sources from being force-appended in group-mode.
                                c = (r.content or "").strip()
                                if c and c in (final_summary or ""):
                                    passthrough_blocks.append(c)
                                continue
                            layout_sub_results.append(r)

                        def _strip_blocks(text: str, blocks: list[str]) -> str:
                            s = text or ""
                            for b in blocks:
                                bb = (b or "").strip()
                                if not bb:
                                    continue
                                # remove with common surrounding newlines to avoid leaving huge blank areas
                                for pat in (
                                    "\n\n" + bb + "\n\n",
                                    "\n\n" + bb + "\n",
                                    "\n" + bb + "\n\n",
                                    "\n" + bb + "\n",
                                    bb + "\n\n",
                                    bb + "\n",
                                    bb,
                                ):
                                    if pat in s:
                                        s = s.replace(pat, "\n\n")
                            return s.strip()

                        layout_markdown = _strip_blocks(
                            final_summary, passthrough_blocks
                        )
                        layout_markdown = sanitize_markdown_for_publish(layout_markdown)

                        astrbot_logger.info(
                            "[dailynews] image_layout start enabled=%s provider=%s",
                            bool(user_config.get("image_layout_enabled", False)),
                            str(user_config.get("image_layout_provider_id") or ""),
                        )
                        final_summary = await ImageLayoutAgent().enhance_markdown(
                            draft_markdown=layout_markdown,
                            sub_results=layout_sub_results,
                            user_config=user_config,
                            astrbot_context=astrbot_context,
                            image_plan=image_plan,
                        )
                        final_summary = sanitize_markdown_for_publish(final_summary)
                        if passthrough_blocks:
                            tail = "\n\n".join(
                                [b for b in passthrough_blocks if (b or "").strip()]
                            ).strip()
                            if tail:
                                final_summary = (
                                    (final_summary or "").strip() + "\n\n" + tail
                                ).strip()
                        final_summary = sanitize_markdown_for_publish(final_summary)
                        astrbot_logger.info(
                            "[dailynews] image_layout done has_image=%s",
                            bool(re.search(r"!\[[^\]]*\]\(", (final_summary or "")))
                            or ("<img" in (final_summary or "")),
                        )
                    except Exception as e:
                        astrbot_logger.warning(
                            "[dailynews] image_layout failed: %s", e, exc_info=True
                        )

                astrbot_logger.info(
                    "[dailynews] [workflow] run_workflow completed successfully from source: %s",
                    source,
                )
                return {
                    "status": "success",
                    "decision": decision,
                    "sub_reports": reports,
                    "sub_results": sub_results,
                    "image_plan": image_plan,
                    "final_summary": final_summary,
                    "timestamp": datetime.now().isoformat(),
                }

            except asyncio.CancelledError:
                astrbot_logger.warning(
                    "[dailynews] [workflow] run_workflow cancelled (source=%s)", source
                )
                await _cancel_tasks(write_tasks, label="write")
                await _cancel_tasks(report_tasks, label="analyze")
                await _cancel_tasks(fetch_tasks, label="fetch_list")
                raise
            except Exception as e:
                import traceback

                await _cancel_tasks(write_tasks, label="write")
                await _cancel_tasks(report_tasks, label="analyze")
                await _cancel_tasks(fetch_tasks, label="fetch_list")
                return {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }
