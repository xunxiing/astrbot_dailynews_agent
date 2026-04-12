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
from ..agents.react.orchestrator import ReActDailyNewsOrchestrator
from ..agents.single_agent.single_agent_writer import SingleAgentNewsWriter
from ..core.config_models import ImageLayoutConfig
from ..core.markdown_sanitizer import sanitize_markdown_for_publish
from ..core.models import NewsSourceConfig


class NewsWorkflowManager:
    """News workflow with auto-discovered sources, single-agent mode, and react mode."""

    def __init__(self):
        # Kept as `sub_agents` for compatibility with main.py source-test helpers.
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
                "[dailynews] [workflow] run_workflow rejected: another workflow is already running. source=%s",
                source,
            )
            return {
                "status": "error",
                "error": "已有日报任务正在运行中，请稍后再试。",
                "timestamp": datetime.now().isoformat(),
            }

        async with self._lock:
            fetch_tasks: list[asyncio.Task] = []

            async def _wait_with_heartbeat(
                tasks: list[asyncio.Task], *, label: str, interval_s: float = 20.0
            ) -> list[Any]:
                pending = set(tasks)
                results: list[Any] = [None] * len(tasks)
                index_map = {task: idx for idx, task in enumerate(tasks)}

                while pending:
                    done, pending = await asyncio.wait(
                        pending, timeout=float(interval_s)
                    )
                    for task in done:
                        idx = index_map.get(task)
                        if idx is None:
                            continue
                        try:
                            results[idx] = task.result()
                        except Exception as exc:
                            results[idx] = exc

                    if pending:
                        names: list[str] = []
                        for task in list(pending)[:6]:
                            try:
                                names.append(task.get_name())
                            except Exception:
                                names.append("task")
                        astrbot_logger.info(
                            "[dailynews] [%s] heartbeat pending=%d examples=%s",
                            label,
                            len(pending),
                            names,
                        )

                return results

            async def _cancel_tasks(tasks: list[asyncio.Task], *, label: str) -> None:
                alive = [task for task in tasks if task is not None and not task.done()]
                if not alive:
                    return
                for task in alive:
                    try:
                        task.cancel()
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

            try:
                all_sources = list(self.news_sources)
                if not all_sources:
                    return {
                        "status": "error",
                        "error": "未配置任何 news_sources",
                        "timestamp": datetime.now().isoformat(),
                    }

                raw_mode = str(user_config.get("news_workflow_mode") or "").strip().lower()
                workflow_mode = (
                    "single"
                    if raw_mode in {"single", "single_agent", "single-agent"}
                    else "react"
                )
                if raw_mode in {"multi", "multi_agent", "multi-agent"}:
                    astrbot_logger.info(
                        "[dailynews] workflow mode `%s` is no longer supported; fallback to react",
                        raw_mode,
                    )

                fetch_concurrency = max(1, min(6, len(all_sources)))
                fetch_timeout_s = 180.0
                fetch_sem = asyncio.Semaphore(fetch_concurrency)
                fetch_task_sources: list[NewsSourceConfig] = []

                async def _timed_fetch(src: NewsSourceConfig, agent: Any) -> Any:
                    start = time.monotonic()
                    astrbot_logger.info(
                        "[dailynews] [fetch] start source=%s type=%s",
                        src.name,
                        src.type,
                    )
                    try:
                        async with fetch_sem:
                            return await asyncio.wait_for(
                                agent.fetch_latest_articles(src, user_config),
                                timeout=float(fetch_timeout_s),
                            )
                    finally:
                        astrbot_logger.info(
                            "[dailynews] [fetch] done source=%s type=%s cost_ms=%s",
                            src.name,
                            src.type,
                            int((time.monotonic() - start) * 1000),
                        )

                for src in all_sources:
                    agent_cls = self.sub_agents.get(src.type)
                    if not agent_cls:
                        astrbot_logger.warning(
                            "[dailynews] skip source without agent: %s (%s)",
                            src.name,
                            src.type,
                        )
                        continue
                    task = asyncio.create_task(_timed_fetch(src, agent_cls()))
                    try:
                        task.set_name(f"fetch:{src.name}")
                    except Exception:
                        pass
                    fetch_tasks.append(task)
                    fetch_task_sources.append(src)

                if not fetch_tasks:
                    return {
                        "status": "error",
                        "error": "没有可用的信息源处理器",
                        "timestamp": datetime.now().isoformat(),
                    }

                astrbot_logger.info(
                    "[dailynews] [workflow] fetching article lists for %d sources (concurrency=%d timeout_s=%s)",
                    len(fetch_tasks),
                    fetch_concurrency,
                    int(fetch_timeout_s),
                )
                fetch_results = await _wait_with_heartbeat(
                    fetch_tasks, label="fetch", interval_s=20.0
                )

                fetched: dict[str, list[dict[str, Any]]] = {}
                active_sources = list(fetch_task_sources)
                for idx, result in enumerate(fetch_results):
                    src = fetch_task_sources[idx]
                    if isinstance(result, Exception):
                        astrbot_logger.warning(
                            "[dailynews] fetch_latest_articles failed for %s: %s",
                            src.name,
                            result,
                            exc_info=True,
                        )
                        continue
                    name, articles = result
                    fetched[name] = articles or []

                if workflow_mode == "single":
                    single_config = dict(user_config or {})
                    single_config["news_workflow_mode"] = "single"
                    final_summary = await SingleAgentNewsWriter().write_report(
                        sources=active_sources,
                        fetched=fetched,
                        user_config=single_config,
                        astrbot_context=astrbot_context,
                    )
                    final_summary = sanitize_markdown_for_publish(
                        str(final_summary or "")
                    ).strip()
                    if not final_summary:
                        return {
                            "status": "error",
                            "error": "single-agent workflow produced empty final summary",
                            "timestamp": datetime.now().isoformat(),
                        }

                    return {
                        "status": "success",
                        "mode": "single",
                        "decision": None,
                        "sub_reports": [],
                        "sub_results": [],
                        "image_plan": None,
                        "final_summary": final_summary,
                        "timestamp": datetime.now().isoformat(),
                    }

                layout_cfg = ImageLayoutConfig.from_mapping(user_config)
                user_goal = str(
                    user_config.get("react_user_goal")
                    or user_config.get("user_goal")
                    or ""
                ).strip()
                if not user_goal:
                    user_goal = (
                        "Generate a daily report that covers today's important updates "
                        "from all configured sources, with concrete details and links."
                    )

                astrbot_logger.info("[dailynews] [workflow] react mode enabled")
                react_result = await ReActDailyNewsOrchestrator(
                    sub_agent_classes=self.sub_agents
                ).run(
                    user_goal=user_goal,
                    user_config=user_config,
                    astrbot_context=astrbot_context,
                    sources=active_sources,
                    fetched=fetched,
                )

                final_md = sanitize_markdown_for_publish(
                    str(react_result.final_markdown or "")
                )
                react_layout_sub_results = list(
                    getattr(react_result, "layout_sub_results", []) or []
                )
                react_layout_guidance = str(
                    getattr(react_result, "image_layout_guidance", "") or ""
                ).strip()

                if layout_cfg.enabled and final_md and react_layout_sub_results:
                    try:
                        astrbot_logger.info(
                            "[dailynews][react] image_layout start sources=%s guidance=%s",
                            len(react_layout_sub_results),
                            bool(react_layout_guidance),
                        )
                        final_md = await ImageLayoutAgent().enhance_markdown(
                            draft_markdown=final_md,
                            sub_results=react_layout_sub_results,
                            user_config=user_config,
                            astrbot_context=astrbot_context,
                            image_plan=None,
                            layout_guidance=react_layout_guidance,
                        )
                        final_md = sanitize_markdown_for_publish(str(final_md or ""))
                        astrbot_logger.info(
                            "[dailynews][react] image_layout done has_image=%s",
                            bool(re.search(r"!\[[^\]]*\]\(", (final_md or "")))
                            or ("<img" in (final_md or "")),
                        )
                    except Exception as exc:
                        astrbot_logger.warning(
                            "[dailynews][react] image_layout failed: %s",
                            exc,
                            exc_info=True,
                        )

                if not final_md:
                    return {
                        "status": "error",
                        "error": "react mode produced empty final summary",
                        "mode": "react",
                        "react_meta": {
                            "steps": react_result.steps,
                            "termination_reason": react_result.termination_reason,
                            "tool_calls": len(react_result.tool_trace),
                            "status": react_result.status,
                        },
                        "timestamp": datetime.now().isoformat(),
                    }

                astrbot_logger.info(
                    "[dailynews] [workflow] run_workflow completed successfully. source=%s mode=react",
                    source,
                )
                return {
                    "status": "success",
                    "mode": "react",
                    "decision": None,
                    "sub_reports": [],
                    "sub_results": [],
                    "image_plan": None,
                    "react_meta": {
                        "steps": react_result.steps,
                        "termination_reason": react_result.termination_reason,
                        "tool_calls": len(react_result.tool_trace),
                        "status": react_result.status,
                    },
                    "final_summary": final_md,
                    "timestamp": datetime.now().isoformat(),
                }

            except asyncio.CancelledError:
                astrbot_logger.warning(
                    "[dailynews] [workflow] run_workflow cancelled. source=%s", source
                )
                await _cancel_tasks(fetch_tasks, label="fetch")
                raise
            except Exception as exc:
                import traceback

                await _cancel_tasks(fetch_tasks, label="fetch")
                return {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }
