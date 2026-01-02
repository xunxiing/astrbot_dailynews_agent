import asyncio
from datetime import datetime
import re
from typing import Any, Dict, List

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .llm import LLMRunner
from .main_agent import MainNewsAgent
from .image_layout_agent import ImageLayoutAgent
from .image_plan_agent import ImagePlanAgent
from .models import NewsSourceConfig, SubAgentResult


class NewsWorkflowManager:
    """新闻多 Agent 工作流：抓取 -> 汇报 -> 分工 -> 写作 -> 汇总"""

    def __init__(self):
        self.sub_agents: Dict[str, Any] = {}
        self.news_sources: List[NewsSourceConfig] = []

    def add_source(self, config: NewsSourceConfig):
        self.news_sources.append(config)

    def register_sub_agent(self, source_type: str, agent_class):
        self.sub_agents[source_type] = agent_class

    async def run_workflow(self, user_config: Dict[str, Any], astrbot_context: Any) -> Dict[str, Any]:
        llm_timeout_s = int(user_config.get("llm_timeout_s", 180))
        llm_write_timeout_s = int(user_config.get("llm_write_timeout_s", max(llm_timeout_s, 360)))
        llm_merge_timeout_s = int(user_config.get("llm_merge_timeout_s", max(llm_timeout_s, 240)))
        main_provider_id = str(user_config.get("main_agent_provider_id") or "").strip()

        llm_scout = LLMRunner(
            astrbot_context,
            timeout_s=llm_timeout_s,
            max_retries=int(user_config.get("llm_max_retries", 1)),
        )
        llm_write = LLMRunner(
            astrbot_context,
            timeout_s=llm_write_timeout_s,
            max_retries=int(user_config.get("llm_max_retries", 1)),
        )
        llm_merge = LLMRunner(
            astrbot_context,
            timeout_s=llm_merge_timeout_s,
            max_retries=int(user_config.get("llm_max_retries", 1)),
        )
        llm_main_decision = LLMRunner(
            astrbot_context,
            timeout_s=llm_timeout_s,
            max_retries=int(user_config.get("llm_max_retries", 1)),
            provider_id=main_provider_id or None,
        )
        llm_main_merge = LLMRunner(
            astrbot_context,
            timeout_s=llm_merge_timeout_s,
            max_retries=int(user_config.get("llm_max_retries", 1)),
            provider_id=main_provider_id or None,
        )
        llm_image_plan = LLMRunner(
            astrbot_context,
            timeout_s=llm_timeout_s,
            max_retries=int(user_config.get("llm_max_retries", 1)),
            provider_id=main_provider_id or None,
        )

        try:
            sources = list(self.news_sources)
            if not sources:
                return {
                    "status": "error",
                    "error": "未配置任何 news_sources",
                    "timestamp": datetime.now().isoformat(),
                }

            # 1) 每个来源抓一次「最新文章列表」
            fetched: Dict[str, List[Dict[str, str]]] = {}
            fetch_tasks = []
            fetch_task_sources: List[NewsSourceConfig] = []
            for source in sources:
                agent_cls = self.sub_agents.get(source.type)
                if not agent_cls:
                    continue
                agent = agent_cls()
                fetch_tasks.append(agent.fetch_latest_articles(source, user_config))
                fetch_task_sources.append(source)

            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for idx, r in enumerate(fetch_results):
                if isinstance(r, Exception):
                    src = fetch_task_sources[idx] if idx < len(fetch_task_sources) else None
                    astrbot_logger.warning(
                        "[dailynews] fetch_latest_articles failed for %s: %s",
                        getattr(src, "name", "unknown"),
                        r,
                        exc_info=True,
                    )
                    continue
                name, articles = r
                fetched[name] = articles

            # 2) 子 Agent 汇报：把文章链接发给各自子 Agent，让其输出“今日看点/主题”
            reports: List[Dict[str, Any]] = []
            report_tasks = []
            report_task_sources: List[NewsSourceConfig] = []
            for source in sources:
                agent_cls = self.sub_agents.get(source.type)
                if not agent_cls:
                    continue
                agent = agent_cls()
                report_tasks.append(agent.analyze_source(source, fetched.get(source.name, []), llm_scout))
                report_task_sources.append(source)
            report_results = await asyncio.gather(*report_tasks, return_exceptions=True)
            for idx, r in enumerate(report_results):
                if isinstance(r, Exception):
                    src = report_task_sources[idx] if idx < len(report_task_sources) else None
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
                                "article_count": len(fetched.get(src.name, []) or []),
                                "topics": [],
                                "quality_score": 0,
                                "today_angle": "",
                                "sample_articles": (fetched.get(src.name, []) or [])[:3],
                                "error": str(r) or type(r).__name__,
                            }
                        )
                    continue
                reports.append(r)

            # 3) 主 Agent 听汇报 -> 做分工决策
            main_agent = MainNewsAgent()
            decision = await main_agent.analyze_sub_agent_reports(reports, user_config, llm_main_decision)

            # 兜底：当 LLM 只选择部分来源时，按“优先类型 + 其它类型”补齐到 max_sources_per_day
            preferred_types = user_config.get("preferred_source_types", ["wechat"])
            if not isinstance(preferred_types, list):
                preferred_types = ["wechat"]
            preferred_set = {str(x).strip() for x in preferred_types if str(x).strip()}
            max_sources = int(user_config.get("max_sources_per_day", 3))
            preferred_sources = [s for s in sources if (not preferred_set) or s.type in preferred_set]
            other_sources = [s for s in sources if preferred_set and s.type not in preferred_set]
            candidate_sources = preferred_sources + other_sources
            wanted = min(len(candidate_sources), max_sources)

            selected: List[str] = []
            for name in decision.sources_to_process or []:
                if not isinstance(name, str):
                    continue
                n = name.strip()
                if not n or n in selected:
                    continue
                if any(s.name == n for s in candidate_sources):
                    selected.append(n)
            if len(selected) < wanted:
                for s in candidate_sources:
                    if s.name not in selected:
                        selected.append(s.name)
                    if len(selected) >= wanted:
                        break
            if selected:
                decision.sources_to_process = selected

            report_map = {
                str(r.get("source_name")): r
                for r in reports
                if isinstance(r, dict) and r.get("source_name")
            }
            for name in decision.sources_to_process:
                instr = (decision.processing_instructions.get(name) or "").strip()
                if instr:
                    continue
                rep = report_map.get(name) or {}
                angle = str(rep.get("today_angle") or "").strip()
                if angle:
                    decision.processing_instructions[name] = angle
                    continue
                topics = rep.get("topics", []) if isinstance(rep.get("topics"), list) else []
                topic_str = " / ".join([str(t) for t in topics[:3]]) if topics else "热点"
                decision.processing_instructions[name] = f"请聚焦{topic_str}，输出精炼要点并附上文章链接。"

            astrbot_logger.info(
                "[dailynews] sources_to_process=%s (max=%s, candidates=%s)",
                decision.sources_to_process,
                max_sources,
                [s.name for s in candidate_sources],
            )

            # 4) 子 Agent 按分工写各自部分（会再次抓取正文内容）
            write_tasks = []
            for source_name in decision.sources_to_process:
                source = next((s for s in sources if s.name == source_name), None)
                if not source:
                    continue
                agent_cls = self.sub_agents.get(source.type)
                if not agent_cls:
                    continue
                agent = agent_cls()
                instruction = decision.processing_instructions.get(source_name, "")
                write_tasks.append(
                    agent.process_source(
                        source,
                        instruction,
                        fetched.get(source_name, []),
                        llm_write,
                        user_config=user_config,
                    )
                )

            astrbot_logger.debug("[dailynews] write_tasks: %s", len(write_tasks))

            sub_results = await asyncio.gather(*write_tasks, return_exceptions=True)

            image_plan = None
            if bool(user_config.get("image_layout_enabled", False)):
                try:
                    img_counts = {
                        r.source_name: len(r.images or [])
                        for r in sub_results
                        if isinstance(r, SubAgentResult)
                    }
                    astrbot_logger.info("[dailynews] sub_results image counts: %s", img_counts)
                except Exception:
                    pass
                try:
                    image_plan = await ImagePlanAgent().decide_plan(
                        reports=reports,
                        decision=decision,
                        sub_results=sub_results,
                        user_config=user_config,
                        llm=llm_image_plan,
                    )
                except Exception as e:
                    astrbot_logger.warning("[dailynews] image_plan failed: %s", e, exc_info=True)

            final_summary = await main_agent.summarize_all_results(sub_results, decision.final_format, llm_main_merge)

            if not (final_summary or "").strip():
                astrbot_logger.warning("[dailynews] final_summary is empty; fallback to concat sections")
                parts: List[str] = [
                    "# 每日资讯日报",
                    f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                    "",
                ]
                failed_msgs: List[str] = []
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
                if failed_msgs:
                    parts.append("## 错误信息")
                    for msg in failed_msgs:
                        parts.append(f"- {msg}")
                final_summary = "\n".join(parts).strip()

            # 5) 图片排版 Agent（可选）：从抓取到的图片 URL 中挑选并插入到日报 Markdown
            if bool(user_config.get("image_layout_enabled", False)) and (final_summary or "").strip():
                try:
                    astrbot_logger.info(
                        "[dailynews] image_layout start enabled=%s provider=%s",
                        bool(user_config.get("image_layout_enabled", False)),
                        str(user_config.get("image_layout_provider_id") or ""),
                    )
                    final_summary = await ImageLayoutAgent().enhance_markdown(
                        draft_markdown=final_summary,
                        sub_results=sub_results,
                        user_config=user_config,
                        astrbot_context=astrbot_context,
                        image_plan=image_plan,
                    )
                    astrbot_logger.info(
                        "[dailynews] image_layout done has_image=%s",
                        bool(re.search(r"!\[[^\]]*\]\(", (final_summary or "")))
                        or ("<img" in (final_summary or "")),
                    )
                except Exception as e:
                    astrbot_logger.warning("[dailynews] image_layout failed: %s", e, exc_info=True)

            return {
                "status": "success",
                "decision": decision,
                "sub_reports": reports,
                "sub_results": sub_results,
                "image_plan": image_plan,
                "final_summary": final_summary,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            import traceback

            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
