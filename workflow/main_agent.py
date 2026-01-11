import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .llm import LLMRunner
from .models import MainAgentDecision, SubAgentResult
from .utils import _json_from_text


class MainNewsAgent:
    async def analyze_sub_agent_reports(
        self, reports: List[Dict[str, Any]], user_config: Dict[str, Any], llm: LLMRunner
    ) -> MainAgentDecision:
        preferred_types = user_config.get("preferred_source_types", ["wechat"])
        max_sources = int(user_config.get("max_sources_per_day", 3))

        system_prompt = (
            "你是每日资讯编辑（主Agent）。"
            "你会收到多个子Agent的汇报（每个来源的最新文章链接与初步主题）。"
            "你的任务是决定：今天要处理哪些来源、每个来源写什么角度、每个来源最多写几条要点。"
            "请严格只输出 JSON（不要 Markdown、不要解释）。"
        )

        prompt = {
            "now": datetime.now().strftime("%Y-%m-%d"),
            "constraints": {
                "preferred_source_types": preferred_types,
                "max_sources_per_day": max_sources,
            },
            "reports": reports,
            "output_schema": {
                "sources_to_process": ["source_name"],
                "processing_instructions": {"source_name": "instruction string"},
                "final_format": "markdown",
            },
        }

        raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        data = _json_from_text(raw)

        if not isinstance(data, dict):
            return self._fallback_decision(reports, user_config)

        sources_to_process = [
            s.strip()
            for s in data.get("sources_to_process", [])
            if isinstance(s, str) and s.strip()
        ]
        processing_instructions = data.get("processing_instructions", {})
        if not isinstance(processing_instructions, dict):
            processing_instructions = {}
        processing_instructions = {str(k): str(v) for k, v in processing_instructions.items()}

        reported_names = {r.get("source_name") for r in reports if isinstance(r, dict)}
        sources_to_process = [s for s in sources_to_process if s in reported_names]

        preferred_set = set(preferred_types) if isinstance(preferred_types, list) else set()
        preferred_reports = [
            r
            for r in reports
            if isinstance(r, dict)
            and r.get("source_name") in reported_names
            and (not preferred_set or r.get("source_type") in preferred_set)
        ]
        other_reports = [
            r
            for r in reports
            if isinstance(r, dict)
            and r.get("source_name") in reported_names
            and preferred_set
            and r.get("source_type") not in preferred_set
        ]

        def _score(r: Dict[str, Any]) -> Tuple[int, int]:
            try:
                return int(r.get("quality_score", 0)), int(r.get("priority", 0))
            except Exception:
                return 0, 0

        preferred_reports.sort(key=_score, reverse=True)
        other_reports.sort(key=_score, reverse=True)
        all_reports = preferred_reports + other_reports
        wanted = min(len(all_reports), max_sources)

        dedup: List[str] = []
        for s in sources_to_process:
            if s not in dedup:
                dedup.append(s)
        sources_to_process = dedup

        if len(sources_to_process) < wanted:
            for r in all_reports:
                name = str(r.get("source_name") or "")
                if not name or name in sources_to_process:
                    continue
                sources_to_process.append(name)
                if len(sources_to_process) >= wanted:
                    break

        sources_to_process = sources_to_process[:max_sources]

        report_map = {
            str(r.get("source_name")): r
            for r in reports
            if isinstance(r, dict) and r.get("source_name")
        }
        for name in sources_to_process:
            if name in processing_instructions and processing_instructions[name].strip():
                continue
            rep = report_map.get(name) or {}
            angle = str(rep.get("today_angle") or "").strip()
            if angle:
                processing_instructions[name] = angle
            else:
                topics = rep.get("topics", []) if isinstance(rep.get("topics"), list) else []
                topic_str = "、".join([str(t) for t in topics[:3]]) if topics else "热点"
                processing_instructions[name] = f"请聚焦 {topic_str}，输出精炼要点并给出文章链接。"

        astrbot_logger.debug("[dailynews] sources_to_process: %s", sources_to_process)

        final_format = str(data.get("final_format") or user_config.get("output_format", "markdown"))
        return MainAgentDecision(
            sources_to_process=sources_to_process,
            processing_instructions=processing_instructions,
            final_format=final_format,
        )

    def _fallback_decision(self, reports: List[Dict[str, Any]], user_config: Dict[str, Any]) -> MainAgentDecision:
        preferred_types = user_config.get("preferred_source_types", ["wechat"])
        max_sources = int(user_config.get("max_sources_per_day", 3))

        preferred_set = set(preferred_types) if isinstance(preferred_types, list) else set()

        def score(r: Dict[str, Any]) -> Tuple[int, int]:
            return int(r.get("quality_score", 0)), int(r.get("priority", 0))

        preferred = [r for r in reports if (not preferred_set) or r.get("source_type") in preferred_set]
        other = [r for r in reports if preferred_set and r.get("source_type") not in preferred_set]
        preferred.sort(key=score, reverse=True)
        other.sort(key=score, reverse=True)
        candidates = preferred + other

        selected: List[str] = []
        instructions: Dict[str, str] = {}
        for r in candidates:
            if len(selected) >= max_sources:
                break
            name = r.get("source_name")
            if not name:
                continue
            selected.append(name)
            topics = r.get("topics", []) if isinstance(r.get("topics"), list) else []
            topic_str = "、".join([str(t) for t in topics[:3]]) if topics else "热点"
            instructions[name] = f"请聚焦 {topic_str}，输出精炼要点并给出文章链接。"

        return MainAgentDecision(
            sources_to_process=selected,
            processing_instructions=instructions,
            final_format=user_config.get("output_format", "markdown"),
        )

    async def summarize_all_results(
        self, sub_results: List[Any], format_type: str, llm: LLMRunner
    ) -> str:
        ok: List[SubAgentResult] = []
        failed: List[str] = []
        for r in sub_results:
            if isinstance(r, Exception):
                failed.append(str(r) or f"{type(r).__name__}")
                continue
            if isinstance(r, SubAgentResult) and not r.error:
                ok.append(r)
            elif isinstance(r, SubAgentResult):
                failed.append(f"{r.source_name}: {r.error}")

        if not ok:
            lines = ["# 每日资讯日报", "", "未收到子Agent的有效内容小节。"]
            if failed:
                lines.extend(["", "## 错误信息"])
                for msg in failed:
                    lines.append(f"- {msg}")
            return "\n".join(lines)

        system_prompt = (
            "你是每日资讯编辑（主Agent）。"
            "你会收到多个子Agent写好的 Markdown 小节。"
            "请合并为一篇结构清晰、去重、可读性强的中文 Markdown 日报。"
            "格式要求（重要）："
            "1) 顶部只保留一个文档标题（# ...），其余各来源必须用二级标题 `## 来源名` 分隔。"
            "2) 不要把所有来源揉成一个连续列表；几个可以整合的来源建议有 `##` 标题。"
            "3) 来源内可以用 `###`/列表组织要点，但请保留原文链接；不要编造未提供的事实。"
        )
        prompt = {
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections": [{"source": r.source_name, "markdown": r.content} for r in ok],
            "failed": failed,
            "output_format": format_type,
        }
        try:
            return await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        except Exception as e:
            astrbot_logger.warning("[dailynews] merge failed, fallback to concat: %s", e, exc_info=True)
            parts = ["# 每日资讯日报", f"*生成时间: {prompt['now']}*", ""]
            for s in ok:
                parts.append(s.content.strip())
                parts.append("")
            if failed:
                parts.append("## 错误信息")
                for msg in failed:
                    parts.append(f"- {msg}")
            return "\n".join([p for p in parts if p is not None])
