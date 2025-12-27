import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from astrbot.api.star import StarTools
except Exception:  # pragma: no cover
    StarTools = None  # type: ignore

try:
    from ..analysis.wechatanalysis.analysis import fetch_wechat_article
    from ..analysis.wechatanalysis.latest_articles import get_album_articles_chasing_latest_with_seed
except Exception:  # pragma: no cover
    from analysis.wechatanalysis.analysis import fetch_wechat_article  # type: ignore
    from analysis.wechatanalysis.latest_articles import get_album_articles_chasing_latest_with_seed  # type: ignore


@dataclass
class NewsSourceConfig:
    """新闻源配置"""

    name: str
    url: str
    type: str = "wechat"  # wechat, rss, etc.
    priority: int = 1
    max_articles: int = 3
    album_keyword: Optional[str] = None


@dataclass
class SubAgentResult:
    """子 Agent 处理结果"""

    source_name: str
    content: str
    summary: str
    key_points: List[str]
    images: Optional[List[str]] = None
    error: Optional[str] = None


@dataclass
class MainAgentDecision:
    """主 Agent 决策结果"""

    sources_to_process: List[str]
    processing_instructions: Dict[str, str]
    final_format: str = "markdown"


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


_SEED_LOCK = asyncio.Lock()
_SEED_CACHE: Optional[Dict[str, Any]] = None


def _seed_state_path() -> Path:
    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir()) / "wechat_seed_state.json"
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "data" / "wechat_seed_state.json"


def _load_seed_state_sync(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_seed_state_sync(path: Path, state: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _get_seed_state() -> Dict[str, Any]:
    global _SEED_CACHE
    async with _SEED_LOCK:
        if _SEED_CACHE is None:
            _SEED_CACHE = await _run_sync(_load_seed_state_sync, _seed_state_path())
        return _SEED_CACHE


async def _set_seed_state(state: Dict[str, Any]):
    global _SEED_CACHE
    async with _SEED_LOCK:
        _SEED_CACHE = state
        await _run_sync(_save_seed_state_sync, _seed_state_path(), state)


async def _update_seed_entry(key: str, entry: Dict[str, Any]):
    global _SEED_CACHE
    async with _SEED_LOCK:
        if _SEED_CACHE is None:
            _SEED_CACHE = await _run_sync(_load_seed_state_sync, _seed_state_path())
        _SEED_CACHE[key] = entry
        await _run_sync(_save_seed_state_sync, _seed_state_path(), _SEED_CACHE)


class LLMRunner:
    def __init__(self, astrbot_context: Any, timeout_s: int = 180, max_retries: int = 1):
        self._ctx = astrbot_context
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    async def ask(self, *, system_prompt: str, prompt: str) -> str:
        provider = self._ctx.get_using_provider()

        last_exc: Optional[BaseException] = None
        for attempt in range(1, int(self._max_retries) + 2):
            try:
                coro = provider.text_chat(
                    prompt=prompt,
                    session_id=None,  # deprecated but kept for compatibility
                    contexts=[],
                    image_urls=[],
                    func_tool=None,
                    system_prompt=system_prompt,
                )
                resp = await asyncio.wait_for(coro, timeout=self._timeout_s)
                if getattr(resp, "role", None) == "assistant":
                    return getattr(resp, "completion_text", "") or ""
                return (
                    getattr(resp, "completion_text", "")
                    or str(getattr(resp, "raw_completion", ""))
                    or ""
                )
            except asyncio.TimeoutError as e:
                last_exc = e
                astrbot_logger.warning(
                    "[dailynews] LLM timeout after %ss (attempt %s/%s)",
                    self._timeout_s,
                    attempt,
                    int(self._max_retries) + 1,
                )
            except Exception as e:
                last_exc = e
                astrbot_logger.warning(
                    "[dailynews] LLM call failed (attempt %s/%s): %s",
                    attempt,
                    int(self._max_retries) + 1,
                    e,
                    exc_info=True,
                )

            # backoff
            await asyncio.sleep(min(2.0 * attempt, 6.0))

        raise RuntimeError(
            f"LLM call failed after {int(self._max_retries) + 1} attempts: "
            f"{type(last_exc).__name__ if last_exc else 'UnknownError'}"
        )


class NewsWorkflowManager:
    """新闻多 Agent 工作流管理器"""

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
            # fetch_latest_articles 返回 (source_name, articles)
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
            decision = await main_agent.analyze_sub_agent_reports(reports, user_config, llm_scout)

            # 兜底：当 LLM 只选择 1 个来源时，按配置补齐到 max_sources_per_day
            preferred_types = user_config.get("preferred_source_types", ["wechat"])
            if not isinstance(preferred_types, list) or not preferred_types:
                preferred_types = ["wechat"]
            max_sources = int(user_config.get("max_sources_per_day", 3))
            candidate_sources = [s for s in sources if s.type in preferred_types]
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

            # 4) 子 Agent 按分工写各自部分（会再次抓取文章正文内容）
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
                    )
                )

            astrbot_logger.debug("[dailynews] write_tasks: %s", len(write_tasks))

            sub_results = await asyncio.gather(*write_tasks, return_exceptions=True)
            final_summary = await main_agent.summarize_all_results(sub_results, decision.final_format, llm_merge)

            if not (final_summary or "").strip():
                astrbot_logger.warning("[dailynews] final_summary is empty; fallback to concat sections")
                parts: List[str] = ["# 每日资讯日报", f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*", ""]
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

            return {
                "status": "success",
                "decision": decision,
                "sub_reports": reports,
                "sub_results": sub_results,
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


class MainNewsAgent:
    """主 Agent：负责听汇报、分工、最终汇总"""

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

        sources_to_process = [s.strip() for s in data.get("sources_to_process", []) if isinstance(s, str) and s.strip()]
        processing_instructions = data.get("processing_instructions", {})
        if not isinstance(processing_instructions, dict):
            processing_instructions = {}
        processing_instructions = {str(k): str(v) for k, v in processing_instructions.items()}

        # 限制数量 + 仅保留已汇报的来源
        reported_names = {r.get("source_name") for r in reports if isinstance(r, dict)}
        sources_to_process = [s for s in sources_to_process if s in reported_names]

        # 当来源数量 <= 上限时，优先保证“每个来源都有子Agent写作”，避免 LLM 随机只挑 1 个来源
        preferred_reports = [
            r
            for r in reports
            if isinstance(r, dict)
            and r.get("source_name") in reported_names
            and r.get("source_type") in preferred_types
        ]

        def _score(r: Dict[str, Any]) -> Tuple[int, int]:
            try:
                return int(r.get("quality_score", 0)), int(r.get("priority", 0))
            except Exception:
                return 0, 0

        preferred_reports.sort(key=_score, reverse=True)
        wanted = min(len(preferred_reports), max_sources)

        # 先按 LLM 选择顺序保留，再用评分补齐到 wanted
        dedup: List[str] = []
        for s in sources_to_process:
            if s not in dedup:
                dedup.append(s)
        sources_to_process = dedup

        if len(sources_to_process) < wanted:
            for r in preferred_reports:
                name = str(r.get("source_name") or "")
                if not name or name in sources_to_process:
                    continue
                sources_to_process.append(name)
                if len(sources_to_process) >= wanted:
                    break

        # 截断到 max_sources
        sources_to_process = sources_to_process[:max_sources]

        # 为缺失的指令补齐：优先用子Agent汇报的 today_angle，其次兜底提示词
        report_map = {str(r.get("source_name")): r for r in reports if isinstance(r, dict) and r.get("source_name")}
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

        def score(r: Dict[str, Any]) -> Tuple[int, int]:
            return int(r.get("quality_score", 0)), int(r.get("priority", 0))

        candidates = [r for r in reports if r.get("source_type") in preferred_types]
        candidates.sort(key=score, reverse=True)

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
            # 兜底：至少返回错误信息，避免 LLM 再次空转
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
            "要求：保留每节的文章链接；不要编造未提供的事实。"
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


class WechatSubAgent:
    """公众号子 Agent：负责抓取最新文章列表、抓取正文并写出小节"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, str]]]:
        limit = max(int(source.max_articles), 5)
        max_hops = int(user_config.get("wechat_chase_max_hops", 6))
        persist_seed = bool(user_config.get("wechat_seed_persist", True))

        album_keyword = source.album_keyword
        key = f"{source.url}||{album_keyword or ''}"

        start_url = source.url
        if persist_seed:
            state = await _get_seed_state()
            entry = state.get(key) if isinstance(state, dict) else None
            if isinstance(entry, dict) and entry.get("seed_url"):
                start_url = str(entry.get("seed_url"))

        seed_url = start_url
        articles: List[Dict[str, str]] = []

        # Playwright 偶发失败 or 该文章不含「合集目录」入口时，会导致抓取为空：增加轻量重试 + 单篇回退
        for attempt in range(1, 3):
            try:
                seed_url, articles = await _run_sync(
                    get_album_articles_chasing_latest_with_seed,
                    start_url,
                    limit,
                    album_keyword=album_keyword,
                    max_hops=max_hops,
                )
            except Exception as e:
                astrbot_logger.warning(
                    "[dailynews] chasing latest failed for %s (attempt %s/2): %s",
                    source.name,
                    attempt,
                    e,
                    exc_info=True,
                )
                seed_url, articles = start_url, []

            if articles:
                break
            await asyncio.sleep(0.8 * attempt)

        if not articles:
            astrbot_logger.warning(
                "[dailynews] %s has no album articles; fallback to configured URL as single article: %s",
                source.name,
                start_url,
            )
            seed_url = start_url
            articles = [{"title": "", "url": start_url}]

        if persist_seed and seed_url:
            await _update_seed_entry(
                key,
                {
                    "seed_url": seed_url,
                    "source_url": source.url,
                    "album_keyword": album_keyword or "",
                    "updated_at": datetime.now().isoformat(),
                },
            )

        return source.name, articles

    async def analyze_source(
        self, source: NewsSourceConfig, articles: List[Dict[str, str]], llm: LLMRunner
    ) -> Dict[str, Any]:
        # 子 Agent 的“汇报”阶段：只看标题+链接，给主 Agent 一个今日角度与主题
        system_prompt = (
            "你是子Agent（信息侦察）。"
            "你将收到某个公众号来源的最新文章标题与链接。"
            "请快速判断今日主要看点/主题，并给出可写作的角度建议。"
            "只输出 JSON，不要输出其它文本。"
        )
        prompt = {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "latest_articles": articles[:10],
            "output_schema": {
                "source_name": source.name,
                "source_type": source.type,
                "priority": source.priority,
                "article_count": len(articles),
                "topics": ["topic"],
                "quality_score": 0,
                "today_angle": "string",
            },
        }

        raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        data = _json_from_text(raw) or {}
        topics = data.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        quality = data.get("quality_score")
        try:
            if isinstance(quality, float) and 0 <= quality <= 1:
                quality_score = int(quality * 100)
            elif isinstance(quality, str):
                q = quality.strip()
                if q.endswith("%"):
                    quality_score = int(float(q[:-1]) * 1)
                else:
                    qf = float(q)
                    quality_score = int(qf * 100) if 0 <= qf <= 1 else int(qf)
            else:
                quality_score = int(quality)
        except Exception:
            quality_score = len(articles) * 2 + len(topics)

        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": len(articles),
            "topics": [str(t) for t in topics[:8]],
            "quality_score": quality_score,
            "today_angle": str(data.get("today_angle") or ""),
            "sample_articles": articles[:3],
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: List[Dict[str, str]],
        llm: LLMRunner,
    ) -> SubAgentResult:
        if not articles:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                error="该来源未抓取到任何最新文章",
            )

        chosen = articles[: max(1, int(source.max_articles))]

        max_fetch_concurrency = 2
        sem = asyncio.Semaphore(max_fetch_concurrency)

        async def _fetch_one(a: Dict[str, str]) -> Dict[str, Any]:
            url = (a.get("url") or "").strip()
            if not url:
                return {"title": (a.get("title") or "").strip(), "url": "", "error": "missing url"}

            # Playwright 偶发失败时重试
            last_err: Optional[str] = None
            for attempt in range(1, 3):
                try:
                    async with sem:
                        detail = await _run_sync(fetch_wechat_article, url)
                    content_text = (detail.get("content_text") or "").strip()
                    # 控制输入长度，降低 LLM 超时概率
                    if len(content_text) > 1500:
                        content_text = content_text[:1500] + "…"
                    return {
                        "title": (detail.get("title") or a.get("title") or "").strip(),
                        "url": url,
                        "author": (detail.get("author") or "").strip(),
                        "publish_time": (detail.get("publish_time") or "").strip(),
                        "content_text": content_text,
                    }
                except Exception as e:
                    last_err = str(e) or type(e).__name__
                    astrbot_logger.warning(
                        "[dailynews] fetch_wechat_article failed (attempt %s/2): %s",
                        attempt,
                        last_err,
                        exc_info=True,
                    )
                    await asyncio.sleep(1.0 * attempt)
            return {"title": (a.get("title") or "").strip(), "url": url, "error": last_err or "unknown"}

        article_details = await asyncio.gather(*[_fetch_one(a) for a in chosen], return_exceptions=False)

        system_prompt = (
            "你是子Agent（写作）。"
            "你会收到：写作指令 + 多篇公众号文章的正文摘要。"
            "请写出该来源在今日日报中的一个 Markdown 小节（含小标题、要点、每条要点尽量附上文章链接）。"
            "同时只输出 JSON，不要输出其它文本。"
        )
        prompt = {
            "source_name": source.name,
            "instruction": instruction,
            "articles": article_details,
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

        try:
            raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
        except Exception as e:
            # 回退：至少输出标题+链接，保证日报可用
            astrbot_logger.warning("[dailynews] subagent write failed, fallback: %s", e, exc_info=True)
            lines = [f"## {source.name}", "", "（模型生成失败/超时，以下为自动回退摘要）", ""]
            for a in chosen:
                t = (a.get("title") or "").strip()
                u = (a.get("url") or "").strip()
                if t and u:
                    lines.append(f"- {t} ({u})")
                elif u:
                    lines.append(f"- {u}")
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
            # 兜底：直接把模型输出当作 Markdown
            return SubAgentResult(
                source_name=source.name,
                content=str(raw),
                summary="",
                key_points=[],
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
