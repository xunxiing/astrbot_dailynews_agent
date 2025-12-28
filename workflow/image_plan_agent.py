import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .llm import LLMRunner
from .models import MainAgentDecision, SubAgentResult
from .utils import _json_from_text


class ImagePlanAgent:
    async def decide_plan(
        self,
        *,
        reports: List[Dict[str, Any]],
        decision: MainAgentDecision,
        sub_results: List[Any],
        user_config: Dict[str, Any],
        llm: LLMRunner,
    ) -> Optional[Dict[str, Any]]:
        if not bool(user_config.get("image_layout_enabled", False)):
            return None
        if not bool(user_config.get("image_plan_enabled", True)):
            return None

        max_images_total = int(user_config.get("image_layout_max_images_total", 6) or 6)
        max_images_per_source = int(user_config.get("image_layout_max_images_per_source", 3) or 3)

        images_by_source: Dict[str, List[str]] = {}
        for r in sub_results:
            if isinstance(r, SubAgentResult) and r.source_name and r.images:
                urls = [str(u) for u in r.images if isinstance(u, str) and u.strip()]
                if urls:
                    images_by_source[r.source_name] = urls

        if not images_by_source:
            return None

        only_sources = user_config.get("image_layout_sources") or []
        if isinstance(only_sources, list) and only_sources:
            only_set = {str(x).strip() for x in only_sources if str(x).strip()}
            if only_set:
                images_by_source = {k: v for k, v in images_by_source.items() if k in only_set}
        if not images_by_source:
            return None

        report_map = {
            str(r.get("source_name")): r
            for r in reports
            if isinstance(r, dict) and r.get("source_name")
        }
        candidates = []
        for name, urls in images_by_source.items():
            rep = report_map.get(name) or {}
            candidates.append(
                {
                    "source": name,
                    "topics": rep.get("topics", []),
                    "today_angle": rep.get("today_angle", ""),
                    "image_count": len(urls),
                }
            )

        system_prompt = (
            "You are the chief editor deciding an image count plan for the layout agent.\n"
            "You will receive per-source image counts and topic hints.\n"
            "Return JSON with a total image cap and per-source image counts.\n"
            "Only output JSON, no explanations.\n"
        )
        payload = {
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "constraints": {
                "max_images_total": max_images_total,
                "max_images_per_source": max_images_per_source,
            },
            "sources_to_process": decision.sources_to_process,
            "candidates": candidates,
            "output_schema": {
                "max_images_total": max_images_total,
                "by_source": {"source_name": 2},
                "notes": "string",
            },
        }

        try:
            raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            astrbot_logger.warning("[dailynews] image_plan llm failed: %s", e, exc_info=True)
            return self._fallback_plan(decision, images_by_source, max_images_total, max_images_per_source)

        data = _json_from_text(raw)
        if not isinstance(data, dict):
            return self._fallback_plan(decision, images_by_source, max_images_total, max_images_per_source)

        plan = self._sanitize_plan(
            data,
            images_by_source,
            max_images_total,
            max_images_per_source,
        )
        return plan

    def _fallback_plan(
        self,
        decision: MainAgentDecision,
        images_by_source: Dict[str, List[str]],
        max_images_total: int,
        max_images_per_source: int,
    ) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        total = 0
        for name in decision.sources_to_process:
            if name not in images_by_source:
                continue
            want = min(max_images_per_source, len(images_by_source[name]))
            if want <= 0:
                continue
            if total + want > max_images_total:
                want = max(0, max_images_total - total)
            if want <= 0:
                break
            by_source[name] = want
            total += want
            if total >= max_images_total:
                break
        return {
            "max_images_total": max_images_total,
            "by_source": by_source,
            "notes": "fallback",
        }

    def _sanitize_plan(
        self,
        raw: Dict[str, Any],
        images_by_source: Dict[str, List[str]],
        max_images_total: int,
        max_images_per_source: int,
    ) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        raw_map = raw.get("by_source")
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                name = str(k).strip()
                if name not in images_by_source:
                    continue
                try:
                    n = int(v)
                except Exception:
                    continue
                n = max(0, min(n, max_images_per_source, len(images_by_source[name])))
                if n > 0:
                    by_source[name] = n

        try:
            plan_total = int(raw.get("max_images_total", max_images_total))
        except Exception:
            plan_total = max_images_total
        if plan_total <= 0:
            plan_total = max_images_total
        plan_total = min(plan_total, max_images_total)

        # clamp by total
        total = 0
        for name in list(by_source.keys()):
            if total >= plan_total:
                by_source.pop(name, None)
                continue
            remain = plan_total - total
            if by_source[name] > remain:
                by_source[name] = remain
            total += by_source[name]

        if not by_source:
            return {
                "max_images_total": plan_total,
                "by_source": {},
                "notes": "empty",
            }

        return {
            "max_images_total": plan_total,
            "by_source": by_source,
            "notes": str(raw.get("notes") or ""),
        }
