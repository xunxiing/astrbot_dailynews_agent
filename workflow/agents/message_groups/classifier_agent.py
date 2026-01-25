from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.llm import LLMRunner
from .tag_store import TagDef


_URL_RE = re.compile(r"https?://[^\s)>\"]+")


def _strip_markdown(md: str, *, max_chars: int) -> str:
    s = (md or "").strip()
    if not s:
        return ""
    s = re.sub(r"```.*?```", " ", s, flags=re.S)  # code fences
    s = re.sub(r"`[^`]+`", " ", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", s)  # images
    s = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", s)  # links text
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[: max(0, int(max_chars))]


def _heuristic_tag(text: str, tags: Sequence[TagDef]) -> str:
    s = (text or "").lower()
    if not s:
        return "[科技新闻]"

    def has_any(words: Sequence[str]) -> bool:
        return any(w.lower() in s for w in words)

    if has_any(["astrbot", "napcat", "onebot", "astr message", "plugin", "插件"]):
        return "[AstrBot]"
    if has_any(["github", "repo", "release", "commit", "pull request", "pr", "issue"]):
        return "[GitHub项目]"
    if has_any(["llm", "gpt", "openai", "模型", "大模型", "aigc", "paper", "论文", "qwen", "llama", "transformer"]):
        return "[AI日报]"
    if has_any(["原神", "崩坏", "游戏", "二次元", "手游", "版本", "攻略"]):
        return "[游戏/二次元]"
    if has_any(["政策", "法规", "条例", "监管", "国务院", "部委", "国内"]):
        return "[国内政策]"
    if has_any(["制裁", "外交", "国际", "联合国", "北约", "美国", "欧盟", "国外", "局势"]):
        return "[国外政策]"

    # fallback to the first configured tag
    return tags[0].tag if tags else "[科技新闻]"


@dataclass(frozen=True)
class ClassifiedItem:
    item_id: str
    tag: str
    topic: str
    summary: str
    suggested_new_tag: str = ""


class TagClassifierAgent:
    """
    Map step: classify each item into a tag using the dynamic TagStore.
    Uses a single LLM call for a batch; falls back to heuristics when LLM is unavailable.
    """

    async def classify(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        tags: Sequence[TagDef],
        llm: Optional[LLMRunner],
        max_excerpt_chars: int = 900,
    ) -> List[ClassifiedItem]:
        allowed = [t.tag for t in tags if (t.tag or "").strip()]
        if not allowed:
            allowed = [
                "[国内政策]",
                "[国外政策]",
                "[科技新闻]",
                "[AI日报]",
                "[AstrBot]",
                "[GitHub项目]",
                "[游戏/二次元]",
                "[待定/新发现]",
            ]

        payload_items = []
        for it in items:
            iid = str(it.get("id") or "").strip()
            text = _strip_markdown(str(it.get("text") or ""), max_chars=max_excerpt_chars)
            if not iid or not text:
                continue
            payload_items.append({"id": iid, "source": str(it.get("source") or ""), "text": text})

        if not payload_items:
            return []

        if llm is None:
            out: List[ClassifiedItem] = []
            for it in payload_items:
                tag = _heuristic_tag(it["text"], tags)
                topic = (it["text"][:30] + "…") if len(it["text"]) > 30 else it["text"]
                out.append(ClassifiedItem(item_id=it["id"], tag=tag, topic=topic, summary=topic))
            return out

        tag_defs = [{"tag": t.tag, "description": t.description} for t in tags]
        system_prompt = (
            "你是一个中文信息分类器（Map）。\n"
            "你会收到：1) 允许的标签列表（带说明）；2) 多条待分类的文本片段。\n"
            "任务：为每条文本选择最合适的一个标签，并产出 topic/summary。\n"
            "严格输出 JSON 数组，不要输出任何多余文本。\n"
            "规则：\n"
            "- tag 必须是 allowed_tags 之一；如果都不匹配，tag 选 [待定/新发现]，并在 suggested_new_tag 给出一个你建议的新标签（也用 [xxx] 形式）。\n"
            "- topic <= 20 字；summary <= 60 字。\n"
        )
        prompt = {
            "allowed_tags": allowed,
            "tag_definitions": tag_defs,
            "items": payload_items,
            "output_schema": [
                {"id": "string", "tag": "string", "topic": "string", "summary": "string", "suggested_new_tag": "string"}
            ],
        }

        try:
            raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(prompt, ensure_ascii=False))
            data = self._json_from_text(raw)
        except Exception as e:
            astrbot_logger.warning("[dailynews] tag classify failed, fallback to heuristic: %s", e, exc_info=True)
            data = None

        if not isinstance(data, list):
            out2: List[ClassifiedItem] = []
            for it in payload_items:
                tag = _heuristic_tag(it["text"], tags)
                topic = (it["text"][:30] + "…") if len(it["text"]) > 30 else it["text"]
                out2.append(ClassifiedItem(item_id=it["id"], tag=tag, topic=topic, summary=topic))
            return out2

        results: List[ClassifiedItem] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            iid = str(row.get("id") or "").strip()
            if not iid:
                continue
            tag = str(row.get("tag") or "").strip()
            if tag not in allowed:
                tag = "[待定/新发现]"
            topic = str(row.get("topic") or "").strip()
            summary = str(row.get("summary") or "").strip()
            sug = str(row.get("suggested_new_tag") or "").strip()
            if tag != "[待定/新发现]":
                sug = ""
            if sug and not (sug.startswith("[") and sug.endswith("]")):
                sug = ""
            results.append(
                ClassifiedItem(
                    item_id=iid,
                    tag=tag,
                    topic=topic[:20],
                    summary=summary[:80],
                    suggested_new_tag=sug,
                )
            )
        return results

    def _json_from_text(self, text: str) -> Any:
        s = (text or "").strip()
        if not s:
            return None
        # Try to locate a JSON array in the output
        start = s.find("[")
        end = s.rfind("]")
        if start >= 0 and end > start:
            s2 = s[start : end + 1]
        else:
            s2 = s
        try:
            return json.loads(s2)
        except Exception:
            return None


def extract_links(md: str, *, max_links: int = 3) -> List[str]:
    found: List[str] = []
    seen = set()
    for m in _URL_RE.finditer(md or ""):
        u = str(m.group(0) or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        found.append(u)
        if len(found) >= max(1, int(max_links)):
            break
    return found

