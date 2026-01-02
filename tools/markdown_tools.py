from __future__ import annotations

import json
import re
import difflib
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from ..workflow.md_doc_store import create_doc, read_doc, write_doc


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


_FUZZY_STRIP_RE = re.compile(r"[\s`*_>#\-\.\(\)\[\]{}:：，,。！？!？“”‘’\"'…—]+")


def _fuzzy_norm(s: str) -> str:
    return _FUZZY_STRIP_RE.sub("", (s or "").strip()).lower()


def _fuzzy_suggest_lines(
    text: str,
    query: str,
    *,
    limit: int = 5,
    min_score: float = 0.55,
) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    qn = _fuzzy_norm(q)
    if not qn:
        return []

    items: List[Tuple[float, int, str]] = []
    for idx, raw_line in enumerate((text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if len(line) > 180:
            continue
        ln = _fuzzy_norm(line)
        if not ln:
            continue
        # quick boosts for substring matches
        if qn in ln or ln in qn:
            score = 0.99
        else:
            score = difflib.SequenceMatcher(a=qn, b=ln).ratio()
        if score < min_score:
            continue
        items.append((score, idx, line))

    items.sort(key=lambda x: (-x[0], x[1]))
    out: List[Dict[str, Any]] = []
    for score, line_no, line in items[: max(1, min(int(limit), 20))]:
        out.append({"line_no": line_no, "text": line, "score": round(float(score), 3)})
    return out


def _normalize_occurrence(value: Any, default: int = 1) -> int:
    try:
        n = int(value)
    except Exception:
        n = int(default)
    return n


def _find_spans(text: str, *, match: str, regex: bool, occurrence: int) -> List[Tuple[int, int]]:
    s = text or ""
    m = (match or "")
    if not s or not m:
        return []

    spans: List[Tuple[int, int]] = []
    if regex:
        if len(m) > 600:
            return []
        try:
            pat = re.compile(m, flags=re.M | re.S)
        except Exception:
            return []
        for it in pat.finditer(s):
            spans.append((it.start(), it.end()))
            if occurrence > 0 and len(spans) >= occurrence:
                break
        return spans

    start = 0
    while True:
        idx = s.find(m, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(m)))
        if occurrence > 0 and len(spans) >= occurrence:
            break
        start = idx + len(m)
        if start >= len(s):
            break
    return spans


def _line_bounds(text: str, pos: int) -> Tuple[int, int]:
    s = text or ""
    pos = max(0, min(int(pos), len(s)))
    ls = s.rfind("\n", 0, pos)
    le = s.find("\n", pos)
    if ls < 0:
        ls = 0
    else:
        ls = ls + 1
    if le < 0:
        le = len(s)
    return ls, le


def _insert_after_line(text: str, *, line_pos: int, insert_text: str, ensure_blank_line: bool) -> str:
    s = text or ""
    _, le = _line_bounds(s, line_pos)
    insert_at = le
    if insert_at < len(s):
        insert_at += 1  # include the newline
    before = s[:insert_at]
    after = s[insert_at:]
    ins = insert_text or ""
    if ensure_blank_line:
        if before and not before.endswith("\n"):
            before += "\n"
        if before and not before.endswith("\n\n"):
            before += "\n"
        if ins and not ins.endswith("\n"):
            ins += "\n"
        if ins and not ins.endswith("\n\n"):
            ins += "\n"
    return before + ins + after


def _apply_edits(text: str, edits: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    md = text or ""
    applied = 0
    errors: List[str] = []
    not_found: List[Dict[str, Any]] = []

    for e in edits or []:
        if not isinstance(e, dict):
            errors.append("invalid edit item (not object)")
            continue
        op = str(e.get("op") or "").strip()
        match = str(e.get("match") or "")
        regex = bool(e.get("regex", False))
        occurrence = _normalize_occurrence(e.get("occurrence", 1), 1)
        if op not in {"replace", "delete", "insert_after", "insert_before"}:
            errors.append(f"unsupported op: {op}")
            continue
        spans = _find_spans(md, match=match, regex=regex, occurrence=occurrence if occurrence != 0 else 10**9)
        if not spans:
            errors.append(f"match not found: {match[:60]}")
            if not regex and match and len(match) <= 240:
                not_found.append(
                    {
                        "match": match,
                        "suggestions": _fuzzy_suggest_lines(md, match, limit=5),
                    }
                )
            continue
        # only apply to the first match unless occurrence==0 (all)
        targets = spans if occurrence == 0 else spans[:1]

        # apply from back to front to keep indices stable
        for start, end in reversed(targets):
            if op == "delete":
                md = md[:start] + md[end:]
                applied += 1
                continue
            if op == "replace":
                repl = str(e.get("replacement") or "")
                md = md[:start] + repl + md[end:]
                applied += 1
                continue
            if op in {"insert_after", "insert_before"}:
                ins = str(e.get("text") or "")
                if op == "insert_after":
                    md = md[:end] + ins + md[end:]
                else:
                    md = md[:start] + ins + md[start:]
                applied += 1
                continue

    return md, {"applied": applied, "errors": errors, "not_found": not_found}


@dataclass
class MarkdownDocCreateTool(FunctionTool[AstrAgentContext]):
    name: str = "md_doc_create"
    description: str = "Create a markdown doc for editing and return {doc_id}."
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "markdown": {"type": "string", "description": "Initial markdown content"},
                "doc_id": {"type": "string", "description": "Optional doc id (leave empty for auto)"},
            },
            "required": ["markdown"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        md = str(kwargs.get("markdown") or "")
        doc_id = str(kwargs.get("doc_id") or "").strip() or None
        did, path = create_doc(md, doc_id=doc_id)
        return _json_dump({"doc_id": did, "path": str(path.resolve()), "length": len(md)})


@dataclass
class MarkdownDocReadTool(FunctionTool[AstrAgentContext]):
    name: str = "md_doc_read"
    description: str = (
        "Read a markdown doc. Use start/max_chars to paginate; avoid repeating the same call."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Doc id"},
                "start": {"type": "integer", "description": "Start offset (chars), default 0"},
                "max_chars": {"type": "integer", "description": "Max chars to return, default 2400"},
            },
            "required": ["doc_id"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        doc_id = str(kwargs.get("doc_id") or "").strip()
        start = int(kwargs.get("start", 0) or 0)
        max_chars = int(kwargs.get("max_chars", 2400) or 2400)
        max_chars = max(200, min(max_chars, 20000))
        start = max(0, start)

        # Anti-loop guard: some models keep calling md_doc_read with the exact same params.
        # We track minimal per-run state in context.extra (stringified JSON) to avoid infinite loops.
        guard_key = "_dailynews_md_doc_read_guard_v1"
        try:
            extra = getattr(getattr(context, "context", None), "extra", None)
        except Exception:
            extra = None

        state: Dict[str, Any] = {}
        if isinstance(extra, dict):
            raw = extra.get(guard_key)
            if isinstance(raw, str) and raw.strip():
                try:
                    state = json.loads(raw) if raw.strip().startswith("{") else {}
                except Exception:
                    state = {}

        sig = f"{doc_id}:{start}:{max_chars}"
        last_sig = str(state.get("last_sig") or "")
        repeats = int(state.get("repeats") or 0)
        total = int(state.get("total") or 0)
        if sig == last_sig:
            repeats += 1
        else:
            repeats = 0
        total += 1

        md = read_doc(doc_id)
        content_start = start
        note = ""
        # After the 2nd identical call, automatically move forward by one page and tell the model.
        if repeats >= 2:
            content_start = min(len(md), start + max_chars)
            note = (
                f"[md_doc_read guard] Detected repeated read for the same params; "
                f"returning the next page (start={content_start}, max_chars={max_chars}).\n"
            )
        # Hard cap: if the model is still stuck, force it to do other actions.
        if total >= 16:
            note = (
                "[md_doc_read guard] Read limit reached. Stop calling md_doc_read; "
                "use md_doc_match_insert_image / md_doc_apply_edits, or finish.\n"
            )
            content_start = start

        if isinstance(extra, dict):
            try:
                extra[guard_key] = json.dumps(
                    {"last_sig": sig, "repeats": repeats, "total": total},
                    ensure_ascii=False,
                )
            except Exception:
                pass

        return note + md[content_start : content_start + max_chars]


@dataclass
class MarkdownDocApplyEditsTool(FunctionTool[AstrAgentContext]):
    name: str = "md_doc_apply_edits"
    description: str = (
        "Apply edit operations (replace/insert/delete) to a markdown doc by matching text/regex."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Doc id"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "description": "replace|delete|insert_after|insert_before",
                            },
                            "match": {
                                "type": "string",
                                "description": "Text or regex to match",
                            },
                            "regex": {
                                "type": "boolean",
                                "description": "Treat match as regex, default false",
                            },
                            "occurrence": {
                                "type": "integer",
                                "description": "1=first, 0=all, 2=second...",
                            },
                            "replacement": {
                                "type": "string",
                                "description": "Replacement text (for replace)",
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to insert (for insert_*)",
                            },
                        },
                        "required": ["op", "match"],
                    },
                    "description": "Edit list",
                },
            },
            "required": ["doc_id", "edits"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        doc_id = str(kwargs.get("doc_id") or "").strip()
        edits = kwargs.get("edits") or []
        if not isinstance(edits, list) or not edits:
            return _json_dump(
                {
                    "ok": False,
                    "doc_id": doc_id,
                    "changed": False,
                    "applied": 0,
                    "errors": ["edits empty"],
                    "not_found": [],
                }
            )
        md = read_doc(doc_id)
        patched, rep = _apply_edits(md, edits=[e for e in edits if isinstance(e, dict)])
        changed = patched != md
        if changed:
            write_doc(doc_id, patched)
        errors = rep.get("errors") if isinstance(rep, dict) else None
        ok = bool(isinstance(errors, list) and len(errors) == 0)
        return _json_dump({"ok": ok, "doc_id": doc_id, "changed": changed, **rep})


@dataclass
class MarkdownDocMatchInsertImageTool(FunctionTool[AstrAgentContext]):
    name: str = "md_doc_match_insert_image"
    description: str = (
        "Match text in markdown and insert an image markdown `![](url)` right after the matched line."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Doc id"},
                "match": {"type": "string", "description": "Text or regex to match"},
                "regex": {"type": "boolean", "description": "Treat match as regex, default false"},
                "occurrence": {"type": "integer", "description": "1=first, 0=all, 2=second..."},
                "image_url": {"type": "string", "description": "Image URL to insert"},
                "alt": {"type": "string", "description": "Alt text, default empty"},
                "ensure_blank_line": {
                    "type": "boolean",
                    "description": "Ensure blank lines around insertion, default true",
                },
                "suggestions_limit": {
                    "type": "integer",
                    "description": "How many fuzzy suggestions to return on failure, default 5",
                },
            },
            "required": ["doc_id", "match", "image_url"],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        doc_id = str(kwargs.get("doc_id") or "").strip()
        match = str(kwargs.get("match") or "")
        regex = bool(kwargs.get("regex", False))
        occurrence = _normalize_occurrence(kwargs.get("occurrence", 1), 1)
        image_url = str(kwargs.get("image_url") or "").strip()
        alt = str(kwargs.get("alt") or "")
        ensure_blank_line = bool(kwargs.get("ensure_blank_line", True))
        suggestions_limit = int(kwargs.get("suggestions_limit", 5) or 5)

        if not image_url:
            return _json_dump(
                {
                    "ok": False,
                    "doc_id": doc_id,
                    "changed": False,
                    "inserted": 0,
                    "error": "image_url empty",
                }
            )

        md = read_doc(doc_id)
        # idempotency: avoid inserting the same image URL repeatedly
        if image_url in md:
            return _json_dump(
                {
                    "ok": True,
                    "doc_id": doc_id,
                    "changed": False,
                    "inserted": 0,
                    "skipped": "already_exists",
                    "image_url": image_url,
                }
            )
        spans = _find_spans(md, match=match, regex=regex, occurrence=occurrence if occurrence != 0 else 10**9)
        if not spans:
            suggestions: List[Dict[str, Any]] = []
            if not regex and match and len(match) <= 240:
                suggestions = _fuzzy_suggest_lines(md, match, limit=suggestions_limit)
            return _json_dump(
                {
                    "ok": False,
                    "doc_id": doc_id,
                    "changed": False,
                    "inserted": 0,
                    "error": "match not found",
                    "requested_match": match,
                    "regex": regex,
                    "occurrence": occurrence,
                    "suggestions": suggestions,
                    "hint": "Try using one of the suggestions as `match`, or set `regex=true` for flexible matching.",
                }
            )

        img_md = f"![{alt}]({image_url})"
        targets = spans if occurrence == 0 else spans[:1]
        patched = md
        inserted = 0
        # apply in reverse order
        for start, _ in reversed(targets):
            ls, le = _line_bounds(patched, start)
            matched_line = patched[ls:le].strip()
            patched = _insert_after_line(
                patched,
                line_pos=start,
                insert_text=img_md,
                ensure_blank_line=ensure_blank_line,
            )
            inserted += 1

        changed = patched != md
        if changed:
            write_doc(doc_id, patched)
        return _json_dump(
            {
                "ok": True,
                "doc_id": doc_id,
                "changed": changed,
                "inserted": inserted,
                "requested_match": match,
                "regex": regex,
                "occurrence": occurrence,
                "image_url": image_url,
                "matched_line": matched_line if inserted == 1 else None,
            }
        )
