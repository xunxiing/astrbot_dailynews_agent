from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional


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


def ensure_section_links(section: str, articles: List[Dict[str, Any]]) -> str:
    """
    Ensure each article URL appears in the section at least once.
    Hard requirement: never output raw http(s) URLs as plain text lines.
    """
    urls: List[str] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        u = str(a.get("url") or "").strip()
        if u and u not in urls:
            urls.append(u)

    if not urls:
        return section or ""

    section_text = section or ""
    missing = [u for u in urls if u not in section_text]
    if not missing:
        return section_text

    # Only fill missing links (avoid duplication), and never show raw URLs.
    lines: List[str] = [section_text.rstrip(), "", "### 来源（补全）"]
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = str(a.get("title") or "").strip()
        url = str(a.get("url") or "").strip()
        if not url or url not in missing:
            continue
        if title:
            lines.append(f"- {title} ([阅读原文]({url}))")
        else:
            lines.append(f"- [阅读原文]({url})")
    return "\n".join([x for x in lines if x is not None]).strip()

