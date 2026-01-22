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
    urls: List[str] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        u = str(a.get("url") or "").strip()
        if u and u not in urls:
            urls.append(u)

    if not urls:
        return section

    section_text = section or ""
    missing = [u for u in urls if u not in section_text]
    if not missing:
        return section_text

    # 只补全缺失的链接，避免重复
    lines: List[str] = [section_text.rstrip(), "", "### 来源链接（补全）"]
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = str(a.get("title") or "").strip()
        url = str(a.get("url") or "").strip()
        if not url:
            continue
        if url not in missing:
            continue
        if title:
            lines.append(f"- {title} {url}")
        else:
            lines.append(f"- {url}")
    return "\n".join([x for x in lines if x is not None]).strip()
