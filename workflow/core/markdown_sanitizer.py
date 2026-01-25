from __future__ import annotations

import re


_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\\\|\\\\Users\\\\", flags=re.I)
_FILE_URI_RE = re.compile(r"file:///", flags=re.I)

# Remove reader-facing debug / internal plumbing lines.
_DEBUG_LINE_RE = re.compile(
    r"(索引文件|快照|对比|baseline|snapshot|created_at|pushed_at|json\s*校验|validation|traceback|\[dailynews\])",
    flags=re.I,
)

_GENERIC_FILLER_RE = re.compile(
    r"(优化(了)?(用户体验|体验)|修复(了)?(部分|一些)?bug|修复(了)?一些问题|提升(了)?稳定性|性能优化|细节优化|优化细节|"
    r"optimized user experience|fix(ed)? some bugs|performance optimizations?)",
    flags=re.I,
)

_URL_RE = re.compile(r"https?://[^\s)<>\"]+", flags=re.I)


def sanitize_markdown_for_publish(text: str) -> str:
    """
    Hard rules for production rendering:
    - Never leak local file paths / file:/// URIs.
    - Never show raw http(s) URLs as plain text (wrap as markdown links).
    - Drop obvious debug / snapshot / validation lines.
    """
    if not text:
        return ""

    raw_lines = (text or "").splitlines()
    out_lines: list[str] = []

    for line in raw_lines:
        s = (line or "").rstrip()
        if not s.strip():
            out_lines.append("")
            continue

        if _WINDOWS_PATH_RE.search(s) or _FILE_URI_RE.search(s):
            continue

        if _DEBUG_LINE_RE.search(s):
            continue
        if _GENERIC_FILLER_RE.search(s):
            # Keep only if the line also contains concrete anchors (numbers / refs).
            if not re.search(r"(\d|#|`|PR|commit|issue)", s, flags=re.I):
                continue

        # Convert lines that are *only* a URL (optionally with list marker) into a markdown link.
        m = re.match(r"^(\s*([-*+]|\d+\.)\s+)?<?(https?://\S+?)>?\s*$", s, flags=re.I)
        if m:
            prefix = m.group(1) or ""
            url = m.group(3)
            if not prefix:
                prefix = "- "
            out_lines.append(f"{prefix}[查看来源]({url})")
            continue

        # Convert inline raw URLs into markdown links (unless already inside a markdown link target).
        if "http://" in s.lower() or "https://" in s.lower():
            parts: list[str] = []
            last = 0
            for m2 in _URL_RE.finditer(s):
                start, end = m2.span()
                url = m2.group(0)

                # Skip markdown link targets: "... ]( https://... )"
                if re.search(r"\]\(\s*$", s[:start]):
                    continue

                # Handle angle-bracket autolinks: "<https://...>"
                if start > 0 and end < len(s) and s[start - 1] == "<" and s[end] == ">":
                    parts.append(s[last : start - 1])
                    parts.append(f"[查看来源]({url})")
                    last = end + 1
                    continue

                parts.append(s[last:start])
                parts.append(f"[查看来源]({url})")
                last = end

            if parts:
                parts.append(s[last:])
                s = "".join(parts)

        out_lines.append(s)

    cleaned = "\n".join(out_lines)

    # Merge link-only bullets into the previous bullet to avoid ugly "查看来源" standalone lines.
    merged_lines: list[str] = []
    for line in cleaned.splitlines():
        s = (line or "").rstrip()
        m = re.match(r"^\s*([-*+]\s+)?\[(查看来源|阅读原文)\]\((https?://[^)]+)\)\s*$", s, flags=re.I)
        if m and merged_lines:
            url = m.group(3)
            prev = merged_lines[-1].rstrip()
            if prev.lstrip().startswith(("-", "*", "+")) and "[查看来源](" not in prev and "[阅读原文](" not in prev:
                merged_lines[-1] = f"{prev} ( [查看来源]({url}) )"
                continue
        m2 = re.match(r"^\s*[-*+]\s+\[(查看来源|阅读原文)\]\((https?://[^)]+)\)\s*$", s, flags=re.I)
        if m2 and merged_lines:
            url = m2.group(2)
            prev = merged_lines[-1].rstrip()
            if prev.lstrip().startswith(("-", "*", "+")) and "[查看来源](" not in prev and "[阅读原文](" not in prev:
                merged_lines[-1] = f"{prev} ( [查看来源]({url}) )"
                continue
        merged_lines.append(s)

    cleaned = "\n".join(merged_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
