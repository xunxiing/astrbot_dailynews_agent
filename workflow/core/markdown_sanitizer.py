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

_TITLE_RE = re.compile(r"(?m)^#\s*每日资讯日报\s*$")
_FIRST_H1_RE = re.compile(r"(?m)^#\s+.+$")
_FIRST_TAG_H2_RE = re.compile(r"(?m)^##\s*\[[^\]]+\]\s*$")

_THINKING_PREAMBLE_RE = re.compile(
    r"(current thinking|analyzing output format|i(?:'|’)m now satisfied|right,\s*let['’]s|"
    r"my goal here|i will now output the content)",
    flags=re.I,
)

_TIME_LINE_RE = re.compile(r"^\*\s*生成时间\s*:\s*.+\*\s*$")

_DOMAIN_LABEL_LINK_RE = re.compile(
    r"\[(?P<label>[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?(?:\.[A-Za-z0-9-]{1,63})+)\]\((?P<url>https?://[^)]+)\)",
    flags=re.I,
)


def sanitize_markdown_for_publish(text: str) -> str:
    """
    Hard rules for production rendering:
    - Never leak local file paths / file:/// URIs.
    - Never show raw http(s) URLs as plain text (wrap as markdown links).
    - Drop obvious debug / snapshot / validation lines.
    """
    if not text:
        return ""

    # Some providers may prepend "thinking"/analysis text before the real markdown.
    # Prefer keeping only the actual report starting from "# 每日资讯日报".
    m_title = _TITLE_RE.search(text)
    if m_title:
        text = text[m_title.start() :]
    else:
        # If we can find the first tagged section, keep from there and normalize header later.
        m_tag = _FIRST_TAG_H2_RE.search(text)
        if m_tag:
            text = text[m_tag.start() :]
        else:
            m_h1 = _FIRST_H1_RE.search(text)
            if m_h1:
                text = text[m_h1.start() :]

    # Normalize H1: if a model outputs an English title, force it to the standard report title.
    lines0 = (text or "").splitlines()
    if lines0:
        first = (lines0[0] or "").strip()
        if first.startswith("#") and not _TITLE_RE.match(first):
            lines0[0] = "# 每日资讯日报"
        # If the model failed to output any H1 (e.g. starts from ## [tag]), prepend a standard header.
        if lines0 and lines0[0].lstrip().startswith("## "):
            lines0.insert(0, "# 每日资讯日报")
    text = "\n".join(lines0)

    raw_lines = (text or "").splitlines()
    out_lines: list[str] = []

    # Drop obvious "thinking" preamble blocks between title and the first real section.
    seen_first_section = False
    for line in raw_lines:
        s = (line or "").rstrip()
        if not s.strip():
            out_lines.append("")
            continue

        if not seen_first_section:
            if s.lstrip().startswith("## "):
                seen_first_section = True
            else:
                # Allow the standard time line, but drop analysis/preamble chatter.
                if _TIME_LINE_RE.match(s.strip()):
                    out_lines.append(s)
                    continue
                if _THINKING_PREAMBLE_RE.search(s):
                    continue
                # If it's mostly ASCII and not part of a section yet, it's likely preface -> drop.
                ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
                if ascii_ratio > 0.85 and not s.lstrip().startswith("#"):
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

    # Replace domain-as-label markdown links with a consistent label to reduce visual noise.
    cleaned = _DOMAIN_LABEL_LINK_RE.sub(
        lambda m: f"[查看来源]({m.group('url')})", cleaned
    )

    # Merge link-only bullets into the previous bullet to avoid ugly "查看来源" standalone lines.
    merged_lines: list[str] = []
    for line in cleaned.splitlines():
        s = (line or "").rstrip()
        m = re.match(
            r"^\s*([-*+]\s+)?\[(查看来源|阅读原文)\]\((https?://[^)]+)\)\s*$",
            s,
            flags=re.I,
        )
        if m and merged_lines:
            url = m.group(3)
            prev = merged_lines[-1].rstrip()
            if (
                prev.lstrip().startswith(("-", "*", "+"))
                and "[查看来源](" not in prev
                and "[阅读原文](" not in prev
            ):
                merged_lines[-1] = f"{prev} ( [查看来源]({url}) )"
                continue
        m2 = re.match(
            r"^\s*[-*+]\s+\[(查看来源|阅读原文)\]\((https?://[^)]+)\)\s*$",
            s,
            flags=re.I,
        )
        if m2 and merged_lines:
            url = m2.group(2)
            prev = merged_lines[-1].rstrip()
            if (
                prev.lstrip().startswith(("-", "*", "+"))
                and "[查看来源](" not in prev
                and "[阅读原文](" not in prev
            ):
                merged_lines[-1] = f"{prev} ( [查看来源]({url}) )"
                continue
        merged_lines.append(s)

    cleaned = "\n".join(merged_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
