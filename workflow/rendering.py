import re
from pathlib import Path
from typing import Dict, Optional


_TEMPLATE_CACHE: Dict[str, str] = {}


def _plugin_root() -> Path:
    # workflow/rendering.py -> <plugin_root>/workflow/rendering.py
    return Path(__file__).resolve().parents[1]


def load_template(rel_path: str) -> str:
    """
    从插件目录读取模板文件并缓存。rel_path 例如：'templates/daily_news.html'
    """
    key = rel_path.replace("\\", "/")
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    path = _plugin_root() / key
    content = path.read_text(encoding="utf-8")
    _TEMPLATE_CACHE[key] = content
    return content


def to_ascii_entities(text: str) -> str:
    """
    将非 ASCII 字符转为 HTML 实体，避免某些网络渲染端出现中文乱码。
    保留原有 ASCII 字符（含 < > & 等），因此仅建议用于已经安全的 HTML 字符串。
    """
    if not text:
        return ""
    out = []
    for ch in str(text):
        code = ord(ch)
        if code > 127:
            out.append(f"&#x{code:x};")
        else:
            out.append(ch)
    return "".join(out)


def escape_html(text: str) -> str:
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# NOTE: 仅避免把 markdown link 的 "(...)" 里 already-bracketed 的 URL 再包一层。
# 用 (?<!\() 判断前一个字符不是 '(' 即可。
_URL_RE = re.compile(r"(?<!\()(?P<url>https?://[^\s<>()]+)")


def _autolink_bare_urls(md: str) -> str:
    """
    把裸 URL 包一层 <...> 让 markdown 渲染为链接。
    - 避免处理在 fenced code block 内的内容
    """
    if not md:
        return ""
    lines = md.splitlines()
    out = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        def repl(m: re.Match) -> str:
            url = m.group("url")
            # 去掉常见结尾标点
            tail = ""
            while url and url[-1] in ".,;:!?)]}，。；：！？】》）":
                tail = url[-1] + tail
                url = url[:-1]
            return f"<{url}>{tail}" if url else m.group(0)

        out.append(_URL_RE.sub(repl, line))
    return "\n".join(out)


def markdown_to_html(md: str) -> str:
    """
    将 Markdown 渲染为 HTML（并做基础安全处理）：
    - 禁止 Markdown 中的原生 HTML（通过转义 < > &）
    - 支持 fenced code block / 列表 / 引用 / 换行
    - 裸 URL 自动转链接
    - 输出会再做一次 to_ascii_entities，绕开网络渲染端的中文乱码
    """
    raw = md or ""
    raw = _autolink_bare_urls(raw)
    safe_md = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html: Optional[str] = None
    try:
        import markdown as _md  # type: ignore

        html = _md.markdown(
            safe_md,
            extensions=[
                "fenced_code",
                "sane_lists",
                "nl2br",
            ],
            output_format="html5",
        )
    except Exception:
        html = None

    if not html:
        # 兜底：最少保证可读
        return f"<pre>{escape_html(raw)}</pre>"

    return to_ascii_entities(html)


def safe_text(text: str) -> str:
    """
    纯文本 -> HTML 安全文本（并实体化非 ASCII）。
    """
    return to_ascii_entities(escape_html(text or ""))
