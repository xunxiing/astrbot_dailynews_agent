import re
import html as _html
from pathlib import Path
from typing import Dict, Optional


_TEMPLATE_CACHE: Dict[str, str] = {}


def _plugin_root() -> Path:
    # workflow/pipeline/rendering.py -> <plugin_root>
    return Path(__file__).resolve().parents[2]


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
    把裸 URL 转成 [url](url) 让 markdown 渲染为链接。
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
            return f"[{url}]({url}){tail}" if url else m.group(0)

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
    # 仅转义可能形成原生 HTML 的字符（<、&），保留 '>' 以支持 Markdown 引用语法。
    safe_md = raw.replace("&", "&amp;").replace("<", "&lt;")

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

    html = _decorate_markdown_html(html)
    return to_ascii_entities(html)


def safe_text(text: str) -> str:
    """
    纯文本 -> HTML 安全文本（并实体化非 ASCII）。
    """
    return to_ascii_entities(escape_html(text or ""))


_DIVIDER_HTML = (
    '<div class="css-snow-divider">'
    '<div class="divider-line"></div>'
    '<div class="divider-mountain"></div>'
    "</div>"
)


_HR_RE = re.compile(r"<hr\s*/?>", re.I)
_H2_RE = re.compile(r"<h2>(.*?)</h2>", re.I | re.S)
_H3_RE = re.compile(r"<h3>(.*?)</h3>", re.I | re.S)
_CODE_LANG_RE = re.compile(r'<pre><code class="language-([^" ]+)">(.*?)</code></pre>', re.I | re.S)
_CODE_NO_LANG_RE = re.compile(r"<pre><code>(.*?)</code></pre>", re.I | re.S)


def _decorate_markdown_html(html: str) -> str:
    """
    给 Markdown 渲染后的 HTML 做轻量“结构增强”，以便用 CSS 还原你截图里的样式：
    - h2/h3 右侧追加雪山分割线
    - <hr> 替换为雪山分割线
    - fenced code block 添加语言标签头
    """
    s = html or ""

    s = _HR_RE.sub(_DIVIDER_HTML, s)

    def wrap_h2(m: re.Match) -> str:
        inner = m.group(1)
        return f'<div class="md-heading md-h2"><h2>{inner}</h2>{_DIVIDER_HTML}</div>'

    def wrap_h3(m: re.Match) -> str:
        inner = m.group(1)
        return f'<div class="md-heading md-h3"><h3>{inner}</h3>{_DIVIDER_HTML}</div>'

    s = _H2_RE.sub(wrap_h2, s)
    s = _H3_RE.sub(wrap_h3, s)

    def _highlight(code_html: str, lang: str) -> str:
        """
        对 fenced code block 做轻量语法高亮（尽量还原你截图里的“彩色代码”效果）。
        - 使用 pygments（可用则启用）
        - 输出 inline style，避免引入大段 CSS
        """
        code_text = _html.unescape(code_html or "")
        try:
            from pygments import highlight  # type: ignore
            from pygments.formatters import HtmlFormatter  # type: ignore
            from pygments.lexers import TextLexer, get_lexer_by_name  # type: ignore

            try:
                lexer = get_lexer_by_name(lang or "text")
            except Exception:
                lexer = TextLexer()
            fmt = HtmlFormatter(nowrap=True, noclasses=True)
            return highlight(code_text, lexer, fmt)
        except Exception:
            return escape_html(code_text)

    def wrap_code_lang(m: re.Match) -> str:
        lang_raw = (m.group(1) or "").strip()
        body = m.group(2)
        lang_label = escape_html(lang_raw) if lang_raw else "code"
        highlighted = _highlight(body, lang_raw)
        return (
            '<div class="codebox">'
            f'<div class="codebox-head"><span class="codebox-lang">{lang_label}</span></div>'
            f'<pre class="codebox-body"><code class="language-{lang_label}">{highlighted}</code></pre>'
            "</div>"
        )

    def wrap_code_no_lang(m: re.Match) -> str:
        body = m.group(1)
        highlighted = _highlight(body, "text")
        return (
            '<div class="codebox">'
            '<div class="codebox-head"><span class="codebox-lang">code</span></div>'
            f'<pre class="codebox-body"><code>{highlighted}</code></pre>'
            "</div>"
        )

    s = _CODE_LANG_RE.sub(wrap_code_lang, s)
    s = _CODE_NO_LANG_RE.sub(wrap_code_no_lang, s)

    return s
