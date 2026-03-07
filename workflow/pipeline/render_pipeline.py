from __future__ import annotations

import asyncio
import base64
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..core.config_models import RenderImageStyleConfig, RenderPipelineConfig
from ..core.image_utils import (
    adaptive_layout_html_images,
    inline_html_remote_images,
)
from .local_render import wait_for_file_ready
from .rendering import markdown_to_html, safe_text


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_b64(filename: str) -> str:
    try:
        p = _plugin_root() / "image" / filename
        if p.exists() and p.is_file():
            return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return ""
    return ""


def _templates_base_href() -> str:
    try:
        # Use plugin root so templates can reference `image/...` and `font/...` paths consistently.
        return _plugin_root().resolve().as_uri().rstrip("/") + "/"
    except Exception:
        return ""


def _root_asset_data_uri(rel_path: str, mime: str) -> str:
    """
    Read an asset under plugin root and return a data URI. Used for templates that rely on local images/fonts.
    """
    rp = (rel_path or "").strip().replace("\\", "/").lstrip("/")
    if not rp:
        return ""
    try:
        p = (_plugin_root() / rp).resolve()
        if not p.exists() or not p.is_file():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


_FONT_MIME_MAP: dict[str, tuple[str, str]] = {
    ".ttf": ("font/ttf", "truetype"),
    ".otf": ("font/otf", "opentype"),
    ".woff": ("font/woff", "woff"),
    ".woff2": ("font/woff2", "woff2"),
}


def _plugin_data_root() -> Path:
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path  # type: ignore

        return Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_dailynews_agent"
    except Exception:
        return _plugin_root() / ".plugin_data" / "astrbot_dailynews_agent"


def _plugin_data_asset_data_uri(rel_path: str) -> tuple[str, str]:
    rp = (rel_path or "").strip().replace("\\", "/")
    if not rp:
        return "", ""

    ext = Path(rp).suffix.lower()
    mime_and_fmt = _FONT_MIME_MAP.get(ext)
    if mime_and_fmt is None:
        return "", ""

    try:
        base = _plugin_data_root().resolve(strict=False)
        raw_path = Path(rp)
        if raw_path.is_absolute():
            p = raw_path.resolve(strict=False)
        else:
            p = (base / rp.lstrip("/")).resolve(strict=False)
        p.relative_to(base)
        if not p.exists() or not p.is_file():
            return "", ""
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        mime, fmt = mime_and_fmt
        return f"data:{mime};base64,{b64}", fmt
    except Exception:
        return "", ""


def _resolve_chenyu_font_asset(font_files: Sequence[str] | str | None) -> tuple[str, str]:
    if isinstance(font_files, str):
        candidates = [font_files]
    else:
        candidates = list(font_files or [])

    for item in candidates:
        data_uri, fmt = _plugin_data_asset_data_uri(str(item or ""))
        if data_uri:
            return data_uri, fmt

    return _root_asset_data_uri("font/HYWenHei-75W-2.ttf", "font/ttf"), "truetype"


def split_pages(text: str, page_chars: int, max_pages: int) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    if page_chars <= 0:
        return [s]

    pages: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in s.splitlines():
        piece = line + "\n"
        if buf_len + len(piece) > page_chars and buf:
            pages.append("".join(buf).rstrip())
            if len(pages) >= max_pages:
                return pages
            buf, buf_len = [], 0
        buf.append(piece)
        buf_len += len(piece)
    if buf and len(pages) < max_pages:
        pages.append("".join(buf).rstrip())
    return pages


def is_valid_image_file(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        if path.stat().st_size < 128:
            return False
        head = path.read_bytes()[:16]
        if head.startswith(b"\xff\xd8\xff"):  # JPEG
            return True
        if head.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return True
        if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
            return True
        return False
    except Exception:
        return False


@dataclass(frozen=True)
class RenderedPage:
    index: int
    total: int
    markdown: str
    image_path: Path | None
    method: str = ""


RenderHtmlFunc = Callable[[dict], Awaitable[Path | None]]
RenderT2IFunc = Callable[[str], Awaitable[Path | None]]


async def _try_with_retries(
    *,
    attempts: int,
    call: Callable[[], Awaitable[Path | None]],
    poll_timeout_s: float,
    poll_interval_ms: int,
    log_level: str = "error",
) -> Path | None:
    last: Exception | None = None
    last_exc_info = None
    for attempt in range(max(0, attempts) + 1):
        try:
            p = await call()
            if p is None:
                raise RuntimeError("render returned None")
            candidate = Path(str(p)).resolve()
            ok = await wait_for_file_ready(
                candidate,
                is_valid=is_valid_image_file,
                timeout_s=poll_timeout_s,
                interval_ms=poll_interval_ms,
            )
            if ok:
                return candidate
            raise RuntimeError("render produced invalid image")
        except Exception as e:
            last = e
            last_exc_info = sys.exc_info()
            if attempt < max(0, attempts):
                await asyncio.sleep(0.6 + 0.6 * attempt)
    if last is not None:
        msg = "[dailynews] render failed after retries: %s"
        lvl = (log_level or "error").lower()
        if lvl == "warning":
            astrbot_logger.warning(msg, last, exc_info=last_exc_info)
        elif lvl == "info":
            astrbot_logger.info(msg, last, exc_info=last_exc_info)
        else:
            astrbot_logger.error(msg, last, exc_info=last_exc_info)
    return None


async def _build_body_html(page_markdown: str, *, style: RenderImageStyleConfig) -> str:
    body_html = await inline_html_remote_images(markdown_to_html(page_markdown))
    return await adaptive_layout_html_images(
        body_html,
        full_max_width=style.full_max_width,
        medium_max_width=style.medium_max_width,
        narrow_max_width=style.narrow_max_width,
        float_if_width_le=style.float_threshold,
        float_enabled=style.float_enabled,
    )


def _looks_like_chenyu_template(template_str: str) -> bool:
    s = template_str or ""
    return ("chenyu_bg_top" in s) or ("bg-container" in s and "chenyu_bg_" in s)


def _promote_headings_for_chenyu(md: str) -> str:
    """
    chenyu-style.html 瀵广€屼竴绾ф爣棰樸€?h1) 鏈夌壒娈婃牱寮忥紝浣嗘棩鎶ユ鏂囬€氬父鍙湪鏈€椤堕儴鏈変竴涓?`#`銆?    杩欓噷鍋氫竴涓粎鍦ㄦ覆鏌撻樁娈电殑鈥滄爣棰樺眰绾ц皟鏁粹€濓細
    - 鍘绘帀绗竴琛屾枃妗ｆ爣棰?`# ...`锛堥伩鍏嶄笌 Hero 閲嶅锛?    - 灏?`##` 鎻愬崌涓?`#`锛宍###` 鎻愬崌涓?`##`锛堣烦杩?fenced code block锛?    """
    text = (md or "").strip("\n")
    if not text:
        return ""

    lines = text.splitlines()

    # Detect whether the doc uses H2/H3 as main headings (skip fenced code).
    has_h2 = False
    has_h3 = False
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^\s*(#{2,4})\s*(\S.*)?$", line)
        if not m:
            continue
        level = len(m.group(1))
        if level == 2:
            has_h2 = True
        elif level == 3:
            has_h3 = True

    # If there are no H2 but there are H3, treat H3 as 鈥渢op sections鈥?
    promote_map: dict[int, int] = {}
    if has_h2:
        promote_map = {2: 1, 3: 2}
    elif has_h3:
        promote_map = {3: 1, 4: 2}

    out: list[str] = []
    in_fence = False
    removed_first_h1 = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        # Remove the very first H1 title to avoid duplicating the template's hero title.
        if not removed_first_h1 and re.match(r"^\s*#(?!#)\s*\S", line):
            removed_first_h1 = True
            continue

        m = re.match(r"^(\s*)(#{1,6})\s*(\S.*)?$", line)
        if not m:
            out.append(line)
            continue

        indent, hashes, rest = m.group(1), m.group(2), (m.group(3) or "").rstrip()
        level = len(hashes)
        new_level = promote_map.get(level)
        if not new_level:
            out.append(line)
            continue
        out.append(f"{indent}{'#' * new_level} {rest}".rstrip())

    return "\n".join(out).strip("\n")


def _wrap_h1_sections_for_chenyu(body_html: str) -> str:
    """
    chenyu-style 鏈熸湜鈥滄瘡涓竴绾ф爣棰樹竴鍧?section-box鈥濄€傝繖閲屾妸娓叉煋鍚庣殑 HTML 鎸?<h1> 鍒嗘鍖呰锛?    - preamble锛堥涓?h1 涔嬪墠鐨勫唴瀹癸級浼氬苟鍏ョ涓€涓?section
    - 娌℃湁 h1 鏃讹紝鏁翠綋鍖呬竴灞?section-box
    """
    s = (body_html or "").strip()
    if not s:
        return ""

    parts = re.split(r"(<h1>.*?</h1>)", s, flags=re.I | re.S)
    if len(parts) <= 1:
        return f'<div class="section-box"><div class="md">{s}</div></div>'

    pre = (parts[0] or "").strip()
    out: list[str] = []
    i = 1
    while i < len(parts):
        h1 = parts[i] or ""
        rest = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (h1 + rest).strip()
        if not out and pre:
            chunk = (pre + "\n" + chunk).strip()
        if chunk:
            out.append(f'<div class="section-box"><div class="md">{chunk}</div></div>')
        i += 2

    if not out and pre:
        out.append(f'<div class="section-box"><div class="md">{pre}</div></div>')

    return "\n".join(out).strip()


async def render_daily_news_pages(
    *,
    pages: Sequence[str],
    template_str: str,
    render_html: RenderHtmlFunc,
    render_t2i: RenderT2IFunc,
    pipeline: RenderPipelineConfig,
    style: RenderImageStyleConfig,
    title: str = "每日资讯日报",
    subtitle_fmt: str = "第{idx}/{total}页",
    chenyu_font_files: Sequence[str] | str | None = None,
) -> list[RenderedPage]:
    pages_list = list(pages)
    if not pages_list:
        return []

    out: list[RenderedPage] = []
    total = len(pages_list)
    for idx, page in enumerate(pages_list, start=1):
        img_path, method = await render_single_page_to_image(
            markdown=page,
            template_str=template_str,
            render_html=render_html,
            render_t2i=render_t2i,
            pipeline=pipeline,
            style=style,
            title=title,
            subtitle=subtitle_fmt.format(idx=idx, total=total),
            idx=idx,
            chenyu_font_files=chenyu_font_files,
        )

        out.append(
            RenderedPage(
                index=idx,
                total=total,
                markdown=page,
                image_path=img_path,
                method=method,
            )
        )

    return out


async def render_single_page_to_image(
    *,
    markdown: str,
    template_str: str,
    render_html: RenderHtmlFunc,
    render_t2i: RenderT2IFunc,
    pipeline: RenderPipelineConfig,
    style: RenderImageStyleConfig,
    title: str = "每日资讯日报",
    subtitle: str = "",
    idx: int = 1,
    chenyu_font_files: Sequence[str] | str | None = None,
) -> tuple[Path | None, str]:
    """
    娓叉煋鍗曢〉 Markdown 涓哄浘鐗囥€?    """
    bg_img = _asset_b64("sunsetbackground.jpg")
    char_img = _asset_b64("transparent_output.png")
    base_href = _templates_base_href()

    # chenyu-style assets
    chenyu_font, chenyu_font_format = _resolve_chenyu_font_asset(chenyu_font_files)
    chenyu_bg_top = _root_asset_data_uri("image/涓婂崐鑳屾櫙.png", "image/png")
    chenyu_bg_middle = _root_asset_data_uri("image/杩囨浮鍥剧墖.png", "image/png")
    chenyu_bg_bottom = _root_asset_data_uri("image/涓嬪崐鍥剧墖.jpg", "image/jpeg")
    chenyu_tower = _root_asset_data_uri("image/tower_no_bg.png", "image/png")

    is_chenyu = _looks_like_chenyu_template(template_str)
    page_md = _promote_headings_for_chenyu(markdown) if is_chenyu else markdown
    body_html = await _build_body_html(page_md, style=style)
    if is_chenyu:
        body_html = _wrap_h1_sections_for_chenyu(body_html)

    portrait_max_h = max(360, min(560, int(style.full_max_width * 0.68)))
    panorama_max_h = max(260, min(420, int(style.medium_max_width * 0.5)))

    ctx = {
        "title": safe_text(title),
        "subtitle": safe_text(subtitle),
        "body_html": body_html,
        "bg_img": bg_img,
        "char_img": char_img,
        "base_href": base_href,
        "chenyu_font": chenyu_font,
        "chenyu_font_format": chenyu_font_format,
        "chenyu_bg_top": chenyu_bg_top,
        "chenyu_bg_middle": chenyu_bg_middle,
        "chenyu_bg_bottom": chenyu_bg_bottom,
        "chenyu_tower": chenyu_tower,
        "img_full_px": int(style.full_max_width),
        "img_medium_px": int(style.medium_max_width),
        "img_narrow_px": int(style.narrow_max_width),
        "img_portrait_max_h": int(portrait_max_h),
        "img_panorama_max_h": int(panorama_max_h),
    }

    img: Path | None = None
    method = ""

    img = await _try_with_retries(
        attempts=pipeline.retries,
        call=lambda: render_html(ctx),
        poll_timeout_s=pipeline.poll_timeout_s,
        poll_interval_ms=pipeline.poll_interval_ms,
    )
    method = "html" if img is not None else method

    if img is None:
        img = await _try_with_retries(
            attempts=pipeline.retries,
            call=lambda: render_t2i(markdown),
            poll_timeout_s=pipeline.poll_timeout_s,
            poll_interval_ms=pipeline.poll_interval_ms,
        )
        method = "t2i" if img is not None else ""

    return (Path(img).resolve() if img is not None else None, method)
