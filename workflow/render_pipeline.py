from __future__ import annotations

import asyncio
import sys
import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Sequence

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .config_models import RenderImageStyleConfig, RenderPipelineConfig
from .image_utils import adaptive_layout_html_images, get_plugin_data_dir, inline_html_remote_images
from .local_render import render_template_to_image_playwright, wait_for_file_ready
from .rendering import markdown_to_html, safe_text


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _asset_b64(filename: str) -> str:
    try:
        p = _plugin_root() / "image" / filename
        if p.exists() and p.is_file():
            return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return ""
    return ""


def split_pages(text: str, page_chars: int, max_pages: int) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    if page_chars <= 0:
        return [s]

    pages: List[str] = []
    buf: List[str] = []
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
        if head.startswith(b"\xFF\xD8\xFF"):  # JPEG
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
    image_path: Optional[Path]
    method: str = ""


RenderHtmlFunc = Callable[[dict], Awaitable[Optional[Path]]]
RenderT2IFunc = Callable[[str], Awaitable[Optional[Path]]]


async def _try_with_retries(
    *,
    attempts: int,
    call: Callable[[], Awaitable[Optional[Path]]],
    poll_timeout_s: float,
    poll_interval_ms: int,
    log_level: str = "error",
) -> Optional[Path]:
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
) -> List[RenderedPage]:
    pages_list = list(pages)
    if not pages_list:
        return []

    bg_img = _asset_b64("sunsetbackground.jpg")
    char_img = _asset_b64("transparent_output.png")

    out: List[RenderedPage] = []
    total = len(pages_list)
    for idx, page in enumerate(pages_list, start=1):
        body_html = await _build_body_html(page, style=style)
        portrait_max_h = max(360, min(560, int(style.full_max_width * 0.68)))
        panorama_max_h = max(260, min(420, int(style.medium_max_width * 0.5)))
        ctx = {
            "title": safe_text(title),
            "subtitle": safe_text(subtitle_fmt.format(idx=idx, total=total)),
            "body_html": body_html,
            "bg_img": bg_img,
            "char_img": char_img,
            "img_full_px": int(style.full_max_width),
            "img_medium_px": int(style.medium_max_width),
            "img_narrow_px": int(style.narrow_max_width),
            "img_portrait_max_h": int(portrait_max_h),
            "img_panorama_max_h": int(panorama_max_h),
        }

        img: Optional[Path] = await _try_with_retries(
            attempts=pipeline.retries,
            call=lambda: render_html(ctx),
            poll_timeout_s=pipeline.poll_timeout_s,
            poll_interval_ms=pipeline.poll_interval_ms,
            log_level="warning" if pipeline.playwright_fallback else "error",
        )
        method = "html"

        if img is None and pipeline.playwright_fallback:
            try:
                out_path = get_plugin_data_dir("render_fallback") / (
                    f"playwright_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
                )
                img = await render_template_to_image_playwright(
                    template_str,
                    ctx,
                    out_path=out_path,
                    viewport=(1080, 720),
                    timeout_ms=pipeline.playwright_timeout_ms,
                    full_page=True,
                )
                img = Path(str(img)).resolve()
                method = "playwright"
            except Exception:
                astrbot_logger.error("[dailynews] playwright render failed", exc_info=True)
                img = None

        if img is None:
            img = await _try_with_retries(
                attempts=pipeline.retries,
                call=lambda: render_t2i(page),
                poll_timeout_s=pipeline.poll_timeout_s,
                poll_interval_ms=pipeline.poll_interval_ms,
            )
            method = "t2i" if img is not None else ""

        out.append(
            RenderedPage(
                index=idx,
                total=total,
                markdown=page,
                image_path=Path(img).resolve() if img is not None else None,
                method=method,
            )
        )

    return out
