from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .config_models import LayoutRefineConfig, RenderImageStyleConfig
from .image_utils import adaptive_layout_html_images, get_plugin_data_dir, inline_html_remote_images, merge_images_vertical
from .local_render import render_template_to_image_playwright
from .rendering import load_template, markdown_to_html, safe_text
from .utils import _json_from_text


class LayoutRefiner:
    def __init__(self, *, system_prompt: str):
        self._system_prompt = (system_prompt or "").strip()

    @staticmethod
    def _asset_b64(filename: str) -> str:
        try:
            root = Path(__file__).resolve().parents[1]
            p = root / "image" / filename
            if p.exists() and p.is_file():
                return base64.b64encode(p.read_bytes()).decode("utf-8")
        except Exception:
            return ""
        return ""

    @staticmethod
    def _split_pages(text: str, page_chars: int, max_pages: int) -> List[str]:
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

    async def _render_markdown_preview(
        self,
        markdown: str,
        *,
        refine: LayoutRefineConfig,
        style: RenderImageStyleConfig,
    ) -> Optional[Path]:
        pages = self._split_pages(
            markdown,
            page_chars=refine.preview_page_chars,
            max_pages=refine.preview_pages,
        )
        if not pages:
            return None

        tmpl = load_template("templates/daily_news.html").strip()
        bg_img = self._asset_b64("sunsetbackground.jpg")
        char_img = self._asset_b64("transparent_output.png")

        out_paths: List[str] = []
        for idx, page in enumerate(pages, start=1):
            body_html = await inline_html_remote_images(markdown_to_html(page))
            body_html = await adaptive_layout_html_images(
                body_html,
                full_max_width=style.full_max_width,
                medium_max_width=style.medium_max_width,
                narrow_max_width=style.narrow_max_width,
                float_if_width_le=style.float_threshold,
                float_enabled=style.float_enabled,
            )
            ctx = {
                "title": safe_text("每日资讯日报"),
                "subtitle": safe_text(f"预览 {idx}/{len(pages)}"),
                "body_html": body_html,
                "bg_img": bg_img,
                "char_img": char_img,
            }
            out_path = get_plugin_data_dir("layout_refine_previews") / (
                f"layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
            )
            rendered = await render_template_to_image_playwright(
                tmpl,
                ctx,
                out_path=out_path,
                viewport=(1080, 720),
                timeout_ms=refine.preview_timeout_ms,
                full_page=True,
            )
            out_paths.append(str(Path(rendered).resolve()))

        if not out_paths:
            return None
        if len(out_paths) == 1:
            return Path(out_paths[0]).resolve()

        merged = get_plugin_data_dir("layout_refine_previews") / (
            f"layout_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        merged_path = await merge_images_vertical(
            out_paths,
            out_path=merged,
            max_width=1080,
            gap=8,
        )
        return Path(merged_path).resolve()

    async def refine(
        self,
        *,
        markdown: str,
        images_by_source: Dict[str, List[str]],
        user_config: Dict[str, Any],
        astrbot_context: Any,
        provider_id: str,
    ) -> str:
        refine_cfg = LayoutRefineConfig.from_mapping(user_config)
        if not refine_cfg.enabled:
            return (markdown or "").strip()

        style_cfg = RenderImageStyleConfig.from_mapping(user_config)
        current = (markdown or "").strip()
        if not current:
            return current

        request_budget = int(refine_cfg.max_requests)

        system_prompt = self._system_prompt.strip()
        if not system_prompt:
            raise RuntimeError("layout refiner system prompt is empty")

        for round_idx in range(1, int(refine_cfg.rounds) + 1):
            try:
                preview_path = await self._render_markdown_preview(
                    current,
                    refine=refine_cfg,
                    style=style_cfg,
                )
            except Exception as e:
                astrbot_logger.warning("[dailynews] layout_refiner render preview failed: %s", e, exc_info=True)
                preview_path = None
            if preview_path is None:
                return current

            preview_url = f"file:///{preview_path.resolve().as_posix()}"
            payload = {
                "round": round_idx,
                "draft_markdown": current,
                "image_candidates": images_by_source,
                "output_schema": {
                    "patched_markdown": "string",
                    "done": "boolean(optional)",
                    "request_image_urls": ["string(optional)"],
                    "notes": "string(optional)",
                },
            }

            resp = await astrbot_context.llm_generate(
                chat_provider_id=provider_id,
                prompt=json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                image_urls=[preview_url],
            )
            data = _json_from_text(getattr(resp, "completion_text", "") or "")
            if not isinstance(data, dict):
                return current

            req = data.get("request_image_urls")
            if request_budget > 0 and isinstance(req, list) and req:
                urls = [str(u).strip() for u in req if str(u).strip()][: int(refine_cfg.request_max_images)]
                request_budget -= 1
                req_url = ""
                try:
                    merged = get_plugin_data_dir("layout_refine_requests") / (
                        f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(urls)}.jpg"
                    )
                    merged_path = await merge_images_vertical(
                        urls,
                        out_path=merged,
                        max_width=1080,
                        gap=8,
                    )
                    req_url = f"file:///{Path(merged_path).resolve().as_posix()}"
                except Exception:
                    req_url = ""

                payload2 = dict(payload)
                payload2["requested_image_urls"] = urls
                payload2["requested_images_preview"] = "已合成预览图(见图片输入)"
                resp2 = await astrbot_context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=json.dumps(payload2, ensure_ascii=False),
                    system_prompt=system_prompt,
                    image_urls=[u for u in [preview_url, req_url] if u],
                )
                data2 = _json_from_text(getattr(resp2, "completion_text", "") or "")
                if isinstance(data2, dict):
                    data = data2

            patched = str(data.get("patched_markdown") or "").strip()
            if patched:
                if patched == current and bool(data.get("done", True)):
                    return current
                current = patched
                if bool(data.get("done", False)):
                    return current

        return current

