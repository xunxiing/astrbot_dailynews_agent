from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from astrbot.core.agent.tool import ToolSet

from ..tools import (
    ImageUrlDownloadTool,
    MarkdownDocApplyEditsTool,
    MarkdownDocMatchInsertImageTool,
    MarkdownDocReadTool,
)
from .config_models import LayoutRefineConfig, RenderImageStyleConfig
from .image_utils import (
    adaptive_layout_html_images,
    get_plugin_data_dir,
    inline_html_remote_images,
    merge_images_vertical,
)
from .internal_event import make_internal_event
from .md_doc_store import create_doc, read_doc, write_doc
from .rendering import load_template, markdown_to_html, safe_text
from .utils import _json_from_text
from .local_render import render_template_to_image_playwright


class ToolBasedLayoutEditor:
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
            out_path = get_plugin_data_dir("layout_tool_previews") / (
                f"tool_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
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

        merged = get_plugin_data_dir("layout_tool_previews") / (
            f"tool_layout_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        merged_path = await merge_images_vertical(
            out_paths,
            out_path=merged,
            max_width=1080,
            gap=8,
        )
        return Path(merged_path).resolve()

    async def _make_candidate_preview(
        self,
        urls: List[str],
        *,
        max_width: int = 1080,
        gap: int = 8,
    ) -> Optional[str]:
        if not urls:
            return None
        out = get_plugin_data_dir("layout_tool_requests") / (
            f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(urls)}.jpg"
        )
        merged = await merge_images_vertical(
            urls,
            out_path=out,
            max_width=max_width,
            gap=gap,
        )
        local_path = Path(merged).resolve().as_posix()
        return f"file:///{local_path}"

    async def edit(
        self,
        *,
        draft_markdown: str,
        images_by_source: Dict[str, List[str]],
        user_config: Dict[str, Any],
        astrbot_context: Any,
        provider_id: str,
        rounds: int = 2,
        max_steps: int = 25,
        request_max_requests: int = 1,
        request_max_images: int = 6,
    ) -> str:
        system_prompt = self._system_prompt.strip()
        if not system_prompt:
            raise RuntimeError("tool layout system prompt is empty")

        did, _ = create_doc((draft_markdown or "").strip())

        refine_cfg = LayoutRefineConfig.from_mapping(user_config)
        style_cfg = RenderImageStyleConfig.from_mapping(user_config)

        # Tools available to the LLM for editing.
        tools = ToolSet(
            [
                MarkdownDocReadTool(),
                MarkdownDocApplyEditsTool(),
                MarkdownDocMatchInsertImageTool(),
                ImageUrlDownloadTool(),
            ]
        )

        allowed_urls: set[str] = set()
        for _, urls in (images_by_source or {}).items():
            for u in urls or []:
                if isinstance(u, str) and u.strip():
                    allowed_urls.add(u.strip())

        candidate_preview_url: Optional[str] = None
        candidate_preview_urls: List[str] = []
        request_budget = max(0, int(request_max_requests))

        for r in range(1, max(1, int(rounds)) + 1):
            current = read_doc(did).strip()
            if not current:
                return current

            preview_path: Optional[Path] = None
            try:
                preview_path = await self._render_markdown_preview(
                    current,
                    refine=refine_cfg,
                    style=style_cfg,
                )
            except Exception as e:
                astrbot_logger.warning("[dailynews] tool_layout preview render failed: %s", e, exc_info=True)

            image_urls: List[str] = []
            if preview_path is not None:
                image_urls.append(f"file:///{preview_path.resolve().as_posix()}")
            if candidate_preview_url:
                image_urls.append(candidate_preview_url)

            payload = {
                "round": r,
                "doc_id": did,
                "doc_excerpt": current[:1800],
                "doc_length": len(current),
                "image_candidates": images_by_source,
                "constraints": {
                    "request_max_requests": request_budget,
                    "request_max_images": request_max_images,
                },
                "candidate_preview": {
                    "provided": bool(candidate_preview_url),
                    "for_urls": candidate_preview_urls,
                    "note": (
                        "候选图预览已作为图片输入提供；请不要重复 request_image_urls，直接开始插图/改文。"
                        if candidate_preview_url
                        else ""
                    ),
                },
                "tools": {
                    "insert_image": "md_doc_match_insert_image(doc_id, match, image_url, ...)",
                    "apply_edits": "md_doc_apply_edits(doc_id, edits=[...])",
                    "read": "md_doc_read(doc_id, start=0, max_chars=2400)",
                    "image_meta": "image_url_download(url) -> {local_path,width,height}",
                },
                "output_schema": {
                    "done": "boolean",
                    "request_image_urls": ["string(optional)"],
                    "notes": "string(optional)",
                    "patched_markdown": "string(optional, ONLY if you decide to overwrite the whole doc)",
                },
            }

            try:
                resp = await astrbot_context.tool_loop_agent(
                    event=make_internal_event(session_id=f"layout:{did}"),
                    chat_provider_id=provider_id,
                    prompt=json.dumps(payload, ensure_ascii=False),
                    system_prompt=system_prompt,
                    tools=tools,
                    image_urls=image_urls,
                    max_steps=max(5, int(max_steps)),
                )
            except Exception as e:
                # If the model never produces a final response (e.g. loops on md_doc_read),
                # salvage current doc and stop the tool-based refinement.
                astrbot_logger.warning("[dailynews] tool_layout tool_loop_agent failed: %s", e, exc_info=True)
                return read_doc(did).strip()

            raw = getattr(resp, "completion_text", "") or ""
            data = _json_from_text(raw) or {}
            if not isinstance(data, dict):
                # If model didn't output JSON, assume it has edited doc via tools.
                return read_doc(did).strip()

            patched_whole = str(data.get("patched_markdown") or "").strip()
            if patched_whole:
                write_doc(did, patched_whole)

            req = data.get("request_image_urls")
            if request_budget > 0 and isinstance(req, list) and req:
                urls = [str(u).strip() for u in req if str(u).strip()]
                if allowed_urls:
                    urls = [u for u in urls if u in allowed_urls]
                urls = urls[: max(1, int(request_max_images))]
                if urls:
                    request_budget -= 1
                    try:
                        candidate_preview_url = await self._make_candidate_preview(urls)
                        candidate_preview_urls = list(urls)
                    except Exception as e:
                        astrbot_logger.warning(
                            "[dailynews] tool_layout candidate preview failed: %s", e, exc_info=True
                        )
                        candidate_preview_url = None
                        candidate_preview_urls = []
                    continue

            # Some models keep requesting images even when budget is 0.
            # If we already provided a candidate preview, force a follow-up round that forbids further requests.
            if (
                isinstance(req, list)
                and req
                and request_budget <= 0
                and candidate_preview_url
                and candidate_preview_urls
            ):
                astrbot_logger.info("[dailynews] tool_layout ignoring repeated request_image_urls; proceed")
                continue

            if bool(data.get("done", False)):
                return read_doc(did).strip()

        return read_doc(did).strip()
