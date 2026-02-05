from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..core.config_models import (
    LayoutRefineConfig,
    RenderImageStyleConfig,
    RenderPipelineConfig,
)
from ..core.image_utils import (
    get_plugin_data_dir,
    merge_images_vertical,
)
from ..core.utils import _json_from_text
from ..pipeline.render_pipeline import render_single_page_to_image, split_pages
from ..pipeline.rendering import load_template


class LayoutRefiner:
    def __init__(self, *, system_prompt: str):
        self._system_prompt = (system_prompt or "").strip()

    async def _render_markdown_preview(
        self,
        markdown: str,
        *,
        refine: LayoutRefineConfig,
        style: RenderImageStyleConfig,
    ) -> Path | None:
        pages = split_pages(
            markdown,
            page_chars=refine.preview_page_chars,
            max_pages=refine.preview_pages,
        )
        if not pages:
            return None

        tmpl = load_template("templates/daily_news.html").strip()
        pipeline_cfg = RenderPipelineConfig()  # Default rendering settings for preview

        out_paths: list[str] = []
        for idx, page in enumerate(pages, start=1):
            # Use unified render pipeline for consistency
            img_path, _ = await render_single_page_to_image(
                markdown=page,
                template_str=tmpl,
                render_html=lambda ctx: None,  # No remote HTML for preview
                render_t2i=lambda md: None,  # No T2I for preview
                pipeline=pipeline_cfg,
                style=style,
                title="每日资讯日报",
                subtitle=f"预览 {idx}/{len(pages)}",
                idx=idx,
            )
            if img_path:
                out_paths.append(str(img_path.resolve()))

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
        images_by_source: dict[str, list[str]],
        image_catalog: list[dict[str, Any]] | None = None,
        user_config: dict[str, Any],
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
                astrbot_logger.warning(
                    "[dailynews] layout_refiner render preview failed: %s",
                    e,
                    exc_info=True,
                )
                preview_path = None
            if preview_path is None:
                return current

            preview_url = f"file:///{preview_path.resolve().as_posix()}"
            payload = {
                "round": round_idx,
                "draft_markdown": current,
                "image_candidates": images_by_source,
                "image_catalog": image_catalog or [],
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
                urls = [str(u).strip() for u in req if str(u).strip()][
                    : int(refine_cfg.request_max_images)
                ]
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
