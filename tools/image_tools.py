import json
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..workflow.core.image_utils import (
    download_image_to_jpeg_file,
    get_plugin_data_dir,
    merge_images_vertical,
    parse_image_urls,
    probe_image_size_from_url,
)
from ..workflow.core.internal_event import make_internal_event

GEMINI_LAYOUT_PLUGIN_NAME = "astrbot_plugin_gemini_image_generation"
GEMINI_LAYOUT_VALID_RESOLUTIONS = ("1K", "2K", "4K")
GEMINI_LAYOUT_VALID_ASPECT_RATIOS = (
    "1:1",
    "16:9",
    "4:3",
    "3:2",
    "9:16",
    "4:5",
    "5:4",
    "21:9",
    "3:4",
    "2:3",
)


@dataclass
class ImageUrlsPreviewTool(FunctionTool[AstrAgentContext]):
    """
    Merge multiple image URLs into a single vertical preview image.
    """

    name: str = "image_urls_preview"
    description: str = "Merge multiple image URLs into a vertical preview image and return a local file path."
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Image URL list, supports http/https, file:///, base64://",
                },
                "max_width": {
                    "type": "integer",
                    "description": "Preview max width (px), default 1080",
                },
                "gap": {
                    "type": "integer",
                    "description": "Gap between images (px), default 8",
                },
            },
            "required": ["urls"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        urls = parse_image_urls(kwargs.get("urls"))
        if not urls:
            return "Invalid params: urls is empty"

        max_width = kwargs.get("max_width", 1080)
        gap = kwargs.get("gap", 8)
        try:
            max_width = int(max_width)
        except Exception:
            max_width = 1080
        try:
            gap = int(gap)
        except Exception:
            gap = 8

        out_dir = get_plugin_data_dir("image_previews")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"preview_{stamp}_{len(urls)}.jpg"
        try:
            merged = await merge_images_vertical(
                urls,
                out_path=out_path,
                max_width=max_width,
                gap=gap,
            )
            return str(Path(merged).resolve())
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] image_urls_preview failed: %s", e, exc_info=True
            )
            return f"Preview merge failed: {e}"


@dataclass
class ImageUrlDownloadTool(FunctionTool[AstrAgentContext]):
    """
    Download an image URL to a local JPEG file (optionally resized) and return basic metadata.
    """

    name: str = "image_url_download"
    description: str = "Download one image URL to a local JPEG file and return {local_path,width,height}."
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Image URL, supports http/https, file:///, base64://, data:image/...",
                },
                "max_width": {
                    "type": "integer",
                    "description": "Resize max width (px), default 1200",
                },
                "quality": {
                    "type": "integer",
                    "description": "JPEG quality (1-95), default 88",
                },
            },
            "required": ["url"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        url = str(kwargs.get("url") or "").strip()
        if not url:
            return "Invalid params: url is empty"

        try:
            max_width = int(kwargs.get("max_width", 1200) or 1200)
        except Exception:
            max_width = 1200
        try:
            quality = int(kwargs.get("quality", 88) or 88)
        except Exception:
            quality = 88
        quality = max(1, min(int(quality), 95))

        out_dir = get_plugin_data_dir("image_cache")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"img_{stamp}.jpg"
        try:
            saved = await download_image_to_jpeg_file(
                url,
                out_path=out_path,
                max_width=max_width,
                quality=quality,
            )
            if not saved:
                return "Download failed: cannot fetch/convert image"
            saved_path, orig = saved
            size = await probe_image_size_from_url(str(saved_path.resolve()))
            w, h = size or orig
            return json.dumps(
                {
                    "local_path": str(saved_path.resolve()),
                    "width": int(w),
                    "height": int(h),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] image_url_download failed: %s", e, exc_info=True
            )
            return f"Download failed: {e}"


@dataclass
class ImageUrlsDownloadBatchTool(FunctionTool[AstrAgentContext]):
    """
    Download multiple image URLs to local JPEG files and return per-item results.
    """

    name: str = "image_urls_download_batch"
    description: str = "Download multiple image URLs to local JPEG files and return JSON items with local_path/size/error."
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Image URL list, supports http/https, file:///, base64://, data:image/...",
                },
                "max_items": {
                    "type": "integer",
                    "description": "Max URLs to process in one call (1-20), default 8.",
                },
                "max_width": {
                    "type": "integer",
                    "description": "Resize max width (px), default 1200",
                },
                "quality": {
                    "type": "integer",
                    "description": "JPEG quality (1-95), default 88",
                },
            },
            "required": ["urls"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        urls = parse_image_urls(kwargs.get("urls"))
        if not urls:
            return "Invalid params: urls is empty"

        try:
            max_items = int(kwargs.get("max_items", 8) or 8)
        except Exception:
            max_items = 8
        max_items = max(1, min(max_items, 20))

        try:
            max_width = int(kwargs.get("max_width", 1200) or 1200)
        except Exception:
            max_width = 1200
        try:
            quality = int(kwargs.get("quality", 88) or 88)
        except Exception:
            quality = 88
        quality = max(1, min(int(quality), 95))

        urls = urls[:max_items]
        out_dir = get_plugin_data_dir("image_cache")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        items: list[dict[str, Any]] = []

        for idx, url in enumerate(urls, start=1):
            out_path = out_dir / f"img_batch_{stamp}_{idx:02d}.jpg"
            try:
                saved = await download_image_to_jpeg_file(
                    url,
                    out_path=out_path,
                    max_width=max_width,
                    quality=quality,
                )
                if not saved:
                    items.append({"url": url, "error": "download_or_convert_failed"})
                    continue
                saved_path, orig = saved
                size = await probe_image_size_from_url(str(saved_path.resolve()))
                w, h = size or orig
                items.append(
                    {
                        "url": url,
                        "local_path": str(saved_path.resolve()),
                        "width": int(w),
                        "height": int(h),
                    }
                )
            except Exception as e:
                items.append({"url": url, "error": str(e) or type(e).__name__})

        ok = sum(1 for x in items if isinstance(x, dict) and x.get("local_path"))
        failed = len(items) - ok
        return json.dumps(
            {
                "requested": len(urls),
                "ok": ok,
                "failed": failed,
                "items": items,
            },
            ensure_ascii=False,
        )


@dataclass
class GeminiLayoutGenerateImageTool(FunctionTool[AstrAgentContext]):
    """
    Generate one supporting image synchronously via the Gemini image plugin.
    """

    name: str = "gemini_image_generation"
    description: str = (
        "Synchronous Gemini image generation for the layout stage. "
        "Generate one supporting image and return {image_url,file_url,local_path,width,height,image_count}."
    )
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Image generation prompt for a supporting illustration.",
                },
                "resolution": {
                    "type": "string",
                    "description": "Optional resolution override.",
                    "enum": list(GEMINI_LAYOUT_VALID_RESOLUTIONS),
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Optional aspect ratio override.",
                    "enum": list(GEMINI_LAYOUT_VALID_ASPECT_RATIOS),
                },
                "use_reference_images": {
                    "type": "boolean",
                    "description": "Reserved for compatibility with gemini_image_generation. Ignored in layout stage.",
                    "default": False,
                },
                "include_user_avatar": {
                    "type": "boolean",
                    "description": "Reserved for compatibility with gemini_image_generation. Ignored in layout stage.",
                    "default": False,
                },
            },
            "required": ["prompt"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return json.dumps(
                {"ok": False, "error": "prompt_is_empty"},
                ensure_ascii=False,
            )

        resolution_raw = str(kwargs.get("resolution") or "").strip().upper()
        resolution = (
            resolution_raw
            if resolution_raw in GEMINI_LAYOUT_VALID_RESOLUTIONS
            else None
        )
        aspect_ratio_raw = str(kwargs.get("aspect_ratio") or "").strip()
        aspect_ratio = (
            aspect_ratio_raw
            if aspect_ratio_raw in GEMINI_LAYOUT_VALID_ASPECT_RATIOS
            else None
        )

        agent_ctx = context.context
        star_ctx = getattr(agent_ctx, "context", None)
        event = getattr(agent_ctx, "event", None) or make_internal_event(
            session_id="gemini-layout-tool"
        )
        plugin_meta = (
            star_ctx.get_registered_star(GEMINI_LAYOUT_PLUGIN_NAME)
            if star_ctx is not None
            else None
        )
        plugin = getattr(plugin_meta, "star_cls", None)
        if plugin is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "gemini_plugin_not_found",
                    "plugin_name": GEMINI_LAYOUT_PLUGIN_NAME,
                },
                ensure_ascii=False,
            )

        try:
            ensure_client = getattr(plugin, "_ensure_api_client", None)
            if callable(ensure_client):
                ensured = ensure_client(quiet=True)
                if inspect.isawaitable(ensured):
                    await ensured

            success, result = await plugin._generate_image_core_internal(
                event=event,
                prompt=prompt,
                reference_images=[],
                avatar_reference=[],
                override_resolution=resolution,
                override_aspect_ratio=aspect_ratio,
            )
            if not success:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "generation_failed",
                        "detail": str(result or ""),
                    },
                    ensure_ascii=False,
                )

            image_urls, image_paths, text_content, thought_signature = result
            resolved_paths = [
                str(Path(str(p)).resolve())
                for p in (image_paths or [])
                if str(p or "").strip()
            ]
            first_image_url = str((image_urls or [None])[0] or "").strip()
            if resolved_paths:
                first_path = Path(resolved_paths[0]).resolve()
                if not first_path.exists():
                    return json.dumps(
                        {
                            "ok": False,
                            "error": "generated_file_missing",
                            "local_path": str(first_path),
                        },
                        ensure_ascii=False,
                    )
                image_url = f"file:///{first_path.as_posix()}"
                file_url = image_url
                local_path = str(first_path)
            elif first_image_url:
                image_url = first_image_url
                file_url = None
                local_path = None
            else:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "no_image_result_returned",
                        "image_urls": image_urls or [],
                    },
                    ensure_ascii=False,
                )

            size = await probe_image_size_from_url(image_url)
            width, height = size or (0, 0)
            return json.dumps(
                {
                    "ok": True,
                    "image_url": image_url,
                    "file_url": file_url,
                    "local_path": local_path,
                    "width": int(width),
                    "height": int(height),
                    "image_count": max(len(resolved_paths), len(image_urls or [])),
                    "image_urls": image_urls or [],
                    "text_content": text_content,
                    "thought_signature": thought_signature,
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] layout gemini_image_generation failed: %s",
                e,
                exc_info=True,
            )
            return json.dumps(
                {
                    "ok": False,
                    "error": type(e).__name__,
                    "detail": str(e),
                },
                ensure_ascii=False,
            )
