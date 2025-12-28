from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict, List

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

from ..workflow.image_utils import get_plugin_data_dir, merge_images_vertical, parse_image_urls
from ..workflow.image_utils import download_image_to_jpeg_file, probe_image_size_from_url


@dataclass
class ImageUrlsPreviewTool(FunctionTool[AstrAgentContext]):
    """
    Merge multiple image URLs into a single vertical preview image.
    """

    name: str = "image_urls_preview"
    description: str = "Merge multiple image URLs into a vertical preview image and return a local file path."
    parameters: Dict[str, Any] = Field(
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
            astrbot_logger.warning("[dailynews] image_urls_preview failed: %s", e, exc_info=True)
            return f"Preview merge failed: {e}"


@dataclass
class ImageUrlDownloadTool(FunctionTool[AstrAgentContext]):
    """
    Download an image URL to a local JPEG file (optionally resized) and return basic metadata.
    """

    name: str = "image_url_download"
    description: str = "Download one image URL to a local JPEG file and return {local_path,width,height}."
    parameters: Dict[str, Any] = Field(
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
            astrbot_logger.warning("[dailynews] image_url_download failed: %s", e, exc_info=True)
            return f"Download failed: {e}"
