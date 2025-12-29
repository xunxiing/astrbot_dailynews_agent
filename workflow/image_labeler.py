from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from PIL import Image

from .config_models import ImageLabelConfig
from .image_label_store import ImageLabelEntry, load_labels, save_labels, upsert_label
from .image_utils import download_image_to_jpeg_file, get_plugin_data_dir
from .utils import _json_from_text


def _shorten_zh(s: str, limit: int = 30) -> str:
    t = (s or "").strip().replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    if len(t) <= limit:
        return t
    return t[:limit]


def _guess_source_from_url(url: str) -> str:
    u = (url or "").lower()
    if "mp.weixin.qq.com" in u or "mmbiz.qpic.cn" in u:
        return "wechat"
    if "miyoushe.com" in u or "mihoyo.com" in u:
        return "miyoushe"
    return ""


class ImageLabeler:
    def __init__(self, *, system_prompt: str):
        self._system_prompt = (system_prompt or "").strip()

    @staticmethod
    def _cache_path_for(url: str) -> Path:
        h = hashlib.sha1((url or "").encode("utf-8", errors="ignore")).hexdigest()[:20]
        return get_plugin_data_dir("image_label_cache") / f"{h}.jpg"

    async def _ensure_local(self, url: str) -> Tuple[str, int, int]:
        out_path = self._cache_path_for(url)
        if out_path.exists() and out_path.is_file() and out_path.stat().st_size > 512:
            try:
                img = Image.open(str(out_path))
                return str(out_path.resolve()), int(img.width), int(img.height)
            except Exception:
                return str(out_path.resolve()), 0, 0
        saved = await download_image_to_jpeg_file(url, out_path=out_path, max_width=1200, quality=88)
        if not saved:
            return "", 0, 0
        path, (w, h) = saved
        return str(Path(path).resolve()), int(w), int(h)

    async def _label_batch(
        self,
        *,
        astrbot_context: Any,
        provider_id: str,
        items: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        imgs: List[str] = []
        meta: List[Dict[str, Any]] = []
        for it in items:
            url = str(it.get("url") or "").strip()
            if not url:
                continue
            local_path = str(it.get("local_path") or "").strip()
            if local_path:
                imgs.append(f"file:///{Path(local_path).resolve().as_posix()}")
            meta.append(
                {
                    "url": url,
                    "source": str(it.get("source") or ""),
                    "source_guess": _guess_source_from_url(url),
                }
            )
        if not imgs or not meta:
            return {}

        prompt = json.dumps(
            {
                "task": "image_labeling",
                "rules": [
                    "为每张图片生成一句话中文描述，<=30字",
                    "必须包含：人物/物品/主题（尽量用特定称呼如“钟离/胡桃/机器人/AI”等），以及来源(wechat/miyoushe/其他)",
                    "偏新闻口吻：突出事件/公告/活动/版本等时效信息（图片可见时）",
                    "不要编造看不见的信息；不确定就用“疑似/可能”",
                    "输出严格 JSON：[{\"url\":\"...\",\"label\":\"...\"}, ...] 与输入顺序对应",
                ],
                "items": meta,
            },
            ensure_ascii=False,
        )
        try:
            resp = await astrbot_context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=self._system_prompt,
                image_urls=imgs,
            )
            raw = getattr(resp, "completion_text", "") or ""
        except Exception as e:
            astrbot_logger.warning("[dailynews] image_label llm_generate failed: %s", e, exc_info=True)
            return {}

        data = _json_from_text(raw)
        out: Dict[str, str] = {}
        if isinstance(data, list):
            for it in data:
                if not isinstance(it, dict):
                    continue
                url = str(it.get("url") or "").strip()
                label = _shorten_zh(str(it.get("label") or ""), 30)
                if url and label:
                    out[url] = label
        elif isinstance(data, dict) and isinstance(data.get("labels"), list):
            for it in data.get("labels") or []:
                if not isinstance(it, dict):
                    continue
                url = str(it.get("url") or "").strip()
                label = _shorten_zh(str(it.get("label") or ""), 30)
                if url and label:
                    out[url] = label
        return out

    async def build_catalog(
        self,
        *,
        images_by_source: Mapping[str, Sequence[str]],
        astrbot_context: Any,
        cfg: ImageLabelConfig,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, ImageLabelEntry]]:
        """
        Returns:
        - catalog: [{source,url,label,width,height}, ...]
        - labels: url -> ImageLabelEntry (includes cached fields)
        """
        if not cfg.enabled or not cfg.provider_id:
            return [], {}

        # Flatten candidates with source.
        items: List[Dict[str, Any]] = []
        for source, urls in (images_by_source or {}).items():
            src = str(source or "").strip()
            for u in urls or []:
                url = str(u or "").strip()
                if not url:
                    continue
                items.append({"source": src, "url": url})
                if cfg.max_images_total > 0 and len(items) >= cfg.max_images_total:
                    break
            if cfg.max_images_total > 0 and len(items) >= cfg.max_images_total:
                break

        if not items:
            return [], {}

        cache = load_labels()
        need: List[Dict[str, Any]] = []
        for it in items:
            url = str(it["url"])
            if (not cfg.force_refresh) and url in cache and (cache[url].label or "").strip():
                continue
            need.append(it)

        # Download missing ones first (parallel).
        sem = asyncio.Semaphore(int(cfg.concurrency))

        async def prepare_one(it: Dict[str, Any]) -> Dict[str, Any]:
            url = str(it.get("url") or "").strip()
            if not url:
                return it
            async with sem:
                local_path, w, h = await self._ensure_local(url)
            it2 = dict(it)
            if local_path:
                it2["local_path"] = local_path
                it2["width"] = int(w)
                it2["height"] = int(h)
            return it2

        prepared = await asyncio.gather(*(prepare_one(dict(it)) for it in need))

        # Label in batches (each call max 2 images) with limited concurrency.
        batches: List[List[Dict[str, Any]]] = []
        bs = max(1, min(2, int(cfg.batch_size)))
        buf: List[Dict[str, Any]] = []
        for it in prepared:
            if not str(it.get("local_path") or "").strip():
                continue
            buf.append(it)
            if len(buf) >= bs:
                batches.append(buf)
                buf = []
        if buf:
            batches.append(buf)

        call_sem = asyncio.Semaphore(int(cfg.concurrency))

        async def call_batch(batch: List[Dict[str, Any]]) -> Dict[str, str]:
            async with call_sem:
                return await self._label_batch(
                    astrbot_context=astrbot_context, provider_id=cfg.provider_id, items=batch
                )

        results: List[Dict[str, str]] = []
        if batches:
            results = await asyncio.gather(*(call_batch(b) for b in batches))

        labeled: Dict[str, str] = {}
        for m in results:
            labeled.update(m or {})

        # Update cache.
        changed = False
        for it in items:
            url = str(it.get("url") or "").strip()
            if not url:
                continue
            src = str(it.get("source") or "").strip()
            entry = cache.get(url)
            if entry and (entry.label or "").strip() and (not cfg.force_refresh):
                continue
            label = (labeled.get(url) or "").strip()
            if not label:
                # fallback: best-effort stub
                label = _shorten_zh(f"{_guess_source_from_url(url) or src or 'other'} 图片", 30)
            local_path = ""
            w = 0
            h = 0
            for pit in prepared:
                if str(pit.get("url") or "").strip() == url:
                    local_path = str(pit.get("local_path") or "")
                    w = int(pit.get("width") or 0)
                    h = int(pit.get("height") or 0)
                    break
            upsert_label(
                cache,
                url=url,
                label=_shorten_zh(label, 30),
                source=src,
                local_path=local_path,
                width=w,
                height=h,
            )
            changed = True

        if changed:
            try:
                save_labels(cache)
            except Exception as e:
                astrbot_logger.warning("[dailynews] save_labels failed: %s", e, exc_info=True)

        # Build catalog with labels (use cache for all).
        catalog: List[Dict[str, Any]] = []
        for it in items:
            url = str(it.get("url") or "").strip()
            if not url:
                continue
            src = str(it.get("source") or "").strip()
            entry = cache.get(url) or ImageLabelEntry(url=url, label="", source=src)
            label = _shorten_zh(entry.label or "", 30) or _shorten_zh(
                f"{_guess_source_from_url(url) or src or 'other'} 图片", 30
            )
            catalog.append(
                {
                    "source": src,
                    "url": url,
                    "label": label,
                    "width": int(entry.width or 0),
                    "height": int(entry.height or 0),
                }
            )

        return catalog, cache
