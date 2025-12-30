from __future__ import annotations

import asyncio
import hashlib
import json
import random
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


def _should_retry_error(e: Exception) -> bool:
    msg = str(e or "").lower()
    hints = [
        "rate limit",
        "ratelimit",
        "too many requests",
        "429",
        "timeout",
        "timed out",
        "overloaded",
        "capacity",
        "busy",
        "temporarily",
        "connection",
        "reset",
        "gateway",
        "502",
        "503",
        "504",
    ]
    return any(h in msg for h in hints)


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
        cfg: ImageLabelConfig,
    ) -> Dict[str, Dict[str, Any]]:
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
                    "width": int(it.get("width") or 0),
                    "height": int(it.get("height") or 0),
                }
            )
        if not imgs or not meta:
            return {}

        prompt = json.dumps(
            {
                "task": "image_labeling",
                "rules": [
                    "为每张图片生成一句话中文描述，<=30字",
                    "必须尽量包含：人物/主体/物品（尽量用特定称呼），以及来源(wechat/miyoushe/other)",
                    "口吻偏新闻：突出事件/公告/活动/版本等时效信息（图片可见时）",
                    "不要编造看不见的信息；不确定就用“疑似/可能/看不清”",
                    "若图片是纯装饰/纯标题封面/长截图文字太多/表格堆叠/说明书式长图/界面表单，请 skip=true",
                    "输出严格 JSON：[{\"url\":\"...\",\"label\":\"...\",\"skip\":false}, ...] 与输入顺序对应",
                ],
                "items": meta,
            },
            ensure_ascii=False,
        )

        raw = ""
        last_err: Optional[Exception] = None
        for attempt in range(max(0, int(cfg.llm_max_retries)) + 1):
            try:
                resp = await astrbot_context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=self._system_prompt,
                    image_urls=imgs,
                )
                raw = getattr(resp, "completion_text", "") or ""
                if (raw or "").strip():
                    break
            except Exception as e:
                last_err = e
                if attempt >= int(cfg.llm_max_retries) or not _should_retry_error(e):
                    break
                delay = min(
                    float(cfg.llm_retry_max_s),
                    float(cfg.llm_retry_base_s) * (2**attempt),
                )
                delay = delay * (0.8 + 0.4 * random.random())
                await asyncio.sleep(max(0.1, delay))

        if not (raw or "").strip():
            if last_err is not None:
                astrbot_logger.warning("[dailynews] image_label llm_generate failed: %s", last_err, exc_info=True)
            return {}

        data = _json_from_text(raw)
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for it in data:
                if not isinstance(it, dict):
                    continue
                url = str(it.get("url") or "").strip()
                label = _shorten_zh(str(it.get("label") or ""), 30)
                skip = bool(it.get("skip") or False)
                if url and label:
                    out[url] = {"label": label, "skip": skip}
        elif isinstance(data, dict) and isinstance(data.get("labels"), list):
            for it in data.get("labels") or []:
                if not isinstance(it, dict):
                    continue
                url = str(it.get("url") or "").strip()
                label = _shorten_zh(str(it.get("label") or ""), 30)
                skip = bool(it.get("skip") or False)
                if url and label:
                    out[url] = {"label": label, "skip": skip}
        return out

    async def build_catalog(
        self,
        *,
        images_by_source: Mapping[str, Sequence[str]],
        astrbot_context: Any,
        cfg: ImageLabelConfig,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, ImageLabelEntry]]:
        if not cfg.enabled or not cfg.provider_id:
            return [], {}

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

        # Determine which images need labeling.
        need: List[Dict[str, Any]] = []
        for it in items:
            url = str(it.get("url") or "").strip()
            if not url:
                continue
            if not cfg.force_refresh and url in cache and ((cache[url].label or "").strip() or bool(cache[url].skip)):
                continue
            need.append(it)

        # Download missing ones first (parallel).
        download_sem = asyncio.Semaphore(int(cfg.concurrency))

        async def prepare_one(it: Dict[str, Any]) -> Dict[str, Any]:
            url = str(it.get("url") or "").strip()
            if not url:
                return it
            async with download_sem:
                local_path, w, h = await self._ensure_local(url)
            it2 = dict(it)
            if local_path:
                it2["local_path"] = local_path
                it2["width"] = int(w)
                it2["height"] = int(h)
            return it2

        prepared: List[Dict[str, Any]] = []
        if need:
            prepared = list(await asyncio.gather(*(prepare_one(dict(it)) for it in need)))

        # Label in batches (each call <=2 images) with limited concurrency.
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

        labeled: Dict[str, Dict[str, Any]] = {}
        if batches:
            astrbot_logger.info(
                "[dailynews] image_label start: %s images -> %s batches (concurrency=%s)",
                len(prepared),
                len(batches),
                int(cfg.concurrency),
            )
            call_sem = asyncio.Semaphore(int(cfg.concurrency))

            async def call_batch(batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
                async with call_sem:
                    return await self._label_batch(
                        astrbot_context=astrbot_context,
                        provider_id=cfg.provider_id,
                        items=batch,
                        cfg=cfg,
                    )

            results = await asyncio.gather(*(call_batch(b) for b in batches))
            for m in results:
                labeled.update(m or {})

        # Update cache for all items (including those already cached).
        prepared_by_url: Dict[str, Dict[str, Any]] = {str(it.get("url") or ""): it for it in prepared}
        changed = False

        for it in items:
            url = str(it.get("url") or "").strip()
            if not url:
                continue
            src = str(it.get("source") or "").strip()

            entry = cache.get(url)
            if entry and (not cfg.force_refresh) and ((entry.label or "").strip() or bool(entry.skip)):
                continue

            pit = prepared_by_url.get(url) or {}
            w = int(pit.get("width") or 0)
            h = int(pit.get("height") or 0)
            local_path = str(pit.get("local_path") or "")

            decision = labeled.get(url) or {}
            label = str(decision.get("label") or "").strip()
            skip = bool(decision.get("skip") or False)

            if not label:
                label = _shorten_zh(f"{_guess_source_from_url(url) or src or 'other'} 图片", 30)

            # Conservative heuristic: drop ultra-tall “text wall” screenshots.
            if not skip and w > 0 and h > 0 and h / max(1.0, float(w)) >= 3.2 and h >= 1400:
                skip = True
                label = _shorten_zh(f"跳过:长截图文字多({_guess_source_from_url(url) or 'other'})", 30)

            upsert_label(
                cache,
                url=url,
                label=_shorten_zh(label, 30),
                source=src,
                local_path=local_path,
                width=w,
                height=h,
                skip=skip,
            )
            changed = True

        if changed:
            try:
                save_labels(cache)
            except Exception as e:
                astrbot_logger.warning("[dailynews] save_labels failed: %s", e, exc_info=True)

        catalog: List[Dict[str, Any]] = []
        for it in items:
            url = str(it.get("url") or "").strip()
            if not url:
                continue
            src = str(it.get("source") or "").strip()
            entry = cache.get(url) or ImageLabelEntry(url=url, label="", source=src)
            if bool(entry.skip):
                continue
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

