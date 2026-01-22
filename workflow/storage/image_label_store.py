from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..core.image_utils import get_plugin_data_dir
from .sqlite_store import image_label_upsert, image_labels_get_all


@dataclass
class ImageLabelEntry:
    url: str
    label: str
    source: str = ""
    updated_at: str = ""
    local_path: str = ""
    width: int = 0
    height: int = 0
    skip: bool = False


def _store_path() -> Path:
    return get_plugin_data_dir("image_labels") / "labels.json"


def load_labels() -> Dict[str, ImageLabelEntry]:
    # Prefer sqlite; fallback to legacy JSON and migrate into sqlite.
    rows = image_labels_get_all()
    if rows:
        return {
            url: ImageLabelEntry(
                url=url,
                label=r.label,
                source=r.source,
                updated_at=r.updated_at,
                local_path=r.local_path,
                width=int(r.width),
                height=int(r.height),
                skip=bool(r.skip),
            )
            for url, r in rows.items()
        }

    path = _store_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, ImageLabelEntry] = {}
        for url, v in raw.items():
            if not isinstance(url, str) or not isinstance(v, dict):
                continue
            ent = ImageLabelEntry(
                url=url,
                label=str(v.get("label") or ""),
                source=str(v.get("source") or ""),
                updated_at=str(v.get("updated_at") or ""),
                local_path=str(v.get("local_path") or ""),
                width=int(v.get("width") or 0),
                height=int(v.get("height") or 0),
                skip=bool(v.get("skip") or False),
            )
            out[url] = ent
            image_label_upsert(
                url=ent.url,
                label=ent.label,
                source=ent.source,
                updated_at=ent.updated_at or None,
                local_path=ent.local_path,
                width=ent.width,
                height=ent.height,
                skip=bool(ent.skip),
            )
        return out
    except Exception as e:
        astrbot_logger.warning("[dailynews] load_labels failed: %s", e, exc_info=True)
        return {}


def save_labels(labels: Dict[str, ImageLabelEntry]) -> None:
    # sqlite is the source of truth; keep legacy JSON only for backward compatibility (do not write by default).
    for url, entry in (labels or {}).items():
        if not isinstance(url, str) or not isinstance(entry, ImageLabelEntry):
            continue
        image_label_upsert(
            url=entry.url,
            label=entry.label,
            source=entry.source,
            updated_at=entry.updated_at or None,
            local_path=entry.local_path,
            width=entry.width,
            height=entry.height,
            skip=bool(entry.skip),
        )


def upsert_label(
    labels: Dict[str, ImageLabelEntry],
    *,
    url: str,
    label: str,
    source: str = "",
    local_path: str = "",
    width: int = 0,
    height: int = 0,
    skip: bool = False,
    updated_at: Optional[str] = None,
) -> ImageLabelEntry:
    ts = updated_at or datetime.now().isoformat(timespec="seconds")
    entry = ImageLabelEntry(
        url=url,
        label=label,
        source=source,
        updated_at=ts,
        local_path=local_path,
        width=int(width or 0),
        height=int(height or 0),
        skip=bool(skip),
    )
    labels[url] = entry
    image_label_upsert(
        url=entry.url,
        label=entry.label,
        source=entry.source,
        updated_at=entry.updated_at,
        local_path=entry.local_path,
        width=entry.width,
        height=entry.height,
        skip=bool(entry.skip),
    )
    return entry
