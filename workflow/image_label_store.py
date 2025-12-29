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

from .image_utils import get_plugin_data_dir


@dataclass
class ImageLabelEntry:
    url: str
    label: str
    source: str = ""
    updated_at: str = ""
    local_path: str = ""
    width: int = 0
    height: int = 0


def _store_path() -> Path:
    return get_plugin_data_dir("image_labels") / "labels.json"


def load_labels() -> Dict[str, ImageLabelEntry]:
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
            out[url] = ImageLabelEntry(
                url=url,
                label=str(v.get("label") or ""),
                source=str(v.get("source") or ""),
                updated_at=str(v.get("updated_at") or ""),
                local_path=str(v.get("local_path") or ""),
                width=int(v.get("width") or 0),
                height=int(v.get("height") or 0),
            )
        return out
    except Exception as e:
        astrbot_logger.warning("[dailynews] load_labels failed: %s", e, exc_info=True)
        return {}


def save_labels(labels: Dict[str, ImageLabelEntry]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        url: {
            "label": entry.label,
            "source": entry.source,
            "updated_at": entry.updated_at,
            "local_path": entry.local_path,
            "width": int(entry.width),
            "height": int(entry.height),
        }
        for url, entry in (labels or {}).items()
        if isinstance(url, str) and isinstance(entry, ImageLabelEntry)
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_label(
    labels: Dict[str, ImageLabelEntry],
    *,
    url: str,
    label: str,
    source: str = "",
    local_path: str = "",
    width: int = 0,
    height: int = 0,
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
    )
    labels[url] = entry
    return entry

