from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ...core.image_utils import get_plugin_data_dir
from ...storage import sqlite_store


def _utc_now_iso() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass(frozen=True)
class TagDef:
    tag: str
    description: str = ""


class TagStore:
    """
    Dynamic tag library (JSON) used by the message-group mode.

    Stored in the plugin sqlite DB (kv table) under key `dailynews.tag_rules`.
    Seeded from `default_tags.json` under this folder (and migrates legacy `tag_rules/tag_rules.json` once).
    """

    _KV_KEY = "dailynews.tag_rules"

    def __init__(self) -> None:
        self._data_dir = get_plugin_data_dir("tag_rules")
        self._legacy_path = self._data_dir / "tag_rules.json"
        self._default_path = Path(__file__).with_name("default_tags.json")

    @property
    def identifier(self) -> str:
        return f"sqlite://kv/{self._KV_KEY}"

    def ensure_initialized(self) -> dict[str, Any]:
        # 1) Load from sqlite kv (single source of truth)
        data = sqlite_store.kv_get(self._KV_KEY, default=None)
        if isinstance(data, dict) and isinstance(data.get("tags"), list):
            return data

        # 2) One-time migration from legacy JSON file if present
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            legacy = _load_json_file(self._legacy_path)
            if isinstance(legacy, dict) and isinstance(legacy.get("tags"), list):
                legacy.setdefault("updated_at", _utc_now_iso())
                sqlite_store.kv_set(self._KV_KEY, legacy)
                try:
                    self._legacy_path.unlink()
                except Exception:
                    pass
                return legacy
        except Exception:
            pass

        # 3) Seed from defaults
        defaults = _load_json_file(self._default_path) or {}
        tags = defaults.get("tags") if isinstance(defaults.get("tags"), list) else []
        if not isinstance(tags, list):
            tags = []

        new_data: dict[str, Any] = {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "tags": tags,
            "learned_tags": [],
        }
        try:
            sqlite_store.kv_set(self._KV_KEY, new_data)
        except Exception:
            astrbot_logger.warning(
                "[dailynews] failed to init tag rules in sqlite kv: %s",
                self._KV_KEY,
                exc_info=True,
            )
        return new_data

    def load_tags(self) -> list[TagDef]:
        data = self.ensure_initialized()
        out: list[TagDef] = []
        for it in data.get("tags") or []:
            if not isinstance(it, dict):
                continue
            tag = str(it.get("tag") or "").strip()
            if not tag:
                continue
            desc = str(it.get("description") or "").strip()
            out.append(TagDef(tag=tag, description=desc))
        return out

    def promote_learned_tags(
        self, *, suggestions: dict[str, int], min_count: int
    ) -> bool:
        if not suggestions:
            return False

        data = self.ensure_initialized()
        existing = {
            str(it.get("tag") or "").strip()
            for it in (data.get("tags") or [])
            if isinstance(it, dict)
        }
        learned: list[dict[str, Any]] = (
            list(data.get("learned_tags") or [])
            if isinstance(data.get("learned_tags"), list)
            else []
        )

        changed = False
        for raw_tag, cnt in sorted(
            suggestions.items(), key=lambda kv: (-int(kv[1]), kv[0])
        ):
            tag = str(raw_tag or "").strip()
            if not tag or tag in existing:
                continue
            if int(cnt) < int(min_count):
                continue

            # Add to tags
            data.setdefault("tags", [])
            if isinstance(data["tags"], list):
                data["tags"].append(
                    {
                        "tag": tag,
                        "description": "",
                        "created_at": _utc_now_iso(),
                        "source": "auto_promote",
                        "count": int(cnt),
                    }
                )
                changed = True

            learned.append(
                {"tag": tag, "count": int(cnt), "promoted_at": _utc_now_iso()}
            )

        if changed:
            data["updated_at"] = _utc_now_iso()
            data["learned_tags"] = learned[-200:]
            try:
                sqlite_store.kv_set(self._KV_KEY, data)
            except Exception:
                astrbot_logger.warning(
                    "[dailynews] failed to save tag rules to sqlite kv: %s",
                    self._KV_KEY,
                    exc_info=True,
                )
        return changed
