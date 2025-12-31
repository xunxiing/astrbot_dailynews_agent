from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .image_utils import get_plugin_data_dir


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _db_path() -> Path:
    d = get_plugin_data_dir("db")
    d.mkdir(parents=True, exist_ok=True)
    return d / "dailynews.sqlite3"


def _connect() -> sqlite3.Connection:
    p = _db_path()
    conn = sqlite3.connect(str(p), timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    _init(conn)
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kv (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS md_docs (
          doc_id TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seed_state (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_labels (
          url TEXT PRIMARY KEY,
          label TEXT NOT NULL,
          source TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          local_path TEXT NOT NULL,
          width INTEGER NOT NULL,
          height INTEGER NOT NULL,
          skip INTEGER NOT NULL
        )
        """
    )


def kv_get(key: str, default: Any = None) -> Any:
    k = str(key or "").strip()
    if not k:
        return default
    try:
        with _connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (k,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(str(row["value"]))
        except Exception:
            return str(row["value"])
    except Exception as e:
        astrbot_logger.warning("[dailynews] kv_get failed: %s", e, exc_info=True)
        return default


def kv_set(key: str, value: Any) -> None:
    k = str(key or "").strip()
    if not k:
        return
    try:
        payload = json.dumps(value, ensure_ascii=False)
    except Exception:
        payload = json.dumps(str(value), ensure_ascii=False)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (k, payload, _now_iso()),
            )
    except Exception as e:
        astrbot_logger.warning("[dailynews] kv_set failed: %s", e, exc_info=True)


def kv_delete(key: str) -> None:
    k = str(key or "").strip()
    if not k:
        return
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM kv WHERE key=?", (k,))
    except Exception as e:
        astrbot_logger.warning("[dailynews] kv_delete failed: %s", e, exc_info=True)


def md_doc_get(doc_id: str) -> Optional[str]:
    did = str(doc_id or "").strip()
    if not did:
        return None
    try:
        with _connect() as conn:
            row = conn.execute("SELECT content FROM md_docs WHERE doc_id=?", (did,)).fetchone()
        if not row:
            return None
        return str(row["content"] or "")
    except Exception as e:
        astrbot_logger.warning("[dailynews] md_doc_get failed: %s", e, exc_info=True)
        return None


def md_doc_set(doc_id: str, content: str) -> None:
    did = str(doc_id or "").strip()
    if not did:
        return
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO md_docs(doc_id,content,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at",
                (did, str(content or ""), _now_iso()),
            )
    except Exception as e:
        astrbot_logger.warning("[dailynews] md_doc_set failed: %s", e, exc_info=True)


def seed_get_all() -> Dict[str, Any]:
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT key,value FROM seed_state").fetchall()
        out: Dict[str, Any] = {}
        for r in rows or []:
            k = str(r["key"])
            v = str(r["value"] or "")
            try:
                out[k] = json.loads(v)
            except Exception:
                out[k] = {}
        return out
    except Exception as e:
        astrbot_logger.warning("[dailynews] seed_get_all failed: %s", e, exc_info=True)
        return {}


def seed_set_entry(key: str, entry: Dict[str, Any]) -> None:
    k = str(key or "").strip()
    if not k:
        return
    try:
        payload = json.dumps(entry or {}, ensure_ascii=False)
    except Exception:
        payload = "{}"
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO seed_state(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (k, payload, _now_iso()),
            )
    except Exception as e:
        astrbot_logger.warning("[dailynews] seed_set_entry failed: %s", e, exc_info=True)


@dataclass
class ImageLabelRow:
    url: str
    label: str
    source: str
    updated_at: str
    local_path: str
    width: int
    height: int
    skip: bool


def image_labels_get_all() -> Dict[str, ImageLabelRow]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT url,label,source,updated_at,local_path,width,height,skip FROM image_labels"
            ).fetchall()
        out: Dict[str, ImageLabelRow] = {}
        for r in rows or []:
            url = str(r["url"] or "")
            if not url:
                continue
            out[url] = ImageLabelRow(
                url=url,
                label=str(r["label"] or ""),
                source=str(r["source"] or ""),
                updated_at=str(r["updated_at"] or ""),
                local_path=str(r["local_path"] or ""),
                width=int(r["width"] or 0),
                height=int(r["height"] or 0),
                skip=bool(int(r["skip"] or 0)),
            )
        return out
    except Exception as e:
        astrbot_logger.warning("[dailynews] image_labels_get_all failed: %s", e, exc_info=True)
        return {}


def image_label_upsert(
    *,
    url: str,
    label: str,
    source: str = "",
    updated_at: Optional[str] = None,
    local_path: str = "",
    width: int = 0,
    height: int = 0,
    skip: bool = False,
) -> None:
    u = str(url or "").strip()
    if not u:
        return
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO image_labels(url,label,source,updated_at,local_path,width,height,skip) "
                "VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(url) DO UPDATE SET "
                "label=excluded.label, source=excluded.source, updated_at=excluded.updated_at, "
                "local_path=excluded.local_path, width=excluded.width, height=excluded.height, skip=excluded.skip",
                (
                    u,
                    str(label or ""),
                    str(source or ""),
                    str(updated_at or _now_iso()),
                    str(local_path or ""),
                    int(width or 0),
                    int(height or 0),
                    1 if skip else 0,
                ),
            )
    except Exception as e:
        astrbot_logger.warning("[dailynews] image_label_upsert failed: %s", e, exc_info=True)

