from __future__ import annotations

import re
import uuid
from pathlib import Path

from ..core.image_utils import get_plugin_data_dir
from .sqlite_store import md_doc_get, md_doc_set

_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _docs_dir() -> Path:
    d = get_plugin_data_dir("md_docs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def doc_path(doc_id: str) -> Path:
    did = str(doc_id or "").strip()
    if not _DOC_ID_RE.match(did):
        raise ValueError("invalid doc_id")
    return _docs_dir() / f"{did}.md"


def create_doc(markdown: str, *, doc_id: str | None = None) -> tuple[str, Path]:
    did = (doc_id or "").strip() or uuid.uuid4().hex
    p = doc_path(did)
    write_doc(did, markdown)
    return did, p


def read_doc(doc_id: str) -> str:
    # Prefer sqlite, fallback to file (and import into sqlite for migration).
    did = str(doc_id or "").strip()
    if not did:
        raise FileNotFoundError("doc not found")
    content = md_doc_get(did)
    if isinstance(content, str):
        return content
    p = doc_path(did)
    if p.exists():
        s = p.read_text(encoding="utf-8", errors="ignore")
        md_doc_set(did, s)
        return s
    raise FileNotFoundError("doc not found")


def write_doc(doc_id: str, markdown: str) -> Path:
    p = doc_path(doc_id)
    tmp = p.with_suffix(".md.tmp")
    tmp.write_text(markdown or "", encoding="utf-8")
    tmp.replace(p)
    # Keep sqlite as the source of truth; file is a readable mirror under plugin_data.
    md_doc_set(str(doc_id or "").strip(), markdown or "")
    return p
