from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional, Tuple

from .image_utils import get_plugin_data_dir

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


def create_doc(markdown: str, *, doc_id: Optional[str] = None) -> Tuple[str, Path]:
    did = (doc_id or "").strip() or uuid.uuid4().hex
    p = doc_path(did)
    write_doc(did, markdown)
    return did, p


def read_doc(doc_id: str) -> str:
    p = doc_path(doc_id)
    if not p.exists():
        raise FileNotFoundError("doc not found")
    return p.read_text(encoding="utf-8", errors="ignore")


def write_doc(doc_id: str, markdown: str) -> Path:
    p = doc_path(doc_id)
    tmp = p.with_suffix(".md.tmp")
    tmp.write_text(markdown or "", encoding="utf-8")
    tmp.replace(p)
    return p

