import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from astrbot.api.star import StarTools
except Exception:  # pragma: no cover
    StarTools = None  # type: ignore

from .sqlite_store import seed_get_all, seed_set_entry


_SEED_LOCK = asyncio.Lock()
_SEED_CACHE: Optional[Dict[str, Any]] = None
_SEED_MIGRATED: bool = False


def _seed_state_path() -> Path:
    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir()) / "wechat_seed_state.json"
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "data" / "wechat_seed_state.json"


def _load_seed_state_sync(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_seed_state_sync(path: Path, state: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _get_seed_state() -> Dict[str, Any]:
    global _SEED_CACHE
    global _SEED_MIGRATED
    async with _SEED_LOCK:
        if _SEED_CACHE is None:
            # Prefer sqlite; fallback to legacy JSON and migrate into sqlite once.
            data = await asyncio.get_running_loop().run_in_executor(None, seed_get_all)
            if isinstance(data, dict) and data:
                _SEED_CACHE = data
                _SEED_MIGRATED = True
            else:
                legacy = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: _load_seed_state_sync(_seed_state_path())
                )
                _SEED_CACHE = legacy if isinstance(legacy, dict) else {}
                if not _SEED_MIGRATED and isinstance(_SEED_CACHE, dict) and _SEED_CACHE:
                    # best-effort migration
                    for k, v in _SEED_CACHE.items():
                        if isinstance(k, str) and isinstance(v, dict):
                            try:
                                await asyncio.get_running_loop().run_in_executor(
                                    None, lambda kk=k, vv=v: seed_set_entry(kk, vv)
                                )
                            except Exception:
                                pass
                    _SEED_MIGRATED = True
        return _SEED_CACHE


async def _update_seed_entry(key: str, entry: Dict[str, Any]):
    global _SEED_CACHE
    global _SEED_MIGRATED
    async with _SEED_LOCK:
        if _SEED_CACHE is None:
            _SEED_CACHE = await asyncio.get_running_loop().run_in_executor(None, seed_get_all)
            if not isinstance(_SEED_CACHE, dict):
                _SEED_CACHE = {}
        entry = dict(entry)
        entry.setdefault("updated_at", datetime.now().isoformat())
        _SEED_CACHE[key] = entry
        # sqlite is the source of truth; keep legacy JSON as a readable mirror (best-effort).
        await asyncio.get_running_loop().run_in_executor(None, lambda: seed_set_entry(key, entry))
        if not _SEED_MIGRATED:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _save_seed_state_sync(_seed_state_path(), _SEED_CACHE or {})
            )
