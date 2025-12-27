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


_SEED_LOCK = asyncio.Lock()
_SEED_CACHE: Optional[Dict[str, Any]] = None


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
    async with _SEED_LOCK:
        if _SEED_CACHE is None:
            _SEED_CACHE = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _load_seed_state_sync(_seed_state_path())
            )
        return _SEED_CACHE


async def _update_seed_entry(key: str, entry: Dict[str, Any]):
    global _SEED_CACHE
    async with _SEED_LOCK:
        if _SEED_CACHE is None:
            _SEED_CACHE = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _load_seed_state_sync(_seed_state_path())
            )
        entry = dict(entry)
        entry.setdefault("updated_at", datetime.now().isoformat())
        _SEED_CACHE[key] = entry
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: _save_seed_state_sync(_seed_state_path(), _SEED_CACHE or {})
        )

