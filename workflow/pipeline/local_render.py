import asyncio
from pathlib import Path
from typing import Any


async def wait_for_file_ready(
    path: Path,
    *,
    is_valid: Any,
    timeout_s: float = 6.0,
    interval_ms: int = 200,
) -> bool:
    """
    Poll until render output is actually ready. Some render services first write tiny placeholder files.
    """
    try:
        loop = asyncio.get_running_loop()
    except Exception:  # pragma: no cover
        loop = asyncio.get_event_loop()

    deadline = loop.time() + float(timeout_s)
    while loop.time() < deadline:
        try:
            if is_valid(path):
                return True
        except Exception:
            pass
        await asyncio.sleep(max(0.05, int(interval_ms)) / 1000.0)

    try:
        return bool(is_valid(path))
    except Exception:
        return False
