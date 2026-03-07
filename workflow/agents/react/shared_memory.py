from __future__ import annotations

from threading import RLock
from typing import Any


class SharedMemory:
    """
    Thread-safe singleton memory store for sub-agent deliveries.
    key: agent_id
    value: Any serializable payload
    """

    _instance: "SharedMemory | None" = None
    _instance_lock = RLock()

    def __new__(cls) -> "SharedMemory":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._lock = RLock()
                    cls._instance._store = {}
        return cls._instance

    @classmethod
    def instance(cls) -> "SharedMemory":
        return cls()

    def write(self, agent_id: str, content: Any) -> None:
        aid = str(agent_id or "").strip()
        if not aid:
            raise ValueError("agent_id is required")
        with self._lock:
            self._store[aid] = content

    def read(self, agent_ids: list[str]) -> dict[str, Any]:
        if not isinstance(agent_ids, list):
            return {}
        out: dict[str, Any] = {}
        with self._lock:
            for agent_id in agent_ids:
                aid = str(agent_id or "").strip()
                if not aid:
                    continue
                if aid in self._store:
                    out[aid] = self._store[aid]
        return out

    def read_all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._store)

    def reset(self, run_id: str | None = None) -> None:
        # run_id is reserved for future scoped memory isolation.
        _ = run_id
        with self._lock:
            self._store.clear()
