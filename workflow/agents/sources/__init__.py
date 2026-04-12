from __future__ import annotations

from functools import lru_cache
import importlib
import inspect
import pkgutil
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


GENERIC_SOURCE_TEMPLATE_KEYS = frozenset(
    {"generic", "generic_source", "custom", "custom_source", "source"}
)


def _iter_source_module_names() -> list[str]:
    names: list[str] = []
    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        name = str(module_info.name or "").strip()
        if not name or module_info.ispkg:
            continue
        if not name.endswith("_agent") or name.startswith("_"):
            continue
        names.append(name)
    return sorted(names)


def _resolve_source_type(module: Any, module_name: str) -> str:
    source_type = str(getattr(module, "SOURCE_TYPE", "") or "").strip().lower()
    if source_type:
        return source_type
    return module_name.removesuffix("_agent").strip().lower()


def _resolve_agent_class(module: Any):
    explicit = getattr(module, "SOURCE_AGENT_CLASS", None)
    if explicit is not None:
        if isinstance(explicit, str):
            explicit = getattr(module, explicit, None)
        if inspect.isclass(explicit):
            return explicit
        raise RuntimeError(
            f"{module.__name__}: SOURCE_AGENT_CLASS must point to a class object"
        )

    candidates = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and callable(getattr(cls, "fetch_latest_articles", None))
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"{module.__name__}: expected exactly one source agent class with "
        f"`fetch_latest_articles`, found {len(candidates)}"
    )


@lru_cache(maxsize=1)
def discover_source_agent_registry() -> dict[str, type[Any]]:
    """
    Discover source agents by convention.

    Development contract:
    - one file == one source, file name should end with `_agent.py`
    - one source agent class per file, the class must implement `fetch_latest_articles`
    - optionally expose `SOURCE_TYPE = "your_type"` when the file name is not enough
    """
    registry: dict[str, type[Any]] = {}

    for module_name in _iter_source_module_names():
        fqmn = f"{__name__}.{module_name}"
        try:
            module = importlib.import_module(fqmn)
            source_type = _resolve_source_type(module, module_name)
            if not source_type:
                raise RuntimeError("source type is empty")
            agent_class = _resolve_agent_class(module)
        except Exception as exc:
            astrbot_logger.warning(
                "[dailynews] skip source module %s: %s",
                fqmn,
                exc,
                exc_info=True,
            )
            continue

        existing = registry.get(source_type)
        if existing is not None and existing is not agent_class:
            astrbot_logger.warning(
                "[dailynews] duplicate source type `%s`: keep %s, skip %s",
                source_type,
                existing.__module__,
                fqmn,
            )
            continue

        registry[source_type] = agent_class

    return registry


def available_source_types() -> list[str]:
    return sorted(discover_source_agent_registry())


__all__ = [
    "GENERIC_SOURCE_TEMPLATE_KEYS",
    "available_source_types",
    "discover_source_agent_registry",
]
