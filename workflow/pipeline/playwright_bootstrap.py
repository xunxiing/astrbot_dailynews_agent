from __future__ import annotations

import asyncio
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

from ..core.image_utils import get_plugin_data_dir

PLAYWRIGHT_CHROMIUM_HEADLESS_URL = (
    "https://cdn.playwright.dev/dbazure/download/playwright/builds/chromium/1200/"
    "chromium-headless-shell-win64.zip"
)

PLAYWRIGHT_INSTALL_WITH_DEPS_CMD = "playwright install --with-deps chromium"

_BOOTSTRAP_LOCK = asyncio.Lock()


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _browsers_dir() -> Path:
    # Look for playwright_browsers under the same directory as plugins/ or plugin_data/
    # Usually it's in the same parent as plugin_data
    return get_plugin_data_dir("playwright_browsers")


def _chromium_root_dir() -> Path:
    # Use generic names first, then specific versions
    return _browsers_dir() / "chromium-1169"


def get_chromium_executable_path() -> Path | None:
    # Our bootstrap download is Windows-only. Never try to use it on Linux/macOS,
    # even if the plugin data directory was copied over from a Windows machine.
    if not _is_windows():
        return None

    root = _chromium_root_dir()
    if not root.exists():
        # Auto-detect any folder starting with "chromium-"
        try:
            candidates = sorted(_browsers_dir().glob("chromium-*"), reverse=True)
            if candidates:
                root = candidates[0]
            else:
                # Fallback to older directory name if any
                alt_root = _browsers_dir() / "chromium-headless-shell-win64-1200"
                if alt_root.exists():
                    root = alt_root
                else:
                    astrbot_logger.debug(
                        f"[dailynews] No chromium directory found in {_browsers_dir()}"
                    )
                    return None
        except Exception as e:
            astrbot_logger.debug(
                f"[dailynews] get_chromium_executable_path glob failed: {e}"
            )
            return None

    astrbot_logger.debug(f"[dailynews] searching chromium in root: {root}")
    candidates = [
        root / "chrome-win" / "headless_shell.exe",
        root / "chrome-win" / "chrome.exe",
        root / "headless_shell.exe",
        root / "chrome.exe",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: find any *headless* exe (best effort)
    try:
        for p in root.rglob("*.exe"):
            name = p.name.lower()
            if "headless" in name and p.exists():
                return p
    except Exception:
        pass
    return None


def get_chromium_executable_path_or_hint() -> str:
    """
    For logging/error messages. Returns an executable path if found, else a hint string.
    """
    exe = get_chromium_executable_path()
    if exe is not None:
        return str(exe)
    return (
        "chromium executable not ready (bootstrap is background). "
        "Wait a bit or restart the plugin so it can finish downloading."
    )


def detect_windows_bootstrap_download_root() -> Path | None:
    """
    Returns the bootstrap root directory if we detect Windows-only artifacts in plugin data.
    This is useful for warning users who migrated their AstrBot data from Windows to Linux.
    """
    try:
        if not _browsers_dir().exists():
            return None
        for root in sorted(_browsers_dir().glob("chromium-*"), reverse=True):
            if not root.is_dir():
                continue
            # Marker or typical layout signals it was installed by our bootstrap flow.
            if (root / ".installed").exists() or (root / "download.zip").exists():
                return root
            if (root / "chrome-win").exists():
                return root
            # Any .exe under the folder is a strong signal of Windows binaries.
            try:
                if next(root.rglob("*.exe"), None) is not None:  # type: ignore[arg-type]
                    return root
            except Exception:
                pass
    except Exception:
        return None
    return None


def config_needs_playwright(cfg: Mapping[str, Any]) -> bool:
    """
    Best-effort check: whether this plugin config likely needs Playwright to work.
    """
    try:
        # Preferred (v4.10.4+): template_list-based sources.
        news_sources = cfg.get("news_sources", []) or []
        if isinstance(news_sources, list):
            for s in news_sources:
                if not isinstance(s, dict):
                    continue
                t = str(s.get("type") or "").strip().lower()
                if t in {"wechat", "miyoushe", "twitter"}:
                    return True

        # Legacy: separate lists.
        if (cfg.get("wechat_sources") or []) and len(
            cfg.get("wechat_sources") or []
        ) > 0:  # type: ignore[arg-type]
            return True
        if (cfg.get("miyoushe_sources") or []) and len(
            cfg.get("miyoushe_sources") or []
        ) > 0:  # type: ignore[arg-type]
            return True

        # Twitter/X.
        if bool(cfg.get("twitter_enabled", False)) and bool(
            cfg.get("twitter_targets", []) or []
        ):
            return True
    except Exception:
        return False
    return False


def build_playwright_chromium_missing_message(
    *, detected_exe: str = "", custom_browser_path: str = ""
) -> str:
    detected_exe = (detected_exe or "").strip()
    custom_browser_path = (custom_browser_path or "").strip()

    lines = [
        "Playwright Chromium 未安装/不可用，已停止本次执行。",
        f"请在 AstrBot 的运行环境内执行：{PLAYWRIGHT_INSTALL_WITH_DEPS_CMD}",
        f"（如命令不可用，可用：python -m {PLAYWRIGHT_INSTALL_WITH_DEPS_CMD}）",
        "安装完成后重启 AstrBot/插件。",
    ]
    if detected_exe:
        lines.append(f"当前 Playwright 期望的可执行文件路径：{detected_exe}")
    if custom_browser_path:
        lines.append(
            f"当前已配置 custom_browser_path：{custom_browser_path}（不建议修改；一般留空即可）"
        )
    return "\n".join(lines).strip()


async def check_playwright_chromium_ready(
    *, custom_browser_path: str = ""
) -> tuple[bool, str]:
    """
    Linux/macOS: rely on official Playwright browsers (e.g. ~/.cache/ms-playwright).
    Returns (ok, message_if_not_ok).
    """
    custom_browser_path = (custom_browser_path or "").strip()
    if custom_browser_path:
        try:
            if Path(custom_browser_path).expanduser().exists():
                return (True, "")
        except Exception:
            pass
        return (
            False,
            build_playwright_chromium_missing_message(
                custom_browser_path=custom_browser_path
            ),
        )

    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception:
        return (
            False,
            build_playwright_chromium_missing_message(),
        )

    detected = ""
    try:
        async with async_playwright() as p:
            detected = str(p.chromium.executable_path or "")
    except Exception:
        detected = ""

    try:
        if detected and Path(detected).expanduser().exists():
            return (True, "")
    except Exception:
        pass

    return (False, build_playwright_chromium_missing_message(detected_exe=detected))


async def _download_file(url: str, out_path: Path, *, timeout_s: int = 1200) -> None:
    if aiohttp is None:
        raise RuntimeError("aiohttp not available; cannot download playwright browser")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    headers = {"User-Agent": "AstrBotDailyNews/1.0"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 256):
                    if chunk:
                        f.write(chunk)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(out_dir))


async def ensure_playwright_chromium_installed(
    *,
    download_url: str = PLAYWRIGHT_CHROMIUM_HEADLESS_URL,
    force: bool = False,
) -> Path | None:
    """
    Ensure we have a Chromium executable available for Playwright fallback rendering.

    We intentionally do NOT call `playwright install` (slow + requires more deps).
    Instead we download a prebuilt `chromium-headless-shell` zip and point Playwright to it.
    """
    if not _is_windows():
        # Linux/macOS: rely on official `playwright install --with-deps chromium`.
        return None

    async with _BOOTSTRAP_LOCK:
        exe = get_chromium_executable_path()
        if exe is not None and not force:
            return exe

        root = _chromium_root_dir()
        if force and root.exists():
            try:
                for p in sorted(root.rglob("*"), reverse=True):
                    try:
                        if p.is_file():
                            p.unlink()
                        else:
                            p.rmdir()
                    except Exception:
                        pass
                root.rmdir()
            except Exception:
                pass

        root.mkdir(parents=True, exist_ok=True)
        zip_path = root / "download.zip"
        marker = root / ".installed"

        if marker.exists() and exe is None and not force:
            # Previous attempt finished, but we failed to locate exe; keep as-is.
            return None

        astrbot_logger.info(
            "[dailynews] downloading playwright chromium (headless-shell) in background..."
        )
        try:
            await _download_file(download_url, zip_path, timeout_s=3600)
            _extract_zip(zip_path, root)
            try:
                zip_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            marker.write_text("ok", encoding="utf-8")
        except Exception:
            astrbot_logger.error(
                "[dailynews] playwright chromium download/extract failed", exc_info=True
            )
            return None

        exe = get_chromium_executable_path()
        if exe is None:
            astrbot_logger.error(
                "[dailynews] playwright chromium installed but executable not found under %s",
                root,
            )
            return None

        astrbot_logger.info("[dailynews] playwright chromium ready: %s", exe)
        return exe
