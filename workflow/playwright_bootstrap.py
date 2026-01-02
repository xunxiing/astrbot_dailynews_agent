from __future__ import annotations

import asyncio
import os
import zipfile
from pathlib import Path
from typing import Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

from .image_utils import get_plugin_data_dir


PLAYWRIGHT_CHROMIUM_HEADLESS_URL = (
    "https://cdn.playwright.dev/dbazure/download/playwright/builds/chromium/1200/"
    "chromium-headless-shell-win64.zip"
)

_BOOTSTRAP_LOCK = asyncio.Lock()


def _browsers_dir() -> Path:
    # Large binaries belong to plugin_data/.../tmp/
    return get_plugin_data_dir("playwright_browsers")


def _chromium_root_dir() -> Path:
    return _browsers_dir() / "chromium-headless-shell-win64-1200"


def get_chromium_executable_path() -> Optional[Path]:
    root = _chromium_root_dir()
    if not root.exists():
        return None

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
) -> Optional[Path]:
    """
    Ensure we have a Chromium executable available for Playwright fallback rendering.

    We intentionally do NOT call `playwright install` (slow + requires more deps).
    Instead we download a prebuilt `chromium-headless-shell` zip and point Playwright to it.
    """
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

        astrbot_logger.info("[dailynews] downloading playwright chromium (headless-shell) in background...")
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
            astrbot_logger.error("[dailynews] playwright chromium download/extract failed", exc_info=True)
            return None

        exe = get_chromium_executable_path()
        if exe is None:
            astrbot_logger.error("[dailynews] playwright chromium installed but executable not found under %s", root)
            return None

        # Hint Playwright to prefer our local binaries (useful for some environments).
        try:
            os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_browsers_dir().resolve()))
        except Exception:
            pass

        astrbot_logger.info("[dailynews] playwright chromium ready: %s", exe)
        return exe
