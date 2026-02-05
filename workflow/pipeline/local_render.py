import asyncio
from pathlib import Path
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

try:
    from jinja2 import BaseLoader, Environment  # type: ignore
except Exception:  # pragma: no cover
    Environment = None  # type: ignore
    BaseLoader = None  # type: ignore

try:
    from playwright.async_api import async_playwright  # type: ignore
except Exception:  # pragma: no cover
    async_playwright = None  # type: ignore


def render_jinja_template(template_str: str, context: dict[str, Any]) -> str:
    """
    AstrBot 的 render_custom_template 使用 Jinja2；这里做本地 Playwright 兜底时也用同样渲染。
    """
    if Environment is None or BaseLoader is None:
        raise RuntimeError("jinja2 not available")
    env = Environment(loader=BaseLoader(), autoescape=False)
    tmpl = env.from_string(template_str)
    return str(tmpl.render(**(context or {})))


async def screenshot_html_playwright(
    html: str,
    *,
    out_path: Path,
    viewport: tuple[int, int] = (1080, 720),
    timeout_ms: int = 20000,
    full_page: bool = True,
    browser_executable_path: str | None = None,
) -> Path:
    if async_playwright is None:
        raise RuntimeError("playwright not available")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        executable_path: Path | None = None
        if browser_executable_path and str(browser_executable_path).strip():
            cand = Path(str(browser_executable_path)).expanduser()
            if cand.exists():
                executable_path = cand
            else:
                astrbot_logger.warning(
                    "[dailynews] custom_browser_path not found: %s (ignored; falling back to Playwright default)",
                    browser_executable_path,
                )
        else:
            try:
                from .playwright_bootstrap import get_chromium_executable_path

                executable_path = get_chromium_executable_path()
            except Exception:
                executable_path = None

        if executable_path is not None:
            browser = await p.chromium.launch(
                headless=True, executable_path=str(executable_path)
            )
        else:
            browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(
                viewport={"width": int(viewport[0]), "height": int(viewport[1])}
            )
            await page.set_content(
                html or "", wait_until="load", timeout=int(timeout_ms)
            )
            await page.wait_for_timeout(200)
            await page.screenshot(
                path=str(out_path),
                full_page=bool(full_page),
                type="jpeg",
                quality=90,
            )
        finally:
            await browser.close()
    return out_path


async def render_template_to_image_playwright(
    template_str: str,
    context: dict[str, Any],
    *,
    out_path: Path,
    viewport: tuple[int, int] = (1080, 720),
    timeout_ms: int = 20000,
    full_page: bool = True,
    browser_executable_path: str | None = None,
) -> Path:
    html = render_jinja_template(template_str, context)
    return await screenshot_html_playwright(
        html,
        out_path=out_path,
        viewport=viewport,
        timeout_ms=timeout_ms,
        full_page=full_page,
        browser_executable_path=browser_executable_path,
    )


async def wait_for_file_ready(
    path: Path,
    *,
    is_valid: Any,
    timeout_s: float = 6.0,
    interval_ms: int = 200,
) -> bool:
    """
    轮询等待渲染产物“落盘且可用”。某些渲染服务会先写一个很小的占位文件。
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
