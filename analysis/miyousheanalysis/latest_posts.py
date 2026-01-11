import time
from typing import Dict, List, Optional

from playwright.sync_api import TimeoutError as PWTimeoutError
from playwright.sync_api import sync_playwright

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


def _try_close_popups(page) -> None:
    candidates = [
        "text=关闭",
        "text=我知道了",
        "text=知道了",
        "text=取消",
        ".mhy-dialog__close",
        ".close",
        "[aria-label='close']",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click(timeout=800)
                time.sleep(0.2)
        except Exception:
            pass


def _find_post_subject_from_nuxt(obj, depth: int = 0, max_depth: int = 10) -> Optional[str]:
    if depth > max_depth:
        return None

    if isinstance(obj, dict):
        if isinstance(obj.get("subject"), str) and obj.get("subject").strip() and "content" in obj:
            return obj.get("subject").strip()
        for v in obj.values():
            if isinstance(v, (dict, list)):
                hit = _find_post_subject_from_nuxt(v, depth + 1, max_depth)
                if hit:
                    return hit

    if isinstance(obj, list):
        for it in obj:
            hit = _find_post_subject_from_nuxt(it, depth + 1, max_depth)
            if hit:
                return hit

    return None


def get_user_latest_posts(
    user_post_list_url: str,
    limit: int = 10,
    *,
    headless: bool = True,
    sleep_between: float = 0.6,
    max_scroll_rounds: int = 20,
) -> List[Dict[str, str]]:
    """
    访问米游社用户帖子列表页（accountCenter/postList），逐个点开卡片获取真实的文章 URL。
    返回：[{title,url}, ...]（按列表出现顺序）
    """
    out: List[Dict[str, str]] = []

    with sync_playwright() as p:
        executable_path = None
        try:
            from ...workflow.playwright_bootstrap import get_chromium_executable_path

            exe = get_chromium_executable_path()
            executable_path = str(exe) if exe else None
        except Exception:
            executable_path = None

        try:
            if executable_path:
                browser = p.chromium.launch(headless=headless, executable_path=executable_path)
            else:
                astrbot_logger.warning(
                    "[dailynews] playwright chromium not ready; falling back to default Playwright browser path (may fail)."
                )
                browser = p.chromium.launch(headless=headless)
        except Exception as e:
            astrbot_logger.error("[dailynews] failed to launch playwright browser: %s", e)
            return []

        context = browser.new_context()
        page = context.new_page()

        astrbot_logger.info("[dailynews] opening miyoushe post list: %s", user_post_list_url)
        try:
            page.goto(user_post_list_url, wait_until="domcontentloaded", timeout=90000)
        except PWTimeoutError:
            astrbot_logger.warning("[dailynews] miyoushe list goto timeout, attempting fallback to networkidle")
            try:
                page.goto(user_post_list_url, wait_until="networkidle", timeout=30000)
            except Exception:
                browser.close()
                return []
        except Exception as e:
            astrbot_logger.error("[dailynews] miyoushe list goto failed: %s", e)
            browser.close()
            return []

        try:
            page.wait_for_selector(".mhy-article-card", timeout=20000)
        except Exception:
            browser.close()
            return []

        card_idx = 0
        scroll_rounds = 0

        while len(out) < max(1, int(limit)) and scroll_rounds <= max_scroll_rounds:
            _try_close_popups(page)

            cards = page.locator(".mhy-article-card")
            count = cards.count()

            if card_idx >= count:
                page.mouse.wheel(0, 1600)
                time.sleep(1.0)
                new_count = page.locator(".mhy-article-card").count()
                if new_count <= count:
                    break
                scroll_rounds += 1
                continue

            card = cards.nth(card_idx)

            try:
                preview = card.locator(".mhy-article-card__h3").inner_text(timeout=2000).strip()
            except Exception:
                preview = ""

            try:
                card.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            article_page = None
            opened_popup = False

            # Prefer opening a new tab by clicking the title (more stable on Nuxt pages).
            try:
                with context.expect_page(timeout=10000) as new_page_info:
                    card.locator(".mhy-article-card__h3").click(timeout=8000)
                article_page = new_page_info.value
                opened_popup = True
            except Exception:
                try:
                    with page.expect_popup(timeout=8000) as pop:
                        card.click(timeout=8000)
                    article_page = pop.value
                    opened_popup = True
                except PWTimeoutError:
                    # Last resort: same-tab navigation
                    try:
                        card.locator(".mhy-article-card__h3").click(timeout=8000)
                        page.wait_for_url("**/article/**", timeout=12000)
                        article_page = page
                        opened_popup = False
                    except Exception:
                        card_idx += 1
                        continue

            try:
                # 避免 about:blank
                try:
                    article_page.wait_for_url("**/article/**", timeout=15000)
                except Exception:
                    pass
                article_page.wait_for_load_state("domcontentloaded", timeout=15000)
                time.sleep(0.6)

                url = (article_page.url or "").strip()
                if url and "/article/" in url:
                    # Try to refine title via window.__NUXT__ (optional).
                    try:
                        article_page.wait_for_function(
                            "() => window.__NUXT__ && Object.keys(window.__NUXT__).length > 0",
                            timeout=5000,
                        )
                        nuxt = article_page.evaluate("() => window.__NUXT__")
                        nuxt_title = _find_post_subject_from_nuxt(nuxt)
                        if nuxt_title:
                            preview = nuxt_title.strip()
                    except Exception:
                        pass
                    out.append({"title": preview, "url": url})
            except Exception:
                pass
            finally:
                if opened_popup and article_page:
                    try:
                        article_page.close()
                    except Exception:
                        pass
                else:
                    # 回到列表页
                    try:
                        page.goto(user_post_list_url, wait_until="domcontentloaded", timeout=60000)
                        page.wait_for_selector(".mhy-article-card", timeout=20000)
                    except Exception:
                        break

            time.sleep(float(sleep_between))
            card_idx += 1

        browser.close()

    # 去重（按出现顺序）
    seen = set()
    dedup: List[Dict[str, str]] = []
    for it in out:
        u = (it.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        dedup.append({"title": (it.get("title") or "").strip(), "url": u})
    return dedup[: max(1, int(limit))]
