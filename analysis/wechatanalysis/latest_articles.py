import sys
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


WECHAT_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Mobile/15E148 MicroMessenger/8.0.40(0x1800282b) "
    "NetType/WIFI Language/zh_CN"
)


def get_album_articles(
    article_url: str,
    limit: int = 5,
    album_keyword: str | None = None,
) -> list[dict[str, str]]:
    """
    无头模式：从公众号文章页打开「合集/专辑目录」弹窗并抓取最新文章。

    参数:
    - article_url: 任意一篇公众号文章链接（建议来自你想抓取的那个合集/专辑）
    - limit: 最多返回多少篇
    - album_keyword: 用于匹配目录入口名称；为空则默认点击第一个目录入口
    """

    with sync_playwright() as p:
        executable_path = None
        try:
            import sys
            from pathlib import Path

            # analysis/wechatanalysis/latest_articles.py -> parents[2] is plugin root
            root = str(Path(__file__).resolve().parents[2])
            if root not in sys.path:
                sys.path.append(root)
            from workflow.pipeline.playwright_bootstrap import (
                get_chromium_executable_path,
            )

            exe = get_chromium_executable_path()
            executable_path = str(exe) if exe else None
        except Exception as e:
            astrbot_logger.debug(
                f"[dailynews] get_chromium_executable_path import failed: {e}"
            )
            executable_path = None

        if executable_path:
            browser = p.chromium.launch(headless=True, executable_path=executable_path)
        else:
            astrbot_logger.warning(
                "[dailynews] playwright chromium not ready; falling back to default Playwright browser path (may fail)."
            )
            browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=WECHAT_MOBILE_UA)
        page = context.new_page()

        page.goto(article_url, timeout=60000, wait_until="domcontentloaded")

        page.wait_for_selector("span.js_album_directory__name", timeout=15000)
        entries = page.query_selector_all("span.js_album_directory__name")
        if not entries:
            browser.close()
            raise RuntimeError("找不到任何「合集/专辑目录」入口")

        target = None
        if album_keyword:
            for el in entries:
                text = (el.inner_text() or "").strip()
                if album_keyword in text:
                    target = el
                    break
        if not target:
            target = entries[0]

        target.click()
        page.wait_for_timeout(1200)

        page.wait_for_selector(".album_read_directory_title", timeout=10000)
        items = page.evaluate(
            """
        () => {
            const out = [];
            const titles = document.querySelectorAll('.album_read_directory_title');
            titles.forEach(el => {
                const title = (el.textContent || '').trim();
                if (!title) return;

                let url = null;
                let node = el;

                // 向上查找 data-link / data-url
                while (node && !url) {
                    if (node.dataset && node.dataset.link) { url = node.dataset.link; break; }
                    if (node.dataset && node.dataset.url) { url = node.dataset.url; break; }
                    const dl = node.getAttribute && node.getAttribute('data-link');
                    if (dl) { url = dl; break; }
                    const du = node.getAttribute && node.getAttribute('data-url');
                    if (du) { url = du; break; }
                    node = node.parentElement;
                }

                // fallback：找附近的 a 标签
                if (!url) {
                    const a = el.closest('a[href*="mp.weixin.qq.com/s"]')
                           || (el.parentElement && el.parentElement.querySelector('a[href*="mp.weixin.qq.com/s"]'));
                    if (a) url = a.href;
                }

                if (url) out.push({title, url});
            });
            return out;
        }
        """
        )

        browser.close()

    if not items:
        raise RuntimeError("没有从目录弹窗里解析到文章链接")

    seen = set()
    articles: list[dict[str, str]] = []
    for item in items:
        url = item.get("url") or ""
        title = item.get("title") or ""
        if not url or not title:
            continue

        if url.startswith("/"):
            url = urljoin("https://mp.weixin.qq.com", url)

        if url in seen:
            continue
        seen.add(url)

        articles.append({"title": title, "url": url})
        if len(articles) >= limit:
            break

    return articles


def get_album_articles_chasing_latest(
    article_url: str,
    limit: int = 5,
    album_keyword: str | None = None,
    max_hops: int = 6,
) -> list[dict[str, str]]:
    """
    解决「目录弹窗默认定位在当前文章附近」导致抓到的并非全局最新的问题：
    每次取“当前窗口里的第一篇(最上面)”作为新种子，再抓一次，直到第一篇不再变化。

    不会修改用户配置；只在本次抓取链路里使用追逐后的种子 URL。
    """

    curr = article_url
    seen = set()
    last_articles: list[dict[str, str]] = []

    for _ in range(max(1, int(max_hops))):
        if curr in seen:
            break
        seen.add(curr)

        try:
            articles = get_album_articles(
                curr, limit=limit, album_keyword=album_keyword
            )
        except Exception:
            # 失败时直接返回上一轮结果（若没有则为空）
            return last_articles

        last_articles = articles
        if not articles:
            return []

        top = (articles[0].get("url") or "").strip()
        if not top:
            return articles

        if top == curr:
            return articles

        curr = top

    return last_articles


def get_album_articles_chasing_latest_with_seed(
    article_url: str,
    limit: int = 5,
    album_keyword: str | None = None,
    max_hops: int = 6,
) -> tuple[str, list[dict[str, str]]]:
    """
    与 get_album_articles_chasing_latest() 一致，但会额外返回稳定后的种子 URL。
    返回: (seed_url, articles)
    """

    curr = article_url
    seen = set()
    last_articles: list[dict[str, str]] = []

    for _ in range(max(1, int(max_hops))):
        if curr in seen:
            break
        seen.add(curr)

        try:
            articles = get_album_articles(
                curr, limit=limit, album_keyword=album_keyword
            )
        except Exception:
            return curr, last_articles

        last_articles = articles
        if not articles:
            return curr, []

        top = (articles[0].get("url") or "").strip()
        if not top:
            return curr, articles

        if top == curr:
            return curr, articles

        curr = top

    return curr, last_articles


if __name__ == "__main__":
    if len(sys.argv) < 2:
        astrbot_logger.info(
            '用法: python latest_articles.py "文章链接" [数量] [目录关键词]'
        )
        sys.exit(1)

    url = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    album_keyword = sys.argv[3] if len(sys.argv) > 3 else None

    articles = get_album_articles(url, limit, album_keyword=album_keyword)

    astrbot_logger.info("获取到最新 %s 篇文章：", len(articles))
    for i, a in enumerate(articles, 1):
        astrbot_logger.info("%s. %s\n   %s", i, a["title"], a["url"])
