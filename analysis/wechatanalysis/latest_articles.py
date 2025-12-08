import sys
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright


WECHAT_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Mobile/15E148 MicroMessenger/8.0.40(0x1800282b) "
    "NetType/WIFI Language/zh_CN"
)


def get_album_articles(article_url: str, limit: int = 5):
    """无头模式：从文章页打开目录弹窗并抓取最新文章"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=WECHAT_MOBILE_UA)
        page = context.new_page()

        print("打开文章页面…")
        page.goto(article_url, timeout=60000, wait_until="domcontentloaded")

        # 等专辑入口
        page.wait_for_selector("span.js_album_directory__name", timeout=15000)

        # 找到匹配 “AI 早报” 文本的入口
        entries = page.query_selector_all("span.js_album_directory__name")
        target = None
        for el in entries:
            text = (el.inner_text() or "").strip()
            if "AI 早报" in text:
                target = el
                break

        if not target:
            raise RuntimeError("找不到『AI 早报 · 目录』入口")

        print("点击目录入口…")
        target.click()
        page.wait_for_timeout(1200)

        # 等弹窗里的标题元素
        page.wait_for_selector(".album_read_directory_title", timeout=10000)

        # 在页面中执行 JS，采集标题与链接
        items = page.evaluate("""
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
                    if (node.dataset && node.dataset.link) {
                        url = node.dataset.link;
                        break;
                    }
                    if (node.dataset && node.dataset.url) {
                        url = node.dataset.url;
                        break;
                    }
                    const dl = node.getAttribute && node.getAttribute('data-link');
                    if (dl) { url = dl; break; }
                    const du = node.getAttribute && node.getAttribute('data-url');
                    if (du) { url = du; break; }
                    node = node.parentElement;
                }

                // fallback：找附近的 a 标签
                if (!url) {
                    const a = el.closest('a[href*="mp.weixin.qq.com/s"]')
                           || el.parentElement?.querySelector('a[href*="mp.weixin.qq.com/s"]');
                    if (a) url = a.href;
                }

                if (url) out.push({title, url});
            });
            return out;
        }
        """)

        browser.close()

    if not items:
        raise RuntimeError("没有从目录弹窗里解析到文章链接")

    # 处理 URL & 去重
    seen = set()
    articles = []

    for item in items:
        url = item["url"]
        if url.startswith("/"):
            url = urljoin("https://mp.weixin.qq.com", url)

        if url in seen:
            continue
        seen.add(url)

        articles.append({
            "title": item["title"],
            "url": url
        })

        if len(articles) >= limit:
            break

    return articles


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python latest_articles.py \"文章链接\" [数量]")
        sys.exit(1)

    url = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    articles = get_album_articles(url, limit)

    print(f"\n获取到最新 {len(articles)} 篇文章：\n")
    for i, a in enumerate(articles, 1):
        print(f"{i}. {a['title']}")
        print(f"   {a['url']}")
