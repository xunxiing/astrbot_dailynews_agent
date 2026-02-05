from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from playwright.async_api import async_playwright  # type: ignore
except Exception:  # pragma: no cover
    async_playwright = None  # type: ignore


@dataclass
class TweetItem:
    url: str
    text: str
    image_urls: list[str]


def _normalize_proxy(proxy: str) -> str | None:
    s = (proxy or "").strip()
    if not s:
        return None
    # Accept `http://host:port` / `socks5://host:port` / `socks5h://host:port`
    if "://" not in s:
        return None
    return s


def _is_proxy_like(s: str) -> bool:
    x = (s or "").strip().lower()
    return (
        x.startswith("socks5://")
        or x.startswith("socks5h://")
        or x.startswith("http://")
        or x.startswith("https://")
        and ("://127." in x or "://localhost" in x)
    )


def _is_twitter_url(s: str) -> bool:
    x = (s or "").strip().lower()
    return (
        x.startswith("https://x.com/")
        or x.startswith("http://x.com/")
        or x.startswith("https://twitter.com/")
        or x.startswith("http://twitter.com/")
    )


async def fetch_latest_tweets(
    target_url: str,
    *,
    limit: int = 3,
    proxy: str = "",
    timeout_ms: int = 60000,
    viewport_width: int = 1280,
    viewport_height: int = 800,
    executable_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch latest non-pinned tweets from a profile page and return lightweight dicts:
    {url, text, image_urls}.
    """
    if async_playwright is None:
        raise RuntimeError("playwright not available")

    url = (target_url or "").strip()
    if not url:
        return []

    if _is_proxy_like(url) and not _is_twitter_url(url):
        raise ValueError(
            "target_url looks like a proxy address. Did you swap `twitter_targets` and `twitter_proxy`?"
        )
    if not _is_twitter_url(url):
        raise ValueError(
            "target_url must be an X/Twitter profile url like https://x.com/openai"
        )

    limit = max(1, min(int(limit), 10))
    server = _normalize_proxy(proxy)

    async with async_playwright() as p:
        launch_kwargs: dict[str, Any] = {"headless": True}
        if executable_path:
            launch_kwargs["executable_path"] = str(executable_path)
        if server:
            launch_kwargs["proxy"] = {"server": server}

        browser = await p.chromium.launch(**launch_kwargs)
        try:
            context = await browser.new_context(
                viewport={"width": int(viewport_width), "height": int(viewport_height)}
            )
            page = await context.new_page()
            await page.goto(url, timeout=int(timeout_ms), wait_until="load")
            await page.wait_for_selector(
                'article[data-testid="tweet"]', timeout=int(timeout_ms)
            )

            tweets = page.locator('article[data-testid="tweet"]')
            total = await tweets.count()
            if total <= 0:
                return []

            out: list[TweetItem] = []
            for i in range(total):
                t = tweets.nth(i)
                # pinned tweets have socialContext
                try:
                    pinned = (
                        await t.locator('div[data-testid="socialContext"]').count()
                    ) > 0
                except Exception:
                    pinned = False
                if pinned:
                    continue

                # text
                try:
                    text_blocks = t.locator('div[data-testid="tweetText"] span')
                    parts = await text_blocks.all_text_contents()
                    text = "\n".join([x for x in parts if isinstance(x, str)]).strip()
                except Exception:
                    text = ""

                # images
                img_urls: list[str] = []
                try:
                    images = t.locator('img[src*="pbs.twimg.com/media"]')
                    img_urls = await images.evaluate_all(
                        "imgs => imgs.map(img => img.src)"
                    )
                    if not isinstance(img_urls, list):
                        img_urls = []
                    img_urls = [
                        str(u) for u in img_urls if isinstance(u, str) and u.strip()
                    ]
                except Exception:
                    img_urls = []

                # best-effort canonical tweet url: first link containing /status/
                tweet_url = ""
                try:
                    links = t.locator('a[href*="/status/"]')
                    hrefs = await links.evaluate_all("as => as.map(a => a.href)")
                    if isinstance(hrefs, list) and hrefs:
                        tweet_url = str(hrefs[0])
                except Exception:
                    tweet_url = ""

                out.append(TweetItem(url=tweet_url, text=text, image_urls=img_urls))
                if len(out) >= limit:
                    break

            return [
                {"url": x.url, "text": x.text, "image_urls": x.image_urls} for x in out
            ]
        finally:
            await browser.close()
