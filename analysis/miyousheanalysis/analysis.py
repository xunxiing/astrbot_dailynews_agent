import html as html_lib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _html_to_text_basic(html: str) -> str:
    if not html:
        return ""

    html = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", "", html)
    html = re.sub(r"(?is)<img\\b[^>]*>", "", html)

    html = re.sub(r"(?i)<br\\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\\s*>", "\n\n", html)
    html = re.sub(r"(?i)<p\\b[^>]*>", "", html)

    text = _strip_tags(html)
    text = html_lib.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _find_post_like(obj: Any, depth: int = 0, max_depth: int = 10) -> Optional[Dict[str, Any]]:
    if depth > max_depth:
        return None

    if isinstance(obj, dict):
        if "post" in obj and isinstance(obj["post"], dict):
            p = obj["post"]
            if any(k in p for k in ["subject", "title"]) and any(
                k in p for k in ["content", "post_content", "body"]
            ):
                return p

        if any(k in obj for k in ["subject", "title"]) and any(
            k in obj for k in ["content", "post_content", "body"]
        ):
            return obj

        if "article" in obj and isinstance(obj["article"], dict):
            a = obj["article"]
            if any(k in a for k in ["subject", "title"]) and any(k in a for k in ["content", "body"]):
                return a

        for v in obj.values():
            hit = _find_post_like(v, depth + 1, max_depth)
            if hit:
                return hit

    if isinstance(obj, list):
        for it in obj:
            hit = _find_post_like(it, depth + 1, max_depth)
            if hit:
                return hit
    return None


def _content_to_text(content: Any) -> Tuple[str, List[str]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        html = content or ""
        # Some Nuxt payloads store rich HTML content, and `image_list` may be empty.
        # Extract inline images from HTML to improve image candidate coverage.
        imgs = re.findall(r'(?:src|data-src)=["\']([^"\']+)["\']', html, flags=re.I)
        seen = set()
        img_urls = []
        for u in imgs:
            s = str(u or "").strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            img_urls.append(s)
        return _html_to_text_basic(html), img_urls
    if isinstance(content, list):
        lines: List[str] = []
        imgs: List[str] = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if "insert" in blk and "type" not in blk:
                ins = blk.get("insert")
                if isinstance(ins, str):
                    lines.append(ins)
                elif isinstance(ins, dict) and "image" in ins:
                    url = ins["image"]
                    imgs.append(url)
                continue
            t = blk.get("type")
            ins = blk.get("insert")
            if t == "text" and isinstance(ins, str):
                lines.append(ins)
            elif t == "image" and isinstance(ins, dict) and "image" in ins:
                url = ins["image"]
                imgs.append(url)
            else:
                if isinstance(ins, str):
                    lines.append(ins)
        text = "\n\n".join([x for x in lines if str(x).strip()]).strip()
        return text, imgs
    return str(content).strip(), []


def fetch_miyoushe_post(article_url: str) -> Dict[str, Any]:
    """
    抓取米游社帖子页：优先抓取页面发起的 bbs-api `getPostFull` 响应（更稳更全），失败则回退到 DOM 容器解析。
    返回：{title, content_text, image_urls}
    """
    m = re.search(r"/article/(\d+)", article_url or "")
    post_id = m.group(1) if m else ""

    with sync_playwright() as p:
        executable_path = None
        try:
            from ...workflow.playwright_bootstrap import get_chromium_executable_path

            exe = get_chromium_executable_path()
            executable_path = str(exe) if exe else None
        except Exception:
            executable_path = None

        if executable_path:
            browser = p.chromium.launch(headless=True, executable_path=executable_path)
        else:
            astrbot_logger.warning(
                "[dailynews] playwright chromium not ready; falling back to default Playwright browser path (may fail)."
            )
            browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        astrbot_logger.info("[dailynews] opening miyoushe article: %s", article_url)
        # 1) 优先：等待页面请求的 getPostFull 返回，再从 JSON 里取内容
        if post_id:
            try:
                with page.expect_response(
                    lambda r: "getPostFull" in r.url and f"post_id={post_id}" in r.url,
                    timeout=30000,
                ) as resp_info:
                    page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
                resp = resp_info.value
                if resp and resp.status == 200:
                    data = resp.json() or {}
                    post_wrap = (data.get("data") or {}).get("post") or {}
                    post = post_wrap.get("post") if isinstance(post_wrap.get("post"), dict) else {}
                    title = (post.get("subject") or post.get("title") or "").strip() or page.title().strip() or "未命名帖子"
                    content_html = (post.get("content") or "").strip()
                    content_text = _html_to_text_basic(content_html)

                    imgs: List[str] = []
                    img_list = post_wrap.get("image_list")
                    if isinstance(img_list, list):
                        for it in img_list:
                            if isinstance(it, dict):
                                u = it.get("url") or it.get("img") or it.get("src")
                                if isinstance(u, str) and u.startswith("http"):
                                    imgs.append(u)
                            elif isinstance(it, str) and it.startswith("http"):
                                imgs.append(it)
                    cover = post_wrap.get("cover")
                    if isinstance(cover, dict):
                        cu = cover.get("url") or cover.get("img") or cover.get("src")
                        if isinstance(cu, str) and cu.startswith("http"):
                            imgs.append(cu)

                    seen = set()
                    images = [u for u in imgs if u and (u not in seen and not seen.add(u))]
                    browser.close()
                    return {"title": title, "content_text": content_text, "image_urls": images}
            except Exception:
                # 继续走 DOM fallback
                pass

        # 确保页面已加载一些内容，避免只拿到 Loading...
        try:
            page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=15000)
            time.sleep(1.2)
        except Exception:
            pass

        # 1) Nuxt 优先
        try:
            page.wait_for_function(
                "() => window.__NUXT__ && Object.keys(window.__NUXT__).length > 0",
                timeout=12000,
            )
            nuxt = page.evaluate("() => window.__NUXT__")
            post = _find_post_like(nuxt)
            if isinstance(post, dict):
                title = (post.get("subject") or post.get("title") or "").strip() or page.title().strip() or "未命名帖子"
                content = post.get("content") or post.get("post_content") or post.get("body")
                content_text, imgs1 = _content_to_text(content)
                imgs2: List[str] = []
                for k in ["images", "image_list", "img_list", "covers", "cover"]:
                    v = post.get(k)
                    if isinstance(v, str) and v.startswith("http"):
                        imgs2.append(v)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, str) and it.startswith("http"):
                                imgs2.append(it)
                            elif isinstance(it, dict):
                                for kk in ["url", "image", "src"]:
                                    u = it.get(kk)
                                    if isinstance(u, str) and u.startswith("http"):
                                        imgs2.append(u)

                browser.close()
                imgs = []
                seen = set()
                for u in (imgs1 + imgs2):
                    if isinstance(u, str) and u and u not in seen:
                        seen.add(u)
                        imgs.append(u)
                return {"title": title, "content_text": content_text, "image_urls": imgs}
        except Exception:
            pass

        # 2) DOM fallback：只抓正文容器
        title = ""
        for sel in [
            ".mhy-article__title",
            ".mhy-article-page__title",
            ".mhy-article-page__title h1",
            "h1",
            "h2",
        ]:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    title = loc.first.inner_text(timeout=5000).strip()
                    if title:
                        break
            except Exception:
                pass
        if not title:
            # 从 body 文本里粗略提取（DOM 结构变化时兜底）
            body = (page.inner_text("body") or "").strip()
            title = page.title().strip() or "未命名帖子"
            if body and "文章发表" in body:
                head = body.splitlines()
                for line in head[:30]:
                    s = (line or "").strip()
                    if s and "文章发表" not in s and "米游社" not in s and len(s) >= 6:
                        title = s
                        break

        content = page.locator(".mhy-img-text-article__content.ql-editor")
        if content.count() == 0:
            content = page.locator(".mhy-article-page__content")
        try:
            content.first.wait_for(state="visible", timeout=15000)
        except Exception:
            pass
        time.sleep(0.6)

        img_urls: List[str] = []
        try:
            ql_images = content.first.locator(".ql-image")
            if ql_images.count() > 0:
                urls = ql_images.evaluate_all(
                    r"""
                    els => els.map(e => {
                      const img = e.querySelector("img");
                      const src = img?.currentSrc || img?.src || img?.getAttribute("data-src") || img?.getAttribute("src");
                      if (src) return src;
                      const bg = getComputedStyle(e).backgroundImage;
                      if (!bg) return null;
                      const m = bg.match(/url\\([\"']?(.*?)[\"']?\\)/i);
                      return m ? m[1] : null;
                    }).filter(Boolean)
                    """
                )
                img_urls.extend(urls)
        except Exception:
            pass

        seen = set()
        img_urls = [u for u in img_urls if u and (u not in seen and not seen.add(u))]

        body_html = ""
        try:
            body_html = content.first.inner_html(timeout=10000)
        except Exception:
            pass
        content_text = _html_to_text_basic(body_html)

        browser.close()
        return {"title": title, "content_text": content_text, "image_urls": img_urls}
