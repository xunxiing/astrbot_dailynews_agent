import asyncio
import os
import sys
from urllib.parse import parse_qs, urlparse

import aiohttp
from playwright.sync_api import sync_playwright

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

# ---------- 工具函数 ----------


def sanitize_filename(name: str) -> str:
    """清理标题，变成安全的文件名（兼容 Windows）"""
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "")
    # 把空白变成下划线
    name = "_".join(name.split())
    # 控制长度，避免路径过长
    return name[:80] or "wechat_article"


# ---------- 第一步：用 Playwright 抓取文章 ----------


def fetch_wechat_article(url: str):
    """用 Playwright 获取文章信息（标题、作者、时间、正文、图片链接）"""
    with sync_playwright() as p:
        executable_path = None
        try:
            from workflow.pipeline.playwright_bootstrap import (
                get_chromium_executable_path,
            )

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
        context = browser.new_context()
        page = context.new_page()

        astrbot_logger.info("[dailynews] opening wechat article: %s", url)
        page.goto(url, timeout=60000)

        # 等标题加载出来
        page.wait_for_selector("#activity-name", timeout=60000)

        title = page.inner_text("#activity-name").strip()

        # 作者
        try:
            author = page.inner_text(".rich_media_meta.rich_media_meta_text").strip()
        except Exception:
            author = ""

        # 发布时间
        try:
            publish_time = page.inner_text("#js_publish_time").strip()
        except Exception:
            publish_time = ""

        # 正文纯文本
        try:
            content_text = page.inner_text("#js_content").strip()
        except Exception:
            content_text = ""

        # 正文里的图片
        img_elements = page.query_selector_all("#js_content img")
        image_urls = []
        for img in img_elements:
            # 微信图一般在 data-src 里
            src = img.get_attribute("data-src") or img.get_attribute("src")
            if src and src not in image_urls:
                image_urls.append(src)

        browser.close()

        return {
            "title": title,
            "author": author,
            "publish_time": publish_time,
            "content_text": content_text,
            "image_urls": image_urls,
        }


# ---------- 第二步：用 aiohttp 并发下载图片 ----------


async def _download_one_image(session, idx, img_url, images_dir, referer_url):
    """下载单张图片，返回在 MD 中用的相对路径；失败则返回 None"""
    # 从 URL 的 wx_fmt 参数推断图片格式
    ext = ".jpg"
    try:
        qs = parse_qs(urlparse(img_url).query)
        fmt_list = qs.get("wx_fmt")
        if fmt_list and fmt_list[0]:
            ext = "." + fmt_list[0]
    except Exception:
        pass

    filename = f"img_{idx}{ext}"
    local_path = os.path.join(images_dir, filename)
    rel_path = os.path.join(os.path.basename(images_dir), filename)  # 相对路径

    try:
        async with session.get(img_url, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.read()
            with open(local_path, "wb") as f:
                f.write(data)
        astrbot_logger.debug("[dailynews] image downloaded: %s", img_url)
        return rel_path
    except Exception as e:
        astrbot_logger.warning("[dailynews] image download failed: %s (%s)", img_url, e)
        return None


async def download_images_async(image_urls, referer_url, images_dir, max_concurrency=8):
    """
    使用 aiohttp 并发下载所有图片。
    返回在 Markdown 中使用的相对路径列表。
    """
    if not image_urls:
        return []

    os.makedirs(images_dir, exist_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": referer_url,  # 很重要，有些图没有 Referer 会 403
    }

    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = []
        for idx, url in enumerate(image_urls):
            tasks.append(
                _download_one_image(session, idx, url, images_dir, referer_url)
            )

        results = await asyncio.gather(*tasks)

    # 去掉失败的 None
    return [r for r in results if r is not None]


def download_images(image_urls, referer_url, images_dir, max_concurrency=8):
    """
    同步封装：在普通函数里调用 asyncio。
    """
    return asyncio.run(
        download_images_async(image_urls, referer_url, images_dir, max_concurrency)
    )


# ---------- 第三步：生成 Markdown ----------


def build_markdown(data, local_image_paths, url):
    """根据文章信息 + 图片本地路径，生成 Markdown 文本"""
    lines = []

    # 标题
    lines.append(f"# {data['title']}".strip())
    lines.append("")

    # 元信息
    if data.get("author"):
        lines.append(f"- 作者：{data['author']}")
    if data.get("publish_time"):
        lines.append(f"- 发布时间：{data['publish_time']}")
    lines.append(f"- 原文链接：{url}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 正文")
    lines.append("")
    lines.append(data.get("content_text", ""))
    lines.append("")

    # 图片区域（统一放在后面；如果你想按正文位置插入，可以再改）
    if local_image_paths:
        lines.append("")
        lines.append("## 图片")
        lines.append("")
        for i, path in enumerate(local_image_paths):
            lines.append(f"![image_{i}]({path})")
        lines.append("")

    return "\n".join(lines)


def save_markdown(md_text, title, output_dir):
    """把 Markdown 文本写入 .md 文件"""
    os.makedirs(output_dir, exist_ok=True)
    safe_title = sanitize_filename(title)
    md_path = os.path.join(output_dir, safe_title + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path


# ---------- 主流程 ----------


def wechat_to_markdown(url: str, output_dir="output", max_concurrency=8):
    """主流程：拉取文章 → 并发下载图片 → 生成并保存 MD"""

    # 1. 拉取文章信息
    data = fetch_wechat_article(url)

    # 2. 并发下载图片
    images_dir = os.path.join(output_dir, "images")
    astrbot_logger.info(
        "[dailynews] found %s images; downloading...", len(data.get("image_urls") or [])
    )
    local_image_paths = download_images(
        data["image_urls"], url, images_dir, max_concurrency=max_concurrency
    )

    # 3. 生成 Markdown
    md_text = build_markdown(data, local_image_paths, url)

    # 4. 保存 Markdown 文件
    md_path = save_markdown(md_text, data["title"], output_dir)

    astrbot_logger.info("[dailynews] wechat markdown saved: %s", md_path)
    return md_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        astrbot_logger.info("用法: python analysis.py <微信公众号文章链接>")
        sys.exit(1)

    article_url = sys.argv[1]
    # 可以根据你网络情况调节 max_concurrency，8~16 一般比较合适
    wechat_to_markdown(article_url, max_concurrency=10)
