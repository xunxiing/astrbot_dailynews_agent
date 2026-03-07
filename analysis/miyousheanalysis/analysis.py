from __future__ import annotations

import html as html_lib
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


DETAIL_API = "https://bbs-api.miyoushe.com/post/wapi/getPostFull"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

KNOWN_GAME_SLUG = {"ys", "sr", "zzz", "bh3", "bh2", "wd", "dby", "hna", "planet"}
SLUG_TO_GID = {
    "ys": 2,
    "sr": 6,
    "bh3": 1,
    "zzz": 8,
}


def _request_json(
    session: requests.Session, url: str, params: dict[str, Any], referer: str
) -> dict[str, Any]:
    headers = dict(DEFAULT_HEADERS)
    headers["Referer"] = referer
    resp = session.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retcode") != 0:
        raise RuntimeError(f"retcode={data.get('retcode')} msg={data.get('message')}")
    return data


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _html_to_text_basic(raw_html: str) -> str:
    if not raw_html:
        return ""

    s = re.sub(r"(?is)<script[^>]*>.*?</script>", "", raw_html)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", "", s)
    s = re.sub(r"(?is)<img\\b[^>]*>", "", s)

    s = re.sub(r"(?i)<br\\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\\s*>", "\n\n", s)
    s = re.sub(r"(?is)</div\\s*>", "\n", s)
    s = re.sub(r"(?is)<li\\b[^>]*>", "\n- ", s)
    s = re.sub(r"(?is)</li\\s*>", "", s)

    s = _strip_tags(s)
    s = html_lib.unescape(s)
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _find_post_container(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if "subject" in obj and "content" in obj:
            return obj
        for v in obj.values():
            hit = _find_post_container(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for item in obj:
            hit = _find_post_container(item)
            if hit is not None:
                return hit
    return None


def _extract_post_id(article_url: str) -> str:
    raw = str(article_url or "").strip()
    m = re.search(r"/article/(\d+)", raw)
    if m:
        return m.group(1)
    parsed = urlparse(raw)
    q = parse_qs(parsed.query)
    for key in ("post_id", "id"):
        vals = q.get(key)
        if vals and str(vals[0]).isdigit():
            return str(vals[0])
    m2 = re.search(r"\b(\d{6,})\b", raw)
    return m2.group(1) if m2 else ""


def _infer_slug(article_url: str) -> str:
    parsed = urlparse(str(article_url or "").strip())
    for seg in parsed.path.split("/"):
        if seg in KNOWN_GAME_SLUG:
            return seg
    return "ys"


def _extract_inline_images(raw_html: str) -> list[str]:
    if not raw_html:
        return []
    imgs = re.findall(r'(?:src|data-src)=["\']([^"\']+)["\']', raw_html, flags=re.I)
    out: list[str] = []
    seen: set[str] = set()
    for u in imgs:
        s = str(u or "").strip()
        if not s or not s.startswith(("http://", "https://")):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_images_from_node(
    node: Any, *, depth: int = 0, max_depth: int = 6
) -> list[str]:
    if depth > max_depth:
        return []

    out: list[str] = []
    if isinstance(node, dict):
        for key in ("url", "src", "img", "image", "image_url"):
            v = node.get(key)
            if isinstance(v, str) and v.startswith(("http://", "https://")):
                out.append(v)

        for key in ("cover", "covers", "image_list", "images", "img_list", "pics"):
            if key in node:
                out.extend(
                    _extract_images_from_node(
                        node.get(key), depth=depth + 1, max_depth=max_depth
                    )
                )

        for v in node.values():
            if isinstance(v, (dict, list)):
                out.extend(
                    _extract_images_from_node(v, depth=depth + 1, max_depth=max_depth)
                )

    elif isinstance(node, list):
        for item in node:
            out.extend(
                _extract_images_from_node(item, depth=depth + 1, max_depth=max_depth)
            )

    return out


def _dedupe_urls(urls: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        s = str(u or "").strip()
        if not s or not s.startswith(("http://", "https://")):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def fetch_miyoushe_post(article_url: str) -> dict[str, Any]:
    """
    Fetch a Miyoushe article by post_id through getPostFull API.
    Return: {"title": str, "content_text": str, "image_urls": list[str]}
    """
    article_url = str(article_url or "").strip()
    if not article_url:
        raise ValueError("empty article_url")

    post_id = _extract_post_id(article_url)
    if not post_id:
        raise ValueError(f"cannot parse post_id from url: {article_url}")

    slug = _infer_slug(article_url)
    gid = int(SLUG_TO_GID.get(slug, 2))

    with requests.Session() as session:
        detail: dict[str, Any] | None = None

        # Try inferred gid first, then fallback gid=2 for compatibility.
        for try_gid in (gid, 2):
            try:
                detail = _request_json(
                    session,
                    DETAIL_API,
                    {"gids": try_gid, "post_id": post_id, "read": 1},
                    article_url,
                )
                break
            except Exception as e:
                astrbot_logger.debug(
                    "[dailynews] miyoushe getPostFull failed post_id=%s gid=%s err=%s",
                    post_id,
                    try_gid,
                    e,
                )

        if not isinstance(detail, dict):
            raise RuntimeError(f"miyoushe detail api failed for post_id={post_id}")

    data = detail.get("data") if isinstance(detail, dict) else {}
    if not isinstance(data, dict):
        data = {}

    post_wrap = data.get("post") if isinstance(data.get("post"), dict) else {}
    base_post = post_wrap.get("post") if isinstance(post_wrap.get("post"), dict) else {}
    post_obj = _find_post_container(data) or base_post

    title = str(
        post_obj.get("subject")
        or post_obj.get("title")
        or base_post.get("subject")
        or base_post.get("title")
        or f"post_{post_id}"
    ).strip()

    raw_content = post_obj.get("content") or base_post.get("content") or ""
    if isinstance(raw_content, list):
        raw_content = "\n".join(str(x) for x in raw_content)
    raw_content = str(raw_content)

    content_text = _html_to_text_basic(raw_content)

    images = _dedupe_urls(
        _extract_images_from_node(post_wrap)
        + _extract_images_from_node(post_obj)
        + _extract_inline_images(raw_content)
    )

    return {
        "title": title,
        "content_text": content_text,
        "image_urls": images,
    }
