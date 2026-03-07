from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


LIST_API = "https://bbs-api.miyoushe.com/painter/wapi/userPostList"
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
SLUG_TO_GID = {"ys": 2}


def _parse_uid_and_slug(account: str) -> tuple[str, str]:
    raw = str(account or "").strip()
    if not raw:
        raise ValueError("empty account")
    if raw.isdigit():
        return raw, "ys"

    parsed = urlparse(raw)
    query = parse_qs(parsed.query)

    uid = ""
    for key in ("id", "uid"):
        values = query.get(key)
        if values and values[0].isdigit():
            uid = values[0]
            break

    if not uid:
        m = re.search(r"\b(\d{6,})\b", raw)
        if m:
            uid = m.group(1)
    if not uid:
        raise ValueError("cannot parse miyoushe uid from input")

    slug = "ys"
    for seg in parsed.path.split("/"):
        if seg in KNOWN_GAME_SLUG:
            slug = seg
            break
    return uid, slug


def _build_post_url(slug: str, post_id: str) -> str:
    return f"https://www.miyoushe.com/{slug}/article/{post_id}"


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


def get_user_latest_posts(
    user_post_list_url: str,
    limit: int = 10,
    *,
    headless: bool = True,
    sleep_between: float = 0.6,
    max_scroll_rounds: int = 20,
) -> list[dict[str, str]]:
    """
    Fetch latest Miyoushe posts via official bbs-api.
    Return: [{"title": "...", "url": "..."}, ...]
    """
    del headless, sleep_between, max_scroll_rounds
    size = max(1, int(limit))

    uid, slug = _parse_uid_and_slug(user_post_list_url)
    referer = (
        user_post_list_url
        if str(user_post_list_url or "").startswith("http")
        else f"https://www.miyoushe.com/{slug}/accountCenter/postList?id={uid}"
    )

    with requests.Session() as session:
        list_data = _request_json(
            session, LIST_API, {"size": size, "uid": uid}, referer
        )
        rows = list_data.get("data", {}).get("list", []) or []

        out: list[dict[str, str]] = []
        for item in rows[:size]:
            base_post = item.get("post", {}) if isinstance(item, dict) else {}
            post_id = str(base_post.get("post_id") or "").strip()
            if not post_id:
                continue

            title = str(base_post.get("subject") or "").strip()
            gid = int(base_post.get("game_id") or SLUG_TO_GID.get(slug, 2))

            # Best effort: resolve the post title from detail API.
            try:
                detail = _request_json(
                    session,
                    DETAIL_API,
                    {"gids": gid, "post_id": post_id, "read": 1},
                    referer,
                )
                post_obj = _find_post_container(detail.get("data")) or {}
                detail_title = str(post_obj.get("subject") or "").strip()
                if detail_title:
                    title = detail_title
            except Exception as e:
                astrbot_logger.debug(
                    "[dailynews] miyoushe detail title fallback post_id=%s err=%s",
                    post_id,
                    e,
                )

            out.append(
                {
                    "title": title or f"post_{post_id}",
                    "url": _build_post_url(slug, post_id),
                }
            )

    dedup: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in out:
        u = str(item.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        dedup.append({"title": str(item.get("title") or "").strip(), "url": u})
    return dedup[:size]
