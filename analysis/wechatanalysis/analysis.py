from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
from datetime import datetime
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

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

TIMEOUT = 25
_ALLOWED_LATEST_SCOPES = {"auto", "account", "album"}


def mobile_headers(referer: str = "https://mp.weixin.qq.com/") -> dict[str, str]:
    return {
        "User-Agent": WECHAT_MOBILE_UA,
        "Referer": referer,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "X-Requested-With": "com.tencent.mm",
    }


def ensure_mobile_article_url(url: str) -> str:
    sp = urlsplit(url)
    query_map = parse_qs(sp.query, keep_blank_values=True)
    query_map["nwr_flag"] = ["1"]
    query = urlencode([(k, v) for k, vals in query_map.items() for v in vals])
    return urlunsplit((sp.scheme, sp.netloc, sp.path, query, ""))


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "")
    name = "_".join(name.split())
    return name[:80] or "wechat_article"


def first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(strip=True)
            if text:
                return text
    return ""


def extract_js_var(page_html: str, name: str) -> str:
    pattern = rf"\bvar\s+{re.escape(name)}\s*=\s*([^;]+);"
    match = re.search(pattern, page_html)
    if not match:
        return ""
    expr = match.group(1).strip()
    quoted = re.search(r'"([^"]*)"|\'([^\']*)\'', expr)
    if quoted:
        return quoted.group(1) or quoted.group(2) or ""
    number = re.search(r"-?\d+", expr)
    return number.group(0) if number else ""


def normalize_url(url: str) -> str:
    url = html.unescape((url or "").strip())
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urljoin("https://mp.weixin.qq.com", url)
    return url


def parse_album_id(page_html: str) -> str:
    patterns = [
        r"album_id=(\d{8,})",
        r"\balbumId\s*:\s*['\"](\d{8,})['\"]",
        r"\balbum_id_str\s*:\s*['\"](\d{8,})['\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, page_html)
        if match:
            return match.group(1)
    return ""


def extract_js_number(page_html: str, name: str) -> int:
    pattern = rf"(?:var|let)\s+{re.escape(name)}\s*=\s*(\d+)"
    match = re.search(pattern, page_html)
    return int(match.group(1)) if match else 0


def _normalize_latest_scope(latest_scope: str) -> str:
    scope = str(latest_scope or "").strip().lower()
    return scope if scope in _ALLOWED_LATEST_SCOPES else "auto"


def parse_article_context(article_url: str, session: requests.Session) -> dict[str, object]:
    article_url = ensure_mobile_article_url(article_url)
    resp = session.get(
        article_url,
        headers=mobile_headers(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    final_url = resp.url
    if "wappoc_appmsgcaptcha" in final_url:
        raise RuntimeError("hit wechat captcha page; retry later or use a different network")

    page_html = resp.text
    soup = BeautifulSoup(page_html, "html.parser")

    content_node = soup.select_one("#js_content")
    image_urls: list[str] = []
    content_text = ""
    if content_node:
        content_text = content_node.get_text("\n", strip=True)
        for img in content_node.select("img"):
            src = img.get("data-src") or img.get("src")
            src = normalize_url(str(src or ""))
            if src and src not in image_urls:
                image_urls.append(src)

    title = first_text(soup, ["#activity-name", ".rich_media_title", "h1"])
    author = first_text(soup, ["#js_name", ".rich_media_meta.rich_media_meta_text"])

    ct = extract_js_var(page_html, "ct")
    publish_time = ""
    if ct.isdigit():
        publish_time = datetime.fromtimestamp(int(ct)).strftime("%Y-%m-%d %H:%M:%S")
    if not publish_time:
        publish_time = first_text(soup, ["#js_publish_time"])

    context: dict[str, object] = {
        "title": title,
        "author": author,
        "publish_time": publish_time,
        "content_text": content_text,
        "image_urls": image_urls,
        "url": final_url,
        "biz": extract_js_var(page_html, "biz"),
        "mid": extract_js_var(page_html, "mid"),
        "sn": extract_js_var(page_html, "sn"),
        "idx": extract_js_var(page_html, "idx") or "1",
        "ct": ct,
        "appmsg_type": extract_js_var(page_html, "appmsg_type") or "9",
        "msg_daily_idx": extract_js_var(page_html, "msg_daily_idx") or "1",
        "comment_id": extract_js_var(page_html, "comment_id"),
        "segment_comment_id": extract_js_var(page_html, "segment_comment_id"),
        "album_id": parse_album_id(page_html),
    }
    return context


def extract_articles_from_general_msg_list(
    msg_obj: dict,
    limit: int,
    seen_urls: set[str],
    out: list[dict[str, str]],
) -> None:
    for item in msg_obj.get("list", []):
        ext = item.get("app_msg_ext_info") or {}
        if ext:
            url = normalize_url(str(ext.get("content_url", "")))
            if url and url not in seen_urls:
                seen_urls.add(url)
                out.append(
                    {
                        "title": str(ext.get("title", "")),
                        "url": url,
                        "create_time": str((item.get("comm_msg_info") or {}).get("datetime", "")),
                    }
                )
                if len(out) >= limit:
                    return

        for sub in ext.get("multi_app_msg_item_list", []) or []:
            url = normalize_url(str(sub.get("content_url", "")))
            if url and url not in seen_urls:
                seen_urls.add(url)
                out.append(
                    {
                        "title": str(sub.get("title", "")),
                        "url": url,
                        "create_time": str((item.get("comm_msg_info") or {}).get("datetime", "")),
                    }
                )
                if len(out) >= limit:
                    return


def fetch_account_latest_articles(
    context: dict[str, object],
    limit: int,
    session: requests.Session,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    biz = str(context.get("biz", ""))
    if not biz:
        return [], {"error": "missing_biz"}

    profile_url = (
        f"https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz={biz}&scene=124#wechat_redirect"
    )
    resp = session.get(
        profile_url,
        headers=mobile_headers(str(context.get("url", "https://mp.weixin.qq.com/"))),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    page_html = resp.text
    lower_html = page_html.lower()

    if (
        "请在微信客户端打开链接" in page_html
        or "please open in wechat client" in lower_html
        or "<title>验证</title>" in page_html
        or "<title>verify</title>" in lower_html
    ):
        return [], {"error": "wechat_client_required", "profile_url": profile_url}

    msg_list_match = re.search(r"var\s+msgList\s*=\s*'(.+?)';", page_html, re.S)
    if not msg_list_match:
        return [], {"error": "msg_list_not_found", "profile_url": profile_url}

    msg_raw = html.unescape(msg_list_match.group(1))
    try:
        msg_obj = json.loads(msg_raw)
    except json.JSONDecodeError:
        return [], {"error": "msg_list_parse_failed", "profile_url": profile_url}

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    extract_articles_from_general_msg_list(msg_obj, limit, seen_urls, results)

    can_continue = extract_js_number(page_html, "can_msg_continue")
    next_offset = extract_js_number(page_html, "next_offset")

    while len(results) < limit and can_continue == 1:
        params = {
            "action": "getmsg",
            "__biz": biz,
            "f": "json",
            "offset": str(next_offset),
            "count": str(min(10, limit - len(results))),
            "is_ok": "1",
            "scene": "124",
            "uin": "",
            "key": "",
            "pass_ticket": "",
            "wxtoken": "",
            "x5": "0",
        }
        r = session.get(
            "https://mp.weixin.qq.com/mp/profile_ext",
            params=params,
            headers=mobile_headers(profile_url),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        general_raw = data.get("general_msg_list", "")
        if not general_raw:
            break

        try:
            general_obj = json.loads(general_raw)
        except json.JSONDecodeError:
            break

        extract_articles_from_general_msg_list(general_obj, limit, seen_urls, results)
        can_continue = int(data.get("can_msg_continue", 0))
        next_offset = int(data.get("next_offset", 0))
        if not next_offset:
            break

    return results, {"profile_url": profile_url}


def build_getappmsgext_request(
    context: dict[str, object],
) -> tuple[str, dict[str, str], dict[str, str]]:
    api = "https://mp.weixin.qq.com/mp/getappmsgext"
    params = {
        "f": "json",
        "mock": "",
        "uin": "",
        "key": "",
        "pass_ticket": "",
        "wxtoken": "777",
        "devicetype": "",
        "clientversion": "",
        "version": "",
        "__biz": str(context.get("biz", "")),
        "appmsg_token": "",
        "x5": "0",
        "user_article_role": "0",
    }
    payload = {
        "r": str(random.random()),
        "__biz": str(context.get("biz", "")),
        "appmsg_type": str(context.get("appmsg_type", "9")),
        "mid": str(context.get("mid", "")),
        "sn": str(context.get("sn", "")),
        "idx": str(context.get("idx", "1")),
        "scene": "",
        "subscene": "0",
        "ascene": "0",
        "title": str(context.get("title", "")),
        "ct": str(context.get("ct", "")),
        "abtest_cookie": "",
        "devicetype": "",
        "version": "",
        "is_need_ticket": "0",
        "is_need_ad": "0",
        "comment_id": str(context.get("comment_id", "")),
        "is_need_reward": "0",
        "both_ad": "0",
        "reward_uin_count": "0",
        "send_time": "",
        "msg_daily_idx": str(context.get("msg_daily_idx", "1")),
        "is_original": "0",
        "is_only_read": "1",
        "req_id": "",
        "pass_ticket": "",
        "is_temp_url": "0",
        "item_show_type": "0",
        "tmp_version": "1",
        "more_read_type": "0",
        "appmsg_like_type": "2",
        "related_video_sn": "",
        "related_video_num": "5",
        "vid": "",
        "is_pay_subscribe": "0",
        "pay_subscribe_uin_count": "0",
        "has_red_packet_cover": "0",
        "album_video_num": "5",
        "cur_album_id": str(context.get("album_id", "")),
        "is_public_related_video": "0",
        "encode_info_by_base64": "",
        "exptype": "",
        "export_key": "",
        "export_key_extinfo": "",
        "segment_comment_id": str(context.get("segment_comment_id", "")),
        "business_type": "0",
    }
    return api, params, payload


def call_getappmsgext(context: dict[str, object], session: requests.Session) -> dict:
    api, params, payload = build_getappmsgext_request(context)
    headers = {
        **mobile_headers(str(context.get("url", "https://mp.weixin.qq.com/"))),
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    resp = session.post(api, params=params, data=payload, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_album_articles(
    context: dict[str, object],
    limit: int,
    session: requests.Session,
) -> list[dict[str, str]]:
    biz = str(context.get("biz", ""))
    album_id = str(context.get("album_id", ""))
    if not biz or not album_id:
        return []

    api = "https://mp.weixin.qq.com/mp/appmsgalbum"
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    begin_msgid = ""
    begin_itemidx = ""

    while len(results) < limit:
        batch_size = min(10, limit - len(results))
        params = {
            "action": "getalbum",
            "__biz": biz,
            "album_id": album_id,
            "count": str(batch_size),
            "f": "json",
        }
        if begin_msgid:
            params["begin_msgid"] = begin_msgid
            params["begin_itemidx"] = begin_itemidx or "1"

        resp = session.get(
            api,
            params=params,
            headers=mobile_headers(str(context.get("url", "https://mp.weixin.qq.com/"))),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        album_resp = data.get("getalbum_resp", {})
        articles = album_resp.get("article_list", [])
        if not articles:
            break

        for article in articles:
            url = normalize_url(str(article.get("url", "")))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "title": str(article.get("title", "")),
                    "url": url,
                    "msgid": str(article.get("msgid", "")),
                    "itemidx": str(article.get("itemidx", "")),
                    "create_time": str(article.get("create_time", "")),
                }
            )
            if len(results) >= limit:
                break

        if str(album_resp.get("continue_flag", "0")) != "1":
            break

        last = articles[-1]
        begin_msgid = str(last.get("msgid", ""))
        begin_itemidx = str(last.get("itemidx", "1"))
        if not begin_msgid:
            break

    return results


def fetch_latest_articles(
    article_url: str,
    limit: int = 5,
    latest_scope: str = "auto",
    session: requests.Session | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    scope = _normalize_latest_scope(latest_scope)
    own_session = session is None
    sess = session or requests.Session()

    try:
        context = parse_article_context(article_url, sess)
        account_meta: dict[str, str] = {}
        account_error = ""
        latest_articles: list[dict[str, str]] = []
        resolved_scope = scope

        if scope in {"auto", "account"}:
            latest_articles, account_meta = fetch_account_latest_articles(context, limit, sess)
            if latest_articles:
                resolved_scope = "account"
            elif scope == "account":
                account_error = account_meta.get("error", "account_fetch_failed")

        if (not latest_articles) and scope in {"auto", "album"}:
            latest_articles = fetch_album_articles(context, limit, sess)
            if latest_articles:
                resolved_scope = "album"

        meta = {
            "scope": resolved_scope,
            "requested_scope": scope,
            "profile_url": account_meta.get("profile_url", ""),
            "account_error": account_error or account_meta.get("error", ""),
            "article_url": str(context.get("url", article_url)),
            "album_id": str(context.get("album_id", "")),
        }
        return latest_articles, meta
    finally:
        if own_session:
            sess.close()


def download_images(
    image_urls: list[str],
    referer_url: str,
    images_dir: str,
    session: requests.Session,
) -> list[str]:
    if not image_urls:
        return []
    os.makedirs(images_dir, exist_ok=True)

    local_paths: list[str] = []
    headers = mobile_headers(referer_url)

    for idx, img_url in enumerate(image_urls):
        ext = ".jpg"
        try:
            qs = parse_qs(urlparse(img_url).query)
            wx_fmt = qs.get("wx_fmt", [""])[0]
            if wx_fmt:
                ext = "." + wx_fmt
        except Exception:
            pass

        filename = f"img_{idx}{ext}"
        abs_path = os.path.join(images_dir, filename)
        rel_path = os.path.join(os.path.basename(images_dir), filename)

        try:
            r = session.get(img_url, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            with open(abs_path, "wb") as f:
                f.write(r.content)
            local_paths.append(rel_path)
        except Exception as exc:
            astrbot_logger.warning(
                "[dailynews] image download failed: %s (%s)", img_url, exc
            )

    return local_paths


def build_markdown(
    article: dict[str, object],
    local_images: list[str],
    latest_articles: list[dict[str, str]],
    latest_label: str,
) -> str:
    lines = [f"# {article.get('title', '')}", ""]
    if article.get("author"):
        lines.append(f"- Author: {article['author']}")
    if article.get("publish_time"):
        lines.append(f"- Publish Time: {article['publish_time']}")
    lines.append(f"- Source: {article.get('url', '')}")
    lines.extend(["", "---", "", "## Content", "", str(article.get("content_text", "")), ""])

    if local_images:
        lines.extend(["## Images", ""])
        for i, path in enumerate(local_images):
            lines.append(f"![image_{i}]({path})")
        lines.append("")

    if latest_articles:
        lines.extend([f"## {latest_label}", ""])
        for i, item in enumerate(latest_articles, 1):
            lines.append(f"{i}. [{item.get('title', '')}]({item.get('url', '')})")
        lines.append("")

    return "\n".join(lines)


def fetch_wechat_article(url: str) -> dict[str, object]:
    session = requests.Session()
    try:
        return parse_article_context(url, session)
    finally:
        session.close()


def wechat_to_markdown(
    url: str,
    output_dir: str = "output",
    limit: int = 5,
    latest_scope: str = "auto",
    download_imgs: bool = True,
) -> str:
    result = run(
        article_url=url,
        output_dir=output_dir,
        limit=max(1, int(limit or 1)),
        download_imgs=bool(download_imgs),
        latest_scope=latest_scope,
    )
    return str(result["md_path"])


def run(
    article_url: str,
    output_dir: str,
    limit: int,
    download_imgs: bool,
    latest_scope: str,
) -> dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    scope = _normalize_latest_scope(latest_scope)

    try:
        context = parse_article_context(article_url, session)

        api, params, payload = build_getappmsgext_request(context)
        ext_data: dict[str, object] = {}
        try:
            ext_data = call_getappmsgext(context, session)
        except Exception as exc:
            ext_data = {"error": str(exc)}

        latest_articles, latest_meta = fetch_latest_articles(
            article_url=str(context.get("url", article_url)),
            limit=max(1, int(limit or 1)),
            latest_scope=scope,
            session=session,
        )
        latest_label_map = {
            "account": "Latest Account Articles",
            "album": "Latest Album Articles",
        }
        latest_label = latest_label_map.get(
            str(latest_meta.get("scope", scope)), "Latest Articles"
        )

        images_dir = os.path.join(output_dir, "images")
        local_images = (
            download_images(
                list(context.get("image_urls") or []),
                str(context.get("url", article_url)),
                images_dir,
                session,
            )
            if download_imgs
            else []
        )

        md_text = build_markdown(context, local_images, latest_articles, latest_label)
        safe_title = sanitize_filename(str(context.get("title") or "wechat_article"))
        md_path = os.path.join(output_dir, safe_title + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        latest_json_path = os.path.join(output_dir, safe_title + "_latest_articles.json")
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(latest_articles, f, ensure_ascii=False, indent=2)

        api_debug = {
            "getappmsgext": {"url": api, "query_params": params, "request_body": payload},
            "appmsgalbum_example": {
                "url": "https://mp.weixin.qq.com/mp/appmsgalbum",
                "query_params": {
                    "action": "getalbum",
                    "__biz": str(context.get("biz", "")),
                    "album_id": str(context.get("album_id", "")),
                    "count": min(limit, 10),
                    "f": "json",
                },
                "pagination_example": {
                    "query_params": {
                        "action": "getalbum",
                        "__biz": str(context.get("biz", "")),
                        "album_id": str(context.get("album_id", "")),
                        "count": min(limit, 10),
                        "begin_msgid": "<last_msgid_from_previous_page>",
                        "begin_itemidx": "<last_itemidx_from_previous_page>",
                        "f": "json",
                    }
                },
            },
            "account_fetch": {
                "mode": scope,
                "profile_url": latest_meta.get("profile_url", ""),
                "error": latest_meta.get("account_error", ""),
            },
        }
        api_debug_path = os.path.join(output_dir, safe_title + "_api_debug.json")
        with open(api_debug_path, "w", encoding="utf-8") as f:
            json.dump(api_debug, f, ensure_ascii=False, indent=2)

        return {
            "context": context,
            "latest_articles": latest_articles,
            "latest_meta": latest_meta,
            "md_path": md_path,
            "latest_json_path": latest_json_path,
            "api_debug_path": api_debug_path,
            "ext_data": ext_data,
            "images_downloaded": len(local_images),
            "images_found": len(list(context.get("image_urls") or [])),
        }
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeChat public-account crawler without Playwright (article + account/album API)."
    )
    parser.add_argument("article_url", help="WeChat article URL")
    parser.add_argument("--output-dir", default="output", help="output directory")
    parser.add_argument("--limit", type=int, default=5, help="latest articles count")
    parser.add_argument(
        "--latest-scope",
        choices=["auto", "account", "album"],
        default="auto",
        help="latest article source: account/profile, album, or auto fallback",
    )
    parser.add_argument(
        "--no-download-images",
        action="store_true",
        help="do not download content images",
    )
    args = parser.parse_args()

    result = run(
        article_url=args.article_url,
        output_dir=args.output_dir,
        limit=max(1, int(args.limit or 1)),
        download_imgs=not bool(args.no_download_images),
        latest_scope=args.latest_scope,
    )

    context = result.get("context") or {}
    latest_meta = result.get("latest_meta") or {}

    print("Title:", context.get("title", ""))
    print("Author:", context.get("author", ""))
    print("Publish Time:", context.get("publish_time", ""))
    print("Article URL:", context.get("url", ""))
    print("Images Found:", result.get("images_found", 0))
    print("Images Downloaded:", result.get("images_downloaded", 0))
    print("Album ID:", context.get("album_id", ""))
    print("Latest Scope:", latest_meta.get("requested_scope", "auto"))
    print("Resolved Scope:", latest_meta.get("scope", ""))
    print("Latest Articles:", len(result.get("latest_articles") or []))
    if latest_meta.get("account_error"):
        print("Account Fetch Error:", latest_meta.get("account_error"))
    print("Markdown:", result.get("md_path", ""))
    print("Latest JSON:", result.get("latest_json_path", ""))
    print("API Debug:", result.get("api_debug_path", ""))


if __name__ == "__main__":
    main()
