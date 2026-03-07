from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

DEFAULT_WECHAT_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Mobile/15E148 MicroMessenger/8.0.40(0x1800282b) "
    "NetType/WIFI Language/zh_CN"
)
DEFAULT_TIMEOUT = 25
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
_ALLOWED_SCOPES = {"auto", "account", "album"}


class WechatCrawlerError(Exception):
    pass


class WechatCaptchaError(WechatCrawlerError):
    pass


@dataclass
class CrawlOptions:
    article_url: str
    output_dir: str = "output"
    limit: int = 5
    latest_scope: str = "auto"
    download_images: bool = True
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    user_agent: str = DEFAULT_WECHAT_MOBILE_UA
    session: requests.Session | None = None


@dataclass
class LatestArticle:
    title: str
    url: str
    msgid: str = ""
    itemidx: str = ""
    create_time: str = ""


@dataclass
class ArticleContext:
    title: str = ""
    author: str = ""
    publish_time: str = ""
    content_text: str = ""
    image_urls: list[str] = field(default_factory=list)
    url: str = ""
    biz: str = ""
    mid: str = ""
    sn: str = ""
    idx: str = "1"
    ct: str = ""
    appmsg_type: str = "9"
    msg_daily_idx: str = "1"
    comment_id: str = ""
    segment_comment_id: str = ""
    album_id: str = ""


@dataclass
class CrawlResult:
    context: ArticleContext
    latest_articles: list[LatestArticle]
    latest_label: str
    account_error: str
    ext_data: dict[str, Any]
    downloaded_images: list[str]
    md_path: str
    latest_json_path: str
    api_debug_path: str


def _normalize_scope(scope: str) -> str:
    s = str(scope or "").strip().lower()
    return s if s in _ALLOWED_SCOPES else "auto"


def _safe_int_ts(value: str | int | None) -> int:
    try:
        n = int(value or 0)
    except Exception:
        return 0
    return n // 1000 if n > 10_000_000_000 else n


def sort_latest_articles_desc(articles: list[LatestArticle]) -> list[LatestArticle]:
    return sorted(articles, key=lambda x: _safe_int_ts(x.create_time), reverse=True)


def mobile_headers(
    user_agent: str, referer: str = "https://mp.weixin.qq.com/", ajax: bool = False
) -> dict[str, str]:
    h = {
        "User-Agent": user_agent,
        "Referer": referer,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "X-Requested-With": "com.tencent.mm",
    }
    if ajax:
        h["X-Requested-With"] = "XMLHttpRequest"
    return h


def ensure_mobile_article_url(url: str) -> str:
    split = urlsplit(url.strip())
    query_map = parse_qs(split.query, keep_blank_values=True)
    query_map["nwr_flag"] = ["1"]
    query = urlencode([(k, v) for k, values in query_map.items() for v in values])
    return urlunsplit((split.scheme, split.netloc, split.path, query, ""))


def normalize_url(url: str) -> str:
    url = html.unescape((url or "").strip())
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urljoin("https://mp.weixin.qq.com", url)
    return url


def first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(strip=True)
            if text:
                return text
    return ""


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "")
    name = "_".join(name.split())
    return name[:80] or "wechat_article"


def extract_js_var(page_html: str, name: str) -> str:
    m = re.search(rf"\bvar\s+{re.escape(name)}\s*=\s*([^;]+);", page_html)
    if not m:
        return ""
    expr = m.group(1).strip()
    q = re.search(r'"([^"]*)"|\'([^\']*)\'', expr)
    if q:
        return q.group(1) or q.group(2) or ""
    n = re.search(r"-?\d+", expr)
    return n.group(0) if n else ""


def extract_js_number(page_html: str, name: str) -> int:
    m = re.search(rf"(?:var|let)\s+{re.escape(name)}\s*=\s*(\d+)", page_html)
    return int(m.group(1)) if m else 0


def parse_album_id(page_html: str) -> str:
    for p in [
        r"album_id=(\d{8,})",
        r"\balbumId\s*[:=]\s*['\"](\d{8,})['\"]",
        r"\balbum_id_str\s*[:=]\s*['\"](\d{8,})['\"]",
    ]:
        m = re.search(p, page_html)
        if m:
            return m.group(1)
    return ""


def _looks_like_wechat_client_only(page_html: str) -> bool:
    lower_html = page_html.lower()
    return (
        "请在微信客户端打开链接" in page_html
        or "please open in wechat client" in lower_html
        or "<title>验证</title>" in page_html
        or "<title>verify</title>" in lower_html
    )


class RequestClient:
    def __init__(
        self,
        timeout: int,
        max_retries: int,
        user_agent: str,
        session: requests.Session | None = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.session = session or requests.Session()

    def request(
        self,
        method: str,
        url: str,
        *,
        referer: str,
        params: dict[str, Any] | None = None,
        data: Any = None,
        ajax: bool = False,
    ) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=mobile_headers(self.user_agent, referer, ajax=ajax),
                    timeout=self.timeout,
                )
                if resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"server_error_{resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                return resp
            except (requests.RequestException, requests.HTTPError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise WechatCrawlerError(f"request_failed: {method} {url} ({last_error})")

    def get_text(
        self,
        url: str,
        *,
        referer: str,
        params: dict[str, Any] | None = None,
        ajax: bool = False,
    ) -> tuple[str, str]:
        r = self.request("GET", url, referer=referer, params=params, ajax=ajax)
        return r.text, r.url

    def get_json(
        self,
        url: str,
        *,
        referer: str,
        params: dict[str, Any] | None = None,
        ajax: bool = False,
    ) -> dict[str, Any]:
        r = self.request("GET", url, referer=referer, params=params, ajax=ajax)
        try:
            return r.json()
        except Exception as exc:
            raise WechatCrawlerError(f"invalid_json_response: GET {url}") from exc

    def post_json(
        self,
        url: str,
        *,
        referer: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        r = self.request(
            "POST", url, referer=referer, params=params, data=data, ajax=True
        )
        try:
            return r.json()
        except Exception as exc:
            raise WechatCrawlerError(f"invalid_json_response: POST {url}") from exc

    def download_bytes(self, url: str, *, referer: str) -> bytes:
        return self.request("GET", url, referer=referer).content


class WechatCrawler:
    def __init__(self, options: CrawlOptions):
        self.options = options
        self.client = RequestClient(
            options.timeout, options.max_retries, options.user_agent, options.session
        )

    def fetch_article_context(self) -> ArticleContext:
        article_url = ensure_mobile_article_url(self.options.article_url)
        page_html, final_url = self.client.get_text(
            article_url, referer="https://mp.weixin.qq.com/"
        )
        if "wappoc_appmsgcaptcha" in final_url:
            raise WechatCaptchaError(
                "hit wechat captcha page; retry later or use different network/session"
            )
        soup = BeautifulSoup(page_html, "html.parser")
        content_node = soup.select_one("#js_content")
        image_urls: list[str] = []
        content_text = ""
        if content_node:
            content_text = content_node.get_text("\n", strip=True)
            for img in content_node.select("img"):
                src = normalize_url(img.get("data-src") or img.get("src"))
                if src and src not in image_urls:
                    image_urls.append(src)
        ct = extract_js_var(page_html, "ct")
        publish_time = (
            datetime.fromtimestamp(int(ct)).strftime("%Y-%m-%d %H:%M:%S")
            if ct.isdigit()
            else first_text(soup, ["#publish_time", "#js_publish_time"])
        )
        return ArticleContext(
            title=first_text(soup, ["#activity-name", ".rich_media_title", "h1"]),
            author=first_text(
                soup, ["#js_name", ".rich_media_meta.rich_media_meta_text"]
            ),
            publish_time=publish_time,
            content_text=content_text,
            image_urls=image_urls,
            url=final_url,
            biz=extract_js_var(page_html, "biz"),
            mid=extract_js_var(page_html, "mid"),
            sn=extract_js_var(page_html, "sn"),
            idx=extract_js_var(page_html, "idx") or "1",
            ct=ct,
            appmsg_type=extract_js_var(page_html, "appmsg_type") or "9",
            msg_daily_idx=extract_js_var(page_html, "msg_daily_idx") or "1",
            comment_id=extract_js_var(page_html, "comment_id"),
            segment_comment_id=extract_js_var(page_html, "segment_comment_id"),
            album_id=parse_album_id(page_html),
        )

    def call_getappmsgext(self, context: ArticleContext) -> dict[str, Any]:
        api = "https://mp.weixin.qq.com/mp/getappmsgext"
        params = {"f": "json", "__biz": context.biz, "wxtoken": "777", "x5": "0"}
        payload = {
            "r": str(random.random()),
            "__biz": context.biz,
            "appmsg_type": context.appmsg_type,
            "mid": context.mid,
            "sn": context.sn,
            "idx": context.idx,
            "title": context.title,
            "ct": context.ct,
            "msg_daily_idx": context.msg_daily_idx,
            "comment_id": context.comment_id,
            "segment_comment_id": context.segment_comment_id,
            "cur_album_id": context.album_id,
        }
        return self.client.post_json(
            api, referer=context.url, params=params, data=payload
        )

    def _extract_articles_from_general_msg_list(
        self,
        msg_obj: dict[str, Any],
        seen_urls: set[str],
        out: list[LatestArticle],
        limit: int,
    ) -> None:
        for item in msg_obj.get("list", []):
            ext = item.get("app_msg_ext_info") or {}
            comm = item.get("comm_msg_info") or {}
            create_time = str(comm.get("datetime", ""))
            if ext:
                url = normalize_url(ext.get("content_url", ""))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    out.append(
                        LatestArticle(
                            title=ext.get("title", ""), url=url, create_time=create_time
                        )
                    )
                    if len(out) >= limit:
                        return
            for sub in ext.get("multi_app_msg_item_list", []) or []:
                url = normalize_url(sub.get("content_url", ""))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    out.append(
                        LatestArticle(
                            title=sub.get("title", ""), url=url, create_time=create_time
                        )
                    )
                    if len(out) >= limit:
                        return

    def fetch_account_latest_articles(
        self, context: ArticleContext, limit: int
    ) -> tuple[list[LatestArticle], dict[str, str]]:
        if not context.biz:
            return [], {"error": "missing_biz"}
        profile_url = f"https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz={context.biz}&scene=124#wechat_redirect"
        page_html, _ = self.client.get_text(profile_url, referer=context.url)
        if _looks_like_wechat_client_only(page_html):
            return [], {"error": "wechat_client_required", "profile_url": profile_url}

        results: list[LatestArticle] = []
        seen_urls: set[str] = set()
        # Primary strategy: always start from offset=0 to fetch newest feed first.
        can_continue = 1
        next_offset = 0
        page_count = 0
        getmsg_ret = 0
        while len(results) < limit and can_continue == 1:
            page_count += 1
            if page_count > 80:
                break
            data = self.client.get_json(
                "https://mp.weixin.qq.com/mp/profile_ext",
                referer=profile_url,
                params={
                    "action": "getmsg",
                    "__biz": context.biz,
                    "f": "json",
                    "offset": str(max(0, next_offset)),
                    "count": str(min(10, max(1, limit - len(results)))),
                    "is_ok": "1",
                    "scene": "124",
                    "uin": "",
                    "key": "",
                    "pass_ticket": "",
                    "wxtoken": "",
                    "x5": "0",
                },
                ajax=True,
            )
            try:
                getmsg_ret = int(data.get("ret", 0))
            except Exception:
                getmsg_ret = 0
            if getmsg_ret == -3 and not results:
                break
            raw = data.get("general_msg_list", "")
            if not raw:
                break
            try:
                obj = json.loads(raw)
            except Exception:
                break
            self._extract_articles_from_general_msg_list(obj, seen_urls, results, limit)
            can_continue = int(data.get("can_msg_continue", 0))
            next_offset = int(data.get("next_offset", 0))
            if not next_offset:
                break
        if results:
            return sort_latest_articles_desc(results)[:limit], {
                "profile_url": profile_url,
                "strategy": "getmsg_offset0",
            }

        # Fallback strategy: parse initial msgList from profile page if getmsg path yields nothing.
        m = re.search(r"var\s+msgList\s*=\s*'(.+?)';", page_html, re.S)
        if not m:
            err = "no_session" if getmsg_ret == -3 else "msg_list_not_found"
            return [], {"error": err, "profile_url": profile_url}
        try:
            msg_obj = json.loads(html.unescape(m.group(1)))
        except Exception:
            return [], {"error": "msg_list_parse_failed", "profile_url": profile_url}

        self._extract_articles_from_general_msg_list(msg_obj, seen_urls, results, limit)
        can_continue = extract_js_number(page_html, "can_msg_continue")
        next_offset = extract_js_number(page_html, "next_offset")
        page_count = 0
        while len(results) < limit and can_continue == 1:
            page_count += 1
            if page_count > 80:
                break
            data = self.client.get_json(
                "https://mp.weixin.qq.com/mp/profile_ext",
                referer=profile_url,
                params={
                    "action": "getmsg",
                    "__biz": context.biz,
                    "f": "json",
                    "offset": str(max(0, next_offset)),
                    "count": str(min(10, max(1, limit - len(results)))),
                    "is_ok": "1",
                    "scene": "124",
                    "uin": "",
                    "key": "",
                    "pass_ticket": "",
                    "wxtoken": "",
                    "x5": "0",
                },
                ajax=True,
            )
            raw = data.get("general_msg_list", "")
            if not raw:
                break
            try:
                obj = json.loads(raw)
            except Exception:
                break
            self._extract_articles_from_general_msg_list(obj, seen_urls, results, limit)
            can_continue = int(data.get("can_msg_continue", 0))
            next_offset = int(data.get("next_offset", 0))
            if not next_offset:
                break
        if results:
            return sort_latest_articles_desc(results)[:limit], {
                "profile_url": profile_url,
                "strategy": "page_msgList",
            }
        err = "no_session" if getmsg_ret == -3 else "no_articles"
        return [], {"error": err, "profile_url": profile_url}

    def fetch_album_latest_articles(
        self, context: ArticleContext, limit: int
    ) -> list[LatestArticle]:
        if not context.biz or not context.album_id:
            return []
        results: list[LatestArticle] = []
        seen_urls: set[str] = set()
        begin_msgid, begin_itemidx = "", ""
        for _ in range(120):
            params = {
                "action": "getalbum",
                "__biz": context.biz,
                "album_id": context.album_id,
                "count": "10",
                "f": "json",
            }
            if begin_msgid:
                params["begin_msgid"] = begin_msgid
                params["begin_itemidx"] = begin_itemidx or "1"
            data = self.client.get_json(
                "https://mp.weixin.qq.com/mp/appmsgalbum",
                referer=context.url,
                params=params,
                ajax=True,
            )
            album_resp = data.get("getalbum_resp", {})
            articles = album_resp.get("article_list", [])
            if not articles:
                break
            for a in articles:
                url = normalize_url(a.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    LatestArticle(
                        title=a.get("title", ""),
                        url=url,
                        msgid=str(a.get("msgid", "")),
                        itemidx=str(a.get("itemidx", "")),
                        create_time=str(a.get("create_time", "")),
                    )
                )
            if str(album_resp.get("continue_flag", "0")) != "1":
                break
            last = articles[-1]
            begin_msgid = str(last.get("msgid", ""))
            begin_itemidx = str(last.get("itemidx", "1"))
            if not begin_msgid:
                break
        return sort_latest_articles_desc(results)[:limit]

    def download_images(
        self, image_urls: list[str], referer_url: str, images_dir: str
    ) -> list[str]:
        if not self.options.download_images or not image_urls:
            return []
        os.makedirs(images_dir, exist_ok=True)
        out: list[str] = []
        for i, img_url in enumerate(image_urls):
            ext = ".jpg"
            try:
                wx_fmt = parse_qs(urlparse(img_url).query).get("wx_fmt", [""])[0]
                if wx_fmt:
                    ext = "." + wx_fmt
            except Exception:
                pass
            name = f"img_{i}{ext}"
            abs_path = os.path.join(images_dir, name)
            rel_path = os.path.join(os.path.basename(images_dir), name)
            try:
                with open(abs_path, "wb") as f:
                    f.write(self.client.download_bytes(img_url, referer=referer_url))
                out.append(rel_path)
            except Exception:
                continue
        return out

    @staticmethod
    def build_markdown(
        article: ArticleContext,
        local_images: list[str],
        latest_articles: list[LatestArticle],
        latest_label: str,
    ) -> str:
        lines = [f"# {article.title}", ""]
        if article.author:
            lines.append(f"- Author: {article.author}")
        if article.publish_time:
            lines.append(f"- Publish Time: {article.publish_time}")
        lines.extend(
            [
                f"- Source: {article.url}",
                "",
                "---",
                "",
                "## Content",
                "",
                article.content_text,
                "",
            ]
        )
        if local_images:
            lines.extend(["## Images", ""])
            lines.extend([f"![image_{i}]({p})" for i, p in enumerate(local_images)])
            lines.append("")
        if latest_articles:
            lines.extend([f"## {latest_label}", ""])
            lines.extend(
                [f"{i}. [{a.title}]({a.url})" for i, a in enumerate(latest_articles, 1)]
            )
            lines.append("")
        return "\n".join(lines)

    def crawl(self) -> CrawlResult:
        os.makedirs(self.options.output_dir, exist_ok=True)
        context = self.fetch_article_context()
        try:
            ext_data = self.call_getappmsgext(context)
        except Exception as exc:
            ext_data = {"error": str(exc)}
        latest_articles: list[LatestArticle] = []
        latest_label = "Latest Articles"
        account_meta: dict[str, str] = {}
        account_error = ""
        if self.options.latest_scope in ("auto", "account"):
            latest_label = "Latest Account Articles"
            latest_articles, account_meta = self.fetch_account_latest_articles(
                context, self.options.limit
            )
            if not latest_articles and self.options.latest_scope == "account":
                account_error = account_meta.get("error", "account_fetch_failed")
        if not latest_articles and self.options.latest_scope in ("auto", "album"):
            latest_articles = self.fetch_album_latest_articles(
                context, self.options.limit
            )
            latest_label = "Latest Album Articles"
        latest_articles = sort_latest_articles_desc(latest_articles)[
            : self.options.limit
        ]
        local_images = self.download_images(
            context.image_urls,
            context.url,
            os.path.join(self.options.output_dir, "images"),
        )
        safe_title = sanitize_filename(context.title or "wechat_article")
        md_path = os.path.join(self.options.output_dir, safe_title + ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(
                self.build_markdown(
                    context, local_images, latest_articles, latest_label
                )
            )
        latest_json_path = os.path.join(
            self.options.output_dir, safe_title + "_latest_articles.json"
        )
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(i) for i in latest_articles], f, ensure_ascii=False, indent=2
            )
        api_debug_path = os.path.join(
            self.options.output_dir, safe_title + "_api_debug.json"
        )
        with open(api_debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "runtime": {
                        "latest_scope": self.options.latest_scope,
                        "timeout": self.options.timeout,
                        "max_retries": self.options.max_retries,
                    },
                    "article_context": asdict(context),
                    "account_fetch": {
                        "mode": self.options.latest_scope,
                        "profile_url": account_meta.get("profile_url", ""),
                        "error": account_error or account_meta.get("error", ""),
                    },
                    "ext_response_keys": list(ext_data.keys())
                    if isinstance(ext_data, dict)
                    else [],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return CrawlResult(
            context,
            latest_articles,
            latest_label,
            account_error or account_meta.get("error", ""),
            ext_data if isinstance(ext_data, dict) else {},
            local_images,
            md_path,
            latest_json_path,
            api_debug_path,
        )


def fetch_wechat_article(
    article_url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    user_agent: str = DEFAULT_WECHAT_MOBILE_UA,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    opts = CrawlOptions(
        article_url=article_url,
        limit=1,
        download_images=False,
        timeout=max(5, int(timeout or DEFAULT_TIMEOUT)),
        max_retries=max(0, int(max_retries or DEFAULT_MAX_RETRIES)),
        user_agent=user_agent or DEFAULT_WECHAT_MOBILE_UA,
        session=session,
    )
    return asdict(WechatCrawler(opts).fetch_article_context())


def fetch_latest_articles(
    article_url: str,
    limit: int = 5,
    latest_scope: str = "auto",
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    user_agent: str = DEFAULT_WECHAT_MOBILE_UA,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    opts = CrawlOptions(
        article_url=article_url,
        limit=max(1, int(limit or 1)),
        latest_scope=_normalize_scope(latest_scope),
        download_images=False,
        timeout=max(5, int(timeout or DEFAULT_TIMEOUT)),
        max_retries=max(0, int(max_retries or DEFAULT_MAX_RETRIES)),
        user_agent=user_agent or DEFAULT_WECHAT_MOBILE_UA,
        session=session,
    )
    crawler = WechatCrawler(opts)
    context = crawler.fetch_article_context()
    rows: list[LatestArticle] = []
    meta: dict[str, str] = {
        "scope": opts.latest_scope,
        "article_url": context.url or article_url,
    }
    if opts.latest_scope in ("auto", "account"):
        rows, account_meta = crawler.fetch_account_latest_articles(context, opts.limit)
        meta.update({k: str(v) for k, v in account_meta.items()})
        if rows:
            meta["scope"] = "account"
    if not rows and opts.latest_scope in ("auto", "album"):
        rows = crawler.fetch_album_latest_articles(context, opts.limit)
        if rows:
            meta["scope"] = "album"
    rows = sort_latest_articles_desc(rows)[: opts.limit]
    return [
        {
            "title": r.title,
            "url": r.url,
            "create_time": r.create_time,
            "msgid": r.msgid,
            "itemidx": r.itemidx,
        }
        for r in rows
        if r.url
    ], meta


def wechat_to_markdown(
    article_url: str,
    output_dir: str = "output",
    *,
    limit: int = 5,
    latest_scope: str = "auto",
    download_images: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    user_agent: str = DEFAULT_WECHAT_MOBILE_UA,
    session: requests.Session | None = None,
) -> str:
    opts = CrawlOptions(
        article_url=article_url,
        output_dir=output_dir or "output",
        limit=max(1, int(limit or 1)),
        latest_scope=_normalize_scope(latest_scope),
        download_images=bool(download_images),
        timeout=max(5, int(timeout or DEFAULT_TIMEOUT)),
        max_retries=max(0, int(max_retries or DEFAULT_MAX_RETRIES)),
        user_agent=user_agent or DEFAULT_WECHAT_MOBILE_UA,
        session=session,
    )
    return WechatCrawler(opts).crawl().md_path


def run(
    article_url: str,
    output_dir: str = "output",
    *,
    limit: int = 5,
    latest_scope: str = "auto",
    download_images: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    user_agent: str = DEFAULT_WECHAT_MOBILE_UA,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    opts = CrawlOptions(
        article_url=article_url,
        output_dir=output_dir or "output",
        limit=max(1, int(limit or 1)),
        latest_scope=_normalize_scope(latest_scope),
        download_images=bool(download_images),
        timeout=max(5, int(timeout or DEFAULT_TIMEOUT)),
        max_retries=max(0, int(max_retries or DEFAULT_MAX_RETRIES)),
        user_agent=user_agent or DEFAULT_WECHAT_MOBILE_UA,
        session=session,
    )
    r = WechatCrawler(opts).crawl()
    return {
        "title": r.context.title,
        "author": r.context.author,
        "publish_time": r.context.publish_time,
        "content_text": r.context.content_text,
        "image_urls": list(r.context.image_urls),
        "url": r.context.url,
        "latest_label": r.latest_label,
        "latest_articles": [asdict(x) for x in r.latest_articles],
        "account_error": r.account_error,
        "md_path": r.md_path,
        "latest_json_path": r.latest_json_path,
        "api_debug_path": r.api_debug_path,
    }


def print_summary(result: CrawlResult, options: CrawlOptions) -> None:
    print("Title:", result.context.title)
    print("Author:", result.context.author)
    print("Publish Time:", result.context.publish_time)
    print("Article URL:", result.context.url)
    print("Images Found:", len(result.context.image_urls))
    print("Images Downloaded:", len(result.downloaded_images))
    print("Album ID:", result.context.album_id)
    print("Latest Scope:", options.latest_scope)
    print("Latest Label:", result.latest_label)
    print("Latest Articles:", len(result.latest_articles))
    if result.account_error:
        print("Account Fetch Error:", result.account_error)
    print("Markdown:", result.md_path)
    print("Latest JSON:", result.latest_json_path)
    print("API Debug:", result.api_debug_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeChat public-account crawler without Playwright (article + account/album APIs)."
    )
    parser.add_argument("article_url", help="WeChat article URL")
    parser.add_argument("--output-dir", default="output", help="output directory")
    parser.add_argument("--limit", type=int, default=5, help="latest articles count")
    parser.add_argument(
        "--latest-scope",
        choices=["auto", "account", "album"],
        default="auto",
        help="latest article source",
    )
    parser.add_argument(
        "--no-download-images",
        action="store_true",
        help="do not download content images",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="request retry count",
    )
    parser.add_argument(
        "--user-agent", default=DEFAULT_WECHAT_MOBILE_UA, help="custom user-agent"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    options = CrawlOptions(
        article_url=args.article_url,
        output_dir=args.output_dir,
        limit=max(1, args.limit),
        latest_scope=_normalize_scope(args.latest_scope),
        download_images=not args.no_download_images,
        timeout=max(5, args.timeout),
        max_retries=max(0, args.max_retries),
        user_agent=args.user_agent,
    )
    try:
        result = WechatCrawler(options).crawl()
    except WechatCaptchaError as exc:
        raise SystemExit(f"CaptchaError: {exc}")
    except WechatCrawlerError as exc:
        raise SystemExit(f"CrawlerError: {exc}")
    print_summary(result, options)


if __name__ == "__main__":
    main()
