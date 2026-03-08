from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


SKLAND_HOME = "https://www.skland.com/"
SKLAND_API_HOST = "https://zonai.skland.com"
SKLAND_ORIGIN = "https://www.skland.com"
SKLAND_PORTAL101_URL = "https://fp-it.portal101.cn/deviceprofile/v4"
SKLAND_ARTICLE_URL = "https://www.skland.com/article?id={item_id}"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "skland_config.json"
PORTAL101_TEMPLATE_PAYLOAD_JSON = r'''{"appId":"default","organization":"UWXspnCCJN4sfYlNfqps","ep":"HenWeH02GuuuKk904A/OhSfKrwHHMoLq5Z5VaZ6qucoLpf+KHPrZEhG79vWkxIvRAJBDZOZmCUtYUjvkNbm0JIAElZFjteUGe8UNZ5cZs7xyvDBJQp4Y08AZS7GsE0xGMW39kDXFQHEpDnwhYMoCBygOXYPrbi8glKsY7TGvAxQ=","data":"243f46e5d86ece6856b659ee4874d2e4f4f645764508ed030af594b3690f4275601c70379bf14a791c5a5351aa2e7284c375f65ac6a91d3529f35bf065772fe0d8503f530e63c8192ab9609fac2fc0efd8d829ee6e191a3b32c8202774aa5a4e13d052f15a968448ce54f06f6e20f96377cc5e655ded17665e22d2ac6e5629f209368acaeed043af62520e6b3c1c4dd689bd21a85ec18f4956c3c8aaf6409e156402c8b92faaa16f5df8fd7cf3e06626805f0a70ce2b7baa02a25584663696aad23d041eb68b39a177b55a60e03df03cedf40864b3ab6c2863627da1944b837e47db235b0e2320449857fb7325ecd0d6f1ad49cb6fbaf056364f560db2bed997c1bda921d5818bec6c22d9d7c402cdc2f3b1552e87c97b00e043fa2afa5e67bf43d03a78ff0ac3eb54b34400f9fa54be377ecc3ed0fcba5f936b53419f61b3431e31c1966b4b453af231d0754d9c6ad49129611da24326f4d96237363e2e7326af29d1eeef18f61364bd40faf56276e4dc3751de61afe11ecdf31076f79057f875f409cf280cbb992ef255b8d1a2a3688e40853182da7a9fd8d1d8f4b46373d2a0ade07927fdb59648beb8b055a2748cdd984a01161434e4042bc11cd1506fa3ddc0f7fe6b24565591bfd52afa4bd5517d924eb82cfeaf724c560072c591339325e43f68d3b2ec20b62d3b3c3de2bc97427e3fed83133e8eefe3fcb4d79eacdc51e97f974045cca8f26d90c503840e0994bd3d745cac270a867386b3405aea2c7a5db2a56765fe3aac0eeb12d73316706cbe71df61b6d1df2071bc45c4b64d0fa21357d13d30dc2c7d78763073f2be9ee4385fe4dfaa3c5629494db57c3dc975832d72bc6c44a953844de65d66179b45662b8dd5e233bfe27266dd497b3cd8d496186de39d25979c1c1fa5e181d16a4a9ed1e1f95e85539246033e32654d0102ed3081519c3cdaf6c42935362cd39de10ccfb232681306b75683d9de22e2bef73def96bc4337ff65b8978e09494c77319460bad8e1da361250fed77de890ccdfb31bad255c975c77b9332a221cc0af6d2fb5fb5763595258adbc987d923cdbabb1e020102f220085b097e0b89f3aa007944b3940b5684585db242c876da2f4425f29883fe07649655c22566af82d48989f00a25640b269bf660ef4ec2e8ea218e033349718a221d662359636e9351f93a962f603e308059a1c69f138fab270e397588b2c36320f03ad9fcaeac28365d79a7a6ecce41767149ef5ba9a816f7a8455dd358bad0c4aeba01b15580a4ea0bea581c9f16d6dfa27918f49ee5e72e12017b284f441ea6de42f196f621a48e35c8fcfb6b00c7f4908ac12781839e9caf28c421658a81914845a9afd777e90474bb987ab75f39ed1300b92e89a944e7279e252d7969ed5634919ecb657c47c8ced9dc52866ab5a90455271a452ebdccbbf39d8fb0de4ec87398b8d640d1610863b8fcebac8bd4712962650452ef5cc23d74599b4e44a835ed36adf807fc0274c9b9f2f2341c1bf3da4a899a7d80a82ece9d878841b447abff3bf816f6e314bfd096bfccd1f718f7da9c7627c41b5e0ebec863f59368d949883beba8b12886297aac88309d5c4a6f09fa70b333c388a63d89dcce3e335eacd5f23548590129c9151d288fcc4b091271c3f70a6911041f7f9135f62ab87a008ccec80976384e302fb565243b14dc46748b3ae47a9e5455635402661dab62b60a9a2502c2914d6772b623ebf5220e09a8ecf9e788398be9b4723ee71fd6491aa9231a319f4b3be504568c9b20c59b839a6620e29f95c4cccaf36e0a1be45ccebe7fe7890836e187ca7212167a175cc1fb5916e77450e5d0b18fd57a0e932f553679f5a9483d7f685d90df2940cdd081882c7d75f41173f4ea5e542a37327a0cd56f477ccfa654ff10798d43cb627cbbc30a36b5e60819552891a1490a1f27afd3c786714748de1d64ac75c9072a9654db29dd7c7a756e529210aaf0bdd1c1333529a672c7c74fd1977c9d53d3675152b3800e8138233890653a3c05bbd650e3e158fdfc7f6b18c779a38aed49d242c1cea89449eb81b9189cf","os":"web","encode":5,"compress":2}'''


@dataclass
class SignContext:
    token: str
    client_time: str
    server_time: str
    d_id: str
    monotonic_time: float = 0.0


class SklandClient:
    def __init__(
        self,
        *,
        d_id: str | None = None,
        thumbcache: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Origin": SKLAND_ORIGIN,
                "Referer": SKLAND_HOME,
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/146.0.0.0 Safari/537.36"
                ),
            }
        )
        self._manual_d_id = self._resolve_manual_d_id(d_id, thumbcache)
        self.sign_context = SignContext(
            token="",
            client_time="",
            server_time="",
            d_id=self._manual_d_id or "",
            monotonic_time=0.0,
        )

    @staticmethod
    def _load_config() -> dict[str, Any]:
        if not CONFIG_PATH.exists():
            return {}
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _resolve_manual_d_id(
        self, d_id: str | None, thumbcache: str | None
    ) -> str | None:
        config = self._load_config()
        final_d_id = d_id or os.getenv("SKLAND_D_ID") or config.get("d_id")
        if final_d_id:
            return final_d_id.strip()

        thumbcache_value = (
            thumbcache or os.getenv("SKLAND_THUMBCACHE") or config.get("thumbcache")
        )
        if thumbcache_value:
            return f"B{thumbcache_value.strip()}"
        return None

    def _load_portal_payload(self) -> dict[str, Any]:
        config = self._load_config()
        payload_text = os.getenv("SKLAND_PORTAL_PAYLOAD_JSON") or config.get(
            "portal_payload_json"
        )
        if payload_text:
            payload = json.loads(payload_text)
        else:
            payload = json.loads(PORTAL101_TEMPLATE_PAYLOAD_JSON.replace('\\"', '"'))
        return copy.deepcopy(payload)

    def bootstrap_stateless_d_id(self) -> str:
        payload = self._load_portal_payload()
        response = self.session.post(
            SKLAND_PORTAL101_URL,
            json=payload,
            headers={"Content-Type": "application/json;charset=utf-8"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        portal_payload = response.json()
        if portal_payload.get("code") != 1100:
            raise RuntimeError(f"portal101 bootstrap failed: {portal_payload}")
        detail = portal_payload.get("detail") or {}
        device_id = detail.get("deviceId")
        if not device_id:
            raise RuntimeError(f"portal101 did not return deviceId: {portal_payload}")
        return f"B{device_id}"

    def ensure_d_id(self, *, refresh: bool = False) -> str:
        if refresh or not self.sign_context.d_id:
            self.sign_context.d_id = self._manual_d_id or self.bootstrap_stateless_d_id()
        return self.sign_context.d_id

    @staticmethod
    def make_list_id(length: int = 16) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(alphabet[secrets.randbelow(len(alphabet))] for _ in range(length))

    @staticmethod
    def build_query(params: dict[str, Any]) -> str:
        pairs = [(key, value) for key, value in params.items() if value is not None]
        return urllib.parse.urlencode(pairs, quote_via=urllib.parse.quote)

    @staticmethod
    def sign(raw: str, token: str) -> str:
        sha_hex = hmac.new(
            token.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return hashlib.md5(sha_hex.encode("utf-8")).hexdigest()

    def get_game_catalog(self) -> dict[str, Any]:
        response = self.session.get(f"{SKLAND_API_HOST}/web/v1/game", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def bootstrap_sign_context(self) -> None:
        game_payload = self.get_game_catalog()
        server_time = str(game_payload["timestamp"])
        path = "/web/v1/auth/refresh"
        last_error: dict[str, Any] | None = None

        for attempt in range(3):
            d_id = self.ensure_d_id(refresh=attempt > 0 and not self._manual_d_id)
            fields = {
                "platform": "3",
                "timestamp": server_time,
                "dId": d_id,
                "vName": "1.0.0",
            }
            raw = path + server_time + json.dumps(
                fields, separators=(",", ":"), ensure_ascii=False
            )
            sign_value = self.sign(raw, "")
            response = self.session.get(
                f"{SKLAND_API_HOST}{path}",
                headers={"Content-Type": "application/json", **fields, "sign": sign_value},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                try:
                    body = response.json()
                except Exception:
                    body = response.text
                raise RuntimeError(f"HTTP {response.status_code} for {path}: {body}")

            payload = response.json()
            if payload.get("code") == 0:
                self.sign_context = SignContext(
                    token=payload["data"]["token"],
                    client_time=server_time,
                    server_time=str(payload["timestamp"]),
                    d_id=d_id,
                    monotonic_time=time.monotonic(),
                )
                return
            last_error = payload
            if payload.get("code") != 10001 or self._manual_d_id:
                break

        raise RuntimeError(f"bootstrap refresh failed: {last_error}")

    def _ensure_sign_context(self) -> None:
        if (
            self.sign_context.token
            and self.sign_context.client_time
            and self.sign_context.server_time
        ):
            return
        self.bootstrap_sign_context()

    def corrected_timestamp(self) -> str:
        self._ensure_sign_context()
        elapsed = max(0, int(time.monotonic() - self.sign_context.monotonic_time))
        return str(int(self.sign_context.server_time) + elapsed)

    def _send_signed_get(self, path: str, params: dict[str, Any]) -> requests.Response:
        query = self.build_query(params)
        timestamp = self.corrected_timestamp()
        sign_fields = {
            "platform": "3",
            "timestamp": timestamp,
            "dId": self.ensure_d_id(),
            "vName": "1.0.0",
        }
        raw = path + query + timestamp + json.dumps(
            sign_fields, separators=(",", ":"), ensure_ascii=False
        )
        headers = {
            "Content-Type": "application/json",
            **sign_fields,
            "sign": self.sign(raw, self.sign_context.token),
        }
        url = f"{SKLAND_API_HOST}{path}"
        return self.session.get(url, params=params, headers=headers, timeout=self.timeout)

    def signed_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_sign_context()

        for attempt in range(3):
            response = self._send_signed_get(path, params)
            if response.status_code == 401 and attempt < 2:
                if not self._manual_d_id:
                    self.ensure_d_id(refresh=True)
                self.bootstrap_sign_context()
                continue

            if response.status_code >= 400:
                try:
                    body = response.json()
                except Exception:
                    body = response.text
                raise RuntimeError(f"HTTP {response.status_code} for {path}: {body}")

            payload = response.json()
            if payload.get("code") == 10000 and attempt < 2:
                self.bootstrap_sign_context()
                continue

            if payload.get("code") == 10001 and attempt < 2 and not self._manual_d_id:
                self.ensure_d_id(refresh=True)
                self.bootstrap_sign_context()
                continue

            return payload

        raise RuntimeError(
            f"signed_get failed after retries: path={path}, params={params}"
        )

    def search_posts(
        self,
        keyword: str,
        *,
        page_size: int = 10,
        page_token: str | None = None,
        sort_type: int = 1,
        view_kind: int = 0,
        list_id: str | None = None,
    ) -> dict[str, Any]:
        params = {
            "keyword": keyword,
            "pageSize": page_size,
            "viewKind": view_kind,
            "sortType": sort_type,
            "listId": list_id or self.make_list_id(),
            "pageToken": page_token,
        }
        return self.signed_get("/web/v1/search/item", params)

    def get_home_feed(
        self,
        *,
        game_id: int,
        cate_id: int,
        page_size: int = 5,
        page_token: str | None = None,
        sort_type: int = 2,
        list_id: str | None = None,
    ) -> dict[str, Any]:
        page_size = min(max(int(page_size), 1), 5)
        params = {
            "gameId": game_id,
            "cateId": cate_id,
            "pageSize": page_size,
            "sortType": sort_type,
            "listId": list_id or self.make_list_id(),
            "pageToken": page_token,
        }
        return self.signed_get("/web/v1/home/index", params)

    def get_item_detail(self, item_id: int | str) -> dict[str, Any]:
        return self.signed_get("/web/v1/item", {"id": str(item_id)})

    @staticmethod
    def extract_item_id(value: int | str) -> str:
        text = str(value).strip()
        if text.isdigit():
            return text
        parsed = urllib.parse.urlparse(text)
        query_id = urllib.parse.parse_qs(parsed.query).get("id")
        if query_id and query_id[0].isdigit():
            return query_id[0]
        match = re.search(r"(?:id=|/article/)(\d+)", text)
        if match:
            return match.group(1)
        raise ValueError(f"cannot extract item id from: {value}")

    @classmethod
    def article_url(cls, item_id_or_url: int | str) -> str:
        text = str(item_id_or_url).strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return SKLAND_ARTICLE_URL.format(item_id=cls.extract_item_id(text))

    @staticmethod
    def safe_filename(value: str) -> str:
        text = re.sub(r'[\\/:*?"<>|]+', '_', value).strip()
        return text[:120] or "article"

    @staticmethod
    def _pick_image_url(image: dict[str, Any]) -> str:
        if image.get("url"):
            return image["url"]
        for info in image.get("displayInfos") or []:
            if info.get("style") == "origin" and info.get("url"):
                return info["url"]
        for info in image.get("displayInfos") or []:
            if info.get("url"):
                return info["url"]
        return ""

    @staticmethod
    def _render_inline_content(
        content: dict[str, Any],
        text_map: dict[str, str],
        link_map: dict[str, dict[str, Any]],
        at_map: dict[str, dict[str, Any]],
        bv_map: dict[str, Any],
    ) -> str:
        content_type = content.get("type")
        if content_type == "text":
            return text_map.get(str(content.get("contentId")), "")
        if content_type == "link":
            link = link_map.get(str(content.get("contentId")), {})
            url = link.get("url") or link.get("link") or ""
            label = link.get("title") or link.get("text") or link.get("display") or url
            return f"[{label}]({url})" if url else label
        if content_type == "at":
            at_info = at_map.get(str(content.get("contentId")), {})
            name = (
                at_info.get("name")
                or at_info.get("nickName")
                or at_info.get("nickname")
                or content.get("name")
                or "user"
            )
            return f"@{name}"
        if content_type == "bv":
            bv_info = bv_map.get(str(content.get("contentId")), {})
            url = bv_info.get("url") or ""
            title = bv_info.get("title") or url or "video"
            return f"[{title}]({url})" if url else title
        return content.get("text") or ""

    def parse_item_detail(self, item_id_or_url: int | str) -> dict[str, Any]:
        item_id = self.extract_item_id(item_id_or_url)
        payload = self.get_item_detail(item_id)
        if payload.get("code") != 0:
            raise RuntimeError(f"item detail failed: {payload}")

        data = payload.get("data") or {}
        item = data.get("item") or {}
        user = data.get("user") or {}
        tags = data.get("tags") or []

        fmt = item.get("format") or "{}"
        try:
            format_data = json.loads(fmt)
        except Exception:
            format_data = {"version": 1, "data": []}

        text_map = {
            str(entry.get("id")): entry.get("c") or entry.get("content") or ""
            for entry in (item.get("textSlice") or [])
        }
        link_map = {str(entry.get("id")): entry for entry in (item.get("linkSlice") or [])}
        at_map = {str(entry.get("id")): entry for entry in (item.get("atSlice") or [])}
        bv_map = {str(entry.get("id")): entry for entry in (item.get("bvSlice") or [])}
        image_map = {
            str(entry.get("id")): entry for entry in (item.get("imageListSlice") or [])
        }

        blocks: list[str] = []
        image_urls: list[str] = []
        for block in format_data.get("data") or []:
            block_type = block.get("type")
            if block_type == "paragraph":
                parts = [
                    self._render_inline_content(content, text_map, link_map, at_map, bv_map)
                    for content in (block.get("contents") or [])
                ]
                text = "".join(parts).strip()
                blocks.append(text)
                continue
            if block_type == "image":
                image = image_map.get(str(block.get("imageId")), {})
                url = self._pick_image_url(image)
                alt = image.get("description") or "image"
                if url:
                    image_urls.append(url)
                blocks.append(f"![{alt}]({url})" if url else "")
                continue
            if block_type in {"divider", "hr"}:
                blocks.append("---")
                continue
            if block_type == "heading":
                parts = [
                    self._render_inline_content(content, text_map, link_map, at_map, bv_map)
                    for content in (block.get("contents") or [])
                ]
                text = "".join(parts).strip()
                blocks.append(f"## {text}" if text else "")
                continue
            if block_type in {"unorderedList", "orderedList"}:
                for row in block.get("items") or []:
                    parts = [
                        self._render_inline_content(content, text_map, link_map, at_map, bv_map)
                        for content in (row.get("contents") or [])
                    ]
                    text = "".join(parts).strip()
                    if text:
                        blocks.append(f"- {text}")
                continue

        body_markdown = "\n\n".join(part for part in blocks if part).strip()
        body_markdown = re.sub(r"\n{3,}", "\n\n", body_markdown)
        content_text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body_markdown)
        content_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content_text)
        content_text = re.sub(r"^#+\s*", "", content_text, flags=re.M)
        content_text = re.sub(r"^-\s*", "", content_text, flags=re.M)
        content_text = re.sub(r"\s+", " ", content_text).strip()

        article = {
            "item_id": item.get("id") or item_id,
            "url": self.article_url(item_id),
            "article_url": self.article_url(item_id),
            "title": item.get("title") or data.get("title"),
            "game_id": item.get("gameId") or data.get("gameId"),
            "cate_id": item.get("cateId") or data.get("cateId"),
            "published_at_ts": item.get("publishedAtTs")
            or item.get("timestamp")
            or data.get("publishedAtTs"),
            "latest_reply_at_ts": item.get("latestReplyAtTs")
            or data.get("latestReplyAtTs"),
            "user_id": user.get("id") or item.get("userId"),
            "user_name": user.get("name")
            or user.get("nickName")
            or user.get("nickname"),
            "ip_location": item.get("ipLocation")
            or data.get("ipLocation")
            or item.get("firstIpLocation"),
            "tags": [
                tag.get("name")
                for tag in tags
                if isinstance(tag, dict) and tag.get("name")
            ],
            "published_at": ts_to_str(
                item.get("publishedAtTs") or item.get("timestamp") or data.get("publishedAtTs")
            ),
            "content_text": content_text,
            "summary": content_text[:220],
            "image_urls": image_urls,
            "markdown": body_markdown,
            "raw": data,
        }
        article["markdown_full"] = build_article_markdown(article, body_markdown)
        return article


def format_item(entry: dict[str, Any]) -> dict[str, Any]:
    item = entry.get("item") or {}
    user = entry.get("user") or {}
    tags = entry.get("tags") or []
    return {
        "item_id": item.get("id"),
        "title": item.get("title"),
        "game_id": item.get("gameId"),
        "cate_id": item.get("cateId"),
        "view_kind": item.get("viewKind"),
        "published_at_ts": item.get("publishedAtTs") or item.get("timestamp"),
        "latest_reply_at_ts": item.get("latestReplyAtTs"),
        "user_id": user.get("id") or item.get("userId"),
        "user_name": user.get("name") or user.get("nickName") or user.get("nickname"),
        "ip_location": item.get("ipLocation"),
        "tags": [
            tag.get("name")
            for tag in tags
            if isinstance(tag, dict) and tag.get("name")
        ],
        "api_detail_url": (
            f"{SKLAND_API_HOST}/web/v1/item?id={item.get('id')}"
            if item.get("id")
            else None
        ),
        "article_url": (
            SKLAND_ARTICLE_URL.format(item_id=item.get("id"))
            if item.get("id")
            else None
        ),
    }


def format_item_detail(payload: dict[str, Any]) -> dict[str, Any]:
    client = SklandClient()
    item_id = ((payload.get("data") or {}).get("item") or {}).get("id")
    if not item_id:
        return {"raw": payload, "content": ""}
    return client.parse_item_detail(str(item_id))


def build_article_markdown(meta: dict[str, Any], body_markdown: str) -> str:
    lines = [f"# {meta.get('title') or 'Untitled'}", ""]
    if meta.get("item_id"):
        lines.append(f"- item_id: `{meta['item_id']}`")
    if meta.get("url"):
        lines.append(f"- URL: {meta['url']}")
    if meta.get("article_url") and meta.get("article_url") != meta.get("url"):
        lines.append(f"- Page: {meta['article_url']}")
    if meta.get("user_name"):
        lines.append(f"- Author: {meta['user_name']}")
    if meta.get("published_at"):
        lines.append(f"- Published: {meta['published_at']}")
    elif meta.get("published_at_ts"):
        lines.append(f"- Published: {ts_to_str(meta['published_at_ts'])}")
    if meta.get("game_name"):
        lines.append(f"- Game: {meta['game_name']}")
    elif meta.get("game_id"):
        lines.append(f"- game_id: {meta['game_id']}")
    if meta.get("cate_name"):
        lines.append(f"- Category: {meta['cate_name']}")
    elif meta.get("cate_id"):
        lines.append(f"- cate_id: {meta['cate_id']}")
    if meta.get("tags"):
        lines.append(f"- Tags: {', '.join(meta['tags'])}")
    lines.append("")
    lines.append(body_markdown.strip() or "[No content parsed]")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def ts_to_str(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def rewrite_markdown_images_to_local(
    markdown: str,
    md_path: Path,
    session: requests.Session,
    timeout: int = 60,
) -> str:
    md_path = Path(md_path)
    asset_dir = md_path.parent / f"{md_path.stem}_assets"
    pattern = re.compile(r'!\[([^\]]*)\]\((https?://[^)]+)\)')
    matches = list(pattern.finditer(markdown))
    if not matches:
        return markdown

    asset_dir.mkdir(parents=True, exist_ok=True)
    rewritten = markdown
    replacements: dict[str, str] = {}

    for index, match in enumerate(matches, start=1):
        alt_text = match.group(1)
        url = match.group(2)
        if url in replacements:
            local_rel = replacements[url]
            rewritten = rewritten.replace(
                f"![{alt_text}]({url})", f"![{alt_text}]({local_rel})"
            )
            continue

        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        parsed = urllib.parse.urlparse(url)
        suffix = Path(parsed.path).suffix or '.bin'
        file_name = f"img_{index:02d}{suffix}"
        file_path = asset_dir / file_name
        file_path.write_bytes(response.content)
        local_rel = file_path.relative_to(md_path.parent).as_posix()
        replacements[url] = local_rel
        rewritten = rewritten.replace(
            f"![{alt_text}]({url})", f"![{alt_text}]({local_rel})"
        )

    return rewritten
