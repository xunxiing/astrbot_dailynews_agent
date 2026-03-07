from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


_BEARER_TOKEN = (
    "AAAAAAAAAAAAAAAAAAAAAFXzAwAAAAAAMHCxpeSDG1gLNLghVe8d74hl6k4%3D"
    "RUMF4xAQLsbeBhTSRrCiQpJtxoGWeyHrDb5te2jpGskWDFW82F"
)
_API_BASES = ("https://api.x.com", "https://api.twitter.com")
_DEFAULT_QUERY_IDS = {
    "UserByScreenName": "pLsOiyHJ1eFwPJlNmLp4Bg",
    "UserTweets": "_9v58axugmURcAmrOi7nxw",
}
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_RESERVED_SEGMENTS = {
    "home",
    "explore",
    "search",
    "settings",
    "notifications",
    "messages",
    "i",
    "compose",
    "share",
    "intent",
}
_SCREEN_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,20}$")
_MAIN_JS_RE = re.compile(
    r"https://abs\.twimg\.com/responsive-web/client-web/main\.[^\"']+\.js"
)


def _normalize_proxy(proxy: str) -> str | None:
    s = (proxy or "").strip()
    if not s:
        return None
    parsed = urlparse(s)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(
            "twitter_proxy only supports http/https in API mode (e.g. http://127.0.0.1:59549)"
        )
    return s


def _is_proxy_like(s: str) -> bool:
    raw = (s or "").strip()
    if not raw:
        return False
    try:
        u = urlparse(raw)
        host = (u.hostname or "").lower()
        scheme = (u.scheme or "").lower()
        if scheme in {"socks5", "socks5h"}:
            return True
        if scheme in {"http", "https"} and host in {"localhost", "127.0.0.1"}:
            return True
        if scheme in {"http", "https"} and host.startswith("127."):
            return True
    except Exception:
        return False
    return False


def _is_twitter_url(s: str) -> bool:
    raw = (s or "").strip()
    if not raw:
        return False
    try:
        u = urlparse(raw)
        host = (u.netloc or "").split(":")[0].lower()
        if host.startswith("www."):
            host = host[4:]
        return host in {"x.com", "twitter.com", "mobile.twitter.com", "m.twitter.com"}
    except Exception:
        return False


def _extract_screen_name(target_url: str) -> str:
    u = urlparse((target_url or "").strip())
    segs = [p for p in (u.path or "").split("/") if p]
    if not segs:
        raise ValueError(
            "target_url must include a profile name, e.g. https://x.com/openai"
        )
    first = segs[0].strip().lstrip("@")
    if not first:
        raise ValueError("invalid X/Twitter profile url")
    if first.lower() in _RESERVED_SEGMENTS:
        raise ValueError("target_url must point to a profile page, not a system route")
    if not _SCREEN_NAME_RE.match(first):
        raise ValueError(f"invalid X/Twitter screen_name: {first}")
    return first


def _auth_headers(guest_token: str | None = None) -> dict[str, str]:
    h = {
        "authorization": f"Bearer {_BEARER_TOKEN}",
        "user-agent": _UA,
        "accept": "application/json, text/plain, */*",
        "x-twitter-active-user": "yes",
        "x-twitter-client-language": "en",
    }
    if guest_token:
        h["x-guest-token"] = guest_token
    return h


def _session_with_proxy(proxy: str | None) -> requests.Session:
    s = requests.Session()
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    return s


def _request_text(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    params: dict[str, Any] | None = None,
) -> tuple[int, str]:
    resp = session.request(
        method.upper(),
        url,
        headers=headers,
        params=params,
        timeout=timeout_s,
    )
    return resp.status_code, resp.text


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    params: dict[str, Any] | None = None,
) -> Any:
    code, text = _request_text(
        session, method, url, headers=headers, timeout_s=timeout_s, params=params
    )
    if code != 200:
        raise RuntimeError(f"{method.upper()} {url} -> {code}: {text[:220]}")
    try:
        return json.loads(text)
    except Exception:
        return {}


def _activate_guest_token(
    session: requests.Session,
    *,
    timeout_s: float,
) -> str:
    last_err: Exception | None = None
    for base in _API_BASES:
        url = f"{base}/1.1/guest/activate.json"
        try:
            data = _request_json(
                session,
                "POST",
                url,
                headers=_auth_headers(),
                timeout_s=timeout_s,
            )
            token = str((data or {}).get("guest_token") or "").strip()
            if token:
                return token
            last_err = RuntimeError(f"guest_token missing from {url}")
        except Exception as e:
            last_err = e if isinstance(e, Exception) else Exception(str(e))
    raise RuntimeError(str(last_err) if last_err else "guest token activation failed")


def _extract_query_id(script_text: str, operation_name: str) -> str:
    pattern = (
        r'queryId:"([^"]+)",operationName:"'
        + re.escape(operation_name)
        + r'"'
    )
    m = re.search(pattern, script_text)
    return (m.group(1) if m else "").strip()


def _discover_query_ids(
    session: requests.Session,
    *,
    target_url: str,
    timeout_s: float,
) -> dict[str, str]:
    code, html = _request_text(
        session,
        "GET",
        target_url,
        headers={"user-agent": _UA},
        timeout_s=timeout_s,
    )
    if code != 200 or not html:
        return dict(_DEFAULT_QUERY_IDS)

    script_urls = list(dict.fromkeys(_MAIN_JS_RE.findall(html)))
    if not script_urls:
        return dict(_DEFAULT_QUERY_IDS)

    js_code, js_text = _request_text(
        session,
        "GET",
        script_urls[0],
        headers={"user-agent": _UA},
        timeout_s=timeout_s,
    )
    if js_code != 200 or not js_text:
        return dict(_DEFAULT_QUERY_IDS)

    out = dict(_DEFAULT_QUERY_IDS)
    for op in ("UserByScreenName", "UserTweets"):
        qid = _extract_query_id(js_text, op)
        if qid:
            out[op] = qid
    return out


def _dig_first(node: Any, *path: str) -> Any:
    cur = node
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_user_id(user_by_name_data: dict[str, Any]) -> str:
    user_result = _dig_first(user_by_name_data, "data", "user", "result")
    if not isinstance(user_result, dict):
        return ""
    return str(user_result.get("rest_id") or "").strip()


def _iter_tweet_holders(node: Any):
    if isinstance(node, dict):
        if isinstance(node.get("tweet_results"), dict):
            yield node
        for v in node.values():
            yield from _iter_tweet_holders(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_tweet_holders(item)


def _unwrap_tweet_result(raw: Any) -> dict[str, Any] | None:
    node = raw if isinstance(raw, dict) else None
    if not node:
        return None
    if node.get("__typename") == "TweetWithVisibilityResults":
        node = node.get("tweet") if isinstance(node.get("tweet"), dict) else None
    if not isinstance(node, dict):
        return None
    if not isinstance(node.get("legacy"), dict):
        return None
    return node


def _tweet_text(tweet_obj: dict[str, Any]) -> str:
    legacy = tweet_obj.get("legacy") if isinstance(tweet_obj.get("legacy"), dict) else {}
    note_text = _dig_first(
        tweet_obj, "note_tweet", "note_tweet_results", "result", "text"
    )
    if isinstance(note_text, str) and note_text.strip():
        return note_text.strip()
    return str(legacy.get("full_text") or legacy.get("text") or "").strip()


def _tweet_images(tweet_obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    legacy = tweet_obj.get("legacy") if isinstance(tweet_obj.get("legacy"), dict) else {}

    for key in ("extended_entities", "entities"):
        bucket = legacy.get(key)
        if not isinstance(bucket, dict):
            continue
        media = bucket.get("media")
        if not isinstance(media, list):
            continue
        for m in media:
            if not isinstance(m, dict):
                continue
            if str(m.get("type") or "").lower() != "photo":
                continue
            url = str(m.get("media_url_https") or m.get("media_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(url)
    return out


def _extract_tweets_from_timeline(
    timeline_data: dict[str, Any], *, screen_name: str, limit: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for holder in _iter_tweet_holders(timeline_data):
        social = holder.get("socialContext")
        if isinstance(social, dict):
            if str(social.get("contextType") or "").lower() == "pin":
                continue
            if "pinned" in str(social.get("text") or "").lower():
                continue

        raw_result = holder.get("tweet_results", {}).get("result")  # type: ignore[union-attr]
        tweet_obj = _unwrap_tweet_result(raw_result)
        if not tweet_obj:
            continue

        legacy = tweet_obj.get("legacy") if isinstance(tweet_obj.get("legacy"), dict) else {}
        tid = str(legacy.get("id_str") or tweet_obj.get("rest_id") or "").strip()
        if not tid or tid in seen_ids:
            continue
        seen_ids.add(tid)

        text = _tweet_text(tweet_obj)
        if not text:
            continue

        out.append(
            {
                "url": f"https://x.com/{screen_name}/status/{tid}",
                "text": text,
                "image_urls": _tweet_images(tweet_obj),
            }
        )
        if len(out) >= limit:
            break
    return out


def _user_by_screen_name_features() -> dict[str, Any]:
    return {
        "hidden_profile_likes_enabled": True,
        "hidden_profile_subscriptions_enabled": True,
        "responsive_web_graphql_exclude_directive_enabled": True,
        "verified_phone_label_enabled": False,
        "subscriptions_verification_info_enabled": True,
        "subscriptions_verification_info_reason_enabled": True,
        "subscriptions_verification_info_verified_since_enabled": True,
        "highlights_tweets_tab_ui_enabled": True,
        "responsive_web_twitter_article_notes_tab_enabled": True,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
        "responsive_web_graphql_timeline_navigation_enabled": True,
    }


def _user_tweets_features() -> dict[str, Any]:
    return {
        "rweb_video_timestamps_enabled": True,
        "profile_label_improvements_pcf_label_in_post_enabled": True,
        "responsive_web_graphql_exclude_directive_enabled": True,
        "verified_phone_label_enabled": False,
        "creator_subscriptions_tweet_preview_api_enabled": True,
        "responsive_web_graphql_timeline_navigation_enabled": True,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
        "premium_content_api_read_enabled": False,
        "communities_web_enable_tweet_community_results_fetch": True,
        "c9s_tweet_anatomy_moderator_badge_enabled": True,
        "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
        "responsive_web_grok_analyze_post_followups_enabled": True,
        "responsive_web_jetfuel_frame": False,
        "responsive_web_grok_share_attachment_enabled": True,
        "articles_preview_enabled": True,
        "responsive_web_edit_tweet_api_enabled": True,
        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
        "view_counts_everywhere_api_enabled": True,
        "longform_notetweets_consumption_enabled": True,
        "responsive_web_twitter_article_tweet_consumption_enabled": True,
        "tweet_awards_web_tipping_enabled": False,
        "responsive_web_grok_show_grok_translated_post": False,
        "responsive_web_grok_analysis_button_from_backend": True,
        "creator_subscriptions_quote_tweet_preview_enabled": False,
        "freedom_of_speech_not_reach_fetch_enabled": True,
        "standardized_nudges_misinfo": True,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
        "longform_notetweets_rich_text_read_enabled": True,
        "longform_notetweets_inline_media_enabled": True,
        "responsive_web_grok_image_annotation_enabled": True,
        "responsive_web_grok_imagine_annotation_enabled": True,
        "responsive_web_grok_community_note_auto_translation_is_enabled": False,
        "responsive_web_enhance_cards_enabled": False,
    }


def _fetch_latest_tweets_sync(
    target_url: str,
    *,
    limit: int,
    proxy: str,
    timeout_ms: int,
) -> list[dict[str, Any]]:
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

    screen_name = _extract_screen_name(url)
    server = _normalize_proxy(proxy)
    limit = max(1, min(int(limit), 10))
    timeout_s = max(8.0, float(timeout_ms) / 1000.0)

    with _session_with_proxy(server) as session:
        query_ids = _discover_query_ids(session, target_url=url, timeout_s=timeout_s)
        guest_token = _activate_guest_token(session, timeout_s=timeout_s)
        headers = _auth_headers(guest_token)

        last_err: Exception | None = None
        for base in _API_BASES:
            try:
                user_params = {
                    "variables": json.dumps(
                        {"screen_name": screen_name, "withSafetyModeUserFields": True},
                        separators=(",", ":"),
                    ),
                    "features": json.dumps(
                        _user_by_screen_name_features(),
                        separators=(",", ":"),
                    ),
                }
                user_url = (
                    f"{base}/graphql/{query_ids['UserByScreenName']}/UserByScreenName"
                )
                user_data = _request_json(
                    session,
                    "GET",
                    user_url,
                    headers=headers,
                    timeout_s=timeout_s,
                    params=user_params,
                )
                user_id = _extract_user_id(user_data)
                if not user_id:
                    raise RuntimeError("UserByScreenName did not return rest_id")

                tweets_params = {
                    "variables": json.dumps(
                        {
                            "userId": user_id,
                            "count": max(5, min(40, limit + 8)),
                            "includePromotedContent": False,
                            "withQuickPromoteEligibilityTweetFields": True,
                            "withVoice": True,
                            "withV2Timeline": True,
                        },
                        separators=(",", ":"),
                    ),
                    "features": json.dumps(
                        _user_tweets_features(),
                        separators=(",", ":"),
                    ),
                }
                tweets_url = f"{base}/graphql/{query_ids['UserTweets']}/UserTweets"
                timeline_data = _request_json(
                    session,
                    "GET",
                    tweets_url,
                    headers=headers,
                    timeout_s=timeout_s,
                    params=tweets_params,
                )
                return _extract_tweets_from_timeline(
                    timeline_data, screen_name=screen_name, limit=limit
                )
            except Exception as e:
                last_err = e if isinstance(e, Exception) else Exception(str(e))
                astrbot_logger.warning(
                    "[dailynews] x graphql request failed (%s): %s", base, last_err
                )
                continue

    raise RuntimeError(str(last_err) if last_err else "x graphql fetch failed")


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
    Pure API mode:
    1) guest token
    2) GraphQL UserByScreenName
    3) GraphQL UserTweets
    """
    _ = (viewport_width, viewport_height, executable_path)
    return await asyncio.to_thread(
        _fetch_latest_tweets_sync,
        target_url,
        limit=limit,
        proxy=proxy,
        timeout_ms=timeout_ms,
    )
