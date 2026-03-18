import asyncio
import importlib
import json
import re
from datetime import datetime
from typing import Any

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

WECHAT_LATEST_SCOPE = "auto"
WECHAT_LATEST_OVERFETCH_FACTOR = 4
WECHAT_ALLOW_SEED_URL_FALLBACK = False
WECHAT_LATEST_MAX_AGE_HOURS = 36
WECHAT_IMAGES_MAX_PER_ARTICLE = 8
WECHAT_IMAGES_MAX_TOTAL = 24
WECHAT_IMAGES_FOR_LLM_PER_ARTICLE = 3
_DATE_IN_TEXT_RE_ASCII = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
_DATE_IN_TEXT_RE = re.compile(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})")


def _safe_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = int(default)
    return max(minimum, min(maximum, n))


def _article_create_ts(item: dict[str, Any]) -> int:
    try:
        raw = str((item or {}).get("create_time") or "").strip()
        if raw.isdigit():
            n = int(raw)
            # Some upstream fields may be milliseconds.
            if n > 10_000_000_000:
                n = n // 1000
            return n
    except Exception:
        pass
    return 0


def _looks_like_today_item(item: dict[str, Any]) -> bool:
    today = datetime.now().date()
    for key in ("title", "name", "date_label", "published", "publish_time"):
        text = str((item or {}).get(key) or "").strip()
        if not text:
            continue
        m = _DATE_IN_TEXT_RE.search(text)
        if not m:
            continue
        try:
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
            if datetime(y, mo, d).date() == today:
                return True
        except Exception:
            continue
    return False


def _fallback_fetch_latest_articles(
    article_url: str,
    limit: int = 5,
    latest_scope: str = "auto",
) -> tuple[list[dict[str, str]], dict[str, str]]:
    from analysis.wechatanalysis.latest_articles import get_album_articles  # type: ignore

    rows = get_album_articles(
        article_url=article_url,
        limit=max(1, int(limit or 1)),
        latest_scope=latest_scope,
    )
    return rows, {"scope": latest_scope, "article_url": article_url}


try:
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.append(root)
    _wechat_analysis = importlib.import_module("analysis.wechatanalysis.analysis")
    fetch_wechat_article = getattr(_wechat_analysis, "fetch_wechat_article")
    fetch_wechat_latest_articles = getattr(
        _wechat_analysis, "fetch_latest_articles", None
    )
    if not callable(fetch_wechat_latest_articles):
        fetch_wechat_latest_articles = _fallback_fetch_latest_articles
except Exception:  # pragma: no cover
    from analysis.wechatanalysis.analysis import fetch_wechat_article  # type: ignore

    try:
        from analysis.wechatanalysis.analysis import (  # type: ignore
            fetch_latest_articles as fetch_wechat_latest_articles,
        )
    except Exception:
        fetch_wechat_latest_articles = _fallback_fetch_latest_articles

from ...core.llm import LLMRunner
from ...core.models import NewsSourceConfig, SubAgentResult
from ...core.utils import _json_from_text, _run_sync, ensure_section_links
from ...core.wechat_freshness import (
    article_is_new_since_recent_baseline as _article_is_new_since_recent_baseline,
    build_wechat_seed_key as _build_wechat_seed_key,
    merge_seen_articles as _merge_seen_articles,
)
from ...storage.seed_store import _get_seed_state, _update_seed_entry


class WechatSubAgent:
    """鍏紬鍙峰瓙 Agent锛氭姄鍙栨渶鏂版枃绔犲垪琛ㄣ€佹姄鍙栨鏂囧苟鍐欏嚭灏忚妭"""

    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, str]]]:
        limit = max(int(source.max_articles), 5)
        latest_overfetch_factor = int(WECHAT_LATEST_OVERFETCH_FACTOR)
        fetch_limit = min(60, max(limit, limit * latest_overfetch_factor))
        persist_seed = bool(user_config.get("wechat_seed_persist", True))
        allow_seed_url_fallback = bool(WECHAT_ALLOW_SEED_URL_FALLBACK)
        latest_max_age_hours = int(WECHAT_LATEST_MAX_AGE_HOURS)
        fetch_timeout_s = 90.0

        album_keyword = source.album_keyword
        latest_scope = str(WECHAT_LATEST_SCOPE)
        effective_max_age_hours = int(latest_max_age_hours)
        latest_max_age_seconds = (
            int(effective_max_age_hours) * 3600 if effective_max_age_hours > 0 else 0
        )

        key = _build_wechat_seed_key(source.url, album_keyword or "", latest_scope)

        state = await _get_seed_state() if persist_seed else {}
        entry = state.get(key) if isinstance(state, dict) else None
        if not isinstance(entry, dict):
            entry = {}

        # Keep a small pool of last-known seed URLs to avoid occasional bad redirects/pages.
        candidates: list[str] = []
        seen = set()

        def _push(u: str):
            s = (u or "").strip()
            if not s or s in seen:
                return
            seen.add(s)
            candidates.append(s)

        # Always try the user-configured URL first, then historical seeds.
        _push(source.url)

        seed_urls = entry.get("seed_urls")
        if isinstance(seed_urls, list):
            for u in seed_urls:
                if isinstance(u, str):
                    _push(u)
        if isinstance(entry.get("seed_url"), str):
            _push(str(entry.get("seed_url")))

        last_good_seed_url = str(entry.get("last_good_seed_url") or "").strip()
        # Only used as a last resort for fallback.
        if last_good_seed_url:
            _push(last_good_seed_url)

        seed_url = ""
        articles: list[dict[str, str]] = []
        raw_articles_for_cache: list[dict[str, str]] = []
        raw_seed_url = ""
        last_err: str = ""

        for start_url in candidates[: max(1, len(candidates))]:
            stale_only_result = False
            for attempt in range(1, 3):
                try:
                    rows, meta = await asyncio.wait_for(
                        _run_sync(
                            fetch_wechat_latest_articles,
                            start_url,
                            fetch_limit,
                            latest_scope=latest_scope,
                        ),
                        timeout=float(fetch_timeout_s),
                    )
                    raw_articles = [
                        {
                            "title": str((x or {}).get("title", "")),
                            "url": str((x or {}).get("url", "")),
                            "create_time": str((x or {}).get("create_time", "")),
                        }
                        for x in (rows or [])
                        if isinstance(x, dict) and str((x or {}).get("url", "")).strip()
                    ]
                    if raw_articles:
                        raw_seed_url = (
                            str((meta or {}).get("article_url") or start_url).strip()
                            or start_url
                        )
                        raw_articles_for_cache = list(raw_articles)
                    articles = list(raw_articles)
                    # Defensive: force newest-first by create_time when available.
                    if any(_article_create_ts(x) > 0 for x in articles):
                        articles = sorted(
                            articles,
                            key=lambda x: _article_create_ts(x),
                            reverse=True,
                        )
                    if latest_max_age_seconds > 0 and articles:
                        cutoff_ts = int(datetime.now().timestamp()) - int(
                            latest_max_age_seconds
                        )
                        stale_count = 0
                        new_override_count = 0
                        filtered_articles: list[dict[str, str]] = []
                        for item in articles:
                            ts = _article_create_ts(item)
                            if ts > 0 and ts < cutoff_ts:
                                is_today = _looks_like_today_item(item)
                                is_new_since_baseline = (
                                    _article_is_new_since_recent_baseline(item, entry)
                                )
                                if not is_today and not is_new_since_baseline:
                                    stale_count += 1
                                    continue
                                if is_new_since_baseline and not is_today:
                                    new_override_count += 1
                            filtered_articles.append(item)
                        if stale_count > 0 or new_override_count > 0:
                            astrbot_logger.warning(
                                "[dailynews] wechat latest stale-filter source=%s start=%s dropped=%s kept=%s new_override=%s max_age_hours=%s max_age_seconds=%s",
                                source.name,
                                start_url,
                                stale_count,
                                len(filtered_articles),
                                new_override_count,
                                effective_max_age_hours,
                                latest_max_age_seconds,
                            )
                        articles = filtered_articles
                        if not articles:
                            stale_only_result = True
                            last_err = f"all articles older than {effective_max_age_hours} hours"
                    if len(articles) > limit:
                        articles = articles[:limit]
                    meta_scope = str((meta or {}).get("scope") or latest_scope).strip()
                    meta_strategy = str((meta or {}).get("strategy") or "").strip()
                    meta_error = str((meta or {}).get("error") or "").strip()
                    newest_ts = _article_create_ts(articles[0]) if articles else 0
                    seed_url = str((meta or {}).get("article_url") or start_url).strip()
                    astrbot_logger.info(
                        "[dailynews] latest fetch ok source=%s start=%s scope=%s strategy=%s meta_err=%s fetch_limit=%s selected=%s newest_ts=%s",
                        source.name,
                        start_url,
                        meta_scope or latest_scope,
                        meta_strategy or "-",
                        meta_error or "-",
                        fetch_limit,
                        len(articles),
                        newest_ts,
                    )
                except asyncio.TimeoutError:
                    last_err = f"timeout>{int(fetch_timeout_s)}s"
                    astrbot_logger.warning(
                        "[dailynews] latest fetch timeout for %s (start=%s, attempt %s/2): %s",
                        source.name,
                        start_url,
                        attempt,
                        last_err,
                        exc_info=True,
                    )
                    seed_url, articles = start_url, []
                except Exception as e:
                    last_err = str(e) or type(e).__name__
                    astrbot_logger.warning(
                        "[dailynews] latest fetch failed for %s (start=%s, attempt %s/2): %s",
                        source.name,
                        start_url,
                        attempt,
                        last_err,
                        exc_info=True,
                    )
                    seed_url, articles = start_url, []

                if articles:
                    break
                if stale_only_result:
                    break
                await asyncio.sleep(0.8 * attempt)

            if articles or raw_articles_for_cache:
                # Update pool: put latest seed_url at front, keep up to 3.
                if persist_seed and (seed_url or raw_seed_url):
                    selected_seed_url = seed_url or raw_seed_url
                    new_pool: list[str] = []
                    for u in [selected_seed_url] + [
                        x for x in candidates if x != selected_seed_url
                    ]:
                        if u and u not in new_pool and "mp.weixin.qq.com/s" in u:
                            new_pool.append(u)
                        if len(new_pool) >= 3:
                            break
                    updated_entry = _merge_seen_articles(
                        {
                            **entry,
                            "seed_url": selected_seed_url,
                            "seed_urls": new_pool,
                            "last_good_seed_url": selected_seed_url,
                            "source_url": source.url,
                            "album_keyword": album_keyword or "",
                            "latest_scope": latest_scope,
                        },
                        raw_articles_for_cache or articles,
                    )
                    await _update_seed_entry(key, updated_entry)
                    entry = updated_entry
                break

        if not articles:
            if allow_seed_url_fallback:
                fallback_url = last_good_seed_url or source.url
                astrbot_logger.warning(
                    "[dailynews] %s has no latest wechat articles after retries (scope=%s); fallback to single seed url: %s (last_err=%s)",
                    source.name,
                    latest_scope,
                    fallback_url,
                    last_err or "unknown",
                )
                seed_url = fallback_url
                articles = [{"title": "", "url": fallback_url, "create_time": ""}]
            else:
                astrbot_logger.warning(
                    "[dailynews] %s has no latest wechat articles after retries (scope=%s); return empty list (fallback disabled). last_err=%s",
                    source.name,
                    latest_scope,
                    last_err or "unknown",
                )
                articles = []

        return source.name, articles

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, str]], llm: LLMRunner
    ) -> dict[str, Any]:
        sample_articles = articles[:3]
        quick_topics: list[str] = []
        for a in articles[:10]:
            if not isinstance(a, dict):
                continue
            t = str(a.get("title") or "").strip()
            if t:
                quick_topics.append(t[:36])
        quick_report: dict[str, Any] = {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": len(articles),
            "topics": quick_topics[:8],
            "quality_score": len(articles) * 2 + len(quick_topics),
            "today_angle": "",
            "sample_articles": sample_articles,
            "error": None,
        }

        system_prompt = (
            "浣犳槸瀛怉gent锛堜俊鎭睛瀵燂級銆?"
            "浣犲皢鏀跺埌鏌愪釜鍏紬鍙锋潵婧愮殑鏈€鏂版枃绔犳爣棰樹笌閾炬帴銆?"
            "璇峰揩閫熷垽鏂粖鏃ヤ富瑕佺湅鐐?涓婚锛屽苟缁欏嚭鍙啓浣滅殑瑙掑害寤鸿銆?"
            "鍙緭鍑?JSON锛屼笉瑕佽緭鍑哄叾瀹冩枃鏈€?"
        )
        system_prompt += (
            "\n\nCRITICAL OUTPUT RULES (must follow):\n"
            "1) Never output raw URLs (no lines starting with http/https). All links must be Markdown links like [闃呰鍘熸枃](URL).\n"
            "2) Ban vague filler like 鈥滀紭鍖栦綋楠?淇閮ㄥ垎bug鈥? Use concrete details from the provided article excerpts: feature name, affected module, behavior change, numbers (limits, performance, versions).\n"
            "3) Prefer 3-6 bullets max. Each bullet:\n"
            "   - **鏍囬**锛氫竴鍙ヨ瘽缁撹銆?( [闃呰鍘熸枃](url) )\n"
            "     - 缁嗚妭锛氳嚦灏?1 鏉″叿浣撶粏鑺傦紱濡傛灉鏈夌増鏈彿/鍙傛暟/鍔熻兘鐐硅鍐欏嚭鏉ャ€傚缓璁啓10-15鏉★紝濡傛灉娌℃湁杩欎箞澶氶厡鎯呰€冭檻锛屾病鏈変环鍊肩殑鍐呭灏变笉鍐橽n"
            "4) If you cannot extract concrete details, output an empty section_markdown (do NOT make up content).\n"
        )
        prompt = {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "latest_articles": articles[:10],
            "output_schema": {
                "source_name": source.name,
                "source_type": source.type,
                "priority": source.priority,
                "article_count": len(articles),
                "topics": ["topic"],
                "quality_score": 0,
                "today_angle": "string",
            },
        }

        try:
            raw = await llm.ask(
                system_prompt=system_prompt,
                prompt=json.dumps(prompt, ensure_ascii=False),
            )
        except Exception as e:
            astrbot_logger.warning(
                "[dailynews] wechat analyze_source llm failed, use quick report source=%s type=%s err=%s",
                source.name,
                type(e).__name__,
                str(e) or repr(e),
            )
            return quick_report

        data = _json_from_text(raw) or {}
        if not isinstance(data, dict):
            return quick_report

        topics = data.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        quality = data.get("quality_score")
        try:
            if isinstance(quality, float) and 0 <= quality <= 1:
                quality_score = int(quality * 100)
            elif isinstance(quality, str):
                q = quality.strip()
                if q.endswith("%"):
                    quality_score = int(float(q[:-1]) * 1)
                else:
                    qf = float(q)
                    quality_score = int(qf * 100) if 0 <= qf <= 1 else int(qf)
            else:
                quality_score = int(quality)
        except Exception:
            quality_score = len(articles) * 2 + len(topics)

        return {
            "source_name": source.name,
            "source_type": source.type,
            "source_url": source.url,
            "priority": source.priority,
            "article_count": len(articles),
            "topics": [str(t) for t in topics[:8]],
            "quality_score": quality_score,
            "today_angle": str(data.get("today_angle") or ""),
            "sample_articles": sample_articles,
            "error": None,
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: list[dict[str, str]],
        llm: LLMRunner,
        user_config: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        if not articles:
            return SubAgentResult(
                source_name=source.name,
                content="",
                summary="",
                key_points=[],
                error="no latest articles fetched from this source",
            )

        chosen = articles[: max(1, int(source.max_articles))]
        max_images_per_article = int(WECHAT_IMAGES_MAX_PER_ARTICLE)
        max_images_total = int(WECHAT_IMAGES_MAX_TOTAL)
        max_images_for_llm_per_article = min(
            int(WECHAT_IMAGES_FOR_LLM_PER_ARTICLE),
            max(0, max_images_per_article),
        )

        max_fetch_concurrency = 2
        sem = asyncio.Semaphore(max_fetch_concurrency)

        async def _fetch_one(a: dict[str, str]) -> dict[str, Any]:
            url = (a.get("url") or "").strip()
            if not url:
                return {
                    "title": (a.get("title") or "").strip(),
                    "url": "",
                    "error": "missing url",
                }

            last_err: str | None = None
            for attempt in range(1, 3):
                try:
                    async with sem:
                        detail = await _run_sync(fetch_wechat_article, url)
                    content_text = (detail.get("content_text") or "").strip()
                    if len(content_text) > 1500:
                        content_text = content_text[:1500] + "..."
                    image_urls = detail.get("image_urls") or []
                    if not isinstance(image_urls, list):
                        image_urls = []
                    image_urls = [
                        str(u) for u in image_urls if isinstance(u, str) and u.strip()
                    ]
                    raw_image_count = len(image_urls)
                    if max_images_per_article > 0:
                        image_urls = image_urls[:max_images_per_article]
                    else:
                        image_urls = []
                    image_urls_for_llm = (
                        image_urls[:max_images_for_llm_per_article]
                        if max_images_for_llm_per_article > 0
                        else []
                    )
                    return {
                        "title": (detail.get("title") or a.get("title") or "").strip(),
                        "url": url,
                        "author": (detail.get("author") or "").strip(),
                        "publish_time": (detail.get("publish_time") or "").strip(),
                        "content_text": content_text,
                        "image_urls": image_urls,
                        "image_urls_for_llm": image_urls_for_llm,
                        "image_urls_raw_count": raw_image_count,
                    }
                except Exception as e:
                    last_err = str(e) or type(e).__name__
                    astrbot_logger.warning(
                        "[dailynews] fetch_wechat_article failed (attempt %s/2): %s",
                        attempt,
                        last_err,
                        exc_info=True,
                    )
                    await asyncio.sleep(1.0 * attempt)
            return {
                "title": (a.get("title") or "").strip(),
                "url": url,
                "error": last_err or "unknown",
            }

        article_details = await asyncio.gather(
            *[_fetch_one(a) for a in chosen], return_exceptions=False
        )
        images_all_unique: list[str] = []
        seen = set()
        raw_total = 0
        prompt_article_details: list[dict[str, Any]] = []
        for d in article_details:
            if not isinstance(d, dict):
                continue
            raw_total += int(d.get("image_urls_raw_count") or 0)
            for u in d.get("image_urls") or []:
                if isinstance(u, str) and u and u not in seen:
                    seen.add(u)
                    images_all_unique.append(u)
            prompt_d = dict(d)
            prompt_d["image_urls"] = list(prompt_d.get("image_urls_for_llm") or [])
            prompt_d.pop("image_urls_for_llm", None)
            prompt_d.pop("image_urls_raw_count", None)
            prompt_article_details.append(prompt_d)

        images = images_all_unique[:max_images_total] if max_images_total > 0 else []
        if raw_total or images_all_unique:
            astrbot_logger.info(
                "[dailynews] %s image urls stats raw_total=%s unique_after_article_cap=%s output_kept=%s article_cap=%s total_cap=%s llm_per_article=%s",
                source.name,
                raw_total,
                len(images_all_unique),
                len(images),
                max_images_per_article,
                max_images_total,
                max_images_for_llm_per_article,
            )
        if len(images_all_unique) > len(images):
            astrbot_logger.warning(
                "[dailynews] %s dropped %s image urls by total cap (%s)",
                source.name,
                len(images_all_unique) - len(images),
                max_images_total,
            )

        system_prompt = (
            "You are a writing sub-agent for one WeChat source. "
            "Given instruction and article details, write one concise markdown section. "
            "Return JSON only."
        )
        prompt = {
            "source_name": source.name,
            "instruction": instruction,
            "articles": prompt_article_details,
            "output_schema": {
                "summary": "string",
                "key_points": ["string"],
                "section_markdown": "markdown string",
            },
        }

        try:
            raw = await llm.ask(
                system_prompt=system_prompt,
                prompt=json.dumps(prompt, ensure_ascii=False),
            )
        except Exception as e:
            err_text = str(e).strip() or repr(e)
            astrbot_logger.warning(
                "[dailynews] subagent write failed, fallback source=%s type=%s err=%s article_count=%s prompt_image_urls=%s",
                source.name,
                type(e).__name__,
                err_text,
                len(prompt_article_details),
                sum(
                    len(d.get("image_urls") or [])
                    for d in prompt_article_details
                    if isinstance(d, dict)
                ),
                exc_info=True,
            )
            lines = [
                f"## {source.name}",
                "",
                "(LLM generation failed or timed out; fallback summary below)",
                "",
            ]
            for a in chosen:
                t = (a.get("title") or "").strip()
                u = (a.get("url") or "").strip()
                if t and u:
                    lines.append(f"- {t} ([闃呰鍘熸枃]({u}))")
                elif u:
                    lines.append(f"- [闃呰鍘熸枃]({u})")
            return SubAgentResult(
                source_name=source.name,
                content="\n".join(lines).strip(),
                summary="",
                key_points=[],
                images=images or None,
                error=None,
            )

        data = _json_from_text(raw)
        if not isinstance(data, dict):
            return SubAgentResult(
                source_name=source.name,
                content=str(raw),
                summary="",
                key_points=[],
                images=images or None,
                error=None,
            )

        summary = str(data.get("summary") or "")
        key_points = data.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        section = str(data.get("section_markdown") or "")
        section = ensure_section_links(section, article_details)

        return SubAgentResult(
            source_name=source.name,
            content=section,
            summary=summary,
            key_points=[str(x) for x in key_points[:10]],
            images=images or None,
            error=None,
        )
