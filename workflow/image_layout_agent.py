import json
from datetime import datetime
from pathlib import Path
import base64
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .image_utils import (
    adaptive_layout_html_images,
    get_plugin_data_dir,
    image_url_to_data_uri,
    inline_html_remote_images,
    merge_images_vertical,
)
from .models import SubAgentResult
from .rendering import load_template, markdown_to_html, safe_text
from .local_render import render_template_to_image_playwright
from .utils import _json_from_text


class ImageLayoutAgent:
    """
    图片排版 Agent：从各来源抓到的图片 URL 中挑选并插入到日报 Markdown 里。
    - 使用用户指定的 provider（可选：支持视觉）
    - 输出为 patched_markdown（JSON）
    """

    async def enhance_markdown(
        self,
        *,
        draft_markdown: str,
        sub_results: List[Any],
        user_config: Dict[str, Any],
        astrbot_context: Any,
        image_plan: Optional[Dict[str, Any]] = None,
        rendered_preview_enabled: Optional[bool] = None,
    ) -> str:
        if not bool(user_config.get("image_layout_enabled", False)):
            return draft_markdown

        provider_id = str(user_config.get("image_layout_provider_id") or "").strip()

        max_images_total = int(user_config.get("image_layout_max_images_total", 6) or 6)
        max_images_per_source = int(user_config.get("image_layout_max_images_per_source", 3) or 3)
        pass_images_to_model = bool(user_config.get("image_layout_pass_images_to_model", True))
        max_images_to_model = int(user_config.get("image_layout_max_images_to_model", 6) or 6)
        preview_enabled = bool(user_config.get("image_layout_preview_enabled", False))
        preview_max_images = int(user_config.get("image_layout_preview_max_images", 6) or 6)
        preview_max_width = int(user_config.get("image_layout_preview_max_width", 1080) or 1080)
        preview_gap = int(user_config.get("image_layout_preview_gap", 8) or 8)
        shuffle_candidates = bool(user_config.get("image_layout_shuffle_candidates", True))
        shuffle_seed = user_config.get("image_layout_shuffle_seed")

        refine_enabled = bool(user_config.get("image_layout_refine_enabled", False))
        refine_rounds = int(user_config.get("image_layout_refine_rounds", 2) or 2)
        refine_max_requests = int(user_config.get("image_layout_refine_max_requests", 2) or 2)
        refine_request_max_images = int(user_config.get("image_layout_refine_request_max_images", 6) or 6)
        refine_preview_page_chars = int(user_config.get("image_layout_refine_preview_page_chars", 2400) or 2400)
        refine_preview_pages = int(user_config.get("image_layout_refine_preview_pages", 1) or 1)
        refine_preview_timeout_ms = int(user_config.get("image_layout_refine_preview_timeout_ms", 20000) or 20000)
        if rendered_preview_enabled is not None:
            refine_enabled = bool(rendered_preview_enabled)

        # 收集图片候选（来自写作子 Agent 抓正文时解析出的 image_urls）
        images_by_source: Dict[str, List[str]] = {}
        for r in sub_results:
            if isinstance(r, SubAgentResult) and r.source_name and r.images:
                urls: List[str] = []
                seen = set()
                for u in r.images:
                    if isinstance(u, str) and u.strip() and u not in seen:
                        seen.add(u)
                        urls.append(u.strip())
                if urls:
                    images_by_source[r.source_name] = urls

        if shuffle_candidates and images_by_source:
            seed = (
                str(shuffle_seed).strip()
                if shuffle_seed is not None and str(shuffle_seed).strip()
                else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            rnd = random.Random(seed)
            items = list(images_by_source.items())
            rnd.shuffle(items)
            shuffled: Dict[str, List[str]] = {}
            for src, urls in items:
                urls2 = list(urls)
                rnd.shuffle(urls2)
                shuffled[src] = urls2
            images_by_source = shuffled

        plan_by_source: Dict[str, int] = {}

        def _fallback_insert(md: str) -> str:
            picked: List[Tuple[str, str]] = []
            total = 0
            for src, urls in images_by_source.items():
                wanted = plan_by_source.get(src)
                if wanted is None or wanted <= 0:
                    wanted = max_images_per_source
                wanted = min(int(wanted), max_images_per_source)
                for u in urls[: max(1, wanted)]:
                    picked.append((src, u))
                    total += 1
                    if total >= max_images_total:
                        break
                if total >= max_images_total:
                    break
            if not picked:
                return md

            src_queues: Dict[str, List[str]] = {}
            for src, url in picked:
                src_queues.setdefault(src, []).append(url)

            lines_out: List[str] = []
            inserted = 0
            for line in (md or "").splitlines():
                lines_out.append(line)
                stripped = line.lstrip()
                is_heading = stripped.startswith("## ") or stripped.startswith("### ") or stripped.startswith("# ")
                if not is_heading or inserted >= len(picked):
                    continue

                matched = False
                for src, queue in list(src_queues.items()):
                    if not queue:
                        continue
                    if src and src in line:
                        url = queue.pop(0)
                        lines_out.append(f"![]({url})")
                        lines_out.append("")
                        inserted += 1
                        matched = True
                        if inserted >= len(picked):
                            break
                if matched:
                    continue

                for src, queue in list(src_queues.items()):
                    if queue:
                        url = queue.pop(0)
                        lines_out.append(f"![]({url})")
                        lines_out.append("")
                        inserted += 1
                        break

            remaining: List[Tuple[str, str]] = []
            for src, queue in src_queues.items():
                for u in queue:
                    remaining.append((src, u))
            if remaining:
                lines_out.extend(["", "## 配图", ""])
                for src, u in remaining:
                    lines_out.append(f"**{src}**")
                    lines_out.append(f"![]({u})")
                    lines_out.append("")

            return "\n".join(lines_out).strip()

        if not images_by_source:
            astrbot_logger.info("[dailynews] image_layout: no image candidates; skip")
            return draft_markdown

        if not provider_id:
            astrbot_logger.warning(
                "[dailynews] image_layout_enabled but image_layout_provider_id empty; fallback insert"
            )
            return _fallback_insert(draft_markdown)

        total_candidates = sum(len(v) for v in images_by_source.values())
        astrbot_logger.info(
            "[dailynews] image_layout candidates: %s sources, %s images",
            len(images_by_source),
            total_candidates,
        )

        # 可选：只对指定来源做图（为空则对全部）
        only_sources = user_config.get("image_layout_sources") or []
        if isinstance(only_sources, list) and only_sources:
            only_set = {str(x).strip() for x in only_sources if str(x).strip()}
            if only_set:
                images_by_source = {k: v for k, v in images_by_source.items() if k in only_set}

        if not images_by_source:
            return draft_markdown

        plan_by_source: Dict[str, int] = {}
        if isinstance(image_plan, dict):
            if isinstance(image_plan.get("by_source"), dict):
                for k, v in image_plan.get("by_source", {}).items():
                    try:
                        plan_by_source[str(k)] = max(0, int(v))
                    except Exception:
                        continue
            elif isinstance(image_plan.get("sources"), list):
                for item in image_plan.get("sources", []):
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("source") or "").strip()
                    if not name:
                        continue
                    try:
                        plan_by_source[name] = max(0, int(item.get("max_images", 0)))
                    except Exception:
                        continue
            try:
                plan_total = int(image_plan.get("max_images_total", max_images_total))
                if plan_total > 0:
                    max_images_total = min(max_images_total, plan_total)
            except Exception:
                pass

        if plan_by_source:
            filtered: Dict[str, List[str]] = {}
            for name, urls in images_by_source.items():
                want = plan_by_source.get(name)
                if want is None or want <= 0:
                    continue
                filtered[name] = urls[: min(int(want), max_images_per_source)]
            if filtered:
                images_by_source = filtered
            else:
                astrbot_logger.warning(
                    "[dailynews] image_layout plan filtered all sources; ignore plan"
                )
                plan_by_source = {}

        # 给模型看的图片（vision）：从每个来源取前 N 张，总量封顶
        image_urls_for_model: List[str] = []
        if pass_images_to_model:
            for src, urls in images_by_source.items():
                for u in urls[: max(1, max_images_per_source)]:
                    image_urls_for_model.append(u)
                    if len(image_urls_for_model) >= max_images_to_model:
                        break
                if len(image_urls_for_model) >= max_images_to_model:
                    break

        preview_hint = ""
        if pass_images_to_model and preview_enabled and image_urls_for_model:
            try:
                preview_urls = image_urls_for_model[: max(1, preview_max_images)]
                preview_out = get_plugin_data_dir("image_previews") / (
                    f"layout_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                merged = await merge_images_vertical(
                    preview_urls,
                    out_path=preview_out,
                    max_width=preview_max_width,
                    gap=preview_gap,
                )
                data_uri = await image_url_to_data_uri(str(Path(merged).resolve()))
                if data_uri:
                    image_urls_for_model = [data_uri]
                else:
                    local_path = Path(merged).resolve().as_posix()
                    image_urls_for_model = [f"file:///{local_path}"]
                preview_hint = "你将收到一张“拼接预览图”，从上到下依次对应候选图片。"
            except Exception as e:
                astrbot_logger.warning("[dailynews] image_layout preview merge failed: %s", e, exc_info=True)

        system_prompt = (
            "你是日报的“图片排版/编辑 Agent”。\n"
            "你会收到：\n"
            "1) 今日日报草稿（Markdown）\n"
            "2) 各来源可用的图片 URL 列表（已经由系统从消息源中提取）\n\n"
            "你的任务：\n"
            "- 选择合适的图片，并把图片以 Markdown 语法插入到草稿中（使用 ![](url)）。\n"
            "- 尽量把图片插入到对应来源/对应小节附近，优先放在相关标题后。\n"
            "- 不要把所有图片集中放在文末；只有在无法匹配时才放到“配图”小节。\n"
            f"- 总图片数不超过 {max_images_total} 张；单个来源不超过 {max_images_per_source} 张。\n"
            "- 不要编造不存在的图片 URL，只能从候选列表中挑选。\n"
            "- 只输出 JSON，不要输出其它文本。\n"
        )

        if image_plan:
            system_prompt += "- image_plan 表示主 Agent 对各来源图片数量的约束，请优先满足。\n"
        if preview_hint:
            system_prompt += f"- {preview_hint}\n"

        system_prompt += "- 在不影响相关性的前提下，避免每次都选候选列表最靠前的图片，尽量做出不同的选择。\n"

        payload = {
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "constraints": {
                "max_images_total": max_images_total,
                "max_images_per_source": max_images_per_source,
            },
            "image_plan": image_plan or {},
            "image_candidates": images_by_source,
            "draft_markdown": draft_markdown,
            "output_schema": {
                "patched_markdown": "string",
                "chosen": [
                    {"source": "source_name", "url": "image_url", "where": "hint(optional)"}
                ],
            },
        }

        try:
            resp = await astrbot_context.llm_generate(
                chat_provider_id=provider_id,
                prompt=json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                image_urls=image_urls_for_model if pass_images_to_model else [],
            )
            raw = getattr(resp, "completion_text", "") or ""
        except Exception as e:
            astrbot_logger.warning("[dailynews] image_layout llm_generate failed: %s", e, exc_info=True)
            return _fallback_insert(draft_markdown)

        data = _json_from_text(raw)
        if not isinstance(data, dict):
            astrbot_logger.warning("[dailynews] image_layout returned non-json; skip")
            return _fallback_insert(draft_markdown)

        patched = str(data.get("patched_markdown") or "").strip()
        if not patched:
            astrbot_logger.warning("[dailynews] image_layout patched_markdown empty; skip")
            return _fallback_insert(draft_markdown)

        if "![](" not in patched and "<img" not in patched:
            astrbot_logger.warning("[dailynews] image_layout produced no images; fallback insert")
            return _fallback_insert(patched)

        if refine_enabled and refine_rounds > 0:
            try:
                refined = await self._refine_with_render_preview(
                    markdown=patched,
                    images_by_source=images_by_source,
                    user_config=user_config,
                    astrbot_context=astrbot_context,
                    provider_id=provider_id,
                    max_rounds=refine_rounds,
                    max_requests=refine_max_requests,
                    request_max_images=refine_request_max_images,
                    preview_page_chars=refine_preview_page_chars,
                    preview_pages=refine_preview_pages,
                    preview_timeout_ms=refine_preview_timeout_ms,
                )
                if (refined or "").strip():
                    patched = refined
            except Exception as e:
                astrbot_logger.warning("[dailynews] image_layout refine failed: %s", e, exc_info=True)

        return patched

    def _asset_b64(self, filename: str) -> str:
        try:
            root = Path(__file__).resolve().parents[1]
            p = root / "image" / filename
            if p.exists() and p.is_file():
                return base64.b64encode(p.read_bytes()).decode("utf-8")
        except Exception:
            return ""
        return ""

    def _split_pages(self, text: str, page_chars: int, max_pages: int) -> List[str]:
        s = (text or "").strip()
        if not s:
            return []
        if page_chars <= 0:
            return [s]
        pages: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for line in s.splitlines():
            piece = line + "\n"
            if buf_len + len(piece) > page_chars and buf:
                pages.append("".join(buf).rstrip())
                if len(pages) >= max_pages:
                    return pages
                buf, buf_len = [], 0
            buf.append(piece)
            buf_len += len(piece)
        if buf and len(pages) < max_pages:
            pages.append("".join(buf).rstrip())
        return pages

    async def _render_markdown_preview(
        self,
        markdown: str,
        *,
        preview_page_chars: int,
        preview_pages: int,
        timeout_ms: int,
        user_config: Dict[str, Any],
    ) -> Optional[Path]:
        pages = self._split_pages(
            markdown,
            page_chars=int(preview_page_chars),
            max_pages=max(1, int(preview_pages)),
        )
        if not pages:
            return None

        tmpl = load_template("templates/daily_news.html").strip()
        bg_img = self._asset_b64("sunsetbackground.jpg")
        char_img = self._asset_b64("transparent_output.png")

        out_paths: List[str] = []
        for idx, page in enumerate(pages, start=1):
            body_html = await inline_html_remote_images(markdown_to_html(page))
            body_html = await adaptive_layout_html_images(
                body_html,
                full_max_width=int(user_config.get("render_img_full_max_width", 1000) or 1000),
                medium_max_width=int(user_config.get("render_img_medium_max_width", 820) or 820),
                narrow_max_width=int(user_config.get("render_img_narrow_max_width", 420) or 420),
                float_if_width_le=int(user_config.get("render_img_float_threshold", 480) or 480),
                float_enabled=bool(user_config.get("render_img_float_enabled", True)),
            )
            ctx = {
                "title": safe_text("每日资讯日报"),
                "subtitle": safe_text(f"预览 {idx}/{len(pages)}"),
                "body_html": body_html,
                "bg_img": bg_img,
                "char_img": char_img,
            }
            out_path = get_plugin_data_dir("layout_refine_previews") / (
                f"layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
            )
            rendered = await render_template_to_image_playwright(
                tmpl,
                ctx,
                out_path=out_path,
                viewport=(1080, 720),
                timeout_ms=int(timeout_ms),
                full_page=True,
            )
            out_paths.append(str(Path(rendered).resolve()))

        if not out_paths:
            return None
        if len(out_paths) == 1:
            return Path(out_paths[0]).resolve()

        merged = get_plugin_data_dir("layout_refine_previews") / (
            f"layout_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        merged_path = await merge_images_vertical(
            out_paths,
            out_path=merged,
            max_width=1080,
            gap=8,
        )
        return Path(merged_path).resolve()

    async def _refine_with_render_preview(
        self,
        *,
        markdown: str,
        images_by_source: Dict[str, List[str]],
        user_config: Dict[str, Any],
        astrbot_context: Any,
        provider_id: str,
        max_rounds: int,
        max_requests: int,
        request_max_images: int,
        preview_page_chars: int,
        preview_pages: int,
        preview_timeout_ms: int,
    ) -> str:
        current = (markdown or "").strip()
        if not current:
            return current

        request_budget = max(0, int(max_requests))

        system_prompt = (
            "你是日报的“排版审稿 Agent”。\n"
            "你会收到：当前 Markdown、候选图片 URL 列表、以及一张“渲染预览图”(作为图片输入)。\n"
            "你的目标：让图片大小/比例自然、位置合理。\n"
            "- 系统会把“窄图”自动做并排(浮动)排版；因此当图片不适合整行大图时，优先改成更适合并排的小图。\n"
            "- 避免超长竖图整行占据过大高度；必要时可替换为更合适的图或移动到更合理的位置。\n"
            "- 只输出 JSON，不要输出其它文字。\n"
            "如果你需要查看某些候选 URL 的原图细节，请输出 JSON：{\"request_image_urls\": [..] }，我会把它们合成预览图再给你。\n"
            "当你认为排版已经“差不多”时，输出 JSON：{\"patched_markdown\": \"...\", \"done\": true}。\n"
        )

        for round_idx in range(1, max(1, int(max_rounds)) + 1):
            try:
                preview_path = await self._render_markdown_preview(
                    current,
                    preview_page_chars=preview_page_chars,
                    preview_pages=preview_pages,
                    timeout_ms=preview_timeout_ms,
                    user_config=user_config,
                )
            except Exception:
                preview_path = None
            if preview_path is None:
                return current

            preview_url = f"file:///{preview_path.resolve().as_posix()}"
            payload = {
                "round": round_idx,
                "draft_markdown": current,
                "image_candidates": images_by_source,
                "output_schema": {
                    "patched_markdown": "string",
                    "done": "boolean(optional)",
                    "request_image_urls": ["string(optional)"],
                    "notes": "string(optional)",
                },
            }

            resp = await astrbot_context.llm_generate(
                chat_provider_id=provider_id,
                prompt=json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                image_urls=[preview_url],
            )
            data = _json_from_text(getattr(resp, "completion_text", "") or "")
            if not isinstance(data, dict):
                return current

            req = data.get("request_image_urls")
            if request_budget > 0 and isinstance(req, list) and req:
                urls = [str(u).strip() for u in req if str(u).strip()][: max(1, int(request_max_images))]
                request_budget -= 1
                req_url = ""
                try:
                    merged = get_plugin_data_dir("layout_refine_requests") / (
                        f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(urls)}.jpg"
                    )
                    merged_path = await merge_images_vertical(
                        urls,
                        out_path=merged,
                        max_width=1080,
                        gap=8,
                    )
                    req_url = f"file:///{Path(merged_path).resolve().as_posix()}"
                except Exception:
                    req_url = ""

                payload2 = dict(payload)
                payload2["requested_image_urls"] = urls
                payload2["requested_images_preview"] = "已合成预览图(见图片输入)"
                resp2 = await astrbot_context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=json.dumps(payload2, ensure_ascii=False),
                    system_prompt=system_prompt,
                    image_urls=[u for u in [preview_url, req_url] if u],
                )
                data2 = _json_from_text(getattr(resp2, "completion_text", "") or "")
                if isinstance(data2, dict):
                    data = data2

            patched = str(data.get("patched_markdown") or "").strip()
            if patched:
                if patched == current and bool(data.get("done", True)):
                    return current
                current = patched
                if bool(data.get("done", False)):
                    return current

        return current
