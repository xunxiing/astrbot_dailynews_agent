import json
from datetime import datetime
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from ..core.config_models import ImageLabelConfig, ImageLayoutConfig, LayoutPrompts, LayoutRefineConfig
from ..core.image_utils import get_plugin_data_dir, merge_images_vertical
from .image_labeler import ImageLabeler
from .layout_refiner import LayoutRefiner
from .tool_based_layout_editor import ToolBasedLayoutEditor
from ..core.models import MainAgentDecision, SubAgentResult
from ..pipeline.rendering import load_template
from ..core.utils import _json_from_text
from ..core.llm import LLMRunner


class ImageLayoutAgent:
    """
    图片排版 Agent：从各来源抓到的图片 URL 中挑选并插入到日报 Markdown 里。
    - 使用用户指定的 provider（可选：支持视觉）
    - 输出为 patched_markdown（JSON）
    """

    async def _decide_image_plan(
        self,
        *,
        sub_results: List[Any],
        user_config: Dict[str, Any],
        astrbot_context: Any,
        max_images_total: int,
        max_images_per_source: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Simplified internal image planning logic, moved from independent ImagePlanAgent.
        """
        if not bool(user_config.get("image_plan_enabled", True)):
            return None

        images_by_source: Dict[str, List[str]] = {}
        for r in sub_results:
            if isinstance(r, SubAgentResult) and r.source_name and r.images:
                urls = [str(u) for u in r.images if isinstance(u, str) and u.strip()]
                if urls:
                    images_by_source[r.source_name] = urls

        if not images_by_source:
            return None

        # Call LLM for planning if provider is available
        provider_id = str(user_config.get("main_agent_provider_id") or "").strip()
        if not provider_id:
            return None

        llm = LLMRunner(astrbot_context, provider_ids=[provider_id])
        
        system_prompt = (
            "You are the chief editor deciding an image count plan for the layout agent.\n"
            "Return JSON with a total image cap and per-source image counts.\n"
            "Only output JSON.\n"
        )
        payload = {
            "constraints": {
                "max_images_total": max_images_total,
                "max_images_per_source": max_images_per_source,
            },
            "candidates": {name: len(urls) for name, urls in images_by_source.items()},
            "output_schema": {
                "max_images_total": "int",
                "by_source": {"source_name": "int"}
            },
        }

        try:
            raw = await llm.ask(system_prompt=system_prompt, prompt=json.dumps(payload, ensure_ascii=False))
            data = _json_from_text(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return None

    async def enhance_markdown(
        self,
        *,
        draft_markdown: str,
        sub_results: List[Any],
        user_config: Dict[str, Any],
        astrbot_context: Any,
        image_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        layout_cfg = ImageLayoutConfig.from_mapping(user_config)
        if not layout_cfg.enabled:
            return draft_markdown

        provider_id = layout_cfg.provider_id
        max_images_total = int(layout_cfg.max_images_total)
        max_images_per_source = int(layout_cfg.max_images_per_source)
        pass_images_to_model = bool(layout_cfg.pass_images_to_model)
        max_images_to_model = int(layout_cfg.max_images_to_model)
        preview_enabled = bool(layout_cfg.preview_enabled)
        preview_max_images = int(layout_cfg.preview_max_images)
        preview_max_width = int(layout_cfg.preview_max_width)
        preview_gap = int(layout_cfg.preview_gap)
        request_max_requests = int(layout_cfg.request_max_requests)
        request_max_images = int(layout_cfg.request_max_images)
        tool_enabled = bool(layout_cfg.tool_enabled)
        tool_rounds = int(layout_cfg.tool_rounds)
        tool_max_steps = int(layout_cfg.tool_max_steps)
        shuffle_candidates = bool(layout_cfg.shuffle_candidates)
        shuffle_seed = str(layout_cfg.shuffle_seed or "").strip()

        refine_cfg = LayoutRefineConfig.from_mapping(user_config)

        _IMG_MD_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")
        _IMG_HTML_RE = re.compile(r"""<img[^>]+src=(?P<q>["'])(?P<url>[^"']+)(?P=q)""", re.I)
        _HEADING_RE = re.compile(r"^(?P<h>#{1,6})\s+(?P<t>.+?)\s*$")

        def _extract_existing_image_urls(md: str) -> List[str]:
            found: List[str] = []
            seen = set()
            s = md or ""
            for m in _IMG_MD_RE.finditer(s):
                u = str(m.group("url") or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    found.append(u)
            for m in _IMG_HTML_RE.finditer(s):
                u = str(m.group("url") or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    found.append(u)
            return found

        _STOP = {
            "今天",
            "今日",
            "资讯",
            "新闻",
            "简报",
            "日报",
            "速递",
            "盘点",
            "链接",
            "文章",
            "内容",
            "来源",
            "更多",
            "详情",
            "介绍",
            "总结",
            "要点",
            "更新",
            "发布",
            "转发",
            "评论",
        }

        # Internal token/section helpers are kept for now but can be further simplified.
        # Future: Move to a shared markdown utility.
        def _tokens(text: str) -> set[str]:
            s = (text or "").strip()
            if not s:
                return set()
            parts = re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", s)
            out: set[str] = set()
            for p in parts:
                t = p.strip().lower()
                if not t:
                    continue
                if t in _STOP:
                    continue
                if len(t) >= 2:
                    out.add(t)
            return out

        def _split_sections(md: str) -> List[Dict[str, Any]]:
            lines = (md or "").splitlines()
            heads: List[Tuple[int, int, str]] = []
            for idx, line in enumerate(lines):
                m = _HEADING_RE.match(line)
                if not m:
                    continue
                level = len(m.group("h") or "")
                title = str(m.group("t") or "").strip()
                if not title:
                    continue
                heads.append((idx, level, title))

            if not heads:
                return []

            sections: List[Dict[str, Any]] = []
            for i, (start, level, title) in enumerate(heads):
                end = len(lines)
                for j in range(i + 1, len(heads)):
                    nxt_start, nxt_level, _ = heads[j]
                    if nxt_level <= level:
                        end = nxt_start
                        break
                body = "\n".join(lines[start:end]).strip()
                sections.append(
                    {
                        "start": start,
                        "end": end,
                        "level": level,
                        "title": title,
                        "body": body,
                        "tokens": _tokens(title + "\n" + body),
                    }
                )
            return sections

        def _best_section_index(
            *,
            source_name: str,
            source_text: str,
            sections: List[Dict[str, Any]],
        ) -> Optional[int]:
            if not sections:
                return None

            name = (source_name or "").strip()
            if name:
                for idx, s in enumerate(sections):
                    if name in str(s.get("title") or ""):
                        return idx

            st = _tokens((source_text or "") + "\n" + name)
            if not st:
                return 0

            best_i = 0
            best = 0.0
            for idx, sec in enumerate(sections):
                tt = sec.get("tokens") or set()
                if not tt:
                    continue
                inter = len(st.intersection(tt))
                if inter <= 0:
                    continue
                score = inter / (max(1.0, (len(st) * len(tt)) ** 0.5))
                title = str(sec.get("title") or "")
                if name and name in title:
                    score += 0.25
                if score > best:
                    best, best_i = score, idx

            if best < 0.06:
                return 0
            return best_i

        def _insert_images_near_sections(
            md: str,
            *,
            picked_by_source: Dict[str, List[str]],
            source_text_by_source: Dict[str, str],
        ) -> str:
            lines = (md or "").splitlines()
            if not lines or not picked_by_source:
                return (md or "").strip()

            existing = set(_extract_existing_image_urls(md))
            sections = _split_sections(md)
            if not sections:
                out = (md or "").rstrip() + "\n\n## 配图\n\n"
                for src, urls in picked_by_source.items():
                    for u in urls:
                        if u in existing:
                            continue
                        out += f"**{src}**\n![]({u})\n\n"
                        existing.add(u)
                return out.strip()

            def _apply_at_line(insert_at: int, url: str) -> None:
                nonlocal lines
                insert_at = max(0, min(int(insert_at), len(lines)))
                if insert_at > 0 and lines[insert_at - 1].strip():
                    lines.insert(insert_at, "")
                    insert_at += 1
                lines.insert(insert_at, f"![]({url})")
                lines.insert(insert_at + 1, "")

            used_any = False
            for src, urls in picked_by_source.items():
                if not urls:
                    continue
                src_text = source_text_by_source.get(src) or ""
                curr_md = "\n".join(lines)
                curr_sections = _split_sections(curr_md)
                idx = _best_section_index(
                    source_name=src,
                    source_text=src_text,
                    sections=curr_sections,
                )
                if idx is None or idx < 0 or idx >= len(curr_sections):
                    idx = 0
                sec = curr_sections[idx]
                insert_at = int(sec["start"]) + 1
                while insert_at < len(lines) and not lines[insert_at].strip():
                    insert_at += 1

                for u in urls:
                    if u in existing:
                        continue
                    _apply_at_line(insert_at, u)
                    used_any = True
                    existing.add(u)
                    insert_at += 2

            if not used_any:
                return (md or "").strip()
            return "\n".join(lines).strip()

        source_text_by_source: Dict[str, str] = {}

        # 收集图片候选（来自写作子 Agent 抓正文时解析出的 image_urls）
        images_by_source: Dict[str, List[str]] = {}
        for r in sub_results:
            if isinstance(r, SubAgentResult) and r.source_name and r.images:
                if r.content and r.content.strip():
                    source_text_by_source[r.source_name] = r.content.strip()
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
                str(shuffle_seed).strip() if str(shuffle_seed).strip() else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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

            picked_by_source: Dict[str, List[str]] = {}
            for src, url in picked:
                picked_by_source.setdefault(src, []).append(url)

            try:
                return _insert_images_near_sections(
                    md,
                    picked_by_source=picked_by_source,
                    source_text_by_source=source_text_by_source,
                )
            except Exception:
                lines_out: List[str] = [(md or "").rstrip(), "", "## 配图", ""]
                for src, urls in picked_by_source.items():
                    for u in urls:
                        lines_out.extend([f"**{src}**", f"![]({u})", ""])
                return "\n".join([x for x in lines_out if x is not None]).strip()

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

        # If no image_plan provided, try internal simplified planning
        if image_plan is None:
            image_plan = await self._decide_image_plan(
                sub_results=sub_results,
                user_config=user_config,
                astrbot_context=astrbot_context,
                max_images_total=max_images_total,
                max_images_per_source=max_images_per_source
            )

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

        prompts = LayoutPrompts.from_files(load_template=load_template)

        image_catalog: List[Dict[str, Any]] = []
        label_cfg = ImageLabelConfig.from_mapping(user_config)
        if label_cfg.enabled and label_cfg.provider_id:
            # When we have per-image labels, avoid feeding the layout model a huge image list (and skip vision previews).
            pass_images_to_model = False
            try:
                labeler = ImageLabeler(system_prompt=prompts.image_labeler_system)
                image_catalog, _ = await labeler.build_catalog(
                    images_by_source=images_by_source,
                    astrbot_context=astrbot_context,
                    cfg=label_cfg,
                )
                if image_catalog:
                    astrbot_logger.info("[dailynews] image_label catalog ready: %s", len(image_catalog))
            except Exception as e:
                astrbot_logger.warning("[dailynews] image_label failed: %s", e, exc_info=True)

        # Tool-based layout editing: let the model call tools to insert images / fix text,
        # then we render a preview each round and feed it back until it says "done".
        if tool_enabled and hasattr(astrbot_context, "tool_loop_agent"):
            try:
                tool_system = (prompts.image_layout_tool_system + "\n").strip() + "\n"
                tool_system += (
                    f"- 总图片数不超过 {max_images_total} 张；单个来源不超过 {max_images_per_source} 张。\n"
                )
                if image_plan:
                    tool_system += "- image_plan 表示主 Agent 对各来源图片数量的约束，请优先满足。\n"

                patched = await ToolBasedLayoutEditor(system_prompt=tool_system).edit(
                    draft_markdown=draft_markdown,
                    images_by_source=images_by_source,
                    image_catalog=image_catalog,
                    user_config=user_config,
                    astrbot_context=astrbot_context,
                    provider_id=provider_id,
                    rounds=tool_rounds,
                    max_steps=tool_max_steps,
                    request_max_requests=request_max_requests,
                    request_max_images=request_max_images,
                    send_images_to_model=bool(pass_images_to_model),
                )
                if (patched or "").strip():
                    has_img = bool(re.search(r"!\[[^\]]*\]\(", patched)) or ("<img" in patched)
                    if has_img:
                        return patched
                    astrbot_logger.info(
                        "[dailynews] tool-based image_layout produced no images; fallback to legacy layout"
                    )
            except Exception as e:
                astrbot_logger.warning("[dailynews] tool-based image_layout failed: %s", e, exc_info=True)

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
                local_path = Path(merged).resolve().as_posix()
                image_urls_for_model = [f"file:///{local_path}"]
                preview_hint = "你将收到一张“拼接预览图”，从上到下依次对应候选图片。"
            except Exception as e:
                astrbot_logger.warning("[dailynews] image_layout preview merge failed: %s", e, exc_info=True)

        system_prompt = (prompts.image_layout_system + "\n").strip() + "\n"
        system_prompt += f"- 总图片数不超过 {max_images_total} 张；单个来源不超过 {max_images_per_source} 张。\n"

        if image_plan:
            system_prompt += "- image_plan 表示主 Agent 对各来源图片数量的约束，请优先满足。\n"
        if preview_hint:
            system_prompt += f"- {preview_hint}\n"

        # Provide the model with structural/context hints to reduce “错位插图”。
        sections_hint: List[Dict[str, Any]] = []
        try:
            for s in _split_sections(draft_markdown):
                sections_hint.append(
                    {
                        "title": str(s.get("title") or ""),
                        "level": int(s.get("level") or 0),
                        "sample": str(s.get("body") or "")[:220],
                    }
                )
        except Exception:
            sections_hint = []

        source_sections_hint: List[Dict[str, Any]] = []
        try:
            for src, txt in source_text_by_source.items():
                source_sections_hint.append(
                    {
                        "source": src,
                        "section_snippet": (txt or "")[:520],
                        "keywords": sorted(list(_tokens(txt)))[:24],
                    }
                )
        except Exception:
            source_sections_hint = []

        baseline_patched = ""
        try:
            picked_by_source_hint: Dict[str, List[str]] = {}
            total = 0
            for src, urls in images_by_source.items():
                wanted = plan_by_source.get(src)
                if wanted is None or wanted <= 0:
                    wanted = max_images_per_source
                wanted = min(int(wanted), max_images_per_source)
                take = urls[: max(1, wanted)]
                if total + len(take) > max_images_total:
                    take = take[: max(0, max_images_total - total)]
                if take:
                    picked_by_source_hint[src] = take
                    total += len(take)
                if total >= max_images_total:
                    break
            if picked_by_source_hint:
                baseline_patched = _insert_images_near_sections(
                    draft_markdown,
                    picked_by_source=picked_by_source_hint,
                    source_text_by_source=source_text_by_source,
                )
        except Exception:
            baseline_patched = ""

        payload = {
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "constraints": {
                "max_images_total": max_images_total,
                "max_images_per_source": max_images_per_source,
            },
            "image_plan": image_plan or {},
            "image_candidates": images_by_source,
            "image_catalog": image_catalog,
            "sections": sections_hint,
            "source_sections": source_sections_hint,
            "draft_markdown": draft_markdown,
            "baseline_patched_markdown": baseline_patched,
            "output_schema": {
                "patched_markdown": "string",
                "request_image_urls": ["string(optional)"],
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

        # Optional tool-like loop: the model can ask to preview certain candidate URLs.
        # We will merge them into one preview image and feed it back as a vision input.
        if pass_images_to_model and int(request_max_requests) > 0:
            request_budget = int(request_max_requests)
            allowed_urls: set[str] = set()
            try:
                for _, urls in images_by_source.items():
                    for u in urls:
                        if isinstance(u, str) and u.strip():
                            allowed_urls.add(u.strip())
            except Exception:
                allowed_urls = set()

            async def _make_request_preview(urls: List[str]) -> Optional[str]:
                if not urls:
                    return None
                out = get_plugin_data_dir("image_layout_requests") / (
                    f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(urls)}.jpg"
                )
                merged = await merge_images_vertical(
                    urls,
                    out_path=out,
                    max_width=preview_max_width,
                    gap=preview_gap,
                )
                local_path = Path(merged).resolve().as_posix()
                return f"file:///{local_path}"

            while request_budget > 0:
                req = data.get("request_image_urls")
                if not isinstance(req, list) or not req:
                    break

                urls = [str(u).strip() for u in req if str(u).strip()]
                if allowed_urls:
                    urls = [u for u in urls if u in allowed_urls]
                urls = urls[: max(1, int(request_max_images))]
                if not urls:
                    break

                request_budget -= 1
                try:
                    req_preview = await _make_request_preview(urls)
                except Exception as e:
                    astrbot_logger.warning("[dailynews] image_layout request preview failed: %s", e, exc_info=True)
                    req_preview = None

                payload2 = dict(payload)
                payload2["requested_image_urls"] = urls
                payload2["requested_images_preview"] = "已合成为候选图预览（见图片输入）"

                try:
                    resp2 = await astrbot_context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=json.dumps(payload2, ensure_ascii=False),
                        system_prompt=system_prompt,
                        image_urls=[
                            *(
                                image_urls_for_model
                                if isinstance(image_urls_for_model, list)
                                else []
                            ),
                            *([req_preview] if req_preview else []),
                        ],
                    )
                    raw2 = getattr(resp2, "completion_text", "") or ""
                    data2 = _json_from_text(raw2)
                    if isinstance(data2, dict):
                        data = data2
                        # If the model already produced the final markdown, stop early.
                        if str(data.get("patched_markdown") or "").strip():
                            break
                    else:
                        break
                except Exception as e:
                    astrbot_logger.warning("[dailynews] image_layout request loop llm failed: %s", e, exc_info=True)
                    break

        patched = str(data.get("patched_markdown") or "").strip()
        if not patched:
            astrbot_logger.warning("[dailynews] image_layout patched_markdown empty; skip")
            return _fallback_insert(draft_markdown)

        if not re.search(r"!\[[^\]]*\]\(", patched) and "<img" not in patched:
            astrbot_logger.warning("[dailynews] image_layout produced no images; fallback insert")
            return _fallback_insert(patched)

        if refine_cfg.enabled and bool(pass_images_to_model):
            try:
                refiner = LayoutRefiner(system_prompt=prompts.layout_refiner_system)
                refined = await refiner.refine(
                    markdown=patched,
                    images_by_source=images_by_source,
                    image_catalog=image_catalog,
                    user_config=user_config,
                    astrbot_context=astrbot_context,
                    provider_id=provider_id,
                )
                if (refined or "").strip():
                    patched = refined
            except Exception as e:
                astrbot_logger.warning("[dailynews] image_layout refine failed: %s", e, exc_info=True)

        return patched
