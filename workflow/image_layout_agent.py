import json
from datetime import datetime
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)

from .config_models import ImageLayoutConfig, LayoutPrompts, LayoutRefineConfig
from .image_utils import get_plugin_data_dir, image_url_to_data_uri, merge_images_vertical
from .layout_refiner import LayoutRefiner
from .models import SubAgentResult
from .rendering import load_template
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
        shuffle_candidates = bool(layout_cfg.shuffle_candidates)
        shuffle_seed = str(layout_cfg.shuffle_seed or "").strip()

        refine_cfg = LayoutRefineConfig.from_mapping(user_config)

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

        prompts = LayoutPrompts.from_files(load_template=load_template)
        system_prompt = (prompts.image_layout_system + "\n").strip() + "\n"
        system_prompt += f"- 总图片数不超过 {max_images_total} 张；单个来源不超过 {max_images_per_source} 张。\n"

        if image_plan:
            system_prompt += "- image_plan 表示主 Agent 对各来源图片数量的约束，请优先满足。\n"
        if preview_hint:
            system_prompt += f"- {preview_hint}\n"

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

        if refine_cfg.enabled:
            try:
                refiner = LayoutRefiner(system_prompt=prompts.layout_refiner_system)
                refined = await refiner.refine(
                    markdown=patched,
                    images_by_source=images_by_source,
                    user_config=user_config,
                    astrbot_context=astrbot_context,
                    provider_id=provider_id,
                )
                if (refined or "").strip():
                    patched = refined
            except Exception as e:
                astrbot_logger.warning("[dailynews] image_layout refine failed: %s", e, exc_info=True)

        return patched
