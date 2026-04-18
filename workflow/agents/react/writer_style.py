from __future__ import annotations

REACT_CONCISE_REPORT_STYLE = """
单行体（One-liner）日报格式，高优先级覆盖其他写作偏好：
1. 每条资讯默认只写一行，严格遵循 `**[主体与动作]**：[硬核数据 1]，[硬核数据 2]；[转折/补充逻辑]。`
2. 严禁输出换行解释、项目符号、小标题或背景段落；把新闻当成纯粹的 changelog 来写。
3. 只提取客观参数和硬信息，如版本号、像素值、倍数、榜单、成本变化、开放状态、兼容性限制。
4. 禁止使用“标志着、预示着、这意味着、值得注意的是、开发者需注意”等主观评价和媒体废话。
5. 当增强与限制并存时，必须用“但”“同时”等连接词缝合成一行。
6. 每条资讯尽量控制在 60-80 字；若素材缺少硬核数据，则退化为 30 字以内的最短可用事实句，不要为凑格式编造对比或补背景。
""".strip()

REACT_CHIEF_EDITOR_CONCISE_HINT = (
    "When planning or asking the writer to draft the report, default to one-liner "
    "changelog-style items. Do not ask for background explainers, scene-setting, or "
    "multi-line expansions unless the user explicitly asks for depth."
)


def compose_react_writer_style_hint(style_hint: str) -> str:
    extra = str(style_hint or "").strip()
    if not extra:
        return REACT_CONCISE_REPORT_STYLE
    return f"{REACT_CONCISE_REPORT_STYLE}\n\n附加风格偏好：\n{extra}"
