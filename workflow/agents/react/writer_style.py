from __future__ import annotations

REACT_CONCISE_REPORT_STYLE = """
短句日报格式（高优先级）：
1. 每个事件优先写成 1 行【核心事件】+ 最多 1 行【核心数据/关键特性】。
2. 第一行【核心事件】必填：用“[主体] + [动作/发布] + [核心标的/状态]”客观陈述事实，不使用华丽形容词。
3. 第二行严格可选：只有原始素材明确包含具体测试成绩/数据指标、影响使用的关键特性、明确开放状态时，才允许输出。
4. 如果没有上述硬信息，绝对不要为了凑格式补第二行；不要解释常识，不要扩写背景，不要写“这意味着”“这标志着”“值得关注的是”之类主观过渡。
5. 除非用户明确要求深度解读，否则不要把单条新闻扩写成小作文。
""".strip()

REACT_CHIEF_EDITOR_CONCISE_HINT = (
    "When planning or asking the writer to draft the report, default to compact event "
    "items: one fact line plus at most one optional hard-detail line. Do not ask for "
    "background explainers or scene-setting unless the user explicitly asks for depth."
)


def compose_react_writer_style_hint(style_hint: str) -> str:
    extra = str(style_hint or "").strip()
    if not extra:
        return REACT_CONCISE_REPORT_STYLE
    return f"{REACT_CONCISE_REPORT_STYLE}\n\n附加风格偏好：\n{extra}"
