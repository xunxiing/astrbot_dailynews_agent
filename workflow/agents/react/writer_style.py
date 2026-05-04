from __future__ import annotations

from ...pipeline.rendering import load_template

REACT_CONCISE_REPORT_STYLE = str(
    load_template("templates/prompts/react_writer_style_coverage.txt") or ""
).strip()

REACT_CHIEF_EDITOR_CONCISE_HINT = str(
    load_template("templates/prompts/react_chief_editor_hint.txt") or ""
).strip()


def compose_react_writer_style_hint(style_hint: str) -> str:
    extra = str(style_hint or "").strip()
    if not extra:
        return REACT_CONCISE_REPORT_STYLE
    return f"{REACT_CONCISE_REPORT_STYLE}\n\nAdditional style preference:\n{extra}"
