from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.plugins.astrbot_dailynews_agent.workflow.agents.react.react_agent import (
    ReActAgent,
)
from data.plugins.astrbot_dailynews_agent.workflow.agents.react.writer_style import (
    REACT_CHIEF_EDITOR_CONCISE_HINT,
    REACT_CONCISE_REPORT_STYLE,
    compose_react_writer_style_hint,
)
from data.plugins.astrbot_dailynews_agent.workflow.core.config_models import (
    ReactAgentConfig,
)


class ReactWriterStyleTests(unittest.TestCase):
    def test_compose_style_hint_returns_default_rules_when_empty(self):
        self.assertEqual(compose_react_writer_style_hint(""), REACT_CONCISE_REPORT_STYLE)

    def test_compose_style_hint_appends_extra_user_preference(self):
        style = compose_react_writer_style_hint("语气再冷静一点。")

        self.assertIn(REACT_CONCISE_REPORT_STYLE, style)
        self.assertIn("附加风格偏好：", style)
        self.assertIn("语气再冷静一点。", style)

    def test_react_agent_system_prompt_contains_concise_dispatch_hint(self):
        agent = ReActAgent(
            astrbot_context=SimpleNamespace(),
            registry=SimpleNamespace(),
            shared_memory=SimpleNamespace(),
            config=ReactAgentConfig(),
        )

        self.assertIn(REACT_CHIEF_EDITOR_CONCISE_HINT, agent._system_prompt())
        self.assertIn("one-liner", agent._system_prompt().lower())

    def test_react_writer_templates_include_compact_line_format(self):
        system_text = (
            PLUGIN_ROOT / "templates" / "prompts" / "react_writer_system.txt"
        ).read_text(encoding="utf-8")
        user_text = (
            PLUGIN_ROOT / "templates" / "prompts" / "react_writer_user.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("One-liner", user_text)
        self.assertIn("短句日报格式", system_text)
        self.assertTrue(
            ("单行体结构" in system_text) or ("One-liner" in system_text)
        )
        self.assertTrue(
            ("默认采用单行体" in user_text) or ("One-liner" in user_text)
        )


if __name__ == "__main__":
    unittest.main()
