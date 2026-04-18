from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.plugins.astrbot_dailynews_agent.workflow.agents.react.orchestrator import (
    _build_target_source_snapshot,
)
from data.plugins.astrbot_dailynews_agent.workflow.core.models import NewsSourceConfig


class ReactSourceContextTests(unittest.TestCase):
    def test_target_source_snapshot_ignores_sources_without_items(self):
        sources = [
            NewsSourceConfig(
                name="AstrBot 插件市场",
                url="official",
                type="plugin_registry",
                priority=1,
                max_articles=20,
                album_keyword=None,
            ),
            NewsSourceConfig(
                name="AstrBot Core",
                url="AstrBotDevs/AstrBot",
                type="github",
                priority=1,
                max_articles=1,
                album_keyword=None,
            ),
        ]
        fetched = {
            "AstrBot 插件市场": [],
            "AstrBot Core": [
                {
                    "title": "fix: repair context pollution",
                    "url": "https://github.com/AstrBotDevs/AstrBot/commit/abc123",
                }
            ],
        }

        snapshot = _build_target_source_snapshot(sources=sources, fetched=fetched)

        self.assertEqual(len(snapshot["sources"]), 1)
        self.assertEqual(snapshot["sources"][0]["name"], "AstrBot Core")
        self.assertEqual(snapshot["target_urls"], ["AstrBotDevs/AstrBot"])


if __name__ == "__main__":
    unittest.main()
