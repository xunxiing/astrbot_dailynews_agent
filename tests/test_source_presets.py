from __future__ import annotations

import asyncio
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.plugins.astrbot_dailynews_agent.workflow.agents.sources.github_source import (
    fetch_github_snapshot_for_source,
)
from data.plugins.astrbot_dailynews_agent.workflow.agents.sources.rss_agent import (
    RssSubAgent,
)
from data.plugins.astrbot_dailynews_agent.workflow.core.config_models import (
    ASTRBOT_RELEASES_ATOM_URL,
    NewsSourcesConfig,
)
from data.plugins.astrbot_dailynews_agent.workflow.core.models import NewsSourceConfig


class SourcePresetConfigTests(unittest.TestCase):
    def test_astrbot_preset_expands_builtin_sources(self):
        sources = NewsSourcesConfig.from_mapping(
            {"source_preset_tags": ["astrbot"]}
        ).sources

        self.assertEqual(
            [(source.type, source.url) for source in sources],
            [
                ("plugin_registry", "official"),
                ("github", "AstrBotDevs/AstrBot"),
                ("rss", ASTRBOT_RELEASES_ATOM_URL),
            ],
        )

        by_key = {(source.type, source.url): source for source in sources}
        self.assertEqual(
            by_key[("plugin_registry", "official")].meta["since_hours"], 24
        )
        self.assertEqual(
            by_key[("github", "AstrBotDevs/AstrBot")].meta["since_hours"], 24
        )
        self.assertEqual(
            by_key[("rss", ASTRBOT_RELEASES_ATOM_URL)].meta["since_hours"], 24
        )

    def test_manual_sources_win_when_presets_expand_duplicates(self):
        sources = NewsSourcesConfig.from_mapping(
            {
                "source_preset_tags": ["astrbot"],
                "news_sources": [
                    {
                        "__template_key": "plugin_registry_official",
                        "name": "Manual Registry",
                        "since_hours": 12,
                        "max_plugins": 9,
                    },
                    {
                        "__template_key": "github",
                        "name": "Manual Core",
                        "repo": "AstrBotDevs/AstrBot",
                        "since_hours": 12,
                        "max_releases": 1,
                        "max_commits": 2,
                        "max_prs": 3,
                    },
                    {
                        "__template_key": "rss",
                        "name": "Manual Releases",
                        "url": ASTRBOT_RELEASES_ATOM_URL,
                        "since_hours": 12,
                        "max_articles": 2,
                    },
                ],
            }
        ).sources

        self.assertEqual(len(sources), 3)

        by_key = {(source.type, source.url): source for source in sources}
        self.assertEqual(by_key[("plugin_registry", "official")].name, "Manual Registry")
        self.assertEqual(
            by_key[("plugin_registry", "official")].meta["since_hours"], 12
        )
        self.assertEqual(by_key[("github", "AstrBotDevs/AstrBot")].name, "Manual Core")
        self.assertEqual(
            by_key[("github", "AstrBotDevs/AstrBot")].meta["since_hours"], 12
        )
        self.assertEqual(
            by_key[("rss", ASTRBOT_RELEASES_ATOM_URL)].name, "Manual Releases"
        )
        self.assertEqual(
            by_key[("rss", ASTRBOT_RELEASES_ATOM_URL)].meta["since_hours"], 12
        )


class SourcePresetRuntimeTests(unittest.TestCase):
    def test_preset_github_source_runs_without_raw_github_template(self):
        source = NewsSourceConfig(
            name="AstrBot Core",
            url="AstrBotDevs/AstrBot",
            type="github",
            priority=1,
            max_articles=1,
            album_keyword=None,
            meta={
                "since_hours": 24,
                "max_releases": 3,
                "max_commits": 10,
                "max_prs": 10,
            },
        )
        captured: dict[str, object] = {}

        async def fake_fetch_repo_snapshot(
            self,
            *,
            owner: str,
            repo: str,
            since: datetime,
            max_commits: int,
            max_prs: int,
            max_releases: int,
        ) -> dict[str, object]:
            captured.update(
                {
                    "owner": owner,
                    "repo": repo,
                    "since": since,
                    "max_commits": max_commits,
                    "max_prs": max_prs,
                    "max_releases": max_releases,
                }
            )
            return {
                "repo": {"full_name": f"{owner}/{repo}"},
                "window": {"hours": 24},
                "releases_recent": [],
                "commits_recent": [],
                "prs_recent": [],
            }

        with patch(
            "data.plugins.astrbot_dailynews_agent.workflow.agents.sources.github_source.GitHubClient.fetch_repo_snapshot",
            new=fake_fetch_repo_snapshot,
        ):
            snapshot = asyncio.run(
                fetch_github_snapshot_for_source(
                    source=source,
                    user_config={
                        "source_preset_tags": ["astrbot"],
                        "news_sources": [],
                        "github_enabled": False,
                    },
                )
            )

        self.assertIsNotNone(snapshot)
        self.assertEqual(captured["owner"], "AstrBotDevs")
        self.assertEqual(captured["repo"], "AstrBot")
        self.assertEqual(captured["max_releases"], 3)
        self.assertEqual(captured["max_commits"], 10)
        self.assertEqual(captured["max_prs"], 10)
        since = captured["since"]
        self.assertIsInstance(since, datetime)
        self.assertLess(
            abs(
                (
                    datetime.now(tz=timezone.utc) - since  # type: ignore[operator]
                ).total_seconds()
                - 24 * 3600
            ),
            10,
        )

    def test_rss_since_hours_filters_outdated_items(self):
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        source = NewsSourceConfig(
            name="AstrBot 版本更新",
            url=ASTRBOT_RELEASES_ATOM_URL,
            type="rss",
            priority=1,
            max_articles=5,
            album_keyword=None,
            meta={"since_hours": 24, "timeout_s": 20},
        )

        async def fake_fetch_rss_feed(url: str, **kwargs):
            self.assertEqual(url, ASTRBOT_RELEASES_ATOM_URL)
            self.assertFalse(kwargs["keep_only_report_day"])
            return {
                "feed_title": "AstrBot Releases",
                "feed_url": url,
                "report_date": "",
                "items": [
                    {
                        "title": "v1 recent",
                        "link": "https://example.invalid/recent",
                        "published_ts": now_ts - 2 * 3600,
                    },
                    {
                        "title": "v0 old",
                        "link": "https://example.invalid/old",
                        "published_ts": now_ts - 48 * 3600,
                    },
                ],
            }

        with patch(
            "data.plugins.astrbot_dailynews_agent.workflow.agents.sources.rss_agent.fetch_rss_feed",
            new=fake_fetch_rss_feed,
        ):
            source_name, articles = asyncio.run(
                RssSubAgent().fetch_latest_articles(source, user_config={})
            )

        self.assertEqual(source_name, "AstrBot 版本更新")
        self.assertEqual([item["title"] for item in articles], ["v1 recent"])


if __name__ == "__main__":
    unittest.main()
