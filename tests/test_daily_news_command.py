from __future__ import annotations

import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_MAIN_MODULE = "data.plugins.astrbot_dailynews_agent.main"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyTask:
    def cancel(self) -> None:
        return None


class _DummyLoop:
    def create_task(self, coro):
        try:
            coro.close()
        except Exception:
            pass
        return _DummyTask()


class _DummyScheduler:
    def __init__(self, context, config):
        self.context = context
        self.config = config
        self.prepare_calls: list[dict[str, object]] = []
        self.publish_calls: list[dict[str, object]] = []
        self.send_calls: list[dict[str, object]] = []

    async def start(self):
        return None

    async def stop(self):
        return None

    def get_config_snapshot(self):
        return {
            "report_cache_enabled": True,
            "source_preset_tags": [],
            "news_sources": [
                {"__template_key": "github", "repo": "owner/repo"},
            ],
        }

    async def prepare_report(self, cfg, *, source="manual", prefer_cache=True):
        self.prepare_calls.append(
            {
                "cfg": dict(cfg),
                "source": source,
                "prefer_cache": prefer_cache,
            }
        )
        return {"content": "tag report", "cache_hit": False}

    async def publish_report_to_astrbook(self, content, *, config):
        self.publish_calls.append({"content": content, "config": dict(config)})
        return {"ok": True}

    async def _send_to_targets(self, content, targets, *, config, prepared=None):
        self.send_calls.append(
            {
                "content": content,
                "targets": list(targets),
                "config": dict(config),
                "prepared": dict(prepared or {}),
            }
        )
        return 0


class _DummyContext:
    def __init__(self):
        self.loop = _DummyLoop()
        self.provider_manager = SimpleNamespace(
            llm_tools=SimpleNamespace(func_list=[])
        )


class _DummyEvent:
    unified_msg_origin = "webchat:FriendMessage:webchat!user!conv"

    def plain_result(self, text):
        return str(text)


async def _collect_async_gen(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


class DailyNewsCommandTests(unittest.TestCase):
    def test_daily_news_with_preset_tag_keeps_cache_enabled_when_not_forced(self):
        module = importlib.import_module(PLUGIN_MAIN_MODULE)
        original_scheduler = module.DailyNewsScheduler
        module.DailyNewsScheduler = _DummyScheduler
        try:
            plugin = module.DailyNewsPlugin(context=_DummyContext(), config={})
        finally:
            module.DailyNewsScheduler = original_scheduler

        event = _DummyEvent()
        results = asyncio.run(_collect_async_gen(plugin.daily_news(event, "astrbot")))

        self.assertEqual(results[0], "正在准备仅包含预设标签 astrbot 的日报，请稍候...")
        self.assertEqual(len(plugin.scheduler.prepare_calls), 1)
        prepare_call = plugin.scheduler.prepare_calls[0]
        self.assertTrue(prepare_call["prefer_cache"])
        self.assertEqual(prepare_call["cfg"]["report_cache_scope"], "preset:astrbot")

    def test_daily_news_force_with_preset_tag_uses_tag_scoped_cache(self):
        module = importlib.import_module(PLUGIN_MAIN_MODULE)
        original_scheduler = module.DailyNewsScheduler
        module.DailyNewsScheduler = _DummyScheduler
        try:
            plugin = module.DailyNewsPlugin(context=_DummyContext(), config={})
        finally:
            module.DailyNewsScheduler = original_scheduler

        event = _DummyEvent()
        results = asyncio.run(_collect_async_gen(plugin.daily_news(event, "force astrbot")))

        self.assertEqual(results[0], "正在强制刷新仅包含预设标签 astrbot 的日报，请稍候...")
        self.assertEqual(results[-1], "tag report")
        self.assertEqual(len(plugin.scheduler.prepare_calls), 1)
        prepare_call = plugin.scheduler.prepare_calls[0]
        cfg = prepare_call["cfg"]
        self.assertEqual(prepare_call["source"], "manual")
        self.assertFalse(prepare_call["prefer_cache"])
        self.assertEqual(cfg["source_preset_tags"], ["astrbot"])
        self.assertEqual(cfg["news_sources"], [])
        self.assertEqual(cfg["report_cache_scope"], "preset:astrbot")
        self.assertEqual(len(plugin.scheduler.publish_calls), 0)
        self.assertEqual(plugin.scheduler.send_calls[0]["config"]["report_cache_scope"], "preset:astrbot")

    def test_daily_news_rejects_unknown_preset_tag(self):
        module = importlib.import_module(PLUGIN_MAIN_MODULE)
        original_scheduler = module.DailyNewsScheduler
        module.DailyNewsScheduler = _DummyScheduler
        try:
            plugin = module.DailyNewsPlugin(context=_DummyContext(), config={})
        finally:
            module.DailyNewsScheduler = original_scheduler

        event = _DummyEvent()
        results = asyncio.run(_collect_async_gen(plugin.daily_news(event, "unknown")))

        self.assertEqual(len(results), 1)
        self.assertIn("未知的信息源预设标签：unknown", results[0])
        self.assertIn("astrbot", results[0])
        self.assertEqual(plugin.scheduler.prepare_calls, [])
        self.assertEqual(plugin.scheduler.send_calls, [])


if __name__ == "__main__":
    unittest.main()
