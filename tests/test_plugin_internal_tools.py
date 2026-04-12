from __future__ import annotations

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
    def __init__(self):
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


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

    async def start(self):
        return None

    async def stop(self):
        return None


class _DummyContext:
    def __init__(self):
        self.loop = _DummyLoop()
        self.provider_manager = SimpleNamespace(
            llm_tools=SimpleNamespace(func_list=[])
        )
        self.add_llm_tools_called = False

    def add_llm_tools(self, *tools) -> None:
        self.add_llm_tools_called = True
        raise AssertionError("plugin should not register internal tools globally")

    def get_config(self):
        return {}


class PluginInternalToolIsolationTests(unittest.TestCase):
    def test_plugin_init_does_not_register_internal_tools_globally(self):
        module = importlib.import_module(PLUGIN_MAIN_MODULE)
        original_scheduler = module.DailyNewsScheduler
        module.DailyNewsScheduler = _DummyScheduler
        try:
            context = _DummyContext()
            plugin = module.DailyNewsPlugin(context=context, config={})
        finally:
            module.DailyNewsScheduler = original_scheduler

        self.assertFalse(context.add_llm_tools_called)
        self.assertEqual(context.provider_manager.llm_tools.func_list, [])
        self.assertIsInstance(plugin.scheduler, _DummyScheduler)


if __name__ == "__main__":
    unittest.main()
