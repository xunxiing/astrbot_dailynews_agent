from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_MAIN_MODULE = "data.plugins.astrbot_dailynews_agent.main"


class PluginImportCycleTests(unittest.TestCase):
    def test_plugin_main_imports_cleanly_in_fresh_interpreter(self):
        # A fresh interpreter exposes circular imports that a warm sys.modules cache can hide.
        code = textwrap.dedent(
            f"""
            import importlib

            importlib.import_module({PLUGIN_MAIN_MODULE!r})
            print("plugin main import ok")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"fresh import failed for {PLUGIN_MAIN_MODULE}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )
        self.assertIn("plugin main import ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
