from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "workflow" / "core" / "wechat_freshness.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "wechat_freshness_test_module", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

article_is_new_since_recent_baseline = _MODULE.article_is_new_since_recent_baseline
build_article_fingerprint = _MODULE.build_article_fingerprint
merge_seen_articles = _MODULE.merge_seen_articles


class WechatFreshnessTests(unittest.TestCase):
    def test_new_article_is_accepted_against_recent_baseline(self):
        baseline_now = datetime(2026, 3, 16, 21, 0, 0)
        today_now = datetime(2026, 3, 17, 9, 30, 0)

        article_a = {
            "title": "baseline article",
            "url": "https://mp.weixin.qq.com/s?__biz=1&mid=100",
            "create_time": "1773600000",
        }
        article_b = {
            "title": "new article",
            "url": "https://mp.weixin.qq.com/s?__biz=1&mid=101",
            "create_time": "1773600000",
        }

        entry = merge_seen_articles({}, [article_a], now=baseline_now)

        self.assertFalse(
            article_is_new_since_recent_baseline(article_a, entry, now=today_now)
        )
        self.assertTrue(
            article_is_new_since_recent_baseline(article_b, entry, now=today_now)
        )

    def test_merge_seen_articles_keeps_multiple_same_day_articles(self):
        now = datetime(2026, 3, 17, 12, 0, 0)
        article_a = {
            "title": "morning post",
            "url": "https://mp.weixin.qq.com/s?__biz=1&mid=200",
            "create_time": "1773687600",
        }
        article_b = {
            "title": "afternoon post",
            "url": "https://mp.weixin.qq.com/s?__biz=1&mid=201",
            "create_time": "1773705600",
        }

        entry = merge_seen_articles({}, [article_a], now=now)
        entry = merge_seen_articles(entry, [article_b], now=now)

        seen = entry.get("seen_articles") or []
        fingerprints = {item["fingerprint"] for item in seen}

        self.assertEqual(len(seen), 2)
        self.assertIn(build_article_fingerprint(article_a), fingerprints)
        self.assertIn(build_article_fingerprint(article_b), fingerprints)

    def test_first_run_without_baseline_does_not_override(self):
        now = datetime(2026, 3, 17, 9, 30, 0)
        article = {
            "title": "unknown older article",
            "url": "https://mp.weixin.qq.com/s?__biz=1&mid=300",
            "create_time": "1773000000",
        }
        self.assertFalse(article_is_new_since_recent_baseline(article, {}, now=now))


if __name__ == "__main__":
    unittest.main()
