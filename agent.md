# DailyNews Agent

日报系统开发约定，面向新增信息源与工作流维护。

## 模式

- `react`: 主推模式。默认模式。
- `single`: 简化模式。用于最小链路生成。
- `multi`: 已移除。旧配置会自动兼容到 `react`，不要再新增 `multi` 分支。

## 核心原则

- 单文件单信息源。
- 不做中心化手工注册。
- 新增信息源时，默认只改 `workflow/agents/sources/<type>_agent.py`。
- 除非在改核心工作流，否则不要改 `workflow/pipeline/scheduler.py`、`workflow/pipeline/workflow_manager.py`、`main.py`。
- 新信息源优先走 `react`，不要为了接入一个源再引入新的 orchestration 分支。

## 自动注册

- 自动发现入口：`workflow/agents/sources/__init__.py`
- 扫描规则：只扫描 `workflow/agents/sources/*_agent.py`
- `source_type` 推导规则：
  - 优先读模块常量 `SOURCE_TYPE`
  - 否则用文件名去掉 `_agent` 后缀
- Agent 类推导规则：
  - 优先读模块常量 `SOURCE_AGENT_CLASS`
  - 否则自动寻找当前文件里唯一一个实现了 `fetch_latest_articles` 的类

## Source 文件约定

路径：

```text
workflow/agents/sources/<type>_agent.py
```

推荐最小骨架：

```python
from __future__ import annotations

from typing import Any

from ...core.models import NewsSourceConfig, SubAgentResult

SOURCE_TYPE = "demo_feed"


class DemoFeedSubAgent:
    async def fetch_latest_articles(
        self, source: NewsSourceConfig, user_config: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        return source.name, []

    async def analyze_source(
        self, source: NewsSourceConfig, articles: list[dict[str, Any]], llm: Any
    ) -> dict[str, Any]:
        return {
            "source_name": source.name,
            "topics": [],
            "today_angle": "",
            "article_count": len(articles or []),
        }

    async def process_source(
        self,
        source: NewsSourceConfig,
        instruction: str,
        articles: list[dict[str, Any]],
        llm: Any,
        user_config: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        return SubAgentResult(
            source_name=source.name,
            content="",
            summary="",
            key_points=[],
        )
```

## React 模式下的信息源接口

`react` 模式实际会调用这三个方法：

- `await agent.fetch_latest_articles(source, user_config)`
- `await agent.analyze_source(source, articles, llm)`
- `await agent.process_source(source, instruction, articles, llm, user_config=user_config)`

约束：

- `fetch_latest_articles` 必须返回 `(source.name, articles)`
- `articles` 必须是 `list[dict]`
- `analyze_source` 返回可序列化字典
- `process_source` 返回 `SubAgentResult`
- 三个方法都必须是 `async def`

结论：

- 如果一个源要支持 `react`，就不要只实现抓取，三个方法都应实现
- 如果只想做最低接入，可以先实现 `single` 可用的抓取能力，但主线功能视为未完成

## 配置约定

新信息源优先使用通用 `news_sources` 结构，不再新增中心化模板分支。

推荐配置：

```json
{
  "__template_key": "demo_feed",
  "type": "demo_feed",
  "name": "Demo Feed",
  "url": "https://example.com/feed.xml",
  "priority": 1,
  "max_articles": 5,
  "meta": {
    "token": "",
    "category": ""
  }
}
```

兼容规则：

- `__template_key` 可以写成自定义值，推荐与 `type` 相同
- `type` 必须与 source 文件注册出来的 `source_type` 一致
- `meta` 用于放 source 私有参数
- 未放进 `meta` 的额外字段也会被吸收到 `source.meta`

不要做的事：

- 不要为了一个新源去扩展 `NewsSourcesConfig.from_mapping()` 的大段 `if/elif`
- 不要为新源新增 `scheduler.register_sub_agent(...)`
- 不要把 source 私有逻辑写进 `workflow_manager`

## 需要改中心文件的唯一场景

- 改自动发现规则：改 `workflow/agents/sources/__init__.py`
- 改工作流模式：改 `workflow/core/config_models.py`、`workflow/pipeline/workflow_manager.py`
- 改调度或配置归一化：改 `workflow/pipeline/scheduler.py`
- 改面板显示项：改 `_conf_schema.json`

## 提交流程

新增一个信息源时，默认检查清单：

- 新增 `workflow/agents/sources/<type>_agent.py`
- 配置里新增一个 `news_sources` 项，`type=<type>`
- 本地通过：

```powershell
python -m py_compile workflow\agents\sources\<type>_agent.py workflow\pipeline\workflow_manager.py workflow\pipeline\scheduler.py workflow\core\config_models.py
```

## 禁止事项

- 禁止恢复 `multi-agent` 主流程
- 禁止新增手工 source 注册表
- 禁止让一个 source 分散在多个文件里
- 禁止把 source 接入建立在修改 `main.py` 的命令逻辑上
