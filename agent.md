# astrbot_dailynews_agent 维护指南

## 架构总览

这是一套“抓取 → 多 Agent 写作/汇总 → 图片插入 → 渲染成图 → 推送”的日报工作流插件。

核心链路（从上到下）：

1. `workflow/scheduler.py`：定时触发生成、向配置的目标会话推送
2. `workflow/workflow_manager.py`：工作流编排（抓取/汇报/写作/汇总/图片排版）
3. `workflow/image_layout_agent.py`：只负责“挑图/插图”（LLM 输出 patched_markdown）
4. `workflow/layout_refiner.py`：可选的“渲染预览回喂迭代”（LLM 看到预览图后继续改 Markdown）
5. `workflow/render_pipeline.py`：统一的“Markdown → HTML → 图片”渲染入口（供 `main.py` 与 `scheduler.py` 共用）

## 入口与职责

- `main.py`
  - 插件入口（Star），注册指令、注册 tools
  - 指令触发的“即时渲染并发送”会走 `workflow/render_pipeline.py`

- `workflow/scheduler.py`
  - 定时任务与推送
  - `_render_content_images()` 只负责把 markdown 渲染成图片路径（统一走 `workflow/render_pipeline.py`）
  - `_normalized_config()` 只做“类型/缺省补齐”（缺省值来自 `workflow/config_models.py`）

- `workflow/workflow_manager.py`
  - 多来源抓取/分析/写作/汇总
  - 图片相关：调用 `ImagePlanAgent`（可选）→ `ImageLayoutAgent`（可选）

## 配置：单一真相（建议）

代码侧请优先使用 `workflow/config_models.py` 的 dataclass 读取配置：

- 渲染相关：`RenderPipelineConfig`、`RenderImageStyleConfig`
- 图片插入相关：`ImageLayoutConfig`
- 迭代审稿相关：`LayoutRefineConfig`

UI/配置文件的 schema 在 `_conf_schema.json`，但**业务代码不要在多个文件里重复硬编码默认值**。

## Prompt 规范

与 LLM 交互的系统提示词外置到：

- `templates/prompts/image_layout_agent_system.txt`（挑图/插图）
- `templates/prompts/layout_refiner_system.txt`（渲染预览回喂迭代）

代码里只做“动态约束拼接”（例如最大图片数），不要把大段 prompt 写在 `.py` 里。

## 调试建议

- 快速看当前生效配置：`/news_config`
- 快速看渲染效果：`/news_test_md html`
- 图片候选检查：`/news_image_debug preview`
- 语法检查：`python -m compileall -q .`

## 代码改动建议（后续）

- 新增配置项：同时更新 `_conf_schema.json` 与 `workflow/config_models.py`
- 新增渲染策略：优先改 `workflow/render_pipeline.py`，不要在 `main.py`/`scheduler.py` 各写一份
- 新增“审稿/迭代”能力：放到 `workflow/layout_refiner.py`，不要再塞回 `workflow/image_layout_agent.py`

