# AI日报插件 - 多Agent工作流版

## 🎯 功能概述

这是一个基于AstrBot的AI日报插件，采用多Agent工作流架构，实现每日自动新闻汇总功能。

### ✨ 核心特性

- **多Agent工作流**: 主Agent调度 + 子Agent并行处理
- **自动日报**: 定时生成并发送日报
- **微信公众号支持**: 解析公众号文章并提取关键信息
- **灵活配置**: 支持自定义新闻源和调度时间
- **Markdown输出**: 美观的日报格式

## 🏗️ 架构设计

### 多Agent工作流

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   主Agent       │───▶│   子Agent集群     │───▶│   结果汇总       │
│   (调度决策)     │    │   (并行处理)      │    │   (生成日报)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### 主Agent职责
- 分析所有可用新闻源
- 制定处理策略和优先级
- 分配任务给子Agent
- 汇总所有结果生成最终日报

#### 子Agent职责
- 处理特定类型的新闻源
- 提取文章内容和关键信息
- 生成摘要和要点
- 返回结构化数据给主Agent

## 🚀 使用指南

### 基本命令

| 命令 | 描述 |
|------|------|
| `/daily_news` | 手动生成日报 |
| `/news_config` | 查看当前配置 |
| `/news_toggle` | 切换自动日报开关 |
| `/news_add_source URL` | 添加新闻源（公众号文章链接） |
| `/news_remove_source URL` | 删除新闻源 |
| `/news_config_help` | 显示配置帮助 |

### 配置示例

#### 1. 添加新闻源
```
/news_add_source AI科技早报 https://mp.weixin.qq.com/s?__biz=xxx wechat
```

#### 2. 查看配置
```
/news_config
```

#### 3. 手动生成日报
```
/daily_news
```

## ⚙️ 插件配置（推荐方式）

本插件使用 AstrBot 的 `_conf_schema.json` 进行可视化配置（WebUI 管理面板里直接改，不需要手改文件）。
AstrBot 会把配置落盘到 `data/config/astrbot_dailynews_agent_config.json` 并在插件 `__init__` 时注入 `config`。

如果你需要看配置结构，可在管理面板的「插件配置」里查看；也可以用 `/news_config` 查看当前配置快照。

关键配置项：
- `enabled`: 是否启用自动推送
- `schedule_time`: 每日生成时间（HH:MM）
- `target_sessions`: 推送目标会话（`unified_msg_origin` 列表；也支持直接填群号/会话ID如 `1030223077`）
- `news_sources`: 新闻源（list，每项填一个公众号文章链接）
- `twitter_enabled`: 是否启用 X/Twitter 信息源（需要代理）
- `twitter_targets`: X/Twitter 主页列表（例如 `https://x.com/openai`）
- `twitter_proxy`: 代理地址（支持 `http://` / `socks5://`）

### 配置说明

- **enabled**: 是否启用自动日报
- **schedule_time**: 日报生成时间 (格式: HH:MM)
- **target_groups**: 目标群组ID列表
- **target_users**: 目标用户ID列表
- **news_sources**: 自定义新闻源配置
- **max_sources_per_day**: 每天最多处理的新闻源数量
- **output_format**: 输出格式 (支持: markdown)
- **preferred_source_types**: 优先处理的新闻源类型

## 🔧 技术实现

### 文件结构
```
astrbot_dailynews_agent/
├── main.py                 # 主插件文件
├── requirements.txt        # 依赖包
├── workflow/              # 工作流模块
│   ├── __init__.py
│   ├── agents.py          # Agent实现
│   └── scheduler.py       # 定时调度器
├── tools/                 # 工具模块
│   ├── wechat_tools.py    # 微信工具
│   └── ...
├── analysis/              # 分析模块
│   └── wechatanalysis/
│       ├── analysis.py
│       └── latest_articles.py
└── data/                  # 数据目录
    └── (legacy) scheduler_config.json
```

### 核心类

#### NewsWorkflowManager
- 管理工作流生命周期
- 协调主Agent和子Agent
- 处理任务调度和结果汇总

#### MainNewsAgent
- 制定处理策略
- 分析新闻源优先级
- 生成最终日报内容

#### WechatSubAgent
- 处理微信公众号源
- 提取文章关键信息
- 生成结构化摘要

#### DailyNewsScheduler
- 定时任务调度
- 配置管理
- 自动发送日报

## 📝 开发说明

### 添加新的子Agent

1. 创建新的子Agent类，继承基础接口
2. 在`NewsWorkflowManager`中注册新的子Agent
3. 实现具体的源处理逻辑

```python
class MySubAgent:
    async def process_source(self, source_config: NewsSourceConfig, instruction: str) -> SubAgentResult:
        # 实现具体的处理逻辑
        pass
```

### 扩展新闻源类型

1. 在`NewsSourceConfig`中添加新的类型
2. 创建对应的子Agent实现
3. 更新工作流管理器的注册逻辑

## 🔍 调试和故障排除

### 常见问题

1. **日报生成失败**
   - 检查网络连接
   - 验证新闻源URL有效性
   - 查看插件日志获取详细错误信息

2. **定时任务不执行**
   - 确认插件已启用
   - 检查配置文件格式
   - 验证系统时间设置

3. **微信公众号解析失败**
   - 确认公众号链接格式正确
   - 检查Playwright依赖安装
   - 验证网络访问权限

### 日志查看

插件会在运行时输出详细日志，可以通过AstrBot的日志系统查看调试信息。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进插件功能。

### 开发环境

1. 安装依赖: `pip install -r requirements.txt`
2. 安装Playwright: `playwright install`
3. 配置开发环境

### 提交规范

- 遵循现有代码风格
- 添加必要的注释和文档
- 测试所有功能变更
- 更新相关文档

## 📄 许可证

MIT License - 详见LICENSE文件
