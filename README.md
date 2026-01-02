# AI日报插件 - 多Agent工作流版

## 🎯 功能概述

这是一个基于AstrBot的AI日报插件，采用多Agent工作流架构，实现每日自动新闻汇总功能。

### ✨ 核心特性

- **多Agent工作流**: 主Agent调度 + 子Agent并行处理
- **自动日报**: 定时生成并发送日报
- **微信公众号支持**: 解析公众号文章并提取关键信息
- **灵活配置**: 支持自定义新闻源和调度时间
- **Markdown输出**: 美观的日报格式

## 信息源推荐：

<details>
<summary>科技类</summary>

这里是被折叠的内容

- https://mp.weixin.qq.com/s/QAVCOnnA5M8olmy7peBqsQ   *每日ai日报*
- 

</details>

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
   - 检查Playwright依赖安装，使用playwright install解决可能存在的playwright安装问题，如果有问题，建议先等待10-20分钟后重启astrbot，因为插件会静默下载playwright。
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

