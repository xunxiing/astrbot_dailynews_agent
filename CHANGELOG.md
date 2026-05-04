# Changelog

所有项目的显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [v1.2.7] - 2024-05-04

### Added
- 新增插件 Logo

## [v1.2.6] - 2024-05-04

### Added
- 新增插件 Logo

## [v1.2.4] - 2024-05-04

### Fixed
- 修复 GitHub Actions 中 Python 脚本传参数问题
- 改用 `-c` 参数传递版本号，避免 heredoc 导致的 shell 语法问题

## [v1.2.3] - 2024-05-04

### Fixed
- 修复 GitHub Actions 自动发版脚本无法正确提取 changelog 的问题
- 优化版本号解析逻辑，支持带注释的版本字段

### Changed
- 使用 Python 脚本替代 shell 命令提取 changelog，提高可靠性

## [v1.2.2] - 2024-05-04

### Added
- 新增 GitHub Actions 自动发版工作流
- 新增 CHANGELOG.md 维护规范

## [v1.2.1] - 2024-05-04

### Changed
- 优化提示词模板

## [v1.2.0] - 2024-05-04

### Added
- 实现React多Agent架构
- 新增多信息源搜索平台支持（Grok、Tavily）
- 新增微信公众号数据采集
- 新增米游社数据提取模块
- 新增图像处理工具(image_tools)
- 新增自动日报生成功能
- 新增RSS订阅支持
- 新增Twitter信息源支持
- 新增GitHub动态追踪
- 新增AstrBook内容获取

### Changed
- 重构微信公众号数据采集逻辑
- 优化渲染模板和配置系统
- 移除Playwright依赖，实现零浏览器组件部署

### Fixed
- 修复微信公众号解析失败的问题

## [v1.0.0] - 2024-05-04

### Added
- 🎉 初始版本发布
