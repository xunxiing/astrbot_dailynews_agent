#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试插件核心功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow.agents import NewsSourceConfig, NewsWorkflowManager, WechatSubAgent

def test_plugin_core():
    """测试插件核心功能"""
    print("开始测试插件核心功能...")
    
    # 测试1: 初始化工作流管理器
    print("\n1. 测试初始化工作流管理器...")
    manager = NewsWorkflowManager()
    print("[OK] 工作流管理器初始化成功")
    
    # 测试2: 注册子Agent
    print("\n2. 测试注册子Agent...")
    manager.register_sub_agent('wechat', WechatSubAgent)
    print("[OK] 子Agent注册成功")
    
    # 测试3: 添加新闻源
    print("\n3. 测试添加新闻源...")
    source_config = NewsSourceConfig(
        name="测试源",
        url="https://mp.weixin.qq.com/s/test",
        type="wechat",
        priority=1,
        max_articles=3
    )
    manager.add_source(source_config)
    print("[OK] 新闻源添加成功")
    print(f"  当前新闻源数量: {len(manager.news_sources)}")
    
    # 测试4: 测试WechatSubAgent初始化
    print("\n4. 测试WechatSubAgent初始化...")
    sub_agent = WechatSubAgent()
    print("[OK] WechatSubAgent初始化成功")
    
    print("\nDONE: 所有核心功能测试通过！")
    
    return True

if __name__ == "__main__":
    test_plugin_core()
