"""
兼容层：历史上插件把所有工作流代码都放在 `workflow/agents.py`。
现在已按职责拆分到多个模块，但保留原导入路径以避免外部代码/文档失效。
"""

from .llm import LLMRunner
from .main_agent import MainNewsAgent
from .miyoushe_agent import MiyousheSubAgent
from .github_agent import GitHubSubAgent
from .models import MainAgentDecision, NewsSourceConfig, SubAgentResult
from .twitter_agent import TwitterSubAgent
from .wechat_agent import WechatSubAgent
from .workflow_manager import NewsWorkflowManager

__all__ = [
    "LLMRunner",
    "MainAgentDecision",
    "MainNewsAgent",
    "GitHubSubAgent",
    "MiyousheSubAgent",
    "NewsSourceConfig",
    "NewsWorkflowManager",
    "SubAgentResult",
    "TwitterSubAgent",
    "WechatSubAgent",
]
