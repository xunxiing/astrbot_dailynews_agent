from astrbot.api.star import Context, Star, register

from .tools import WechatArticleMarkdownTool, WechatAlbumLatestArticlesTool


@register(
    "astrbot_plugin_dailynews",          # 插件名
    "your_name",                         # 作者
    "AI 早报 / 日报插件：提供微信公众号文章解析工具。",  # 描述
    "0.1.0",                             # 版本
    "https://github.com/your/repo",      # 仓库（可以先随便填）
)
class DailyNewsPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

        tools = [
            WechatArticleMarkdownTool(),
            WechatAlbumLatestArticlesTool(),
        ]

        # >= v4.5.1 推荐写法
        if hasattr(self.context, "add_llm_tools"):
            self.context.add_llm_tools(*tools)
        else:
            # 兼容老版本
            tool_mgr = self.context.provider_manager.llm_tools
            tool_mgr.func_list.extend(tools)

    async def terminate(self):
        """插件被卸载/停用时调用（目前不需要清理什么，就留空）。"""
        pass