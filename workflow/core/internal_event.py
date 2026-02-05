from __future__ import annotations

from typing import Any


def make_internal_event(*, session_id: str = "internal") -> Any:
    """
    Create a minimal AstrMessageEvent instance for internal tool-loop calls.

    AstrBot's tool_loop_agent requires an event, but our workflow does not run inside a real
    platform message handler.
    """
    try:
        from astrbot.core.platform.astr_message_event import AstrMessageEvent
        from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
        from astrbot.core.platform.message_type import MessageType
        from astrbot.core.platform.platform_metadata import PlatformMetadata
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"astrbot core event classes unavailable: {e}") from e

    msg = AstrBotMessage()
    msg.type = MessageType.OTHER_MESSAGE
    msg.self_id = "internal"
    msg.session_id = session_id
    msg.message_id = "internal"
    msg.sender = MessageMember(user_id="internal", nickname="internal")
    msg.message = []
    msg.message_str = ""
    msg.raw_message = {}

    platform = PlatformMetadata(
        name="internal",
        description="internal",
        id="internal",
    )
    return AstrMessageEvent(
        message_str="internal",
        message_obj=msg,
        platform_meta=platform,
        session_id=session_id,
    )
