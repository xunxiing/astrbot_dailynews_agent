import asyncio
from typing import Any, Optional

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


class LLMRunner:
    def __init__(
        self,
        astrbot_context: Any,
        timeout_s: int = 180,
        max_retries: int = 1,
        provider_id: Optional[str] = None,
    ):
        self._ctx = astrbot_context
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._provider_id = provider_id or None

    async def ask(self, *, system_prompt: str, prompt: str) -> str:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, int(self._max_retries) + 2):
            try:
                if self._provider_id:
                    coro = self._ctx.llm_generate(
                        chat_provider_id=self._provider_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    )
                    resp = await asyncio.wait_for(coro, timeout=self._timeout_s)
                    return getattr(resp, "completion_text", "") or ""

                provider = self._ctx.get_using_provider()
                coro = provider.text_chat(
                    prompt=prompt,
                    session_id=None,  # deprecated but kept for compatibility
                    contexts=[],
                    image_urls=[],
                    func_tool=None,
                    system_prompt=system_prompt,
                )
                resp = await asyncio.wait_for(coro, timeout=self._timeout_s)
                if getattr(resp, "role", None) == "assistant":
                    return getattr(resp, "completion_text", "") or ""
                return (
                    getattr(resp, "completion_text", "")
                    or str(getattr(resp, "raw_completion", ""))
                    or ""
                )
            except asyncio.TimeoutError as e:
                last_exc = e
                astrbot_logger.warning(
                    "[dailynews] LLM timeout after %ss (attempt %s/%s)",
                    self._timeout_s,
                    attempt,
                    int(self._max_retries) + 1,
                )
            except Exception as e:
                last_exc = e
                astrbot_logger.warning(
                    "[dailynews] LLM call failed (attempt %s/%s): %s",
                    attempt,
                    int(self._max_retries) + 1,
                    e,
                    exc_info=True,
                )

            await asyncio.sleep(min(2.0 * attempt, 6.0))

        raise RuntimeError(
            f"LLM call failed after {int(self._max_retries) + 1} attempts: "
            f"{type(last_exc).__name__ if last_exc else 'UnknownError'}"
        )
