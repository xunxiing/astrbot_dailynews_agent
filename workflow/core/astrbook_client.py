from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover
    import logging

    astrbot_logger = logging.getLogger(__name__)


ASTRBOOK_API_BASE = "https://book.astrbot.app"


class AstrBookClient:
    def __init__(self, *, token: str):
        self._token = (token or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        timeout_s: float = 15.0,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"error": "token_missing"}

        url = ASTRBOOK_API_BASE.rstrip("/") + "/" + endpoint.lstrip("/")
        timeout = aiohttp.ClientTimeout(total=float(timeout_s))
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method.upper(),
                    url,
                    headers=self._headers(),
                    params=params,
                    json=json_data,
                ) as resp:
                    if resp.status == 200:
                        try:
                            j = await resp.json()
                            return j if isinstance(j, dict) else {"data": j}
                        except Exception:
                            txt = await resp.text()
                            return {"error": "invalid_json", "text": txt[:400]}
                    if resp.status == 401:
                        return {"error": "token_invalid"}
                    if resp.status == 404:
                        return {"error": "not_found"}
                    txt = await resp.text()
                    return {
                        "error": f"http_{resp.status}",
                        "text": (txt or "")[:400],
                    }
        except asyncio.TimeoutError:
            return {"error": "timeout"}
        except aiohttp.ClientConnectorError:
            return {"error": "connect_failed"}
        except Exception as e:
            astrbot_logger.debug("[dailynews] astrbook request error: %s", e)
            return {"error": "request_error", "detail": str(e)[:200]}

    async def list_threads(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        category: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": max(1, int(page)), "page_size": page_size}
        if category:
            params["category"] = category
        return await self._request_json("GET", "/api/threads", params=params)

    async def read_thread(
        self,
        thread_id: int,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": max(1, int(page)),
            "page_size": max(1, int(page_size)),
        }
        return await self._request_json("GET", f"/api/threads/{int(thread_id)}", params=params)

    async def create_thread(
        self, *, title: str, content: str, category: str = "tech"
    ) -> dict[str, Any]:
        payload = {"title": title, "content": content, "category": category}
        return await self._request_json("POST", "/api/threads", json_data=payload, timeout_s=25.0)

