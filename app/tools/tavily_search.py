from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class TavilySearchTool:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.tavily.com",
        timeout_seconds: int = 15,
        max_results: int = 3,
        search_depth: str = "basic",
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.search_depth = search_depth

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int | None = None) -> dict[str, Any]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": max_results or self.max_results,
            "include_answer": True,
        }
        req = request.Request(
            url=f"{self.base_url}/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Tavily HTTP {exc.code}: {detail[:300]}") from exc
        except Exception as exc:
            raise RuntimeError(f"Tavily request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Tavily returned non-JSON response.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Tavily returned unexpected payload.")
        return parsed

