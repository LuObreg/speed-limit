"""Type T transport for the configured local Ollama model."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol

LOG = logging.getLogger("speedlimit.llm")


class LLMClientError(RuntimeError):
    """Transport problem or unusable response from Ollama."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")[:200]
        raise LLMClientError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"invalid JSON from {url}: {exc}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMClientError(f"cannot reach {url}: {exc}") from exc


class OllamaClient:
    """Client for the local Ollama chat API."""

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self._url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s

    def complete(self, system: str, user: str) -> str:
        data = _post_json(
            f"{self._url}/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            self._timeout,
        )
        try:
            content = str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LLMClientError(f"unexpected Ollama response: {str(data)[:200]}") from exc
        LOG.info("Ollama response received: endpoint=%s model=%s", self._url, self._model)
        return content
