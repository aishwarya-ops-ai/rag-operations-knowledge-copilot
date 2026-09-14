from __future__ import annotations

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS


class GenerationUnavailableError(RuntimeError):
    """Raised when the configured local generation service cannot answer."""


class GenerationProvider(Protocol):
    def generate(self, system_instruction: str, user_prompt: str) -> str:
        ...


class OllamaProvider:
    """Minimal local Ollama client using only Python's standard library."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0,
                "num_predict": 220,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            answer = body["message"]["content"].strip()
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
            raise GenerationUnavailableError(
                f"Local Ollama model '{self.model}' is unavailable at {self.base_url}."
            ) from error

        if not answer:
            raise GenerationUnavailableError("The local model returned an empty response.")
        return answer

