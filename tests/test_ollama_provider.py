import json
from urllib.error import URLError

import pytest

import src.generation.provider as provider_module
from src.generation.provider import GenerationUnavailableError, OllamaProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_provider_sends_grounding_messages(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:11434/api/chat"
        assert timeout == 5
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0
        assert payload["messages"] == [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "question and evidence"},
        ]
        return FakeResponse({"message": {"content": "Grounded answer [E1]"}})

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    provider = OllamaProvider(model="test-model", timeout_seconds=5)

    assert provider.generate("system rules", "question and evidence") == (
        "Grounded answer [E1]"
    )


def test_ollama_provider_converts_connection_failure(monkeypatch):
    def unavailable(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(provider_module, "urlopen", unavailable)
    provider = OllamaProvider(model="test-model", timeout_seconds=5)

    with pytest.raises(GenerationUnavailableError, match="test-model"):
        provider.generate("system", "prompt")
