"""Tests for the Ollama transport."""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

import llm_clients
from llm_clients import LLMClientError, OllamaClient


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


def test_complete_posts_chat_json_and_returns_content(caplog):
    caplog.set_level("INFO")
    response = Response(json.dumps({"message": {"content": "answer"}}).encode())
    with patch.object(llm_clients.urllib.request, "urlopen", return_value=response) as urlopen:
        result = OllamaClient("http://localhost:11434/", "model", 7).complete("system", "user")

    assert result == "answer"
    (request,) = urlopen.call_args.args
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["model"] == "model"
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert payload["stream"] is False and payload["format"] == "json"
    assert urlopen.call_args.kwargs["timeout"] == 7
    assert "Ollama response received: endpoint=http://localhost:11434 model=model" in caplog.text


@pytest.mark.parametrize("body", [b"{", b"[]"])
def test_complete_rejects_bad_json_and_wrong_response_shape(body):
    with patch.object(llm_clients.urllib.request, "urlopen", return_value=Response(body)):
        with pytest.raises(LLMClientError):
            OllamaClient("http://localhost", "model", 1).complete("s", "u")


def test_complete_converts_non_string_content_to_string():
    response = Response(b'{"message":{"content":15}}')
    with patch.object(llm_clients.urllib.request, "urlopen", return_value=response):
        assert OllamaClient("http://localhost", "model", 1).complete("s", "u") == "15"


def test_http_error_includes_status_and_response_body():
    error = urllib.error.HTTPError("url", 503, "unavailable", {}, io.BytesIO(b"busy"))
    with patch.object(llm_clients.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(LLMClientError, match="HTTP 503.*busy"):
            OllamaClient("http://localhost", "model", 1).complete("s", "u")


@pytest.mark.parametrize("error", [
    urllib.error.URLError("offline"), TimeoutError("timed out"), OSError("socket closed"),
])
def test_network_errors_are_wrapped(error):
    with patch.object(llm_clients.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(LLMClientError, match="cannot reach"):
            OllamaClient("http://localhost", "model", 1).complete("s", "u")
