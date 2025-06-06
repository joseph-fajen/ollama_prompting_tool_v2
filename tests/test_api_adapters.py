import pytest
from unittest.mock import patch, MagicMock
import requests # Add this import
from api_adapters import OllamaAdapter, OpenAIAdapter, HuggingFaceAdapter, LLMAdapterFactory

# Mock data for API responses
MOCK_OLLAMA_MODELS = [{"name": "llama3:latest"}, {"name": "codellama:latest"}]
MOCK_OPENAI_MODELS = {"data": [{"id": "gpt-4"}, {"id": "gpt-3.5-turbo"}]}
MOCK_HF_MODELS = [
    "google/flan-t5-small",
    "distilgpt2",
    "gpt2",
    "facebook/opt-125m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen1.5-0.5B-Chat",
    "microsoft/phi-1_5",
    "facebook/opt-350m",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Llama-2-7b-chat-hf"
]

@pytest.fixture
def ollama_adapter():
    return OllamaAdapter(base_url="http://mock-ollama")

@pytest.fixture
def openai_adapter():
    return OpenAIAdapter(base_url="http://mock-openai", api_key="fake_api_key")

@pytest.fixture
def huggingface_adapter():
    return HuggingFaceAdapter(api_key="fake_api_key")

# Tests for OllamaAdapter
def test_ollama_adapter_get_available_models(ollama_adapter):
    with patch("requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": MOCK_OLLAMA_MODELS}
        mock_get.return_value = mock_response
        models = ollama_adapter.get_available_models()
        assert "llama3:latest" in models
        assert "codellama:latest" in models

def test_ollama_adapter_generate(ollama_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Ollama response"}
        mock_post.return_value = mock_response
        response = ollama_adapter.generate("llama3:latest", [{"role": "user", "content": "Hello"}])
        assert response == "Ollama response"

# Tests for OpenAIAdapter
def test_openai_adapter_get_available_models(openai_adapter):
    with patch("requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_OPENAI_MODELS
        mock_get.return_value = mock_response
        models = openai_adapter.get_available_models()
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

def test_openai_adapter_generate(openai_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OpenAI response"}}]}
        mock_post.return_value = mock_response
        response = openai_adapter.generate("gpt-4", [{"role": "user", "content": "Hello"}])
        assert response == "OpenAI response"

# Tests for HuggingFaceAdapter
def test_huggingface_adapter_get_available_models(huggingface_adapter):
    models = huggingface_adapter.get_available_models()
    assert "google/flan-t5-small" in models
    assert "distilgpt2" in models

def test_huggingface_adapter_generate(huggingface_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = [{"generated_text": "HuggingFace response"}]
        mock_post.return_value = mock_response
        response = huggingface_adapter.generate("google/flan-t5-small", [{"role": "user", "content": "Hello"}])
        assert response == "HuggingFace response"

# Tests for LLMAdapterFactory
def test_llm_adapter_factory_create_ollama_adapter():
    adapter = LLMAdapterFactory.create_adapter("ollama")
    assert isinstance(adapter, OllamaAdapter)

def test_llm_adapter_factory_create_openai_adapter():
    adapter = LLMAdapterFactory.create_adapter("openai", api_key="fake_api_key")
    assert isinstance(adapter, OpenAIAdapter)

def test_llm_adapter_factory_create_huggingface_adapter():
    adapter = LLMAdapterFactory.create_adapter("huggingface", api_key="fake_api_key")
    assert isinstance(adapter, HuggingFaceAdapter)

def test_llm_adapter_factory_unsupported_provider():
    with pytest.raises(ValueError):
        LLMAdapterFactory.create_adapter("unsupported_provider")

# Tests for OllamaAdapter streaming
def test_ollama_adapter_generate_stream(ollama_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        # Simulate streaming response
        mock_response.iter_lines.return_value = [
            b'{"response": "Ollama ", "done": false}',
            b'{"response": "stream ", "done": false}',
            b'{"response": "response", "done": true}',
        ]
        mock_post.return_value = mock_response

        response_chunks = list(ollama_adapter.generate("llama3:latest", [{"role": "user", "content": "Hello"}], stream=True))

        assert len(response_chunks) == 3
        assert "".join(response_chunks) == "Ollama stream response"

# Tests for OpenAIAdapter streaming
def test_openai_adapter_generate_stream(openai_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_response = MagicMock()
        # Simulate streaming response
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "OpenAI "}}]}',
            b'data: {"choices": [{"delta": {"content": "stream "}}]}',
            b'data: {"choices": [{"delta": {"content": "response"}}]}',
            b'data: [DONE]',
        ]
        mock_post.return_value = mock_response

        response_chunks = list(openai_adapter.generate("gpt-4", [{"role": "user", "content": "Hello"}], stream=True))

        assert len(response_chunks) == 3
        assert "".join(response_chunks) == "OpenAI stream response"

# Test for OllamaAdapter API error
def test_ollama_adapter_api_error(ollama_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("API error")
        with pytest.raises(requests.exceptions.RequestException):
            ollama_adapter.generate("llama3:latest", [{"role": "user", "content": "Hello"}])

# Test for OpenAIAdapter API error
def test_openai_adapter_api_error(openai_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("API error")
        with pytest.raises(requests.exceptions.RequestException):
            openai_adapter.generate("gpt-4", [{"role": "user", "content": "Hello"}])

# Test for HuggingFaceAdapter API error
def test_huggingface_adapter_api_error(huggingface_adapter):
    with patch("requests.Session.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("API error")
        with pytest.raises(requests.exceptions.RequestException):
            huggingface_adapter.generate("google/flan-t5-small", [{"role": "user", "content": "Hello"}])
