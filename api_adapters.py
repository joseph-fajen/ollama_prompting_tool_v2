import requests
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Generator
from abc import ABC, abstractmethod
from conversation_manager import Message

class LLMAdapter(ABC):
    """Abstract base class for LLM API adapters"""
    
    @abstractmethod
    def generate(self, model: str, messages: List[Dict[str, str]], 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get a list of available models"""
        pass

class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, model: str, messages: List[Dict[str, str]], 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using Ollama API"""
        # For chat mode
        if len(messages) > 1:
            return self._generate_chat(model, messages, stream, **kwargs)
        # For completion mode (traditional)
        else:
            prompt = messages[0]["content"] if messages else ""
            return self._generate_completion(model, prompt, stream, **kwargs)
    
    def _generate_chat(self, model: str, messages: List[Dict[str, str]], 
                      stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using Ollama chat API"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": kwargs.get("options", {})
        }
        
        if stream:
            return self._stream_response(url, payload)
        else:
            response = self.session.post(url, json=payload, timeout=kwargs.get("timeout", 600))
            response.raise_for_status()
            response_json = response.json()
            
            if "message" in response_json:
                return response_json["message"]["content"]
            return ""
    
    def _generate_completion(self, model: str, prompt: str, 
                           stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using Ollama completion API"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": kwargs.get("options", {})
        }
        
        if stream:
            return self._stream_response(url, payload)
        else:
            response = self.session.post(url, json=payload, timeout=kwargs.get("timeout", 600))
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("response", "")
    
    def _stream_response(self, url: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response tokens"""
        with self.session.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    response_json = json.loads(line)
                    if "message" in response_json:  # Chat API
                        chunk = response_json["message"].get("content", "")
                    else:  # Generate API
                        chunk = response_json.get("response", "")
                    yield chunk
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            models = [model["name"] for model in data.get("models", []) 
                     if "embed" not in model["name"].lower()]
            return models
        except:
            # Fall back to command line
            import subprocess
            try:
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                
                # Skip the header line
                if len(lines) > 1:
                    models = []
                    for line in lines[1:]:  # Skip header row
                        parts = line.split()
                        if len(parts) >= 1:
                            model_name = parts[0]
                            # Skip embedding models
                            if "embed" not in model_name.lower():
                                models.append(model_name)
                    return models
            except:
                pass
            return []

class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs"""
    
    def __init__(self, base_url: str, api_key: str):
        # Ensure base_url is not None to avoid "No scheme supplied" errors
        if base_url is None:
            base_url = "https://api.openai.com"
            
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def generate(self, model: str, messages: List[Dict[str, str]], 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using OpenAI-compatible API"""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        try:
            if stream:
                return self._stream_response(url, payload)
            else:
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                response_json = response.json()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                return ""
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"\n[ERROR] Model '{model}' not found or not accessible.")
                print("\nRecommended models to try instead:")
                print("  • gpt-4o (newest and fastest)")
                print("  • gpt-4-turbo (good balance of capability and speed)")
                print("  • gpt-3.5-turbo (fastest)")
                print("\nNote: Models with 'moderation' in the name are for content filtering, not text generation.")
            elif e.response.status_code == 401:
                print("\n[ERROR] Authentication error. Your API key may be invalid or expired.")
            elif e.response.status_code == 429:
                print("\n[ERROR] Rate limit exceeded. You've sent too many requests to the API.")
                print("Consider using a different model, waiting, or checking your usage limits.")
            elif e.response.status_code == 400:
                print("\n[ERROR] Bad request. The API couldn't process your request.")
                print("This could be due to invalid parameters, model limitations, or content policy violations.")
                
            # Re-raise so our main error handler can deal with it
            raise
        except Exception as e:
            print(f"\n[ERROR] Unexpected error when calling OpenAI API: {str(e)}")
            raise
    
    def _stream_response(self, url: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response tokens from OpenAI API"""
        with self.session.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from OpenAI API"""
        try:
            url = f"{self.base_url}/v1/models"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            # List all models from the API
            all_models = [model["id"] for model in data.get("data", [])]
            
            # If we successfully got models from the API, return them
            if all_models:
                # Order models to put common text generation models first
                recommended_models = [
                    "gpt-4o", 
                    "gpt-4-turbo", 
                    "gpt-3.5-turbo", 
                    "gpt-4", 
                    "gpt-4-1106-preview",
                    "gpt-3.5-turbo-16k"
                ]
                # Put recommended models first, then add all other models
                ordered_models = []
                for model in recommended_models:
                    if any(m.startswith(model) for m in all_models):
                        matches = [m for m in all_models if m.startswith(model)]
                        ordered_models.extend(matches)
                
                # Add remaining models
                for model in all_models:
                    if model not in ordered_models:
                        ordered_models.append(model)
                
                return ordered_models
            
            # Fallback in case no models returned from API
            return [
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-1106-preview",
                "gpt-3.5-turbo-16k"
            ]
        except Exception as e:
            print(f"[Debug] Error getting OpenAI models: {str(e)}")
            # Return a default list of common models if API call fails
            return [
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-1106-preview",
                "gpt-3.5-turbo-16k"
            ]

class HuggingFaceAdapter(LLMAdapter):
    """Adapter for Hugging Face Inference API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def generate(self, model: str, messages: List[Dict[str, str]], 
                 stream: bool = False, **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using Hugging Face API"""
        url = f"{self.base_url}/{model}"
        
        # Store model name for use in _messages_to_prompt
        self.model = model
        
        # Convert messages to prompt that HF understands
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": kwargs.get("return_full_text", False)
            }
        }
        
        # HF API doesn't support true streaming, so ignore stream parameter
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            response_json = response.json()
            
            if isinstance(response_json, list) and len(response_json) > 0:
                return response_json[0].get("generated_text", "")
            return ""
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print("\n[ERROR] Forbidden access to this model. Possible reasons:")
                print("1. Your API key might not have access to this specific model")
                print("2. The model might be gated (requiring Pro subscription or special access)")
                print("3. Your account might have usage limits")
                print("\nSuggestions:")
                print("- Try a different model")
                print("- Check your API key permissions at https://huggingface.co/settings/tokens")
                print("- For Mistral models, check available models at https://huggingface.co/mistralai")
                print("- Consider using a non-gated model like 'facebook/opt-350m' or 'distilgpt2'")
            elif e.response.status_code == 401:
                print("\n[ERROR] Unauthorized. Your API key might be invalid or expired.")
            elif e.response.status_code == 429:
                print("\n[ERROR] Too many requests. You've hit the rate limit for this model.")
            elif e.response.status_code == 422:
                print("\n[ERROR] Unprocessable Entity. The model couldn't process your input.")
                print("\nPossible reasons:")
                print("1. The system+user prompt format might not be compatible with this model")
                print("2. The input might be too long for the model's context window")
                print("3. The model might need a different input format")
                print("\nSuggestions:")
                print("- Try a model specifically designed for instruction following like facebook/opt-350m")
                print("- Try without a system prompt (select 'None' for system prompt)")
                print("- Try a simpler, shorter user prompt")
            
            # Re-raise the exception so the calling code can handle it
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages to a prompt string for HF models"""
        model = self.model if hasattr(self, 'model') else None
        model_lower = str(model).lower() if model else ""
        
        # For T5 and FLAN-T5 models (instruction-tuned)
        if "t5" in model_lower:
            # FLAN-T5 works well with simple instruction format
            system = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            
            if system:
                return f"Instructions: {system}\n\nQuestion: {user}"
            else:
                return f"Question: {user}"
        
        # TinyLlama chat models
        elif "tinyllama" in model_lower and "chat" in model_lower:
            result = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    result += f"<|system|>\n{content}\n"
                elif role == "user":
                    result += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    result += f"<|assistant|>\n{content}\n"
            
            # Add final assistant marker for generation
            if result and not result.endswith("<|assistant|>\n"):
                result += "<|assistant|>\n"
            
            return result
            
        # For Qwen chat models
        elif "qwen" in model_lower and "chat" in model_lower:
            result = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    result += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    result += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    result += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            # Add final assistant marker for generation
            if not result.endswith("<|im_start|>assistant\n<|im_end|>\n"):
                result += "<|im_start|>assistant\n"
            
            return result
            
        # Check if we're dealing with a basic language model
        basic_lm_models = ['gpt2', 'distilgpt2', 'opt-125m', 'opt-350m', 'pythia']
        is_basic_lm = any(model_name in model_lower for model_name in basic_lm_models)
        
        # For basic LMs, just concatenate all messages simply
        if is_basic_lm:
            # Get user message
            user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            
            # For basic models, it's better to just use the user prompt without system prompt
            # as they're not designed to handle chat/instruction formats
            return user_content
        
        # For chat/instruct models like Mistral, Llama, etc.
        result = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                result += f"<|system|>\n{content}\n"
            elif role == "user":
                result += f"<|user|>\n{content}\n"
            elif role == "assistant":
                result += f"<|assistant|>\n{content}\n"
        
        # Add final assistant marker for generation
        if result and not result.endswith("<|assistant|>\n"):
            result += "<|assistant|>\n"
            
        return result
    
    def get_available_models(self) -> List[str]:
        """Get available models (only returns a predefined list as HF has thousands)"""
        # HF has thousands of models, so we just return popular ones that are more likely to be accessible
        # In a real implementation, you might want to allow specifying a model directly
        return [
            # Most likely to succeed with basic API access
            "google/flan-t5-small",  # Small instruction-tuned model
            "distilgpt2",            # Smaller version of GPT-2, widely accessible
            "gpt2",                  # Base GPT-2, usually accessible
            "facebook/opt-125m",     # Smaller OPT model
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small, optimized model
            # May work depending on account access level
            "Qwen/Qwen1.5-0.5B-Chat",      # Small Qwen chat model
            "microsoft/phi-1_5",           # Microsoft's small but effective model
            "facebook/opt-350m",           # Medium OPT model
            # Likely needs Pro subscription or special access
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Llama-2-7b-chat-hf"
        ]

class LLMAdapterFactory:
    """Factory for creating API adapters"""
    
    @staticmethod
    def create_adapter(provider: str, **kwargs) -> LLMAdapter:
        """Create an adapter for the specified provider"""
        if provider == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434")
            return OllamaAdapter(base_url)
        elif provider == "openai":
            base_url = kwargs.get("base_url", "https://api.openai.com")
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key is required for OpenAI adapter")
            return OpenAIAdapter(base_url, api_key)
        elif provider == "huggingface":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key is required for HuggingFace adapter")
            return HuggingFaceAdapter(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")