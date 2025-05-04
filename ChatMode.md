# Chat Mode & API Compatibility

This document explains how to use the new Chat Mode and API compatibility features.

## Chat Mode

The new `ollama_chat.py` script provides an interactive chat interface with conversation history.

### Basic Usage

```bash
# Start a new chat with the default model
python ollama_chat.py

# Use a specific model
python ollama_chat.py --model llama3:8b

# Stream tokens in real-time (better for chat experience)
python ollama_chat.py --model phi3:mini --stream
```

### Available Chat Commands

During a chat session, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help message with all available commands |
| `/list` | List all saved conversations |
| `/load [id]` | Load a conversation by ID or select from a list |
| `/new [title]` | Start a new conversation with optional title |
| `/delete [id]` | Delete a conversation by ID or select from a list |
| `/system [prompt]` | Set or update the system prompt with direct text |
| `/system file:filename.md` | Load system prompt from system_prompts directory |
| `/clear` | Clear the current conversation history |
| `/show` | Show the current conversation |
| `/model [name]` | Switch to a different model |
| `/q`, `exit`, `quit` | Exit chat mode |

### Managing Conversations

```bash
# List all saved conversations
python ollama_chat.py --list-conversations

# Load a specific conversation
python ollama_chat.py --load-conversation abc123

# Start with a system prompt
python ollama_chat.py --system-file system_prompts/blockchain_educator.md
```

## API Compatibility

The chat tool supports multiple LLM providers through an adapter pattern.

### API Key Management

The chat tool offers several methods to securely manage your API keys:

```bash
# Interactive API key setup (recommended)
python ollama_chat.py --setup-keys

# Provide key directly on command line (not recommended for shared systems)
python ollama_chat.py --provider openai --api-key your_api_key

# Use environment variable
export OLLAMA_TOOL_OPENAI_API_KEY=your_api_key
python ollama_chat.py --provider openai
```

API keys are stored securely using:
1. System keyring (most secure, uses OS credential storage)
2. Encrypted config file in user's home directory
3. Environment variables

### OpenAI-Compatible APIs

```bash
# Use OpenAI API with stored key
python ollama_chat.py --provider openai

# Custom OpenAI-compatible endpoint (like LMStudio, LocalAI, or Ollama with OpenAI compatibility layer)
python ollama_chat.py --provider openai --base-url http://localhost:8080 --api-key none
```

### Hugging Face Inference API

```bash
# Use Hugging Face models with stored key
python ollama_chat.py --provider huggingface --model google/flan-t5-small
```

**Notes on Hugging Face Models:**
- Many models have access restrictions or require Pro subscription
- Best for specialized use cases or research
- Most reliable models for general use:
  - `google/flan-t5-small` (instruction-tuned)
  - `distilgpt2` (smaller but accessible)
  - `facebook/opt-125m` (smaller OPT model)
- For basic models, avoid using system prompts for best results

## One-Shot Mode (Backward Compatibility)

The chat tool still supports one-shot prompts without entering chat mode:

```bash
# One-shot prompt without chat mode
python ollama_chat.py --prompt "Explain quantum computing in simple terms"

# One-shot prompt from file
python ollama_chat.py --prompt-file user_prompts/smart_contract_explanation.md
```

## Conversation History Format

Conversations are saved in JSON format in the `ollama_conversations` directory. Each conversation includes:

- Unique ID
- Title
- Creation and update timestamps
- Complete message history with roles (system, user, assistant)
- Support for OpenAI-compatible message format

## Configuration

You can configure default API settings in your config file:

```yaml
# Example configuration with multiple providers
base_url: http://localhost:11434
default_provider: ollama
api_key: null  # For OpenAI or HuggingFace
openai_base_url: https://api.openai.com
default_chat_mode: true  # Always start in chat mode
```

## Implementation Details

The implementation uses an adapter pattern to support different LLM APIs:

1. **Base Interface**: `LLMAdapter` defines the common interface for all adapters
2. **Provider Adapters**: 
   - `OllamaAdapter`: For the local Ollama API
   - `OpenAIAdapter`: For OpenAI-compatible APIs
   - `HuggingFaceAdapter`: For Hugging Face Inference API
3. **Factory**: `LLMAdapterFactory` creates the appropriate adapter based on configuration

## Output File Naming

When using `ollama_prompt.py`, the response filenames now include the provider, model, and both the system prompt and user prompt filenames:

```
provider_modelname_sys-prompt_filename_usr-prompt_filename_timestamp.md
```

For example:
```
ollama_llama3_8b_sys-blockchain_educator_usr-consensus_mechanism_comparison_20250417_120000.md
huggingface_mistral_7b_instruct_sys-blockchain_educator_usr-smart_contract_explanation_20250417_120000.md
```

This makes it easier to identify which system and user prompts were used for each response without having to open the file.

Inside the output files, the metadata section now includes:
```
**Provider:** ollama
**Model:** llama3:8b
**System Prompt:** blockchain_educator
**User Prompt:** consensus_mechanism_comparison
```

For the chat mode in `ollama_chat.py`, this information is stored in the conversation metadata.

## Common Issues

- **API Keys**: OpenAI and Hugging Face providers require API keys
- **Message Format**: Different providers may handle system prompts differently
- **Chat History**: Some models have context length limitations affecting how much history can be retained

## Available Prompts

### System Prompts
- `blockchain_educator.md` - Expert in explaining blockchain concepts clearly
- `blockchain_tech_writer.md` - Technical writer specializing in blockchain documentation
- `blockchain_ux_writer.md` - Focuses on user-friendly blockchain explanations
- `scientific_explainer.md` - Technical expert for precise scientific explanations
- `creative_content_developer.md` - Creates engaging, imaginative content for complex topics

### User Prompts
- `smart_contract_explanation.md` - Requests explanation of smart contracts
- `consensus_mechanism_comparison.md` - Asks for comparison of blockchain consensus mechanisms
- `wallet_setup_guide.md` - Prompts for wallet setup instructions
- `quantum_resistant_blockchain.md` - Technical analysis of post-quantum cryptography for blockchains
- `blockchain_parallel_society.md` - Creative narrative about blockchain's future impact on society