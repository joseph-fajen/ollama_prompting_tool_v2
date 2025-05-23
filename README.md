# LLM Prompt Experimentation Tool

A focused Python utility for experimenting with different combinations of system and user prompts across multiple LLM models, with emphasis on Ollama and OpenAI providers.

## Quick Start

See [QuickStart.md](QuickStart.md) for common commands and basic usage.

## Configuration Management

The tool includes a robust configuration system with multiple ways to manage settings:

1. Command Line Arguments:
   ```bash
   python ollama_prompt.py --provider openai --model gpt-4o
   ```

2. Environment Variables:
   ```bash
   export OLLAMA_TOOL_PROVIDER="openai"
   export OLLAMA_TOOL_MODEL="gpt-4o"
   ```

3. Configuration File:
   - Located at `config/ollama_config.yaml`
   - Supports all configuration options
   - Can be updated with `--update-config` flag

4. Interactive Menu:
   - Provides easy access to common settings
   - Shows available models and providers
   - Allows quick switching between configurations

## Features

- Enhanced Interactive CLI menu with:
  - Model categories (Fast, Detailed Analysis)
  - Prompt categories (Technical Writing, Code Analysis, Security)
  - Keyboard shortcuts (q=quit, b=back, h=help, r=reset, s=save)
  - Current selections display
  - Comprehensive help system

- Model Comparison with:
  - Technical accuracy scoring (0-100)
  - Clarity of explanation scoring (0-100)
  - Response time measurement
  - Word count analysis
  - Multiple output modes (Real-time, Complete, Both, Save only)

- Prompt Management:
  - Organized prompt categories
  - Custom prompt creation
  - System prompt support
  - Prompt preview

- Response Options:
  - Real-time streaming
  - Complete formatted response
  - Both (streaming + complete)
  - Save only

- Configuration Management:
  - Command line arguments
  - Environment variables
  - Configuration file
  - Interactive menu
  - Default settings
  - Type-safe configuration

## Enhanced Model Comparison Features

The tool now includes advanced model comparison capabilities:

1. **Detailed Response Analysis**
   - Technical accuracy scoring (0-100)
   - Clarity of explanation scoring (0-100)
   - Response time measurement
   - Word count analysis

2. **Comparison Metrics**
   - Technical Score: Evaluates technical accuracy and depth
   - Clarity Score: Measures clarity of explanation
   - Response Time: Performance measurement
   - Word Count: Response length analysis

3. **Parallel Execution**
   - Run multiple models simultaneously
   - Real-time progress tracking
   - Detailed comparison summary
   - Sortable results by metrics

## Prerequisites

- Python 3.8+
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```
- For Ollama provider:
  - [Ollama](https://ollama.ai/) installed locally
  - At least one model installed
  - Ollama service running on http://localhost:11434
- For other providers:
  - Valid API keys for OpenAI or HuggingFace
  - API keys stored securely (recommended: use `--setup-keys` command)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/joseph-fajen/ollama_prompting_tool_v2.git
   cd ollama_prompting_tool_v2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys (if using OpenAI or HuggingFace):
   ```bash
   # Interactive setup
   python ollama_prompt.py --setup-keys
   
   # Or use environment variables
   export OLLAMA_TOOL_OPENAI_API_KEY="your_openai_key"
   export OLLAMA_TOOL_HUGGINGFACE_API_KEY="your_hf_key"
   ```

### For Ollama Provider
Before using the Ollama provider (default), ensure the Ollama service is running:

```bash
ollama serve
```

This starts the Ollama API server on http://localhost:11434.

### For Other Providers
For OpenAI or HuggingFace, set up your API keys (only needed once):

```bash
python ollama_prompt.py --setup-keys
```

## Provider Compatibility Guide

### Ollama (Recommended for Most Users)
- **Pros**: 
  - No API keys required
  - Local execution
  - No usage costs
  - Fast performance
- **Cons**: 
  - Requires local resources
  - Limited to installed models
- **Best Models**: 
  - Fast: phi3:mini (recommended for quick responses)
  - Detailed Analysis: llama3:8b (recommended for detailed analysis)
- **Use Case**: 
  - Everyday usage
  - Development
  - Testing
  - Quick prototyping

### OpenAI
- **Pros**: 
  - Professional-grade results
  - Reliable API
  - Consistent performance
- **Cons**: 
  - Requires API key with billing
  - Cost-based usage
- **Best Models**: 
  - gpt-4o
  - gpt-3.5-turbo
  - gpt-4-turbo
- **Use Case**: 
  - Production use
  - High-quality outputs
  - Enterprise applications

## Common Issues

- **404 Not Found**: Ollama service not running or model name mismatch
- **400 Bad Request**: Trying to use an embedding model for text generation
- **401 Unauthorized**: Invalid API key for OpenAI or HuggingFace
- **403 Forbidden**: Insufficient permissions with the provided API key

## Prompt Organization

The script uses two directories for managing prompts:

- `system_prompts/`: Contains AI role or persona definitions
- `user_prompts/`: Contains specific tasks or questions

List available prompts with: `python ollama_prompt.py --list-prompts`

## Response Files

- Responses are saved in timestamped batch folders in `ollama_responses/`
- Files include model info, timing, original prompt, and complete response
- Batch organization makes comparing results across models easy

## Examples

### Run with interactive menu
```bash
python ollama_prompt.py
```

### Run a specific model with system and user prompts
```bash
python ollama_prompt.py --model llama3:8b --system-file system_prompts/blockchain_educator.md --prompt-file user_prompts/smart_contract_explanation.md
```

### Run on multiple models in parallel
```bash
python ollama_prompt.py --models llama3:8b mixtral:latest phi3:mini --max-workers 4
```

### Use different providers
```bash
# OpenAI provider
python ollama_prompt.py --provider openai --model gpt-3.5-turbo --system-file system_prompts/blockchain_educator.md

# HuggingFace provider
python ollama_prompt.py --provider huggingface --model mistralai/Mistral-7B-Instruct-v0.1
```

For more examples, see [test_examples.md](test_examples.md).

## Default Behavior

- If no options are specified, uses values from configuration
- If no configuration exists, falls back to these defaults:
  - Provider: Ollama
  - Base URL: http://localhost:11434
  - If no prompt file is specified, interactive input is requested
  - If no system prompt is specified, only the user prompt is used
  - If no model is specified:
    - For Ollama: runs on all available models
    - For OpenAI: uses gpt-3.5-turbo
    - For HuggingFace: uses mistralai/Mistral-7B-Instruct-v0.1

## Configuration Commands

```bash
# Show current configuration
python ollama_prompt.py --show-config

# Save current settings as default
python ollama_prompt.py --save-config

# Update specific settings
python ollama_prompt.py --update-config provider=openai model=gpt-4o

# Reset to defaults
python ollama_prompt.py --reset-config
```

## Error Handling

The tool implements comprehensive error handling with:

1. Custom Exception Classes:
   - `LLMError`: Base exception for all LLM-related errors
   - `ConfigurationError`: For configuration issues
   - `APIError`: For API-related errors

2. User-friendly error messages with:
   - Clear context information
   - Suggested solutions
   - Error type classification

3. Error Categories:
   - Configuration Errors
   - API Errors
   - Network Issues
   - Authentication Problems
   - Model Compatibility Issues


## API Compatibility

Support for multiple LLM providers through an adapter pattern:

```bash
# Set up API keys securely (recommended)
python ollama_prompt.py --setup-keys

# OpenAI-compatible API with stored key
python ollama_prompt.py --provider openai

# Hugging Face models with stored key
python ollama_prompt.py --provider huggingface
```

**Secure API Key Management:**
- System keyring integration (macOS Keychain, Windows Credential Manager, etc.)
- Environment variables: `OLLAMA_TOOL_OPENAI_API_KEY`, `OLLAMA_TOOL_HUGGINGFACE_API_KEY`
- Encrypted local configuration in `~/.ollama_tool/`
- Interactive setup wizard

All providers implement a common interface, making it easy to switch between them with minimal code changes.

## Configuration System

The tool uses a robust, type-safe configuration system that:

1. Stores settings in `config/ollama_config.yaml`
2. Supports environment variables:
   - `OLLAMA_TOOL_PROVIDER`: Default provider (ollama, openai, huggingface)
   - `OLLAMA_TOOL_BASE_URL`: Custom API base URL
   - `OLLAMA_TOOL_DEFAULT_MODEL`: Default model name
   - `OLLAMA_TOOL_API_KEY`: API key for OpenAI/HuggingFace

3. Provides configuration commands:
   ```bash
   # Show current configuration
   python ollama_prompt.py --show-config
   
   # Save current settings as defaults
   python ollama_prompt.py --save-config
   
   # Reset to default configuration
   python ollama_prompt.py --reset-config
   
   # Update specific settings
   python ollama_prompt.py --update-config provider=openai model=gpt-3.5-turbo
   ```

4. Features:
   - Type-safe configuration validation
   - Automatic default value handling
   - Secure API key storage
   - Persistent settings across sessions

The script now includes a configuration system that persists your preferred settings:

- Configurations are stored in `config/ollama_config.yaml`
- Display current settings with `python ollama_prompt.py --show-config`
- Save current run settings as defaults with `python ollama_prompt.py --save-config`
- Reset configuration to defaults with `python ollama_prompt.py --reset-config`

You can configure default values for:
- Default provider (Ollama, OpenAI, HuggingFace)
- Default model(s) to use for each provider
- Default system and user prompts
- Output preferences (streaming, saving responses)
- Performance options (max workers, timeout)
- Custom API URLs for different providers
- API key handling preferences

## Best Practices

1. **Performance Optimization**:
   - Use appropriate model sizes for your use case
   - Set reasonable timeouts for API calls
   - Use streaming mode for chat interactions
   - Configure max workers based on system resources
   - Use environment variables for configuration

2. **Security**:
   - Never hardcode API keys in code
   - Use environment variables for API keys
   - Use the keyring system for secure storage
   - Regularly rotate API keys
   - Monitor API usage and costs

3. **Development Workflow**:
   - Use system prompts to define consistent behavior
   - Organize prompts in `system_prompts/` and `user_prompts/`
   - Save important conversations for reference
   - Use chat mode for iterative development
   - Run tests before major changes