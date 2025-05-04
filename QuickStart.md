# LLM Prompt CLI - QuickStart

## Basic Usage

```bash
# Interactive menu
python ollama_prompt.py

# Bypass menu
python ollama_prompt.py --no-menu

# Specific model
python ollama_prompt.py --model llama3:8b

# Multiple models
python ollama_prompt.py --models llama3:8b mixtral:latest phi3:mini

# All models (Ollama provider only)
python ollama_prompt.py --all-models

# Using different providers
python ollama_prompt.py --provider openai --model gpt-4o

# Force display of interactive menu
python ollama_prompt.py --force-menu

# Set up API keys securely (for non-Ollama providers)
python ollama_prompt.py --setup-keys

# Use environment variables for API keys
export OLLAMA_TOOL_OPENAI_API_KEY="your_openai_key"
export OLLAMA_TOOL_HUGGINGFACE_API_KEY="your_hf_key"
```

## Provider and Model Recommendations

### Ollama (Recommended for Most Users)
- No API key required
- Local execution
- Good starter models: `llama3:8b`, `phi3:mini`, `codellama:7b-instruct`
- Supports streaming for real-time responses
- Best for: Everyday usage, development, testing

### OpenAI
- Requires API key with billing set up
- Recommended models: `gpt-4o`, `gpt-3.5-turbo`, `gpt-4-turbo`
- Supports streaming for real-time responses
- Best for: Production use, high-quality outputs
```bash
python ollama_prompt.py --provider openai --model gpt-4o --stream
```

### Hugging Face
- Requires API key, many models have access restrictions
- Best for specialized use cases or research
- Most reliable models:
  - `google/flan-t5-small`
  - `distilgpt2`
  - `facebook/opt-125m` 
- Supports streaming for real-time responses
- Use without system prompt for best results with basic models
- Best for: Research, specialized applications

## Model Comparison

```bash
# Compare responses from different models
python ollama_prompt.py --models llama3:8b phi3:mini --prompt-file user_prompts/example.md

# Run with streaming for real-time comparison
python ollama_prompt.py --models llama3:8b phi3:mini --prompt-file user_prompts/example.md --stream
```

## Custom Prompts

```bash
# List available prompts
python ollama_prompt.py --list-prompts

# Use custom user prompt
python ollama_prompt.py --prompt-file user_prompts/smart_contract_explanation.md

# Add system prompt
python ollama_prompt.py --system-file system_prompts/blockchain_educator.md
```

## Output Options

```bash
# Stream tokens in real-time
python ollama_prompt.py --stream

# Don't save response to file
python ollama_prompt.py --no-save

# Parallel execution control
python ollama_prompt.py --max-workers 4

# Adjust timeout (in seconds)
python ollama_prompt.py --timeout 600
```

## Configuration Options

```bash
# Show current configuration
python ollama_prompt.py --show-config

# Save current run settings as default
python ollama_prompt.py --model llama3:8b --stream --save-config

# Save provider settings as default
python ollama_prompt.py --provider openai --model gpt-4o --save-config

# Update specific settings
python ollama_prompt.py --update-config provider=openai model=gpt-4o

# Reset configuration to defaults
python ollama_prompt.py --reset-config

# Set custom API URLs
python ollama_prompt.py --base-url http://192.168.1.100:11434

# Use environment variables
export OLLAMA_TOOL_PROVIDER="openai"
export OLLAMA_TOOL_MODEL="gpt-4o"
export OLLAMA_TOOL_BASE_URL="http://localhost:8080"

# Environment variable precedence:
# 1. Command line arguments
# 2. Environment variables
# 3. Configuration file
# 4. Default values
```

See all options: `python ollama_prompt.py --help`