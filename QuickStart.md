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

## Prompt Experimentation Examples

### 1. Technical Documentation Generation
```bash
# Create a system prompt that defines the AI's role
mkdir -p system_prompts
mkdir -p user_prompts
echo "You are a technical writer specializing in blockchain technology. Your task is to create clear, concise, and accurate documentation that explains complex concepts in simple terms." > system_prompts/blockchain_writer.md

echo "Explain how blockchain consensus mechanisms work, focusing on Proof of Work and Proof of Stake. Include examples and use clear, technical language." > user_prompts/consensus_mechanisms.md

# Run with multiple models for comparison
python ollama_prompt.py \
    --models llama3:8b phi3:mini \
    --system-file system_prompts/blockchain_writer.md \
    --prompt-file user_prompts/consensus_mechanisms.md \
    --stream
```

### 2. Code Explanation
```bash
# Create a system prompt for code explanation
echo "You are a senior developer explaining code to junior developers. Focus on clarity and practical examples." > system_prompts/code_explainer.md

echo "Explain this Python function that implements quicksort. Include time complexity analysis and practical use cases." > user_prompts/quicksort_explanation.md

# Run with different models to compare explanations
python ollama_prompt.py \
    --models phi3:mini llama3:8b \
    --system-file system_prompts/code_explainer.md \
    --prompt-file user_prompts/quicksort_explanation.md
```

### 3. Technical Analysis
```bash
# Create a system prompt for technical analysis
echo "You are a technical analyst specializing in blockchain security. Your task is to analyze potential vulnerabilities and provide recommendations." > system_prompts/security_analyst.md

echo "Analyze the security implications of using Proof of Stake consensus in enterprise blockchain networks. Consider both technical and business aspects." > user_prompts/stake_security.md

# Run with streaming for real-time analysis
python ollama_prompt.py \
    --models llama3:8b phi3:mini \
    --system-file system_prompts/security_analyst.md \
    --prompt-file user_prompts/stake_security.md \
    --stream
```

## Best Practices for Prompt Experimentation

1. **System Prompt Design**
   - Clearly define the AI's role and expertise
   - Specify the desired output format
   - Include any specific constraints or requirements
   - Use consistent terminology

2. **User Prompt Structure**
   - Start with a clear, specific question
   - Provide necessary context
   - Specify desired depth of explanation
   - Include any specific requirements

3. **Model Comparison Strategy**
   - Use different models for different aspects:
     - Phi3:mini for quick, general explanations
     - Llama3:8b for more detailed analysis
   - Compare outputs for consistency and depth
   - Use streaming mode for real-time comparison

4. **Response Analysis**
   - Compare clarity and accuracy of explanations
   - Evaluate technical depth and completeness
   - Check for consistency across models
   - Consider response time vs. quality trade-offs

5. **Prompt Refinement**
   - Start with broad prompts and refine
   - Use model outputs to identify gaps
   - Iterate on system prompts for better results
   - Document successful prompt combinations

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