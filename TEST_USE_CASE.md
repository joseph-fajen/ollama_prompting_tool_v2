# LLM Prompt CLI - Test Use Case

This document outlines a comprehensive test case for verifying the core functionality of the LLM Prompt CLI tool.

## Prerequisites

1. Ensure Ollama is installed and running locally
2. Have at least one model installed in Ollama (e.g., llama3:8b)
3. For OpenAI testing:
   - Have a valid OpenAI API key
   - Set up billing on your OpenAI account

## Test Case: Technical Documentation Generator

### Objective
Test the tool's ability to generate high-quality technical documentation using multiple LLM providers.

### Step 1: Basic Functionality with Ollama

```bash
# Create system prompt
mkdir -p system_prompts
mkdir -p user_prompts
echo "You are a technical writer specializing in blockchain technology. Your task is to create clear, concise, and accurate documentation that explains complex concepts in simple terms." > system_prompts/blockchain_writer.md

echo "Explain how blockchain consensus mechanisms work, focusing on Proof of Work and Proof of Stake. Include examples and use clear, technical language." > user_prompts/consensus_mechanisms.md

# Run with Ollama
python ollama_prompt.py \
    --system-file system_prompts/blockchain_writer.md \
    --prompt-file user_prompts/consensus_mechanisms.md \
    --model llama3:8b \
    --stream
```

### Step 2: Advanced Testing with Multiple Providers

```bash
# Set up API keys (if using OpenAI)
export OLLAMA_TOOL_OPENAI_API_KEY="your_openai_key"

# Run with OpenAI
python ollama_prompt.py \
    --provider openai \
    --model gpt-4o \
    --system-file system_prompts/blockchain_writer.md \
    --prompt-file user_prompts/consensus_mechanisms.md \
    --stream

# Compare responses
python ollama_prompt.py \
    --models llama3:8b mixtral:latest \
    --system-file system_prompts/blockchain_writer.md \
    --prompt-file user_prompts/consensus_mechanisms.md \
    --max-workers 2
```

### Step 3: Chat Mode Testing

```bash
# Start chat mode
python ollama_chat.py \
    --model llama3:8b \
    --stream

# In chat mode, try:
1. Ask follow-up questions about the documentation
2. Request clarification on specific points
3. Ask for examples
4. Request different perspectives
```

### Step 4: Configuration Testing

```bash
# Show current configuration
python ollama_prompt.py --show-config

# Save current settings as default
python ollama_prompt.py --save-config

# Reset to defaults
python ollama_prompt.py --reset-config

# Update specific settings
python ollama_prompt.py --update-config provider=openai model=gpt-4o
```

### Expected Results

1. The tool should:
   - Load system and user prompts correctly
   - Generate coherent, technical documentation
   - Handle streaming responses smoothly
   - Save responses to appropriate files
   - Maintain conversation context in chat mode
   - Handle configuration updates properly

2. The output should:
   - Be technically accurate
   - Use appropriate terminology
   - Be well-structured and organized
   - Include examples and explanations
   - Be consistent across different providers

3. The tool should:
   - Provide clear error messages if something goes wrong
   - Handle API rate limits gracefully
   - Maintain state between chat sessions
   - Allow easy switching between providers

### Verification Steps

1. Review generated documentation for:
   - Technical accuracy
   - Clarity of explanations
   - Proper use of terminology
   - Logical flow

2. Test chat mode for:
   - Context preservation
   - Responsiveness
   - Error handling
   - Command recognition

3. Verify configuration:
   - Settings are persisted correctly
   - Environment variables work as expected
   - Default values are appropriate

### Optional Extensions

1. Try different system prompts:
   - Technical writer with different specializations
   - Different tones (formal vs casual)
   - Different levels of detail

2. Test with different models:
   - Compare output quality
   - Measure response times
   - Test memory usage

3. Experiment with chat mode features:
   - Conversation history
   - System prompt updates
   - Model switching

This test case covers the core functionality of the tool while providing a practical use case that demonstrates its capabilities in a real-world scenario.
