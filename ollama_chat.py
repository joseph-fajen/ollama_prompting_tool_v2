import requests
import json
import time
import argparse
import os
import subprocess
import sys
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from error_handler import ErrorHandler

# Import custom modules
import config_manager
from conversation_manager import ConversationManager, Message, Conversation
from api_adapters import LLMAdapterFactory
from api_key_manager import ApiKeyManager

class ChatClient:
    def __init__(self, provider="ollama", **kwargs):
        # Validate configuration
        config = config_manager.load_config()
        if not ErrorHandler.validate_config(config):
            sys.exit(1)

        # Set up configuration with error handling
        try:
            self.base_url = kwargs.get("base_url") or config.get("base_url", "http://localhost:11434")
        except Exception as e:
            ErrorHandler.handle_api_error(e, "Loading configuration")
            sys.exit(1)
        # Set up configuration
        config = config_manager.load_config()
        self.base_url = kwargs.get("base_url") or config.get("base_url", "http://localhost:11434")
        
        # Set up API key management
        self.key_manager = ApiKeyManager()
        supplied_key = kwargs.get("api_key")
        
        # Get API key (prioritize command line arg, then get from manager)
        if supplied_key:
            self.api_key = supplied_key
        else:
            self.api_key = self.key_manager.get_api_key(provider)
        
        # Set up adapter with error handling
        try:
            self.provider = provider
            self.adapter = LLMAdapterFactory.create_adapter(
                provider,
                base_url=self.base_url,
                api_key=self.api_key
            )
        except Exception as e:
            ErrorHandler.handle_api_error(e, "Initializing LLM adapter")
            sys.exit(1)
        
        # Set up conversation manager
        self.conversation_manager = ConversationManager()
        
        # Set up UI
        self.console = Console()
        self.system_prompt = None
        
    def get_installed_models(self):
        """Get a list of all available models"""
        return self.adapter.get_available_models()
    
    def set_system_prompt(self, system_prompt, system_filename=None):
        """Set a system prompt that will be used for all conversations"""
        self.system_prompt = system_prompt
        self.system_filename = system_filename or "direct_input"
        
        # Add to current conversation if it exists
        current_conv = self.conversation_manager.get_current_conversation()
        if current_conv and current_conv.messages:
            # Check if first message is already a system message
            if current_conv.messages[0].role == "system":
                # Update existing system message
                current_conv.messages[0].content = system_prompt
                # Add metadata for filename
                current_conv.metadata = current_conv.metadata or {}
                current_conv.metadata["system_filename"] = self.system_filename
            else:
                # Insert system message at beginning
                system_msg = Message("system", system_prompt)
                current_conv.messages.insert(0, system_msg)
                # Add metadata for filename
                current_conv.metadata = current_conv.metadata or {}
                current_conv.metadata["system_filename"] = self.system_filename
    
    def start_new_conversation(self, title=None):
        """Start a new conversation"""
        conversation = self.conversation_manager.create_conversation(title)
        
        # Add system message if we have one
        if self.system_prompt:
            conversation.add_message("system", self.system_prompt)
            
        return conversation
    
    def generate_response(self, model, prompt, stream=False, save=True, timeout=300):
        """Generate a one-shot response (compatible with original API)"""
        if not self.conversation_manager.get_current_conversation():
            self.start_new_conversation()
            
        # Add user message
        self.conversation_manager.add_message_to_current("user", prompt)
        
        # Generate response
        conversation = self.conversation_manager.get_current_conversation()
        messages = [m.to_openai_format() for m in conversation.messages]
        
        start_time = time.time()
        
        try:
            if stream:
                # Stream the response
                self.console.print(f"\n[bold blue]Model:[/bold blue] {model}")
                
                response_text = ""
                for chunk in self.adapter.generate(model, messages, stream=True, timeout=timeout):
                    response_text += chunk
                    self.console.print(chunk, end="")
                    
                self.console.print("\n")
            else:
                # Get the full response at once
                self.console.print(f"\n[bold blue]Model:[/bold blue] {model}")
                
                response_text = self.adapter.generate(model, messages, stream=False, timeout=timeout)
                
                # Print as formatted markdown
                self.console.print(Markdown(response_text))
            
            # Add assistant response to conversation
            self.conversation_manager.add_message_to_current("assistant", response_text)
            
            # Calculate and display timing
            elapsed_time = time.time() - start_time
            self.console.print(f"[dim]Response time: {elapsed_time:.2f} seconds[/dim]")
            
            # Save conversation if requested
            if save:
                self.conversation_manager.save_conversation()
                
            return response_text
                
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return None
    
    def chat_mode(self, model, stream=False, timeout=300):
        """Interactive chat mode with conversation history"""
        # Check if we need to create a new conversation
        if not self.conversation_manager.get_current_conversation():
            title = Prompt.ask("[bold cyan]Enter a title for this conversation[/bold cyan]", default=f"Chat {datetime.now().strftime('%Y-%m-%d')}")
            self.start_new_conversation(title)
        
        self.console.print(f"\n[bold blue]Chat Mode - Model:[/bold blue] {model}")
        self.console.print("[bold cyan]Type 'exit', 'quit', or '/q' to end the chat[/bold cyan]")
        self.console.print("[bold cyan]Type '/help' to see available commands[/bold cyan]\n")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                
                # Check for commands
                if user_input.lower() in ["exit", "quit", "/q"]:
                    break
                elif user_input.startswith("/"):
                    self._handle_command(user_input, model, stream, timeout)
                    continue
                
                # Add user message to conversation
                self.conversation_manager.add_message_to_current("user", user_input)
                
                # Generate response
                conversation = self.conversation_manager.get_current_conversation()
                messages = [m.to_openai_format() for m in conversation.messages]
                
                self.console.print("\n[bold blue]Assistant[/bold blue]")
                
                start_time = time.time()
                response_text = ""
                
                try:
                    if stream:
                        # Stream the response
                        for chunk in self.adapter.generate(model, messages, stream=True, timeout=timeout):
                            response_text += chunk
                            self.console.print(chunk, end="")
                    else:
                        # Get the full response
                        response_text = self.adapter.generate(model, messages, stream=False, timeout=timeout)
                        self.console.print(Markdown(response_text))
                
                    # Add assistant response to conversation
                    self.conversation_manager.add_message_to_current("assistant", response_text)
                    
                    # Calculate and display timing
                    elapsed_time = time.time() - start_time
                    self.console.print(f"\n[dim]Response time: {elapsed_time:.2f} seconds[/dim]")
                    
                    # Save conversation
                    self.conversation_manager.save_conversation()
                
                except requests.exceptions.RequestException as e:
                    self.console.print(f"\n[bold red]Error generating response:[/bold red] {str(e)}")
            
            except KeyboardInterrupt:
                # Allow graceful exit with Ctrl+C
                self.console.print("\n\n[bold yellow]Chat session interrupted.[/bold yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
    
    def _handle_command(self, command, model, stream, timeout):
        """Handle chat commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            self._show_help()
        elif cmd == "/list":
            self._list_conversations()
        elif cmd == "/load":
            if len(parts) > 1:
                self._load_conversation(parts[1])
            else:
                self._select_conversation_to_load()
        elif cmd == "/new":
            title = " ".join(parts[1:]) if len(parts) > 1 else None
            self.start_new_conversation(title)
            self.console.print(f"[bold green]Started new conversation: {self.conversation_manager.get_current_conversation().title}[/bold green]")
        elif cmd == "/delete":
            if len(parts) > 1:
                self._delete_conversation(parts[1])
            else:
                self._select_conversation_to_delete()
        elif cmd == "/system":
            if len(parts) > 1 and parts[1].startswith("file:"):
                # Load system prompt from file
                filename = parts[1][5:]  # Remove the "file:" prefix
                system_content = _get_prompt_content(f"system_prompts/{filename}")
                if system_content:
                    system_filename = os.path.splitext(os.path.basename(filename))[0]
                    self.set_system_prompt(system_content, system_filename)
                    self.console.print(f"[bold green]System prompt loaded from file: {filename}[/bold green]")
                else:
                    self.console.print(f"[bold red]Could not load system prompt from file: {filename}[/bold red]")
            else:
                # Direct system prompt text
                new_system = " ".join(parts[1:])
                if new_system:
                    self.set_system_prompt(new_system, "direct_input")
                    self.console.print("[bold green]System prompt updated[/bold green]")
                else:
                    self.console.print("[bold yellow]Please provide a system prompt or use file:[filename][/bold yellow]")
        elif cmd == "/clear":
            current = self.conversation_manager.get_current_conversation()
            if current:
                # Keep system message if it exists
                system_msg = None
                system_filename = None
                if current.messages and current.messages[0].role == "system":
                    system_msg = current.messages[0].content
                    system_filename = current.metadata.get("system_filename", "direct_input")
                
                current.clear()
                
                # Re-add system message if it existed
                if system_msg:
                    current.add_message("system", system_msg)
                    # Restore metadata
                    current.metadata = current.metadata or {}
                    current.metadata["system_filename"] = system_filename
                
                self.console.print("[bold green]Conversation cleared[/bold green]")
        elif cmd == "/show":
            self._show_conversation()
        elif cmd == "/model":
            if len(parts) > 1:
                new_model = parts[1]
                self.console.print(f"[bold green]Switched to model: {new_model}[/bold green]")
                return new_model
            else:
                models = self.get_installed_models()
                self.console.print("\n[bold]Available models:[/bold]")
                for i, m in enumerate(models):
                    self.console.print(f"  {i+1}. {m}")
        else:
            self.console.print(f"[bold red]Unknown command: {cmd}[/bold red]")
            self.console.print("Type [bold]/help[/bold] to see available commands")
        
        return model
    
    def _show_help(self):
        """Show help for chat commands"""
        table = Table(title="Chat Commands", box=box.ROUNDED)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        
        commands = [
            ("/help", "Show this help message"),
            ("/list", "List saved conversations"),
            ("/load [id]", "Load a conversation by ID or select from list"),
            ("/new [title]", "Start a new conversation with optional title"),
            ("/delete [id]", "Delete a conversation by ID or select from list"),
            ("/system [prompt]", "Set or update the system prompt with direct text"),
            ("/system file:filename.md", "Load system prompt from system_prompts directory"),
            ("/clear", "Clear the current conversation history"),
            ("/show", "Show the current conversation"),
            ("/model [name]", "Switch to a different model"),
            ("/q, exit, quit", "Exit chat mode")
        ]
        
        for cmd, desc in commands:
            table.add_row(cmd, desc)
            
        self.console.print(table)
    
    def _list_conversations(self):
        """List all saved conversations"""
        conversations = self.conversation_manager.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found[/yellow]")
            return
        
        table = Table(title="Saved Conversations", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Messages", justify="right")
        table.add_column("Last Updated")
        
        for conv in conversations:
            updated = datetime.fromtimestamp(conv.get("updated_at", 0)).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                conv.get("id", ""),
                conv.get("title", "Untitled"),
                str(conv.get("message_count", 0)),
                updated
            )
            
        self.console.print(table)
    
    def _load_conversation(self, conversation_id):
        """Load a conversation by ID"""
        conversation = self.conversation_manager.load_conversation(conversation_id)
        
        if conversation:
            self.conversation_manager.set_current_conversation(conversation.id)
            self.console.print(f"[bold green]Loaded conversation: {conversation.title}[/bold green]")
        else:
            self.console.print(f"[bold red]Conversation not found: {conversation_id}[/bold red]")
    
    def _select_conversation_to_load(self):
        """Interactive conversation selection"""
        conversations = self.conversation_manager.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found[/yellow]")
            return
        
        self.console.print("\n[bold]Select a conversation to load:[/bold]")
        for i, conv in enumerate(conversations):
            updated = datetime.fromtimestamp(conv.get("updated_at", 0)).strftime("%Y-%m-%d %H:%M")
            self.console.print(f"  {i+1}. {conv.get('title')} ({updated}) - {conv.get('message_count')} messages")
        
        try:
            choice = int(Prompt.ask("\nEnter number")) - 1
            if 0 <= choice < len(conversations):
                self._load_conversation(conversations[choice]["id"])
            else:
                self.console.print("[bold red]Invalid selection[/bold red]")
        except ValueError:
            self.console.print("[bold red]Please enter a number[/bold red]")
    
    def _delete_conversation(self, conversation_id):
        """Delete a conversation by ID"""
        if self.conversation_manager.delete_conversation(conversation_id):
            self.console.print(f"[bold green]Deleted conversation: {conversation_id}[/bold green]")
        else:
            self.console.print(f"[bold red]Failed to delete conversation: {conversation_id}[/bold red]")
    
    def _select_conversation_to_delete(self):
        """Interactive conversation deletion"""
        conversations = self.conversation_manager.list_saved_conversations()
        
        if not conversations:
            self.console.print("[yellow]No saved conversations found[/yellow]")
            return
        
        self.console.print("\n[bold]Select a conversation to delete:[/bold]")
        for i, conv in enumerate(conversations):
            updated = datetime.fromtimestamp(conv.get("updated_at", 0)).strftime("%Y-%m-%d %H:%M")
            self.console.print(f"  {i+1}. {conv.get('title')} ({updated}) - {conv.get('message_count')} messages")
        
        try:
            choice = int(Prompt.ask("\nEnter number")) - 1
            if 0 <= choice < len(conversations):
                conv_id = conversations[choice]["id"]
                if Confirm.ask(f"Are you sure you want to delete conversation '{conversations[choice]['title']}'?"):
                    self._delete_conversation(conv_id)
            else:
                self.console.print("[bold red]Invalid selection[/bold red]")
        except ValueError:
            self.console.print("[bold red]Please enter a number[/bold red]")
    
    def _show_conversation(self):
        """Show the current conversation"""
        conversation = self.conversation_manager.get_current_conversation()
        
        if not conversation or not conversation.messages:
            self.console.print("[yellow]No active conversation or empty conversation[/yellow]")
            return
        
        self.console.print(f"\n[bold]Conversation: {conversation.title}[/bold]")
        
        # Skip system message in display
        start_idx = 1 if conversation.messages and conversation.messages[0].role == "system" else 0
        
        for i, msg in enumerate(conversation.messages[start_idx:], 1):
            if msg.role == "user":
                self.console.print(Panel(
                    msg.content,
                    title=f"[bold green]You ({i}/[/bold green][bold]{len(conversation.messages)-start_idx}[/bold][bold green])[/bold green]",
                    border_style="green"
                ))
            elif msg.role == "assistant":
                self.console.print(Panel(
                    Markdown(msg.content),
                    title=f"[bold blue]Assistant ({i}/[/bold blue][bold]{len(conversation.messages)-start_idx}[/bold][bold blue])[/bold blue]",
                    border_style="blue"
                ))

def _get_prompt_content(file_path):
    """Read content from a file"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Interactive chat with local LLMs")
    
    # Model selection arguments
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--model", type=str, help="Model to use (default: use config or first available)")
    model_group.add_argument("--provider", type=str, choices=["ollama", "openai", "huggingface"],
                          default="ollama", help="API provider to use (default: ollama)")
    
    # Prompt arguments
    prompt_group = parser.add_argument_group("Prompt Options")
    prompt_group.add_argument("--system-file", type=str, help="Path to file containing the system prompt")
    prompt_group.add_argument("--prompt", type=str, help="One-shot prompt (won't start chat mode)")
    prompt_group.add_argument("--prompt-file", type=str, help="Path to file containing a one-shot prompt")
    
    # Chat options
    chat_group = parser.add_argument_group("Chat Options")
    chat_group.add_argument("--interactive", action="store_true", 
                         help="Start in interactive chat mode (default if no prompt is provided)")
    chat_group.add_argument("--stream", action="store_true", help="Stream the response token by token")
    chat_group.add_argument("--load-conversation", type=str, help="Load a specific conversation by ID")
    chat_group.add_argument("--list-conversations", action="store_true", help="List saved conversations and exit")
    
    # API options
    api_group = parser.add_argument_group("API Options")
    api_group.add_argument("--base-url", type=str, help="API base URL")
    api_group.add_argument("--api-key", type=str, help="API key (required for OpenAI and HuggingFace)")
    api_group.add_argument("--timeout", type=int, default=300, help="Timeout for API requests in seconds")
    api_group.add_argument("--setup-keys", action="store_true", help="Run interactive API key setup and exit")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle API key setup if requested
    if args.setup_keys:
        from api_key_manager import setup_api_keys
        setup_api_keys()
        return
    
    # Create directories if they don't exist
    os.makedirs("ollama_conversations", exist_ok=True)
    
    # Create client
    client = ChatClient(
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    # Check for API key if required provider
    if args.provider in ["openai", "huggingface"] and not client.api_key:
        # Try to get it interactively
        key_manager = ApiKeyManager()
        api_key = key_manager.prompt_for_api_key(args.provider)
        if api_key:
            client.api_key = api_key
            # Recreate the adapter with the new key
            client.adapter = LLMAdapterFactory.create_adapter(
                args.provider,
                base_url=client.base_url,
                api_key=api_key
            )
        else:
            print(f"\nError: {args.provider} requires an API key.")
            print(f"You can provide it with --api-key, or set up API keys using --setup-keys")
            return
    
    # Handle list conversations command
    if args.list_conversations:
        client._list_conversations()
        return
    
    # Get system prompt if specified
    if args.system_file:
        system_prompt = _get_prompt_content(args.system_file)
        if system_prompt:
            # Extract filename without extension
            system_filename = os.path.splitext(os.path.basename(args.system_file))[0]
            client.set_system_prompt(system_prompt, system_filename)
    
    # Get available models
    available_models = client.get_installed_models()
    if not available_models:
        client.console.print("[bold red]No models found![/bold red]")
        if args.provider == "ollama":
            client.console.print("Please install models with 'ollama pull <model>' and try again.")
        return
    
    # Determine which model to use
    model = None
    if args.model:
        model = args.model
    else:
        config_model = config_manager.get_config_value("default_model")
        if config_model:
            model = config_model
        else:
            # Use first available model
            model = available_models[0]
    
    # Load conversation if specified
    if args.load_conversation:
        client._load_conversation(args.load_conversation)
    
    # Handle one-shot prompt
    if args.prompt or args.prompt_file:
        prompt = args.prompt
        prompt_filename = "direct_input"
        
        if args.prompt_file:
            prompt = _get_prompt_content(args.prompt_file)
            prompt_filename = os.path.splitext(os.path.basename(args.prompt_file))[0]
            
        if not prompt:
            client.console.print("[bold red]No prompt provided![/bold red]")
            return
        
        # Store prompt filename in conversation metadata
        current_conv = client.conversation_manager.get_current_conversation()
        if current_conv:
            current_conv.metadata = current_conv.metadata or {}
            current_conv.metadata["prompt_filename"] = prompt_filename
            
        client.generate_response(model, prompt, stream=args.stream, timeout=args.timeout)
        
    # Interactive chat mode
    elif args.interactive or (not args.prompt and not args.prompt_file):
        client.chat_mode(model, stream=args.stream, timeout=args.timeout)

if __name__ == "__main__":
    main()