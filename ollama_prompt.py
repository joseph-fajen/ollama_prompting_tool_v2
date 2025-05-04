"""
Ollama Prompt Tool - A CLI tool for running prompts on models from different providers.
"""

import requests
import json
import time
import argparse
import os
import subprocess
import concurrent.futures
import glob
import sys
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Import configuration manager and API adapters
import config_manager
import api_adapters
import api_key_manager

class AdapterWrapper:
    """Wrapper class to make LLMAdapter compatible with OllamaClient interface"""
    
    def __init__(self, adapter, provider_name):
        self.adapter = adapter
        self.provider_name = provider_name
        self.console = Console()
        self._system_prompt = None
        # For output filename usage
        self._system_filename = "no_system"
        self._user_filename = "direct_input"
        self._batch_folder = None
    
    def get_installed_models(self):
        """Get available models from the adapter"""
        return self.adapter.get_available_models()
    
    def generate_response(self, model, prompt, stream=False, save=True, timeout=300):
        """Generate a response using the adapter"""
        start_time = time.time()
        
        # Convert prompt to messages format
        messages = []
        
        # Extract system prompt if present
        system_prompt = None
        main_prompt = prompt
        
        if hasattr(self, '_system_prompt') and self._system_prompt:
            system_prompt = self._system_prompt
            # If the prompt starts with the system prompt, extract just the user prompt
            if prompt.startswith(system_prompt):
                main_prompt = prompt[len(system_prompt):].lstrip('\n')
        
        # Add system message if available
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": main_prompt})
        
        try:
            self.console.print(f"\n[bold green]Model:[/bold green] {model}")
            self.console.print("[bold green]Response:[/bold green]\n")
            
            # Generate response using the adapter
            if stream:
                response_text = ""
                for chunk in self.adapter.generate(model, messages, stream=True, timeout=timeout):
                    response_text += chunk
                    self.console.print(chunk, end="")
                self.console.print("\n")
            else:
                response_text = self.adapter.generate(model, messages, stream=False, timeout=timeout)
                # Print as formatted markdown
                self.console.print(Markdown(response_text))
            
            # Calculate and display timing
            elapsed_time = time.time() - start_time
            self.console.print(f"\n[bold blue]Response time:[/bold blue] {elapsed_time:.2f} seconds")
            
            # Save response if requested
            if save:
                filepath = self._save_response(model, prompt, response_text, elapsed_time)
                return response_text, filepath
            
            return response_text, None
            
        except requests.exceptions.HTTPError as http_err:
            # HTTP errors should already be handled by the adapters with specific messages
            self.console.print(f"[bold red]Error:[/bold red] {str(http_err)}")
            
            if self.provider_name == "huggingface":
                self.console.print("\n[bold yellow]Suggestions for Hugging Face:[/bold yellow]")
                self.console.print("1. If using a basic model like gpt2:")
                self.console.print("  • Try without a system prompt (select 'None' for system prompt)")
                self.console.print("  • Use a simpler, shorter user prompt")
                self.console.print("\n2. Try one of these alternative models:")
                self.console.print("  • google/flan-t5-small (instruction-following model)")
                self.console.print("  • facebook/opt-125m (smaller version that's often accessible)")
                self.console.print("  • distilgpt2 (smaller, usually accessible)")
                self.console.print("\n3. Or try with Ollama models which work without API keys")
            
            return None, None
            
        except Exception as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            
            # Provider-specific error suggestions
            if self.provider_name == "openai":
                self.console.print("[bold yellow]OpenAI troubleshooting:[/bold yellow]")
                self.console.print("1. Check that your API key is correct")
                self.console.print("2. Verify you have billing set up at https://platform.openai.com/account/billing")
                self.console.print("3. Make sure the model name is valid")
            
            return None, None
    
    def _save_response(self, model, prompt, response, elapsed_time):
        """Save the response to a file in a timestamped batch folder"""
        # Create responses directory if it doesn't exist
        os.makedirs("ollama_responses", exist_ok=True)
        
        # Create a timestamped batch folder if it doesn't exist yet
        if not hasattr(self, '_batch_folder') or not self._batch_folder:
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._batch_folder = f"ollama_responses/batch_{batch_timestamp}"
            os.makedirs(self._batch_folder, exist_ok=True)
        
        # Extract system prompt and user prompt if combined
        system_prompt = None
        main_prompt = prompt
        
        # Check if this is a combined prompt (system + user)
        if hasattr(self, '_system_prompt') and self._system_prompt:
            system_prompt = self._system_prompt
            # If the prompt starts with the system prompt, extract just the user prompt
            if prompt.startswith(system_prompt):
                main_prompt = prompt[len(system_prompt):].lstrip('\n')
        
        # Get prompt filenames if available (or 'direct_input' if not from file)
        system_filename = getattr(self, '_system_filename', 'no_system')
        user_filename = getattr(self, '_user_filename', 'direct_input') 
        
        # Create a filename based on the model and prompt filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._batch_folder}/{self.provider_name}_{model.replace(':', '_').replace('/', '_')}_sys-{system_filename}_usr-{user_filename}_{timestamp}.md"
        
        # Use context manager for file I/O
        try:
            with open(filename, "w") as f:
                f.write(f"# {self.provider_name.capitalize()} Response - {model}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Model:** {model}\n")
                f.write(f"**System Prompt:** {system_filename}\n")
                f.write(f"**User Prompt:** {user_filename}\n")
                f.write(f"**Response Time:** {elapsed_time:.2f} seconds\n\n")
                
                # Add system prompt section if available
                if system_prompt:
                    f.write("## System Prompt\n\n")
                    f.write(f"```\n{system_prompt}\n```\n\n")
                
                # Add user prompt section
                f.write("## User Prompt\n\n")
                f.write(f"```\n{main_prompt}\n```\n\n")
                
                # Add response section
                f.write("## Response\n\n")
                f.write(response)
            
            print(f"\nResponse saved to {filename}")
            return filename
        except Exception as e:
            self.console.print(f"[bold red]Error saving response:[/bold red] {str(e)}")
            return None
    
    def _process_model(self, model, prompt, stream=False, save=True, timeout=600):
        """Process a single model (helper method for parallel execution)"""
        response, filepath = self.generate_response(model, prompt, stream=stream, save=save, timeout=timeout)
        if response:
            return {
                "model": model,
                "response": response,
                "filepath": filepath
            }
        return None
    
    def run_prompt_on_all_models(self, prompt, models=None, save=True, stream=False, max_workers=None, timeout=600):
        """
        Run a prompt on all available models or a specified list of models in parallel.
        
        Args:
            prompt: The prompt to run
            models: List of models to run (None for all available models)
            save: Whether to save the responses
            stream: Whether to use streaming mode
            max_workers: Maximum number of concurrent workers
            timeout: Timeout for each model response
        
        Returns:
            List of dictionaries containing:
            - model: Model name
            - response: Generated response text
            - response_time: Time taken to generate response
            - word_count: Number of words in the response
            - technical_score: Score based on technical accuracy (0-100)
            - clarity_score: Score based on clarity of explanation (0-100)
        """
        console = Console()
        
        # Get list of models to run
        if models is None:
            models = self.get_installed_models()
        
        # Filter out embedding models
        models = [m for m in models if not m.endswith("-embedding")]
        
        if not models:
            console.print("[bold red]No models available to run.[/bold red]")
            return []
        
        console.print(f"\n[bold]Running prompt on {len(models)} models:[/bold]")
        for model in models:
            console.print(f"  • {model}")
        
        # Create progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("[cyan]Generating responses...", total=len(models))
            
            # Run in parallel with ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {}
                
                # Submit tasks
                for model in models:
                    future = executor.submit(
                        self.generate_response,
                        model=model,
                        prompt=prompt,
                        stream=stream,
                        save=save,
                        timeout=timeout
                    )
                    future_to_model[future] = model
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all jobs
                    for model in models:
                        future = executor.submit(self._process_model, model, prompt, stream, save, timeout)
                        futures.append((future, model))
                    
                    # Process as they complete
                    for future, model in futures:
                        progress.update(task, description=f"[green]Running {model}...")
                        result = future.result()
                        if result:
                            results.append(result)
                        progress.update(task, advance=1)
        
        total_time = time.time() - total_start_time
        
        # Print summary
        self.console.print(f"\n[bold]Finished running {len(models)} models in {total_time:.2f} seconds[/bold]")
        self.console.print("\n[bold]Summary of results:[/bold]")
        
        for result in results:
            self.console.print(f"[green]{result['model']}[/green]: Response saved to {result['filepath']}")
        
        return results


class OllamaClient:
    def __init__(self, base_url=None):
        # Use configured base_url or fall back to default
        self.base_url = base_url or config_manager.get_config_value("base_url", "http://localhost:11434")
        self.console = Console()
        self.session = requests.Session()
        self._system_prompt = None
        
    def get_installed_models(self):
        """Get a list of all installed Ollama models, excluding embedding models"""
        try:
            # Debug: Print before running command
            print("[Debug] Running 'ollama list' to get models...")
            
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            # Debug: Print raw output
            print(f"[Debug] Raw output: {result.stdout}")
            
            # Skip the header line
            if len(lines) > 1:
                models = []
                for line in lines[1:]:  # Skip header row
                    parts = line.split()
                    print(f"[Debug] Processing line: {line}, parts: {parts}")
                    if len(parts) >= 1:
                        model_name = parts[0]
                        # Skip embedding models which cannot generate text
                        if "embed" not in model_name.lower():
                            models.append(model_name)  # Get just the model name
                            print(f"[Debug] Added model: {model_name}")
                
                print(f"[Debug] Final models list: {models}")
                return models
            
            print("[Debug] No models found in output")
            return []
        except Exception as e:
            print(f"[Debug] Exception: {str(e)}")
            self.console.print(f"[bold red]Error getting models:[/bold red] {str(e)}")
            return []
    
    def generate_response(self, model, prompt, stream=False, save=True, timeout=300):
        """Generate a response from an Ollama model"""
        start_time = time.time()
        
        # Create API request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            if stream:
                # Stream the response
                self.console.print(f"\n[bold green]Model:[/bold green] {model}")
                self.console.print("[bold green]Response:[/bold green]\n")
                
                response_text = ""
                with self.session.post(url, json=payload, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            response_json = json.loads(line)
                            chunk = response_json.get("response", "")
                            response_text += chunk
                            self.console.print(chunk, end="")
                
                self.console.print("\n")
            else:
                # Get the full response at once
                self.console.print(f"\n[bold green]Model:[/bold green] {model}")
                self.console.print("[bold green]Response:[/bold green]\n")
                
                response = self.session.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                
                response_json = json.loads(response.text)
                response_text = response_json.get("response", "")
                
                # Print as formatted markdown
                self.console.print(Markdown(response_text))
            
            # Calculate and display timing
            elapsed_time = time.time() - start_time
            self.console.print(f"\n[bold blue]Response time:[/bold blue] {elapsed_time:.2f} seconds")
            
            # Save response if requested
            if save:
                filepath = self._save_response(model, prompt, response_text, elapsed_time)
                return response_text, filepath
            
            return response_text, None
            
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return None, None
    
    def _save_response(self, model, prompt, response, elapsed_time):
        """Save the response to a file in a timestamped batch folder"""
        # Create responses directory if it doesn't exist
        os.makedirs("ollama_responses", exist_ok=True)
        
        # Create a timestamped batch folder if it doesn't exist yet
        # We store this as a class attribute so all responses in one run use the same folder
        if not hasattr(self, '_batch_folder'):
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._batch_folder = f"ollama_responses/batch_{batch_timestamp}"
            os.makedirs(self._batch_folder, exist_ok=True)
        
        # Extract system prompt and user prompt if combined
        system_prompt = None
        main_prompt = prompt
        
        # Check if this is a combined prompt (system + user)
        if hasattr(self, '_system_prompt') and self._system_prompt:
            system_prompt = self._system_prompt
            # If the prompt starts with the system prompt, extract just the user prompt
            if prompt.startswith(system_prompt):
                main_prompt = prompt[len(system_prompt):].lstrip('\n')
        
        # Get prompt filenames if available (or 'direct_input' if not from file)
        system_filename = getattr(self, '_system_filename', 'no_system')
        user_filename = getattr(self, '_user_filename', 'direct_input') 
        
        # Create a filename based on the model and prompt filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._batch_folder}/{model.replace(':', '_')}_sys-{system_filename}_usr-{user_filename}_{timestamp}.md"
        
        # Use context manager for file I/O
        try:
            with open(filename, "w") as f:
                f.write(f"# Ollama Response - {model}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Model:** {model}\n")
                f.write(f"**System Prompt:** {system_filename}\n")
                f.write(f"**User Prompt:** {user_filename}\n")
                f.write(f"**Response Time:** {elapsed_time:.2f} seconds\n\n")
                
                # Add system prompt section if available
                if system_prompt:
                    f.write("## System Prompt\n\n")
                    f.write(f"```\n{system_prompt}\n```\n\n")
                
                # Add user prompt section
                f.write("## User Prompt\n\n")
                f.write(f"```\n{main_prompt}\n```\n\n")
                
                # Add response section
                f.write("## Response\n\n")
                f.write(response)
            
            print(f"\nResponse saved to {filename}")
            return filename
        except Exception as e:
            self.console.print(f"[bold red]Error saving response:[/bold red] {str(e)}")
            return None

    def _process_model(self, model, prompt, stream=False, save=True, timeout=600):
        """Process a single model (helper method for parallel execution)"""
        response, filepath = self.generate_response(model, prompt, stream=stream, save=save, timeout=timeout)
        if response:
            return {
                "model": model,
                "response": response,
                "filepath": filepath
            }
        return None
        
    def run_prompt_on_all_models(self, prompt, models=None, save=True, stream=False, max_workers=None, timeout=600):
        """Run a prompt on all installed models or a specific list of models in parallel"""
        if not models:
            models = self.get_installed_models()
            
        if not models:
            self.console.print("[bold red]No models found![/bold red]")
            return
            
        self.console.print(f"[bold]Running prompt on {len(models)} models:[/bold] {', '.join(models)}\n")
        
        results = []
        total_start_time = time.time()
        
        # If stream is True, we can't use parallel execution (would mix outputs)
        if stream:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[green]Running models...", total=len(models))
                
                for model in models:
                    progress.update(task, description=f"[green]Running {model}...")
                    result = self._process_model(model, prompt, stream=stream, save=save)
                    if result:
                        results.append(result)
                    progress.update(task, advance=1)
        else:
            # Use parallel execution for non-streaming mode
            futures = []
            
            # Default to number of CPU cores if max_workers not specified
            if max_workers is None:
                # Use CPU count or 2, whichever is higher
                max_workers = max(2, os.cpu_count())
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[green]Running models in parallel...", total=len(models))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all jobs
                    for model in models:
                        future = executor.submit(self._process_model, model, prompt, stream, save, timeout)
                        futures.append((future, model))
                    
                    # Process as they complete
                    for future, model in futures:
                        progress.update(task, description=f"[green]Running {model}...")
                        result = future.result()
                        if result:
                            results.append(result)
                        progress.update(task, advance=1)
        
        total_time = time.time() - total_start_time
        
        # Print summary
        self.console.print(f"\n[bold]Finished running {len(models)} models in {total_time:.2f} seconds[/bold]")
        self.console.print("\n[bold]Summary of results:[/bold]")
        
        for result in results:
            self.console.print(f"[green]{result['model']}[/green]: Response saved to {result['filepath']}")
        
        return results


def get_available_files(directory, extensions=None):
    """Get available files in a directory with specific extensions"""
    if not os.path.exists(directory):
        return []
        
    if extensions is None:
        extensions = [".txt", ".md"]
        
    files = []
    for ext in extensions:
        files.extend(glob.glob(f"{directory}/*{ext}"))
    
    return sorted(files)

def get_prompt_content(file_path):
    """Read content from a user prompt file"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def display_interactive_cli_menu(client, provider="ollama"):
    """Display an interactive CLI menu for selecting models and prompts"""
    console = Console()
            
    # Keyboard shortcuts
    shortcuts = {
        'q': 'Quit menu',
        'b': 'Go back to previous menu',
        'h': 'Show help',
        'r': 'Reset to defaults',
        's': 'Save current settings'
    }
            
    # Help system
    def show_help():
        console.print("\n[bold]Keyboard Shortcuts:[/bold]")
        for key, description in shortcuts.items():
            console.print(f"  • [bold]{key}[/bold]: {description}")
        console.print("\n[bold]Navigation:[/bold]")
        console.print("  • Use numbers to select options")
        console.print("  • Press Enter to accept default")
        console.print("  • Use comma-separated numbers for multiple selections")
        console.print("\n[bold]Model Selection:[/bold]")
        console.print("  • Fast models: phi3:mini (recommended for quick responses)")
        console.print("  • Detailed analysis: llama3:8b (recommended for detailed analysis)")
        console.print("\n[bold]Prompt Categories:[/bold]")
        console.print("  • Technical Writing: For creating documentation")
        console.print("  • Code Analysis: For code explanations")
        console.print("  • Security: For security analysis")
        console.print("  • Performance: For performance analysis")
        console.print("\n[bold]Output Modes:[/bold]")
        console.print("  • Real-time: See tokens as they're generated")
        console.print("  • Complete: See the full formatted response")
        console.print("  • Both: See both streaming and complete response")
        console.print("  • Save only: Save response to file without displaying")
            
    # Show help at start
    show_help()
    console.print("\n[bold blue]===== LLM Prompt CLI =====\n[/bold blue]")
    console.print("[yellow]Welcome to the Prompt CLI![/yellow]")
    console.print("This tool helps you run prompts on models from different providers.")
            
    # Show configuration status
    config = config_manager.load_config()
    if config.get("default_model") or config.get("default_models") or config.get("default_provider"):
        console.print("\n[bold magenta]CONFIG:[/bold magenta]")
        if config.get("default_provider"):
            console.print(f"  • Default provider: [cyan]{config['default_provider'].capitalize()}[/cyan]")
        if config.get("default_model"):
            console.print(f"  • Default model: [green]{config['default_model']}[/green]")
        if config.get("default_models"):
            models_str = ", ".join(config["default_models"])
            console.print(f"  • Default models: [green]{models_str}[/green]")
        if config.get("default_system_prompt"):
            console.print(f"  • Default system: [green]{os.path.basename(config['default_system_prompt'])}[/green]")
        if config.get("default_user_prompt"):
            console.print(f"  • Default user: [green]{os.path.basename(config['default_user_prompt'])}[/green]")
        console.print("  • Use [bold]--show-config[/bold] to see all settings")
    
    console.print("\n[bold cyan]TIPS:[/bold cyan]")
    console.print("  • For advanced usage: [bold]python ollama_prompt.py --help[/bold]")
    console.print("  • Skip this menu: [bold]python ollama_prompt.py --no-menu[/bold]")
    console.print("  • Quick reference: [bold]cat QuickStart.md[/bold]")
    console.print("  • Save settings: [bold]python ollama_prompt.py --save-config[/bold]")
    console.print("\n[bold blue]==============================\n[/bold blue]")
    
    # Get available models
    console.print("\n[bold blue]DEBUG: Getting available models...[/bold blue]")
    available_models = client.get_installed_models()
    console.print(f"[bold blue]DEBUG: Available models: {available_models}[/bold blue]")
    if not available_models:
        console.print("[bold red]No Ollama models found![/bold red]")
        console.print("Please install models with 'ollama pull <model>' and try again.")
        console.print("[bold red]DEBUG: Returning None from menu[/bold red]")
        return None
    
    # 0. Select provider
    console.print("\n[bold]0. Select a provider:[/bold]")
    providers = ["ollama", "openai"]
    for i, provider_name in enumerate(providers):
        console.print(f"  {i+1}. {provider_name.capitalize()}")
    
    provider_choice = input("\nEnter your provider choice (number or press Enter for default): ")
    if provider_choice:
        try:
            selected_provider = providers[int(provider_choice) - 1]
            console.print(f"[green]Selected provider: {selected_provider.capitalize()}[/green]")
        except (ValueError, IndexError):
            console.print("[bold red]Invalid choice. Using default provider.[/bold red]")
            selected_provider = provider
    else:
        selected_provider = provider
    
    # 1. Select models
    console.print("\n[bold]1. Select models:[/bold]")
    
    # Show model categories with descriptions
    def show_model_categories():
        console.print("\n[bold]Model Categories:[/bold]")
        console.print("[yellow]Fast Models:[/yellow]")
        console.print("  1. phi3:mini - Fast, good for quick explanations")
        console.print("[cyan]Detailed Analysis:[/cyan]")
        console.print("  2. llama3:8b - Detailed analysis, slightly slower")
        console.print("[green]Recommended:[/green]")
        console.print("  3. phi3:mini (Recommended for most uses)")
        console.print("  4. llama3:8b (Recommended for detailed analysis)")
        
        return {
            "1": "phi3:mini",
            "2": "llama3:8b",
            "3": "phi3:mini",
            "4": "llama3:8b"
        }
    
    # Show model categories
    model_categories = show_model_categories()
    
    # Handle keyboard shortcuts
    while True:
        choice = input("\nEnter your model choice (number or press Enter for recommended): ").lower()
        
        if choice == 'q':
            console.print("[bold yellow]Quitting menu...[/bold yellow]")
            return None
        elif choice == 'b':
            console.print("[bold yellow]Going back to previous menu...[/bold yellow]")
            return None
        elif choice == 'h':
            show_help()
            continue
        elif choice == 'r':
            console.print("[bold yellow]Resetting to defaults...[/bold yellow]")
            selected_models = ["phi3:mini", "llama3:8b"]
            break
        elif choice == 's':
            console.print("[bold yellow]Saving current settings...[/bold yellow]")
            config_manager.save_config({
                "default_models": selected_models,
                "default_provider": selected_provider,
                "default_stream": stream,
                "default_save": save_config
            })
            continue
        else:
            break
    
    if choice:
        try:
            selected_model = model_categories[choice]
            console.print(f"[green]Selected model: {selected_model}[/green]")
            selected_models = [selected_model]
        except KeyError:
            console.print("[bold red]Invalid choice. Using recommended models.[/bold red]")
            selected_models = ["phi3:mini", "llama3:8b"]
    else:
        selected_models = ["phi3:mini", "llama3:8b"]
    
    # 2. Select prompt category
    console.print("\n[bold]2. Select a prompt category:[/bold]")
    
    prompt_categories = show_prompt_categories()
    
    # Handle keyboard shortcuts
    while True:
        choice = input("\nEnter your category choice (number): ").lower()
        
        if choice == 'q':
            console.print("[bold yellow]Quitting menu...[/bold yellow]")
            return None
        elif choice == 'b':
            console.print("[bold yellow]Going back to previous menu...[/bold yellow]")
            return None
        elif choice == 'h':
            show_help()
            continue
        elif choice == 'r':
            console.print("[bold yellow]Resetting to defaults...[/bold yellow]")
            selected_prompt = None
            break
        elif choice == 's':
            console.print("[bold yellow]Saving current settings...[/bold yellow]")
            config_manager.save_config({
                "default_prompt_category": category,
                "default_provider": selected_provider,
                "default_stream": stream,
                "default_save": save_config
            })
            continue
        else:
            break
    
    if choice:
        try:
            category = prompt_categories[choice]
            if category == "custom":
                console.print("\nEnter your custom prompt (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if not line and lines and not lines[-1]:
                        break
                    lines.append(line)
                selected_prompt = "\n".join(lines)
            else:
                # Look for matching prompt file
                prompt_files = glob.glob(f"user_prompts/*{category}*.md")
                if prompt_files:
                    selected_prompt = prompt_files[0]
                else:
                    console.print("[yellow]No matching prompt found. Using custom prompt.[/yellow]")
                    selected_prompt = None
        except KeyError:
            console.print("[bold red]Invalid choice. Using custom prompt.[/bold red]")
            selected_prompt = None
    if prompt_choice:
        try:
            category = prompt_categories[prompt_choice]
            if category == "custom":
                console.print("\nEnter your custom prompt (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if not line and lines and not lines[-1]:
                        break
                    lines.append(line)
                selected_prompt = "\n".join(lines)
            else:
                # Look for matching prompt file
                prompt_files = glob.glob(f"user_prompts/*{category}*.md")
                if prompt_files:
                    selected_prompt = prompt_files[0]
                else:
                    console.print("[yellow]No matching prompt found. Using custom prompt.[/yellow]")
                    selected_prompt = None
        except KeyError:
            console.print("[bold red]Invalid choice. Using custom prompt.[/bold red]")
        try:
            prompt_idx = int(prompt_choice) - 1
            if 0 <= prompt_idx < len(all_user_prompts):
                selected_prompt = all_user_prompts[prompt_idx][1]
                prompt_name = all_user_prompts[prompt_idx][0]
                # Store user prompt filename (for output naming)
                client._user_filename = os.path.splitext(os.path.basename(selected_prompt))[0]
                console.print(f"[green]Selected user prompt: {prompt_name}[/green]")
            else:
                console.print("[bold red]Invalid choice. No prompt selected. You will be asked to enter a prompt directly.[/bold red]")
                selected_prompt = None
                client._user_filename = "direct_input"
        except ValueError:
            console.print("[bold red]Invalid input. No prompt selected. You will be asked to enter a prompt directly.[/bold red]")
            selected_prompt = None
            client._user_filename = "direct_input"
    else:
        console.print("[bold yellow]No user prompts available. You will be asked to enter a prompt directly.[/bold yellow]")
        selected_prompt = None
        client._user_filename = "direct_input"
    
    # 3. Select system prompt (optional)
    system_prompt_options = [("None", None)]
    if system_files:
        for file in system_files:
            system_prompt_options.append((os.path.basename(file), file))
    
    console.print("\n[bold]3. Select a system prompt (optional):[/bold]")
    for i, (system_name, _) in enumerate(system_prompt_options):
        console.print(f"  {i+1}. {system_name}")
    
    system_choice = input("\nEnter your system prompt choice (number): ")
    try:
        system_idx = int(system_choice) - 1
        if 0 <= system_idx < len(system_prompt_options):
            selected_system = system_prompt_options[system_idx][1]
            system_name = system_prompt_options[system_idx][0]
            # Store system filename (basename without extension) for output naming
            if selected_system:
                client._system_filename = os.path.splitext(system_name)[0]
            else:
                client._system_filename = "no_system"
            console.print(f"[green]Selected system prompt: {system_name}[/green]")
        else:
            console.print("[bold red]Invalid choice. Using no system prompt.[/bold red]")
            selected_system = None
            client._system_filename = "no_system"
    except ValueError:
        console.print("[bold red]Invalid input. Using no system prompt.[/bold red]")
        selected_system = None
        client._system_filename = "no_system"
    
    # 4. Output mode
    console.print("\n[bold]4. Output mode:[/bold]")
    console.print("[yellow]Streaming:[/yellow]")
    console.print("  1. Real-time - see tokens as they're generated")
    console.print("  2. Complete - see the full formatted response when finished")
    console.print("[cyan]Additional Options:[/cyan]")
    console.print("  3. Both - see streaming output followed by complete response")
    console.print("  4. Save only - save response to file without displaying")
    
    # Handle keyboard shortcuts
    while True:
        choice = input("\nEnter your choice (number or press Enter for default): ").lower()
        
        if choice == 'q':
            console.print("[bold yellow]Quitting menu...[/bold yellow]")
            return None
        elif choice == 'b':
            console.print("[bold yellow]Going back to previous menu...[/bold yellow]")
            return None
        elif choice == 'h':
            show_help()
            continue
        elif choice == 'r':
            console.print("[bold yellow]Resetting to defaults...[/bold yellow]")
            stream = False
            break
        elif choice == 's':
            console.print("[bold yellow]Saving current settings...[/bold yellow]")
            config_manager.save_config({
                "default_stream": stream,
                "default_provider": selected_provider,
                "default_save": save_config
            })
            continue
        else:
            break
    
    if choice == "1":
        stream = True
        console.print("[green]Output mode: Real-time streaming[/green]")
    elif choice == "2":
        stream = False
        console.print("[green]Output mode: Complete formatted response[/green]")
    elif choice == "3":
        stream = True
        console.print("[green]Output mode: Streaming + Complete[/green]")
    else:
        stream = False
        console.print("[green]Output mode: Complete formatted response[/green]")
    if stream_choice == "1":
        stream = True
        console.print("[green]Output mode: Real-time streaming[/green]")
    elif stream_choice == "2":
        stream = False
        console.print("[green]Output mode: Complete formatted response[/green]")
    elif stream_choice == "3":
        stream = True
        console.print("[green]Output mode: Streaming + Complete[/green]")
    else:
        stream = False
        console.print("[green]Output mode: Complete formatted response[/green]")
    
    # 5. Save as defaults?
    console.print("\n[bold]5. Save these settings as defaults?[/bold]")
    console.print("  1. Yes - save these selections as default configuration")
    console.print("  2. No - don't save (default)")
    
    save_config_choice = input("\nEnter your choice (number or press Enter for default): ")
    save_config = save_config_choice == "1"
    if save_config:
        console.print("[green]These settings will be saved as defaults.[/green]")
    
    console.print(f"\n[bold blue]Starting {selected_provider.capitalize()} generation...[/bold blue]\n")
    
    console.print(f"\n[bold blue]Starting {selected_provider.capitalize()} generation...[/bold blue]")
    
    return selected_provider, selected_models, selected_prompt, selected_system, stream, save_config

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run prompts with LLM models from different providers")
    
    # Model selection arguments
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--model", type=str,
                        help="Model to use (default: use config or all models)")
    model_group.add_argument("--all-models", action="store_true",
                        help="Run the prompt on all installed models (Ollama provider only)")
    model_group.add_argument("--models", type=str, nargs="+",
                        help="List of specific models to run the prompt on")
    
    # Provider arguments
    provider_group = parser.add_argument_group("Provider Options")
    provider_group.add_argument("--provider", type=str, choices=["ollama", "openai"],
                        help="The LLM provider to use (default: ollama)")
    provider_group.add_argument("--api-key", type=str,
                        help="API key for OpenAI (recommended to use environment variables instead)")
    provider_group.add_argument("--setup-keys", action="store_true",
                        help="Set up API keys for providers interactively")
                        
    # Prompt arguments
    prompt_group = parser.add_argument_group("Prompt Selection")
    prompt_group.add_argument("--prompt-file", type=str,
                        help="Path to file containing the user prompt")
    prompt_group.add_argument("--system-file", type=str,
                        help="Path to file containing the system prompt")
    prompt_group.add_argument("--list-prompts", action="store_true",
                        help="List available prompt files and exit")
    
    # Output arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--stream", action="store_true",
                        help="Stream the response token by token")
    output_group.add_argument("--no-save", dest="save", action="store_false",
                        help="Don't save the response to a file")
    output_group.add_argument("--max-workers", type=int,
                        help="Maximum number of worker threads for parallel execution")
    output_group.add_argument("--timeout", type=int,
                        help="Timeout in seconds for each model request (default from config)")
    output_group.add_argument("--no-menu", action="store_true",
                        help="Skip the interactive CLI menu")
    
    # Configuration arguments
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--show-config", action="store_true",
                        help="Display current configuration and exit")
    config_group.add_argument("--save-config", action="store_true",
                        help="Save current run settings as default configuration")
    config_group.add_argument("--reset-config", action="store_true",
                        help="Reset configuration to defaults")
    config_group.add_argument("--base-url", type=str,
                        help="Set API base URL (for Ollama default: http://localhost:11434)")
    config_group.add_argument("--force-menu", action="store_true",
                        help="Force the interactive menu to be displayed")
    config_group.add_argument("--update-config", nargs="*", action="append",
                        help="Update configuration settings. Format: --update-config key=value key2=value2")
    
    # Set defaults from configuration
    config = config_manager.load_config()
    parser.set_defaults(
        save=config["default_save"],
        stream=config["default_stream"],
        all_models=False,
        timeout=config["default_timeout"]
    )
    
    # Set provider default to ollama if not specified in config
    parser.set_defaults(provider=config.get("default_provider", "ollama"))
    
    args = parser.parse_args()
    
    # Create necessary directories if they don't exist
    os.makedirs("system_prompts", exist_ok=True)
    os.makedirs("user_prompts", exist_ok=True)
    
    # Run API key setup if requested
    if args.setup_keys:
        api_key_manager.setup_api_keys()
        return
    
    # Handle configuration commands
    if args.show_config:
        config_manager.display_config()
        return
    
    if args.reset_config:
        if config_manager.reset_config():
            console = Console()
            console.print("[bold green]Configuration reset to defaults.[/bold green]")
        return
    
    if args.save_config:
        # Save the current configuration without running any prompts
        config = config_manager.load_config()
        config_manager.save_config(config)
        console = Console()
        console.print("[bold green]Current configuration saved.[/bold green]")
        return
        
    if args.update_config:
        # Flatten the list of lists from nargs="*" action="append"
        update_items = [item for sublist in args.update_config for item in sublist]
        
        # Parse key=value pairs
        updates = {}
        for item in update_items:
            if "=" not in item:
                console = Console()
                console.print(f"[bold red]Error:[/bold red] Invalid format: {item}")
                console.print("Expected format: key=value")
                return
            
            key, value = item.split("=", 1)
            updates[key] = value
        
        # Update configuration
        config = config_manager.load_config()
        for key, value in updates.items():
            if key not in config:
                console = Console()
                console.print(f"[bold yellow]Warning:[/bold yellow] Unknown config key: {key}")
                continue
            
            if isinstance(config[key], bool):
                # Convert string to boolean for boolean fields
                value = value.lower() in ['true', '1', 't', 'y', 'yes']
            
            config[key] = value
        
        if config_manager.save_config(config):
            console = Console()
            console.print("[bold green]Configuration updated successfully.[/bold green]")
            return
    
    # Update base URL in configuration if specified
    if args.base_url:
        config_manager.update_config("base_url", args.base_url)
    
    # List available prompt files if requested
    if args.list_prompts:
        system_files = get_available_files("system_prompts")
        user_prompt_files = get_available_files("user_prompts")
        
        console = Console()
        console.print("\n[bold]Available System Prompts:[/bold]")
        if system_files:
            for file in system_files:
                console.print(f"  - {file}")
        else:
            console.print("  No system prompt files found in system_prompts/ directory")
            
        console.print("\n[bold]Available User Prompts:[/bold]")
        if user_prompt_files:
            for file in user_prompt_files:
                console.print(f"  - {file}")
        else:
            console.print("  No user prompt files found in user_prompts/ directory")
        
        return
    
    # Get provider and possibly API key
    provider = args.provider
    
    # Handle API key for non-Ollama providers
    api_key = None
    if provider != "ollama":
        # Get API key from arg, environment, or keyring
        api_key = args.api_key
        if not api_key:
            # Try to get from key manager
            key_manager = api_key_manager.ApiKeyManager()
            api_key = key_manager.get_api_key(provider)
            
            # If still no API key, prompt user
            if not api_key:
                api_key = key_manager.prompt_for_api_key(provider)
                
            # If still no API key, exit with error
            if not api_key:
                console = Console()
                console.print(f"[bold red]Error:[/bold red] {provider.capitalize()} API key is required")
                console.print(f"Please provide an API key using --api-key or set up keys with --setup-keys")
                return
    
    # Create the appropriate client/adapter for the selected provider
    if provider == "ollama":
        # Use the original OllamaClient for backward compatibility
        client = OllamaClient(base_url=args.base_url)
    else:
        # For other providers, use the adapter factory
        try:
            adapter = api_adapters.LLMAdapterFactory.create_adapter(
                provider, 
                api_key=api_key,
                base_url=args.base_url
            )
            # Create a thin wrapper to make adapter look like OllamaClient for compatibility
            client = AdapterWrapper(adapter, provider)
        except ValueError as e:
            console = Console()
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            return
    
    # Check if any relevant command line arguments were provided
    # Only check for arguments that would affect model selection or prompt content
    relevant_arg_names = ['model', 'all_models', 'prompt_file', 'system_file', 'models', 'provider']
    args_dict = vars(args)
    
    # Debug print
    console = Console()
    console.print("\n[bold blue]DEBUG: Command line arguments:[/bold blue]")
    for arg_name in relevant_arg_names:
        value = args_dict.get(arg_name)
        console.print(f"  • {arg_name}: {value}")
    console.print(f"  • no_menu: {args_dict.get('no_menu')}")
    
    # More detailed check to help debug
    console.print("\n[bold blue]DEBUG: More details on arguments:[/bold blue]")
    args_values = []
    for arg_name in relevant_arg_names:
        value = args_dict.get(arg_name)
        # Print the value and its type
        console.print(f"  • {arg_name}: {value} (type: {type(value)})")
        args_values.append(value)
    
    # Python treats empty lists, None, False, and 0 as falsy
    has_relevant_args = any(arg for arg in args_values if arg)
    console.print(f"[bold blue]DEBUG: has_relevant_args: {has_relevant_args}[/bold blue]")
    
    # Check if we should use menu based on config and args
    config_use_menu = config_manager.get_config_value("use_menu", True)
    use_menu = config_use_menu and not has_relevant_args and not args.no_menu
    console.print(f"[bold blue]DEBUG: config_use_menu: {config_use_menu}, use_menu: {use_menu}[/bold blue]")
    
    # Force display of menu in these cases:
    # 1. Explicit force-menu flag
    # 2. No command line arguments (just the script name)
    # 3. Config says to use menu and no relevant args provided
    force_menu = args.force_menu or len(sys.argv) == 1
    
    if force_menu or use_menu:
        # Use interactive CLI menu
        console.print("\n[bold green]Displaying interactive menu...[/bold green]")
        menu_results = display_interactive_cli_menu(client, provider)
        if menu_results is None:
            console.print("[bold red]Menu returned None - no models found or user cancelled[/bold red]")
            return
        
        try:
            console.print("\n[bold blue]DEBUG: Got menu results, unpacking...[/bold blue]")
            menu_provider, models_to_run, prompt_file, system_file, stream, save_config_from_menu = menu_results
            console.print(f"[bold blue]DEBUG: Menu selections:[/bold blue]")
            console.print(f"  • Provider: {menu_provider}")
            console.print(f"  • Models: {models_to_run}")
            console.print(f"  • Prompt file: {prompt_file}")
            console.print(f"  • System file: {system_file}")
            
            # If provider changed in the menu, recreate the client
            if menu_provider != provider:
                provider = menu_provider
                if provider == "ollama":
                    client = OllamaClient(base_url=args.base_url)
                else:
                    # Get API key (should be already set from the menu)
                    key_manager = api_key_manager.ApiKeyManager()
                    api_key = key_manager.get_api_key(provider)
                    
                    # Create adapter with the appropriate base URL
                    base_url = args.base_url
                    if provider == "openai":
                        base_url = config_manager.get_config_value("openai_base_url", "https://api.openai.com")
                        if args.base_url:  # Use command line arg if provided
                            base_url = args.base_url
                        console.print(f"[bold blue]DEBUG: Using OpenAI base URL: {base_url}[/bold blue]")
                            
                    adapter = api_adapters.LLMAdapterFactory.create_adapter(
                        provider, 
                        api_key=api_key,
                        base_url=base_url
                    )
                    client = AdapterWrapper(adapter, provider)
        except Exception as e:
            # If there was an issue with menu results
            console = Console()
            console.print(f"[bold red]Error with menu selection: {str(e)}. Exiting.[/bold red]")
            return
        save = True  # Always save in interactive mode
        max_workers = None  # Use default
        timeout = config_manager.get_config_value("default_timeout", 1200)
        
        # If user chose to save settings from menu, set the save_config flag
        if save_config_from_menu:
            args.save_config = True
    else:
        # Use command line arguments and fall back to config values
        # Get system prompt if specified
        system_prompt = None
        if args.system_file:
            system_file = args.system_file
        else:
            system_file = config_manager.get_config_value("default_system_prompt")
            
        # Get user prompt file
        if args.prompt_file:
            prompt_file = args.prompt_file
        else:
            prompt_file = config_manager.get_config_value("default_user_prompt")
        
        # Determine which models to use based on provider
        # For non-Ollama providers, --all-models doesn't make sense since they may have thousands
        if provider != "ollama" and args.all_models:
            client.console.print("[bold yellow]Warning:[/bold yellow] --all-models is only supported for Ollama provider")
            client.console.print("[bold yellow]Using default models for provider instead[/bold yellow]")
            args.all_models = False
        
        if args.all_models and provider == "ollama":
            # Run on all installed Ollama models (excluding embedding models)
            models_to_run = client.get_installed_models()
        elif args.models:
            # Run on specified list of models
            if provider == "ollama":
                # Filter out embedding models for Ollama
                models_to_run = [model for model in args.models if "embed" not in model.lower()]
                if len(models_to_run) < len(args.models):
                    client.console.print("[bold yellow]Warning:[/bold yellow] Skipped embedding models which cannot generate text")
            else:
                # For other providers, use models as specified
                models_to_run = args.models
        elif args.model:
            # Run on a single model
            if provider == "ollama" and "embed" in args.model.lower():
                client.console.print("[bold red]Error:[/bold red] Cannot run text generation on embedding model")
                return
            else:
                models_to_run = [args.model]
        else:
            # Check config for default model or models
            default_model = config_manager.get_config_value("default_model")
            default_models = config_manager.get_config_value("default_models", [])
            
            if default_model:
                if provider == "ollama" and "embed" not in default_model.lower():
                    models_to_run = [default_model]
                elif provider != "ollama":
                    models_to_run = [default_model]
                else:
                    models_to_run = client.get_installed_models()
            elif default_models:
                if provider == "ollama":
                    models_to_run = [model for model in default_models if "embed" not in model.lower()]
                else:
                    models_to_run = default_models
            else:
                models_to_run = client.get_installed_models()
            
        # Get output options
        stream = args.stream
        save = args.save
        
        # For max_workers and timeout, prefer command line args, then config, then default
        if args.max_workers is not None:
            max_workers = args.max_workers
        else:
            max_workers = config_manager.get_config_value("default_max_workers")
            
        if args.timeout is not None:
            timeout = args.timeout
        else:
            timeout = config_manager.get_config_value("default_timeout", 1200)
    
    # Get system prompt content if specified
    system_prompt = None
    if system_file and os.path.exists(system_file):
        system_prompt = get_prompt_content(system_file)
        if system_prompt is None:
            return
        # Store system filename (basename without extension) for output naming
        client._system_filename = os.path.splitext(os.path.basename(system_file))[0]
    else:
        client._system_filename = "no_system"
    
    # Get user prompt content from file or ask for direct input
    if prompt_file and os.path.exists(prompt_file):
        main_prompt = get_prompt_content(prompt_file)
        if main_prompt is None:
            return
        # Store user prompt filename (basename without extension) for output naming
        client._user_filename = os.path.splitext(os.path.basename(prompt_file))[0]
    else:
        # Since we don't have a valid prompt file, and we're supposed to use menu,
        # Let's force display menu instead of proceeding to direct input
        if config_use_menu and not args.no_menu and not has_relevant_args:
            # Force display of interactive menu
            menu_results = display_interactive_cli_menu(client, provider)
            if menu_results is None:
                return
            
            provider, models_to_run, prompt_file, system_file, stream, save_config = menu_results
            
            # Now try again with the prompt file from the menu
            if prompt_file and os.path.exists(prompt_file):
                main_prompt = get_prompt_content(prompt_file)
                if main_prompt is None:
                    return
                client._user_filename = os.path.splitext(os.path.basename(prompt_file))[0]
            else:
                # Still no prompt file, go to direct input
                console = Console()
                console.print("\n[bold]No prompt file selected. Please enter your prompt directly:[/bold]")
                console.print("[dim]Type your prompt below. When finished, press Enter twice (leave a blank line).[/dim]")
                
                lines = []
                while True:
                    line = input()
                    if not line and lines and not lines[-1]:  # Two consecutive empty lines
                        break
                    lines.append(line)
                
                main_prompt = "\n".join(lines).strip()
                if not main_prompt:
                    console.print("[bold red]No prompt provided. Exiting.[/bold red]")
                    return
                
                client._user_filename = "direct_input"
        else:
            # Ask user for direct input
            console = Console()
            console.print("\n[bold]No prompt file selected. Please enter your prompt directly:[/bold]")
            console.print("[dim]Type your prompt below. When finished, press Enter twice (leave a blank line).[/dim]")
            
            lines = []
            while True:
                line = input()
                if not line and lines and not lines[-1]:  # Two consecutive empty lines
                    break
                lines.append(line)
            
            main_prompt = "\n".join(lines).strip()
            if not main_prompt:
                console.print("[bold red]No prompt provided. Exiting.[/bold red]")
                return
                
            client._user_filename = "direct_input"
    
    # Store the system prompt in the client object for access in save_response
    client._system_prompt = system_prompt
    
    # Combine system prompt and user prompt if system prompt is provided
    if system_prompt:
        prompt = f"{system_prompt}\n\n{main_prompt}"
    else:
        prompt = main_prompt
        
    # Generate responses
    if not models_to_run:
        console = Console()
        console.print("[bold red]No models specified or found![/bold red]")
        console.print("Try installing models with: [bold]ollama pull llama3:8b[/bold] (or another model)")
        return
        
    # Display a summary of what will run
    console = Console()
    model_str = models_to_run[0] if len(models_to_run) == 1 else f"{len(models_to_run)} models"
    prompt_str = os.path.basename(prompt_file) if prompt_file else "Direct user input"
    system_str = os.path.basename(system_file) if system_file else "None"
    
    console.print("\n[bold]Running with:[/bold]")
    console.print(f"  • Provider: [cyan]{provider.capitalize()}[/cyan]")
    console.print(f"  • Model(s): [green]{model_str}[/green]")
    console.print(f"  • User prompt: [green]{prompt_str}[/green]")
    console.print(f"  • System prompt: [green]{system_str}[/green]")
    console.print(f"  • Output mode: [green]{'Streaming' if stream else 'Complete'}[/green]")
    console.print(f"  • Save responses: [green]{'Yes' if save else 'No'}[/green]\n")
    
    # Helper function to show current selections
    def show_current_selections():
        console.print("\n[bold]Current Selections:[/bold]")
        console.print(f"  • Provider: [green]{selected_provider.capitalize()}[/green]")
        console.print(f"  • Models: [green]{', '.join(selected_models)}[/green]")
        console.print(f"  • User prompt: [green]{selected_prompt or 'None'}[/green]")
        console.print(f"  • System prompt: [green]{selected_system or 'None'}[/green]")
        console.print(f"  • Output mode: [green]{'Streaming' if stream else 'Complete'}[/green]")
        console.print(f"  • Save config: [green]{'Yes' if save_config else 'No'}[/green]")
    
    # Helper function to show model categories
    def show_model_categories():
        console.print("\n[bold]Model Categories:[/bold]")
        console.print("[yellow]Fast Models:[/yellow]")
        console.print("  1. phi3:mini - Fast, good for quick explanations")
        console.print("[cyan]Detailed Analysis:[/cyan]")
        console.print("  2. llama3:8b - Detailed analysis, slightly slower")
        console.print("[green]Recommended:[/green]")
        console.print("  3. phi3:mini (Recommended for most uses)")
        console.print("  4. llama3:8b (Recommended for detailed analysis)")
        
        return {
            "1": "phi3:mini",
            "2": "llama3:8b",
            "3": "phi3:mini",
            "4": "llama3:8b"
        }\n")
    
    # Run the model(s)
    if len(models_to_run) == 1:
        # Just run a single model normally
        client.generate_response(models_to_run[0], prompt, stream=stream, save=save, timeout=timeout)
    else:
        # Run on multiple models
        client.run_prompt_on_all_models(
            prompt, 
            models=models_to_run, 
            save=save, 
            stream=stream,
            max_workers=max_workers,
            timeout=timeout
        )
        
    # Save configuration if requested
    if args.save_config:
        # Create a config dictionary from the actual values used in this run
        # This way, even menu-selected options are saved correctly
        run_config = {
            "default_provider": provider,
            "default_model": models_to_run[0] if len(models_to_run) == 1 else None,
            "default_models": models_to_run if len(models_to_run) > 1 else [],
            "default_system_prompt": system_file,
            "default_user_prompt": prompt_file,
            "default_stream": stream,
            "default_save": save,
            "default_max_workers": max_workers,
            "default_timeout": timeout,
            "use_menu": not args.no_menu if hasattr(args, 'no_menu') else True
        }
        
        # Add base URL if specified
        if args.base_url:
            if provider == "ollama":
                run_config["base_url"] = args.base_url
            elif provider == "openai":
                run_config["openai_base_url"] = args.base_url
        
        # Load existing config and update with current run values
        config = config_manager.load_config()
        for key, value in run_config.items():
            if value is not None:  # Only update non-None values
                config[key] = value
                
        if config_manager.save_config(config):
            console = Console()
            console.print("\n[bold green]Current settings saved as default configuration.[/bold green]")
            console.print("[green]Future runs will use these settings unless overridden.[/green]")
        
if __name__ == "__main__":
    main()