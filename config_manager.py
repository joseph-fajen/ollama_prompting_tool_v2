import os
import yaml
from rich.console import Console

# Default configuration
DEFAULT_CONFIG = {
    "base_url": "http://localhost:11434",
    "default_model": None,  # Will use all models if None
    "default_system_prompt": None,
    "default_user_prompt": None,
    "default_stream": False,
    "default_save": True,
    "default_max_workers": None,  # Use CPU count
    "default_timeout": 1200,
    "use_menu": True,
    "default_models": [],  # Specific list of models to use
    "default_provider": "ollama",  # API provider (ollama, openai, huggingface)
    "api_key": None,  # API key for OpenAI or HuggingFace
    "openai_base_url": "https://api.openai.com",  # Base URL for OpenAI API
    "default_chat_mode": False,  # Use chat mode by default
    "config_version": "1.1"  # Updated version
}

# Constants
CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "ollama_config.yaml")
console = Console()

def ensure_config_dir():
    """Ensure the config directory exists"""
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config():
    """Load configuration from file, or create default if not exists"""
    ensure_config_dir()
    
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle version updates or missing keys
        if not config:
            return DEFAULT_CONFIG.copy()
            
        # Ensure all expected keys exist, fill in defaults if missing
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = default_value
                
        return config
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {str(e)}")
        console.print("[yellow]Using default configuration[/yellow]")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file"""
    ensure_config_dir()
    
    try:
        # Make config directory writable if it exists
        if os.path.exists(CONFIG_DIR):
            os.chmod(CONFIG_DIR, 0o755)  # rwxr-xr-x
            
        # Make config file writable if it exists
        if os.path.exists(CONFIG_FILE):
            os.chmod(CONFIG_FILE, 0o644)  # rw-r--r--
            
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        # Ensure the file has proper permissions
        os.chmod(CONFIG_FILE, 0o644)  # rw-r--r--
            
        console.print(f"[dim]Configuration saved to {CONFIG_FILE}[/dim]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error saving config:[/bold red] {str(e)}")
        return False

def update_config(key, value):
    """Update a specific configuration value"""
    config = load_config()
    config[key] = value
    return save_config(config)

def reset_config():
    """Reset configuration to defaults"""
    return save_config(DEFAULT_CONFIG.copy())

def display_config():
    """Display current configuration values"""
    config = load_config()
    
    console.print("\n[bold blue]Current Configuration:[/bold blue]")
    for key, value in config.items():
        # Skip config_version in display
        if key == "config_version":
            continue
        
        # Format default_models as a comma-separated string
        if key == 'default_models':
            if isinstance(value, list):
                value = ', '.join(value)
            else:
                value = str(value)
        
        if value is None:
            value_str = "[italic]None[/italic]"
        elif isinstance(value, bool):
            value_str = "[green]True[/green]" if value else "[red]False[/red]"
        elif isinstance(value, list) and not value:
            value_str = "[italic]Empty list[/italic]"
        else:
            value_str = str(value)
            
        console.print(f"  • {key}: {value_str}")
    console.print("")

def get_config_value(key, default=None):
    """Get a specific configuration value with fallback default"""
    config = load_config()
    return config.get(key, default)

def save_current_run(args):
    """Save current run parameters to configuration"""
    config = load_config()
    
    # Only save non-None values
    if args.model:
        config["default_model"] = args.model
    
    if args.models:
        config["default_models"] = args.models
    
    if args.system_file:
        config["default_system_prompt"] = args.system_file
    
    if args.prompt_file:
        config["default_user_prompt"] = args.prompt_file
    
    # Save explicit boolean flags
    if hasattr(args, 'stream'):
        config["default_stream"] = args.stream
    
    if hasattr(args, 'save'):
        config["default_save"] = args.save
    
    if args.max_workers:
        config["default_max_workers"] = args.max_workers
    
    if args.timeout:
        config["default_timeout"] = args.timeout
    
    if hasattr(args, 'no_menu'):
        config["use_menu"] = not args.no_menu
    
    return save_config(config)