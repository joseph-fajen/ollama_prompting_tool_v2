import os
import json
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
import keyring

class ApiKeyManager:
    """Manages API keys for different providers securely"""
    
    # Constants
    KEYRING_SERVICE = "ollama_prompting_tool"
    ENV_PREFIX = "OLLAMA_TOOL_"
    CONFIG_DIR = os.path.join(str(Path.home()), ".ollama_tool")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "api_config.json")
    
    def __init__(self):
        """Initialize the API key manager"""
        # Create config directory if it doesn't exist
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        
        # Initialize config file if it doesn't exist
        if not os.path.exists(self.CONFIG_FILE):
            self._save_config({})
            
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file (only non-sensitive data)"""
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
                
            # Set secure permissions
            os.chmod(self.CONFIG_FILE, 0o600)  # User read/write only
        except Exception as e:
            print(f"Error saving API configuration: {str(e)}")
            
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading API configuration: {str(e)}")
            return {}
            
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for the specified provider using multiple sources
        
        Checks in the following order:
        1. Environment variable (OLLAMA_TOOL_<PROVIDER>_API_KEY)
        2. Keyring/system credential store
        3. Config file
        4. User prompt if interactive is enabled
        """
        provider = provider.lower()
        
        # Check environment variable
        env_var = f"{self.ENV_PREFIX}{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
            
        # Check keyring
        try:
            api_key = keyring.get_password(self.KEYRING_SERVICE, provider)
            if api_key:
                return api_key
        except:
            # Keyring might not be available, continue to next method
            pass
            
        # Check config file
        config = self._load_config()
        if provider in config and "api_key" in config[provider]:
            return config[provider]["api_key"]
            
        return None
        
    def set_api_key(self, provider: str, api_key: str, use_keyring: bool = True) -> bool:
        """Set API key for the specified provider
        
        Args:
            provider: The provider name (e.g., "openai", "huggingface")
            api_key: The API key
            use_keyring: Whether to store in system keyring (True) or config file (False)
            
        Returns:
            bool: True if successful, False otherwise
        """
        provider = provider.lower()
        
        if use_keyring:
            try:
                keyring.set_password(self.KEYRING_SERVICE, provider, api_key)
                print(f"{provider.capitalize()} API key stored securely in system keyring")
                return True
            except Exception as e:
                print(f"Error storing API key in keyring: {str(e)}")
                print("Falling back to config file storage")
                # Fall back to config file if keyring fails
        
        # Store in config file
        config = self._load_config()
        if provider not in config:
            config[provider] = {}
        config[provider]["api_key"] = api_key
        self._save_config(config)
        print(f"{provider.capitalize()} API key stored in config file")
        return True
        
    def clear_api_key(self, provider: str) -> bool:
        """Clear API key for the specified provider"""
        provider = provider.lower()
        
        # Clear from keyring
        try:
            keyring.delete_password(self.KEYRING_SERVICE, provider)
        except:
            # Ignore if not in keyring
            pass
            
        # Clear from config file
        config = self._load_config()
        if provider in config and "api_key" in config[provider]:
            del config[provider]["api_key"]
            self._save_config(config)
            
        print(f"{provider.capitalize()} API key cleared")
        return True
        
    def prompt_for_api_key(self, provider: str, use_keyring: bool = True) -> Optional[str]:
        """Prompt user for API key interactively"""
        provider = provider.lower()
        
        print(f"\nAPI key for {provider.capitalize()} not found.")
        print(f"You can set it now or use the environment variable {self.ENV_PREFIX}{provider.upper()}_API_KEY")
        
        try:
            api_key = getpass.getpass(f"Enter {provider.capitalize()} API key (input will be hidden): ")
            if not api_key:
                print("No API key provided.")
                return None
                
            save = input("Save this API key for future use? (y/n): ").lower() == 'y'
            if save:
                use_secure = input("Use system secure storage (y) or config file (n)? (y/n): ").lower() == 'y'
                self.set_api_key(provider, api_key, use_secure)
                
            return api_key
        except (KeyboardInterrupt, EOFError):
            print("\nAPI key input cancelled.")
            return None
        
def setup_api_keys():
    """Interactive setup for API keys"""
    manager = ApiKeyManager()
    
    print("\n=== API Key Setup ===\n")
    print("This will help you set up API keys for different LLM providers.")
    print("Keys can be stored in your system's secure keyring or in a config file.")
    
    providers = ["openai", "huggingface"]
    
    for provider in providers:
        setup = input(f"\nSet up {provider.capitalize()} API key? (y/n): ").lower() == 'y'
        if setup:
            secure = input("Use system secure storage (recommended) (y) or config file (n)? (y/n): ").lower() == 'y'
            api_key = getpass.getpass(f"Enter {provider.capitalize()} API key (input will be hidden): ")
            if api_key:
                manager.set_api_key(provider, api_key, secure)
    
    print("\nAPI key setup complete!")
    print(f"You can also set keys using environment variables:")
    for provider in providers:
        print(f"  {manager.ENV_PREFIX}{provider.upper()}_API_KEY=your_key_here")
        
if __name__ == "__main__":
    setup_api_keys()