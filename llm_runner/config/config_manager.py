from dataclasses import dataclass
from typing import Dict, Optional, Any
import yaml
import os
from pathlib import Path

@dataclass
class Config:
    base_url: str = "http://localhost:11434"
    provider: str = "ollama"
    api_key: Optional[str] = None
    model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.model
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            base_url=data.get("base_url", "http://localhost:11434"),
            provider=data.get("provider", "ollama"),
            api_key=data.get("api_key"),
            model=data.get("model")
        )

class ConfigManager:
    def __init__(self, config_path: str = "config/ollama_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Config:
        """Load configuration from file"""
        if not self.config_path.exists():
            # Create default config file if it doesn't exist
            default_config = Config()
            self.save_config(default_config)
            return default_config
            
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
                return Config.from_dict(config_data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return Config()

    def save_config(self, config: Config) -> None:
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, "w") as f:
                yaml.dump(config.to_dict(), f)
            self.config = config
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_config(self) -> Config:
        """Get current configuration"""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        current_config = self.get_config()
        for key, value in kwargs.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
        self.save_config(current_config)

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        return self.config.api_key if self.config.provider == provider else None
