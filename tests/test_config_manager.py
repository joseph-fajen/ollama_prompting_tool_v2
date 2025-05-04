import pytest
import os
from llm_runner.config.config_manager import ConfigManager, Config

def test_load_config_default():
    """Test loading default config when file doesn't exist"""
    # Clean up any existing config file
    config_path = "config/ollama_config.yaml"
    if os.path.exists(config_path):
        os.remove(config_path)
    
    manager = ConfigManager()
    config = manager.get_config()
    assert config.base_url == "http://localhost:11434"
    assert config.provider == "ollama"
    assert config.api_key is None

def test_save_and_load_config():
    """Test saving and loading config"""
    # Clean up any existing config file
    config_path = "config/ollama_config.yaml"
    if os.path.exists(config_path):
        os.remove(config_path)
    
    manager = ConfigManager()
    test_config = Config(
        base_url="http://test:1234",
        provider="test",
        api_key="test_key",
        model="test_model"
    )
    
    manager.save_config(test_config)
    loaded_config = manager.get_config()
    
    assert loaded_config.base_url == "http://test:1234"
    assert loaded_config.provider == "test"
    assert loaded_config.api_key == "test_key"
    assert loaded_config.model == "test_model"

def test_update_config():
    """Test updating config"""
    # Clean up any existing config file
    config_path = "config/ollama_config.yaml"
    if os.path.exists(config_path):
        os.remove(config_path)
    
    manager = ConfigManager()
    
    # First set some initial config
    manager.update_config(
        base_url="http://initial:1234",
        provider="initial",
        api_key="initial_key"
    )
    
    # Now update only specific fields
    manager.update_config(
        base_url="http://new:1234",
        provider="new"
    )
    
    config = manager.get_config()
    assert config.base_url == "http://new:1234"
    assert config.provider == "new"
    assert config.api_key == "initial_key"  # Should keep previous value

def test_get_api_key():
    """Test getting API key for provider"""
    manager = ConfigManager()
    manager.update_config(
        provider="test",
        api_key="test_key"
    )
    
    assert manager.get_api_key("test") == "test_key"
    assert manager.get_api_key("other") is None
