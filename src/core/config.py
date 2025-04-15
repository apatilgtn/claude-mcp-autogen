"""
Configuration management for the MCP system.
This module provides functionality to load and access configuration settings.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List

from loguru import logger
from pydantic import BaseModel, Field


class MCPSystemConfig(BaseModel):
    """MCP system configuration model."""
    name: str = "Claude-Inspired MCP"
    version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    data_dir: str = "./data"
    temp_dir: str = "./tmp"
    enable_docker: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: int = 600
    api_reload: bool = True
    
    # UI settings
    ui_enabled: bool = True
    ui_host: str = "0.0.0.0"
    ui_port: int = 8501
    
    # Security settings
    secret_key: str = ""
    token_expiration: int = 86400  # 24 hours
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    # LLM settings
    default_llm_provider: str = "anthropic"
    default_llm_model: str = "claude-3-7-sonnet-20250219"
    
    # Agent settings
    max_agents: int = 10
    default_agent_timeout: int = 60
    
    # MCP settings
    max_message_size: int = 1024 * 1024  # 1 MB
    max_conversation_history: int = 100
    
    # Extension settings
    extensions_dir: str = "./extensions"
    enable_extensions: bool = True
    
    # Additional settings
    extra: Dict[str, Any] = Field(default_factory=dict)


class ConfigLoader:
    """Configuration loader for the MCP system."""
    
    def __init__(self):
        """Initialize the configuration loader."""
        self.config: Dict[str, Any] = {}
        self.loaded_files: List[str] = []
    
    def load_from_env(self, prefix: str = "MCP_") -> None:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
        """
        # Convert environment variables to config
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert key format: MCP_API_HOST -> api_host
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys: MCP_API_TIMEOUT -> api.timeout
                if '__' in config_key:
                    parts = config_key.split('__')
                    current = self.config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    self.config[config_key] = self._parse_env_value(value)
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found: {file_path}")
            return False
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    config = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {file_path}")
                return False
            
            # Update config with loaded values
            self._deep_update(self.config, config)
            self.loaded_files.append(file_path)
            logger.info(f"Loaded configuration from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Handle nested keys: api.host
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        # Handle nested keys: api.host
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def to_model(self) -> MCPSystemConfig:
        """
        Convert the configuration to a Pydantic model.
        
        Returns:
            MCPSystemConfig model
        """
        # Convert nested dictionaries to flat keys
        flat_config = self._flatten_dict(self.config)
        
        # Create model from flat config
        config_model = MCPSystemConfig(**flat_config)
        
        return config_model
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse an environment variable value.
        
        Args:
            value: Environment variable value
            
        Returns:
            Parsed value
        """
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Check for boolean values
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Check for numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update a nested dictionary.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested values
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# Create global configuration instance
settings = ConfigLoader()

# Load configuration from default locations
default_config_files = [
    "./config/default.yaml",
    "./config/default.json",
    f"./config/{os.environ.get('ENVIRONMENT', 'development')}.yaml",
    f"./config/{os.environ.get('ENVIRONMENT', 'development')}.json",
    "./config/local.yaml",
    "./config/local.json"
]

# Load from config files
for config_file in default_config_files:
    if os.path.exists(config_file):
        settings.load_from_file(config_file)

# Load from environment variables
settings.load_from_env(prefix="MCP_")

# Ensure sensitive settings are loaded from environment
sensitive_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SECRET_KEY"]
for key in sensitive_keys:
    if os.environ.get(key):
        settings.set(key.lower(), os.environ.get(key))
