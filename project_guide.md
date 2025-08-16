I built an Incident_response_agent : Directory structure:
└── src/
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   ├── config_manager.py
    │   └── settings.py
    ├── main/
    │   ├── __init__.py
    │   ├── agent_serve.py
    │   ├── cli.py
    │   ├── graph.py
    │   ├── nodes.py
    │   ├── react_agent.py
    │   └── state.py
    └── utils/
        ├── __init__.py
        ├── logging_config.py
        ├── rate_limiter.py
        └── retry.py


Files Content:

================================================
FILE: src/__init__.py
================================================
[Empty file]


================================================
FILE: src/config/__init__.py
================================================
[Empty file]


================================================
FILE: src/config/config_manager.py
================================================
"""
Configuration management utilities for the DevOps Agent.
This module provides utilities for managing different environment configurations
and demonstrates modern Pydantic validation patterns.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import ValidationError, field_validator, model_validator, Field
from pydantic_core import PydanticCustomError

from .settings import Settings, get_settings, reload_settings


class ConfigurationManager:
    """
    Configuration manager for handling different environment configurations
    and providing utilities for configuration management.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self._settings: Optional[Settings] = None

    def load_settings(self, env_file: Optional[str] = None) -> Settings:
        """
        Load settings from environment file or use default.

        Args:
            env_file: Path to .env file to load

        Returns:
            Settings instance
        """
        if env_file:
            self._settings = reload_settings(env_file)
        else:
            self._settings = get_settings()

        return self._settings

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration and return validation results.

        Returns:
            Dictionary containing validation results
        """
        if not self._settings:
            self._settings = get_settings()

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "available_providers": [],
            "missing_required": [],
        }

        try:
            # Validate required settings
            missing_settings = self._settings.validate_required_settings()
            if missing_settings:
                validation_results["valid"] = False
                validation_results["missing_required"] = missing_settings
                validation_results["errors"].extend(missing_settings)

            # Check available LLM providers
            available_providers = self._settings.get_available_llm_providers()
            validation_results["available_providers"] = available_providers

            if not available_providers:
                validation_results["warnings"].append("No LLM providers configured")

            # Validate log file path
            try:
                log_path = self._settings.get_log_file_path()
                validation_results["log_file"] = str(log_path)
            except Exception as e:
                validation_results["warnings"].append(
                    f"Log file validation failed: {e}"
                )

            # Environment-specific validations
            if self._settings.is_production():
                if self._settings.debug:
                    validation_results["warnings"].append(
                        "Debug mode enabled in production"
                    )

                if self._settings.logging.log_console:
                    validation_results["warnings"].append(
                        "Console logging enabled in production"
                    )

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Configuration validation failed: {e}")

        return validation_results

    def export_configuration(self, format: str = "json") -> str:
        """
        Export current configuration to specified format.

        Args:
            format: Output format ('json', 'env')

        Returns:
            Configuration as string
        """
        if not self._settings:
            self._settings = get_settings()

        if format.lower() == "json":
            # Export as JSON (excluding sensitive data)
            config_dict = {
                "environment": self._settings.environment,
                "debug": self._settings.debug,
                "logging": {
                    "log_level": self._settings.logging.log_level,
                    "log_file": self._settings.logging.log_file,
                    "log_console": self._settings.logging.log_console,
                    "log_file_enabled": self._settings.logging.log_file_enabled,
                    "log_rotation": self._settings.logging.log_rotation,
                    "log_retention": self._settings.logging.log_retention,
                },
                "llm": {
                    "default_model": self._settings.llm.default_model,
                    "default_temperature": self._settings.llm.default_temperature,
                    "default_max_tokens": self._settings.llm.default_max_tokens,
                    "default_timeout": self._settings.llm.default_timeout,
                    "openrouter_base_url": self._settings.llm.openrouter_base_url,
                    "available_providers": self._settings.get_available_llm_providers(),
                },
                "github": {
                    "github_api_base_url": self._settings.github.github_api_base_url,
                    "github_api_timeout": self._settings.github.github_api_timeout,
                    "github_rate_limit_warning": self._settings.github.github_rate_limit_warning,
                },
                "agent": {
                    "agent_name": self._settings.agent.agent_name,
                    "agent_version": self._settings.agent.agent_version,
                    "max_concurrent_sessions": self._settings.agent.max_concurrent_sessions,
                    "session_timeout": self._settings.agent.session_timeout,
                    "enable_debug_mode": self._settings.agent.enable_debug_mode,
                },
            }
            return json.dumps(config_dict, indent=2)

        elif format.lower() == "env":
            # Export as .env format (excluding sensitive data)
            env_lines = [
                "# DevOps Agent Configuration Export",
                "# Generated configuration (sensitive data excluded)",
                "",
                f"ENVIRONMENT={self._settings.environment}",
                f"DEBUG={str(self._settings.debug).lower()}",
                "",
                "# Logging Configuration",
                f"LOG_LEVEL={self._settings.logging.log_level}",
                f"LOG_FILE={self._settings.logging.log_file}",
                f"LOG_CONSOLE={str(self._settings.logging.log_console).lower()}",
                f"LOG_FILE_ENABLED={str(self._settings.logging.log_file_enabled).lower()}",
                f"LOG_ROTATION={self._settings.logging.log_rotation}",
                f"LOG_RETENTION={self._settings.logging.log_retention}",
                "",
                "# LLM Configuration",
                f"DEFAULT_MODEL={self._settings.llm.default_model}",
                f"DEFAULT_TEMPERATURE={self._settings.llm.default_temperature}",
                f"DEFAULT_MAX_TOKENS={self._settings.llm.default_max_tokens}",
                f"DEFAULT_TIMEOUT={self._settings.llm.default_timeout}",
                f"OPENROUTER_BASE_URL={self._settings.llm.openrouter_base_url}",
                "",
                "# GitHub Configuration",
                f"GITHUB_API_BASE_URL={self._settings.github.github_api_base_url}",
                f"GITHUB_API_TIMEOUT={self._settings.github.github_api_timeout}",
                f"GITHUB_RATE_LIMIT_WARNING={self._settings.github.github_rate_limit_warning}",
                "",
                "# Agent Configuration",
                f"AGENT_NAME={self._settings.agent.agent_name}",
                f"AGENT_VERSION={self._settings.agent.agent_version}",
                f"MAX_CONCURRENT_SESSIONS={self._settings.agent.max_concurrent_sessions}",
                f"SESSION_TIMEOUT={self._settings.agent.session_timeout}",
                f"ENABLE_DEBUG_MODE={str(self._settings.agent.enable_debug_mode).lower()}",
                "",
                "# Note: API keys and tokens are not exported for security reasons",
                "# Please set them manually in your .env file",
            ]
            return "\n".join(env_lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_environment_config(
        self, environment: str, output_file: Optional[str] = None
    ) -> str:
        """
        Create environment-specific configuration template.

        Args:
            environment: Environment name (development, staging, production)
            output_file: Optional output file path

        Returns:
            Configuration template as string
        """
        templates = {
            "development": {
                "ENVIRONMENT": "development",
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "LOG_CONSOLE": "true",
                "LOG_FILE_ENABLED": "true",
                "LOG_ROTATION": "1 MB",
                "LOG_RETENTION": "7 days",
                "DEFAULT_TEMPERATURE": "0.7",
                "ENABLE_DEBUG_MODE": "true",
            },
            "staging": {
                "ENVIRONMENT": "staging",
                "DEBUG": "false",
                "LOG_LEVEL": "INFO",
                "LOG_CONSOLE": "false",
                "LOG_FILE_ENABLED": "true",
                "LOG_ROTATION": "50 MB",
                "LOG_RETENTION": "30 days",
                "DEFAULT_TEMPERATURE": "0.4",
                "ENABLE_DEBUG_MODE": "false",
            },
            "production": {
                "ENVIRONMENT": "production",
                "DEBUG": "false",
                "LOG_LEVEL": "WARNING",
                "LOG_CONSOLE": "false",
                "LOG_FILE_ENABLED": "true",
                "LOG_ROTATION": "100 MB",
                "LOG_RETENTION": "90 days",
                "DEFAULT_TEMPERATURE": "0.3",
                "ENABLE_DEBUG_MODE": "false",
            },
        }

        if environment not in templates:
            raise ValueError(f"Unknown environment: {environment}")

        template = templates[environment]

        # Create .env template
        env_lines = [
            f"# DevOps Agent Configuration - {environment.upper()}",
            f"# Generated configuration template",
            "",
            "# Environment Configuration",
            f"ENVIRONMENT={template['ENVIRONMENT']}",
            f"DEBUG={template['DEBUG']}",
            "",
            "# Logging Configuration",
            f"LOG_LEVEL={template['LOG_LEVEL']}",
            f"LOG_FILE=logs/devops_agent_{environment}.log",
            f"LOG_CONSOLE={template['LOG_CONSOLE']}",
            f"LOG_FILE_ENABLED={template['LOG_FILE_ENABLED']}",
            f"LOG_ROTATION={template['LOG_ROTATION']}",
            f"LOG_RETENTION={template['LOG_RETENTION']}",
            "",
            "# LLM Configuration",
            "GOOGLE_API_KEY=your_google_api_key_here",
            "OPENROUTER_API_KEY=your_openrouter_api_key_here",
            "CLAUDE_API_KEY=your_claude_api_key_here",
            "CEREBRAS_API_KEY=your_cerebras_api_key_here",
            "MISTRAL_API_KEY=your_mistral_api_key_here",
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
            "DEFAULT_MODEL=gemini-2.5-flash",
            f"DEFAULT_TEMPERATURE={template['DEFAULT_TEMPERATURE']}",
            "DEFAULT_MAX_TOKENS=4096",
            "DEFAULT_TIMEOUT=30",
            "",
            "# GitHub Configuration",
            "GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here",
            "GITHUB_API_BASE_URL=https://api.github.com",
            "GITHUB_API_TIMEOUT=30",
            "GITHUB_RATE_LIMIT_WARNING=100",
            "",
            "# Agent Configuration",
            "AGENT_NAME=DevOps Incident Response Agent",
            "AGENT_VERSION=1.0.0",
            "MAX_CONCURRENT_SESSIONS=10",
            "SESSION_TIMEOUT=300",
            f"ENABLE_DEBUG_MODE={template['ENABLE_DEBUG_MODE']}",
            "",
            "# Security Note: Replace placeholder values with actual credentials",
        ]

        config_content = "\n".join(env_lines)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(config_content)

        return config_content


class EnhancedSettings(Settings):
    """
    Enhanced settings class with additional validation and utilities.
    Demonstrates modern Pydantic validation patterns.
    """

    # Additional fields with custom validation
    api_rate_limit: int = Field(
        default=100, ge=1, le=10000, description="API rate limit per minute"
    )

    cache_enabled: bool = Field(
        default=True, description="Enable caching for API responses"
    )

    cache_ttl: int = Field(default=300, ge=1, description="Cache TTL in seconds")

    # Custom field validators using modern patterns
    @field_validator("api_rate_limit", mode="after")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate API rate limit is reasonable."""
        if v > 1000:
            raise PydanticCustomError(
                "rate_limit_too_high",
                "Rate limit cannot exceed 1000 requests per minute",
                {"value": v, "max": 1000},
            )
        return v

    @field_validator("cache_ttl", mode="after")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate cache TTL is reasonable."""
        if v > 3600:  # 1 hour
            raise ValueError("Cache TTL cannot exceed 1 hour (3600 seconds)")
        return v

    # Model-level validation
    @model_validator(mode="after")
    def validate_production_settings(self) -> "EnhancedSettings":
        """Validate production-specific settings."""
        if self.is_production():
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")

            if self.logging.log_console:
                raise ValueError("Console logging should be disabled in production")

            if self.cache_ttl < 60:
                raise ValueError(
                    "Cache TTL should be at least 60 seconds in production"
                )

        return self

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "enabled": self.cache_enabled,
            "ttl": self.cache_ttl,
            "max_size": 1000,
        }

    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "requests_per_minute": self.api_rate_limit,
            "burst_limit": self.api_rate_limit * 2,
            "window_size": 60,
        }


# Utility functions
def validate_env_file(env_file_path: str) -> Dict[str, Any]:
    """
    Validate an environment file without loading sensitive data.

    Args:
        env_file_path: Path to .env file

    Returns:
        Validation results
    """
    results = {"valid": True, "errors": [], "warnings": [], "variables": {}}

    try:
        env_path = Path(env_file_path)
        if not env_path.exists():
            results["valid"] = False
            results["errors"].append(f"Environment file not found: {env_file_path}")
            return results

        with open(env_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Parse variable assignment
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")

                    results["variables"][key] = value

                    # Check for common issues
                    if key.endswith("_KEY") or key.endswith("_TOKEN"):
                        if value in ["your_key_here", "your_token_here", ""]:
                            results["warnings"].append(
                                f"Line {line_num}: {key} appears to be a placeholder"
                            )

                    if key == "ENVIRONMENT" and value not in [
                        "development",
                        "staging",
                        "production",
                    ]:
                        results["warnings"].append(
                            f"Line {line_num}: Unknown environment '{value}'"
                        )

                    if key == "LOG_LEVEL" and value.upper() not in [
                        "DEBUG",
                        "INFO",
                        "WARNING",
                        "ERROR",
                        "CRITICAL",
                    ]:
                        results["errors"].append(
                            f"Line {line_num}: Invalid log level '{value}'"
                        )

                else:
                    results["warnings"].append(
                        f"Line {line_num}: Invalid format (no '=' found)"
                    )

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Error reading environment file: {e}")

    return results


def create_config_summary() -> str:
    """
    Create a summary of the current configuration.

    Returns:
        Configuration summary as string
    """
    try:
        settings = get_settings()
        manager = ConfigurationManager()
        validation = manager.validate_configuration()

        summary_lines = [
            "DevOps Agent Configuration Summary",
            "=" * 40,
            f"Environment: {settings.environment}",
            f"Debug Mode: {settings.debug}",
            f"Valid Configuration: {validation['valid']}",
            "",
            "Logging Configuration:",
            f"  Level: {settings.logging.log_level}",
            f"  File: {settings.logging.log_file}",
            f"  Console: {settings.logging.log_console}",
            f"  File Enabled: {settings.logging.log_file_enabled}",
            "",
            "LLM Configuration:",
            f"  Available Providers: {', '.join(validation['available_providers']) if validation['available_providers'] else 'None'}",
            f"  Default Model: {settings.llm.default_model}",
            f"  Temperature: {settings.llm.default_temperature}",
            "",
            "GitHub Configuration:",
            f"  API Base URL: {settings.github.github_api_base_url}",
            f"  Timeout: {settings.github.github_api_timeout}s",
            "",
            "Agent Configuration:",
            f"  Name: {settings.agent.agent_name}",
            f"  Version: {settings.agent.agent_version}",
            f"  Max Sessions: {settings.agent.max_concurrent_sessions}",
            f"  Session Timeout: {settings.agent.session_timeout}s",
        ]

        if validation["errors"]:
            summary_lines.extend(
                [
                    "",
                    "Configuration Errors:",
                    *[f"  - {error}" for error in validation["errors"]],
                ]
            )

        if validation["warnings"]:
            summary_lines.extend(
                [
                    "",
                    "Configuration Warnings:",
                    *[f"  - {warning}" for warning in validation["warnings"]],
                ]
            )

        return "\n".join(summary_lines)

    except Exception as e:
        return f"Error generating configuration summary: {e}"


if __name__ == "__main__":
    # Example usage
    print("Configuration Manager Demo")
    print("=" * 30)

    # Create configuration manager
    manager = ConfigurationManager()

    # Load and validate settings
    settings = manager.load_settings()
    validation = manager.validate_configuration()

    print(f"Configuration valid: {validation['valid']}")
    print(f"Available providers: {validation['available_providers']}")

    if validation["errors"]:
        print("Errors:")
        for error in validation["errors"]:
            print(f"  - {error}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    # Export configuration
    print("\nConfiguration Export (JSON):")
    print(manager.export_configuration("json"))

    # Create environment config
    print("\nDevelopment Environment Template:")
    print(manager.create_environment_config("development"))


================================================
FILE: src/config/settings.py
================================================
import os
from pathlib import Path
from typing import Optional, List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    # Log level configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Log file configuration
    log_file: Optional[str] = Field(
        default="logs/devops_agent.log", description="Path to log file"
    )

    # Console logging
    log_console: bool = Field(default=True, description="Enable console logging")

    # File logging
    log_file_enabled: bool = Field(default=True, description="Enable file logging")

    # Log rotation and retention
    log_rotation: str = Field(
        default="10 MB",
        description="Log rotation size or time (e.g., '10 MB', '1 day', '00:00')",
    )

    log_retention: str = Field(
        default="30 days",
        description="Log retention period (e.g., '30 days', '1 week')",
    )

    @field_validator("log_level", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"log_level must be one of {allowed_levels}")
        return v.upper()


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    # API Keys
    google_api_key: Optional[str] = Field(
        default=None, description="Google API key for Gemini models"
    )

    openrouter_api_key: Optional[str] = Field(
        default=None, description="OpenRouter API key for OpenAI models"
    )

    claude_api_key: Optional[str] = Field(default=None, description="Claude API key")

    cerebras_api_key: Optional[str] = Field(
        default=None, description="Cerebras API key for Llama models"
    )

    mistral_api_key: Optional[str] = Field(default=None, description="Mistral API key")

    # OpenRouter configuration
    openrouter_base_url: Optional[str] = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter base URL"
    )

    default_model: str = Field(
        default="gemini-2.5-flash", description="Default model to use"
    )

    default_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Default temperature for model inference",
    )

    default_max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description="Default maximum tokens for model inference",
    )

    default_timeout: int = Field(
        default=30, ge=1, description="Default timeout for API calls in seconds"
    )

    @field_validator("default_temperature", mode="after")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class GitHubSettings(BaseSettings):
    """GitHub configuration settings."""

    github_personal_access_token: Optional[str] = Field(
        default=None, description="GitHub Personal Access Token"
    )

    github_api_base_url: str = Field(
        default="https://api.github.com", description="GitHub API base URL"
    )

    github_api_timeout: int = Field(
        default=30, ge=1, description="GitHub API timeout in seconds"
    )

    github_rate_limit_warning: int = Field(
        default=100, ge=1, description="GitHub API rate limit warning threshold"
    )


class AgentSettings(BaseSettings):
    """Agent-specific configuration settings."""

    agent_name: str = Field(
        default="DevOps Incident Response Agent", description="Name of the agent"
    )

    agent_version: str = Field(default="1.0.0", description="Agent version")

    max_concurrent_sessions: int = Field(
        default=10, ge=1, description="Maximum number of concurrent sessions"
    )

    session_timeout: int = Field(
        default=300, ge=60, description="Session timeout in seconds"
    )

    enable_debug_mode: bool = Field(
        default=False, description="Enable debug mode for development"
    )


class Settings(BaseSettings):
    """
    Main settings class that combines all configuration sections.

    This class uses Pydantic BaseSettings to automatically load configuration
    from environment variables and .env files with type validation.
    """

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",  # Ignore extra fields in .env file
        case_sensitive=False,  # Allow case-insensitive environment variables
    )

    # Configuration sections
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings, description="Logging configuration"
    )

    llm: LLMSettings = Field(
        default_factory=LLMSettings, description="LLM configuration"
    )

    github: GitHubSettings = Field(
        default_factory=GitHubSettings, description="GitHub configuration"
    )

    agent: AgentSettings = Field(
        default_factory=AgentSettings, description="Agent configuration"
    )

    # Environment and deployment settings
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )

    debug: bool = Field(default=False, description="Enable debug mode")

    # Validation methods
    @field_validator("environment", mode="after")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        allowed_envs = ["development", "staging", "production"]
        if v.lower() not in allowed_envs:
            raise ValueError(f"environment must be one of {allowed_envs}")
        return v.lower()

    def validate_required_settings(self) -> List[str]:
        """
        Validate that all required settings are present.

        Returns:
            List of missing required settings
        """
        missing_settings = []

        # Check for at least one LLM API key
        llm_keys = [
            self.llm.google_api_key,
            self.llm.openrouter_api_key,
            self.llm.claude_api_key,
            self.llm.cerebras_api_key,
            self.llm.mistral_api_key,
        ]

        if not any(key for key in llm_keys if key):
            missing_settings.append("At least one LLM API key is required")

        # Check GitHub token
        if not self.github.github_personal_access_token:
            missing_settings.append("GitHub Personal Access Token is required")

        return missing_settings

    def get_available_llm_providers(self) -> List[str]:
        """
        Get list of available LLM providers based on configured API keys.

        Returns:
            List of available provider names
        """
        providers = []

        if self.llm.google_api_key:
            providers.append("gemini")

        if self.llm.openrouter_api_key:
            providers.append("openai")

        if self.llm.claude_api_key:
            providers.append("claude")

        if self.llm.cerebras_api_key:
            providers.append("llama")

        if self.llm.mistral_api_key:
            providers.append("mistral")

        return providers

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def get_log_file_path(self) -> Path:
        """Get the log file path, creating directories if needed."""
        log_path = Path(self.logging.log_file or "logs/devops_agent.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance, creating it if it doesn't exist.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings(env_file: Optional[str] = None) -> Settings:
    """
    Reload settings from environment variables and .env file.

    Args:
        env_file: Optional path to .env file to load

    Returns:
        New Settings instance
    """
    global _settings

    if env_file:
        _settings = Settings(_env_file=env_file)
    else:
        _settings = Settings()

    return _settings


def validate_settings() -> None:
    """
    Validate all required settings are present.

    Raises:
        ValueError: If required settings are missing
    """
    settings = get_settings()
    missing_settings = settings.validate_required_settings()

    if missing_settings:
        raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")


# Convenience functions for accessing common settings
def get_logging_config() -> LoggingSettings:
    """Get logging configuration."""
    return get_settings().logging


def get_llm_config() -> LLMSettings:
    """Get LLM configuration."""
    return get_settings().llm


def get_github_config() -> GitHubSettings:
    """Get GitHub configuration."""
    return get_settings().github


def get_agent_config() -> AgentSettings:
    """Get agent configuration."""
    return get_settings().agent


# Environment variable mapping for backward compatibility
def get_env_var_mapping() -> dict:
    """
    Get mapping of settings to environment variables for backward compatibility.

    Returns:
        Dictionary mapping setting names to environment variable names
    """
    return {
        # Logging
        "LOG_LEVEL": "logging.log_level",
        "LOG_FILE": "logging.log_file",
        "LOG_CONSOLE": "logging.log_console",
        "LOG_FILE_ENABLED": "logging.log_file_enabled",
        "LOG_ROTATION": "logging.log_rotation",
        "LOG_RETENTION": "logging.log_retention",
        # LLM
        "GOOGLE_API_KEY": "llm.google_api_key",
        "OPENROUTER_API_KEY": "llm.openrouter_api_key",
        "CLAUDE_API_KEY": "llm.claude_api_key",
        "CEREBRAS_API_KEY": "llm.cerebras_api_key",
        "MISTRAL_API_KEY": "llm.mistral_api_key",
        "OPENROUTER_BASE_URL": "llm.openrouter_base_url",
        # GitHub
        "GITHUB_PERSONAL_ACCESS_TOKEN": "github.github_personal_access_token",
        "GITHUB_API_BASE_URL": "github.github_api_base_url",
        # Agent
        "ENVIRONMENT": "environment",
        "DEBUG": "debug",
    }


if __name__ == "__main__":
    # Example usage and validation
    try:
        settings = get_settings()
        print("✅ Settings loaded successfully")
        print(f"Environment: {settings.environment}")
        print(f"Available LLM providers: {settings.get_available_llm_providers()}")
        print(f"Log file: {settings.get_log_file_path()}")

        # Validate settings
        validate_settings()
        print("✅ All required settings are present")

    except Exception as e:
        print(f"❌ Settings validation failed: {e}")
        exit(1)


================================================
FILE: src/main/__init__.py
================================================
[Empty file]


================================================
FILE: src/main/agent_serve.py
================================================
[Empty file]


================================================
FILE: src/main/cli.py
================================================
[Empty file]


================================================
FILE: src/main/graph.py
================================================

import os
import sys
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional

# --- Core LangGraph and LangChain Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import BaseMessage, SystemMessage

# --- Add project root to path for local imports ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Local Project Imports ---
# Configuration and Utilities
from src.utils.logging_config import setup_logging_from_env, get_logger
from src.config.settings import get_settings, validate_settings

# LLM Factory
from llms.factory import LLMFactory, LLMArchitecture
from llms.base import ModelConfig, LLMProviderError

# Tool Factories
from tools.github.factory import GitHubToolset
from tools.kubernetes.factory import KubernetesToolset
from tools.prometheus.factory import PrometheusToolsetFactory
from tools.jenkins.factory import JenkinsToolFactory
from tools.Loki.loki_log_aggregation_tool import retrieve_job_logs
from tools.powershell.factory import create_powershell_tools
from tools.slack.factory import SlackToolsetFactory

# ==============================================================================
# 1. INITIAL SETUP & CONFIGURATION
# ==============================================================================
load_dotenv()
logger = get_logger(__name__)

try:
    validate_settings()
    settings = get_settings()
    logger.success("Configuration settings loaded and validated successfully.")
except ValueError as e:
    logger.critical(f"FATAL: Missing required configuration. {e}")
    sys.exit(1)

# ==============================================================================
# 2. DEFINING THE AGENT'S STATE
# ==============================================================================
class AgentState(TypedDict):
    """
    Defines the persistent state passed between graph nodes.
    The `add_messages` function ensures new messages are appended to the list.
    """
    messages: Annotated[List[BaseMessage], add_messages]

# ==============================================================================
# 3. LOADING TOOLS AND THE LLM
# ==============================================================================
slack_user_id = os.getenv("SLACK_USER_ID")
logger.info("Loading all available tools for the agent...")
all_tools = []
try:
    all_tools.extend(GitHubToolset(github_token=settings.github.github_personal_access_token).tools)
    all_tools.extend(KubernetesToolset.from_env().tools)
    all_tools.extend(PrometheusToolsetFactory.create_toolset_from_env())
    all_tools.extend(JenkinsToolFactory(
        base_url=os.getenv("JENKINS_URL"),
        username=os.getenv("JENKINS_USERNAME"),
        api_token=os.getenv("JENKINS_API_TOKEN")
    ).create_all_tools())
    all_tools.extend(create_powershell_tools())
    all_tools.extend(SlackToolsetFactory(slack_bot_token=os.getenv("SLACK_BOT_TOKEN")).tools)
    all_tools.append(retrieve_job_logs)
    
    logger.success(f"Successfully loaded {len(all_tools)} tools.")
except (ValueError, KeyError, TypeError) as e:
    logger.critical(f"Failed to initialize a toolset. Check .env file and settings. Error: {e}", exc_info=True)
    sys.exit(1)

logger.info("Initializing LLM...")
try:
    llm_config = ModelConfig(
        model_name=settings.llm.default_model or "gemini-1.5-flash-latest",
        api_key=settings.llm.google_api_key,
        temperature=settings.llm.default_temperature,
    )
    provider = LLMFactory.create_provider(LLMArchitecture.GEMINI, config=llm_config)
    model_with_tools = provider.get_model().bind_tools(all_tools)
    logger.success("LLM initialized and tools are bound.")
except LLMProviderError as e:
    logger.critical(f"Failed to create LLM provider: {e}")
    sys.exit(1)

# ==============================================================================
# 4. DEFINING THE GRAPH NODES
# ==============================================================================
def call_model_node(state: AgentState) -> dict:
    """Invokes the LLM to decide the next action or respond to the user."""
    logger.info("Node: call_model_node")
    response = model_with_tools.invoke(state["messages"])
    logger.debug(f"LLM Response: {response.content} | Tools: {response.tool_calls}")
    return {"messages": [response]}

# This node is responsible for executing the tools chosen by the agent
tool_node = ToolNode(all_tools)

# ==============================================================================
# 5. DEFINING THE GRAPH EDGES (LOGIC FLOW)
# ==============================================================================
def should_continue_edge(state: AgentState) -> str:
    """
    Routes the graph after the agent's decision.
    If the agent generated tool calls, route to the 'action' node.
    Otherwise, the conversation is over, so route to END.
    """
    logger.info("Edge: should_continue_edge")
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "action"

# ==============================================================================
# 6. ASSEMBLING AND COMPILING THE GRAPH (Simplified)
# ==============================================================================
logger.info("Assembling the agent graph...")
workflow = StateGraph(AgentState)

# Define the nodes: the agent and the tool executor
workflow.add_node("agent", call_model_node)
workflow.add_node("action", tool_node)

# Set the entry point to the agent
workflow.set_entry_point("agent")

# Define the main conditional edge.
# After the agent speaks, decide whether to call a tool or end.
workflow.add_conditional_edges(
    "agent",
    should_continue_edge,
    {
        "action": "action", # If tool calls exist, go to the action node
        "end": END,         # Otherwise, finish.
    },
)

# After any action is taken, the results are processed by looping back to the agent.
workflow.add_edge("action", "agent")

# Compile the graph
# Add a checkpointer for conversation memory
graph = workflow.compile(checkpointer=MemorySaver())
logger.success("DevOps Agent graph assembled and compiled successfully.")


def create_agent_prompt(slack_user_id: Optional[str] = None) -> str:
    """
    Creates the comprehensive system prompt for the AIDE agent.
    """
    slack_user_context = (
        f"The user you are currently interacting with has the Slack ID '{slack_user_id}'. "
        f"To mention them in a message, use the format '<@{slack_user_id}>'."
    ) if slack_user_id else (
        "No specific user is identified for this session. Send messages to public channels like '#devops-alerts'."
    )
    
    return f"""# AIDE - Autonomous Incident Diagnostic Engineer

You are **AIDE (Autonomous Incident Diagnostic Engineer)**, a highly advanced SRE agent. Your primary mission is to autonomously investigate, diagnose, and remediate production incidents with speed, precision, and safety. You operate as a trusted member of the engineering team.

**Your goal is to restore service functionality by systematically identifying the root cause of an issue and executing the most effective remediation plan.**

## Core Operating Principles
1.  **Systematic Investigation:** Always follow a logical, evidence-driven path.
2.  **Observe, Orient, Decide, Act (OODA Loop):** Observe data, form a hypothesis, decide on a tool, and then act.
3.  **Least-Impact First:** Always prefer read-only "investigation" tools before using any tools that change state.
4.  **Clarity and Justification:** Clearly state your hypothesis and why you are choosing a specific tool.
5.  **Assume Nothing, Verify Everything:** After an action, use observability tools to verify the outcome.
6.  **Recognize Limits:** If you are stuck or need a high-risk decision, state that you require human intervention.

## Incident Response Workflow
1.  **Initial Triage & Assessment:** Use Prometheus tools to understand the immediate impact.
2.  **Data Gathering & Correlation:** Form a hypothesis and use Kubernetes, Loki, Jenkins, or GitHub tools to investigate.
3.  **Remediation:** Based on a verified hypothesis, choose the most appropriate remediation tool.
4.  **Verification & Reporting:** Confirm system health with Prometheus/Kubernetes and report via Slack.

## Tooling Cheatsheet & Capabilities

### Observability & Monitoring

#### Prometheus (`prometheus_*`)
*Your primary toolset for observing system and application health. It's crucial to use the right tool for the right job.*

**IMPORTANT: There are two types of services you can monitor:**
1.  **Application Services** (like `fastapi-app`): These are identified by a `service_name` (which corresponds to the `job` label in Prometheus). They provide metrics about application behavior (e.g., HTTP requests).
2.  **System Infrastructure** (like `node-exporter`): These are identified by an `instance` label (e.g., `node-exporter:9100`). They provide metrics about the machine's resources (e.g., CPU, memory).

**Tool Cheatsheet:**

- **`check_service_health`**: **(Use this first for application issues)**. Checks the overall health of a specific application using its `service_name`. It gives you availability (is it up?), latency (is it slow?), and request rate (is it busy?).
  - **Example**: `service_name="fastapi-app"`

- **`analyze_errors`**: Analyzes HTTP error rates for an application using its `service_name`. Use this to find out if an application is failing and which URL paths are causing the most errors.
  - **Example**: `service_name="fastapi-app"`

- **`analyze_performance`**: **(Use this for infrastructure issues)**. Analyzes low-level system resource usage (CPU, memory, disk, network) for a specific machine or container that is running a `node-exporter`. **This tool requires an `instance` label**, not a service name.
  - **Example**: If you suspect the machine running a service is overloaded, you would find its `instance` label first, then use this tool. `instance="node-exporter:9100"`, `metric_type="cpu"`.
  - **DO NOT** use this tool with an application's `service_name`. It will find no data.

- **`investigate_alerts`**: Checks for currently firing alerts in Alertmanager. Use this to see what Prometheus currently thinks is an active, critical problem across the entire system.

- **`custom_prometheus_query`**: **(Expert Use Only)**. Executes a raw PromQL query. Only use this if the specialized tools above cannot provide the information you need. You should be able to solve most problems without this.

#### Grafana Loki (`retrieve_job_logs`)
- Retrieves logs for a specific `job_name`. **MANDATORY: NEVER use the `additional_filters` parameter.**

### CI/CD & Deployments (Jenkins) & Source Code (GitHub)
- Use Jenkins tools (`jenkins_job_status`, `jenkins_console_output`, etc.) to investigate deployments.
- Use GitHub tools (`list_commits`, `get_file_content`, etc.) to investigate code changes.

### Infrastructure & Runtime (Kubernetes, PowerShell)
- Use Kubernetes tools (`list_k8s_pods`, `get_k8s_pod_logs`, etc.) to inspect the live state of applications.
- Use PowerShell tools (`powershell_tofu_plan`, `powershell_git_status`) for IaC and local git checks.

### Communication & Collaboration (Slack)
- Use Slack tools (`slack_send_message`, `slack_create_channel`, etc.) to notify the team. {slack_user_context}

## ACTION Format
Structure your tool calls as a JSON object.
```json
{{
  "tool_name": "name_of_the_tool",
  "parameters": {{
    "param1": "value1"
  }}
}}
Remember: You are a systematic, methodical engineer. Always justify your actions.
"""
# ==============================================================================
# 7. EXAMPLE USAGE SCRIPT
# ==============================================================================
if __name__ == "__main__":
    try:
        system_prompt = create_agent_prompt(slack_user_id=slack_user_id)
        logger.info("System prompt loaded successfully.")
    except FileNotFoundError:
        logger.critical("FATAL: 'prompts/system_prompt.md' not found.")
        sys.exit(1)

    config = {"configurable": {"thread_id": "devops-thread-main"}}

    print("\n🤖 DevOps Agent is ready. Let's solve some incidents!")
    print("   Type 'exit' or 'quit' to end.")

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("🤖 Agent session ended. Goodbye!")
            break

        inputs = {"messages": [SystemMessage(content=system_prompt), {"role": "user", "content": user_input}]}

        try:
            # Invoke the graph. It will run until it hits an interrupt or finishes.
            result = graph.invoke(inputs, config)

            # Check if the graph was interrupted for human input.
            if "__interrupt__" in result:
                interrupt_info = result["__interrupt__"][0].value
                tool_calls_str = json.dumps(interrupt_info.get("tool_calls", []), indent=2)

                print(f"\n🚨 CONFIRM ACTION 🚨\nAgent wants to run the following tool(s):\n{tool_calls_str}")
                confirmation = input("\nDo you approve? (yes/no): ").strip().lower()

                if confirmation == "yes":
                    print("✅ Action approved. Resuming execution...")
                    # Resume with `True`. The graph will continue from the interrupt.
                    final_result = graph.invoke(Command(resume=True), config)
                else:
                    print("❌ Action denied. Informing agent to re-plan...")
                    # Resume with `False`. The graph will loop back to the agent.
                    final_result = graph.invoke(Command(resume=False), config)
                
                # Print the final output after resuming
                if final_result and not final_result.get("__interrupt__"):
                    final_message = final_result.get("messages", [])[-1]
                    if final_message.content:
                        print(f"🤖 Agent: {final_message.content}")

            # If there was no interrupt, print the final message directly.
            elif result:
                final_message = result.get("messages", [])[-1]
                if final_message.content:
                    print(f"🤖 Agent: {final_message.content}")

        except Exception as e:
            logger.exception("An error occurred during graph execution.")
            print(f"An error occurred: {e}")


================================================
FILE: src/main/nodes.py
================================================
[Empty file]


================================================
FILE: src/main/react_agent.py
================================================
#!/usr/bin/env python3
"""
DevOps Incident Response Agent - Main Script
This script initializes and runs the ReAct agent with all required tools.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from loguru import logger
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from src.utils.logging_config import setup_logging_from_env
from src.config.settings import Settings

from llms.factory import LLMFactory, LLMArchitecture
from llms.base import ModelConfig, LLMProviderError
from src.utils.logging_config import setup_logging_from_env, get_logger

from src.config.settings import (
    get_settings,
    validate_settings,
    get_llm_config,
    get_github_config,
)

from tools.github.factory import GitHubToolset
from tools.kubernetes.factory import KubernetesToolset
from tools.prometheus.factory import PrometheusToolBuilder, PrometheusToolsetFactory
from tools.jenkins.factory import JenkinsToolFactory
from tools.Loki.loki_log_aggregation_tool import retrieve_job_logs
from tools.powershell.factory import create_powershell_tools
from tools.slack.factory import SlackToolsetFactory


load_dotenv()

def validate_tools_structure(tools: List, source_name: str) -> None:
    """Validate that tools is a flat list of tool instances, not nested lists."""
    logger.info(f"Validating {source_name} tools structure...")
    
    if not isinstance(tools, list):
        raise ValueError(f"{source_name} tools must be a list, got {type(tools)}")
    
    for i, tool in enumerate(tools):
        if isinstance(tool, list):
            raise ValueError(
                f"{source_name} tool at index {i} is a list, not a tool instance. "
                f"This suggests improper tool collection - use extend() not append()."
            )
        
        if not hasattr(tool, "name"):
            logger.warning(f"{source_name} tool at index {i} missing 'name' attribute")
    
    logger.success(f"{source_name} tools structure validated: {len(tools)} tools")

def load_kubernetes_tools() -> List:
    """Load and return Kubernetes tools."""
    try:
        logger.info("Loading Kubernetes tools...")
        k8s_toolset = KubernetesToolset.from_env()
        k8s_tools = k8s_toolset.tools
        validate_tools_structure(k8s_tools, "Kubernetes")
        logger.success(f"Successfully loaded {len(k8s_tools)} Kubernetes tools")
        return k8s_tools
    except Exception as e:
        logger.error(f"Failed to load Kubernetes tools: {e}")
        return []


def load_prometheus_tools() -> List:
    """Load and return Prometheus tools."""
    try:
        logger.info("Loading Prometheus tools...")
        # The factory method returns a list of tools. Use one consistent name.
        prometheus_tools = PrometheusToolsetFactory.create_toolset_from_env()
        
        # Now validation will work correctly.
        validate_tools_structure(prometheus_tools, "Prometheus")
        logger.success(f"Successfully loaded {len(prometheus_tools)} Prometheus tools")
        return prometheus_tools
    except Exception as e:
        # If anything goes wrong during loading, log it and return an empty list.
        logger.error(f"Failed to load Prometheus tools: {e}")
        return []

def load_jenkins_tools() -> List:
    """Load and return Jenkins tools."""
    try:
        logger.info("Loading Jenkins tools...")
        JENKINS_URL = os.getenv("JENKINS_URL")
        JENKINS_USERNAME = os.getenv("JENKINS_USERNAME")
        JENKINS_API_TOKEN = os.getenv("JENKINS_API_TOKEN")
        
        logger.info("Initializing JenkinsToolFactory...")
        factory = JenkinsToolFactory(
            base_url=JENKINS_URL, username=JENKINS_USERNAME, api_token=JENKINS_API_TOKEN
        )
        jenkins_tools = factory.create_all_tools()
        print(type(jenkins_tools))
        validate_tools_structure(jenkins_tools, "Jenkins")
        logger.success(f"Successfully loaded {len(jenkins_tools)} Jenkins tools")
        return jenkins_tools
    except Exception as e:
        # If anything goes wrong during loading, log it and return an empty list.
        logger.error(f"Failed to load Jenkins tools: {e}")
        return []

def load_powershell_tools() -> List:
    """Load and return PowerShell tools."""
    try:
        logger.info("Loading PowerShell tools...")
        # Call the factory function to get the list of tools
        ps_tools = create_powershell_tools()
        
        # Reuse your validation logic
        validate_tools_structure(ps_tools, "PowerShell")
        
        logger.success(f"Successfully loaded {len(ps_tools)} PowerShell tools")
        return ps_tools
    except Exception as e:
        # If anything goes wrong, log it and return an empty list
        logger.error(f"Failed to load PowerShell tools: {e}")
        return []

def load_slack_tools() -> List:
    """Load and return Slack tools."""
    try:
        logger.info("Loading Slack tools...")
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        slack_toolset = SlackToolsetFactory(slack_bot_token=slack_token)
        slack_tools = slack_toolset.tools
        
        validate_tools_structure(slack_tools, "Slack")
        
        logger.success(f"Successfully loaded {len(slack_tools)} Slack tools")
        return slack_tools
    except Exception as e:
        # If anything goes wrong, log it and return an empty list
        logger.error(f"Failed to load Slack tools: {e}")
        return []

def create_agent_prompt(slack_user_id: Optional[str] = None) -> str:
    """Create the system prompt for the DevOps agent with optional Slack user ID."""
    return """
    # AIDE - Autonomous Incident Diagnostic Engineer

You are **AIDE (Autonomous Incident Diagnostic Engineer)**, a highly advanced SRE agent. Your primary mission is to autonomously investigate, diagnose, and remediate production incidents with speed, precision, and safety. You operate as a trusted member of the engineering team.

**Your goal is to restore service functionality by systematically identifying the root cause of an issue and executing the most effective remediation plan.**

## **CORE DIRECTIVE: LOG ANALYSIS PROTOCOL**

This is a strict and non-negotiable rule for using the `retrieve_job_logs` tool.

**RULE: YOU ARE FORBIDDEN FROM USING THE `additional_filters` PARAMETER.**

- To ensure you always have the complete and unbiased context, you **MUST NEVER** provide a value for the `additional_filters` parameter when calling the `retrieve_job_logs` tool.
- Always call the tool with only the `job_name` and, if necessary, the `hours_back` parameter.
- You must analyze the full, unfiltered log output returned by the tool to form your conclusions. Do not attempt to filter logs at the query level.

**Any deviation from this rule is a protocol violation.**

## Tooling Cheatsheet & Capabilities

### Grafana Loki (`retrieve_job_logs`)

- **Function**: Retrieves a complete, unfiltered set of logs for a specific `job_name` from Loki.
- **MANDATORY USAGE**:
    - **ALWAYS** call this tool by providing only the `job_name`.
    - The `additional_filters` parameter is **PROHIBITED** and **MUST** be omitted from all calls.
- **Parameters**:
    - `job_name`: (Required) The name of the service, e.g., `"fastapi-app"`.
    - `hours_back`: (Optional) How far back to search. Defaults to 1.
    - `limit`: (Optional) The maximum number of log lines to return.
    - `additional_filters`: **FORBIDDEN. DO NOT USE.**


### 1. Systematic Investigation
Always follow a logical, evidence-driven path. Do not jump to conclusions. Start broad, then narrow your focus.

### 2. Observe, Orient, Decide, Act (OODA Loop)
- **Observe**: Gather data about the current state of the system using observability tools
- **Orient**: Analyze the data, correlate it with recent changes, and form a hypothesis
- **Decide**: Propose a clear plan of action with justification
- **Act**: Execute the plan using your operational tools

### 3. Least-Impact First
Always prefer read-only operations (list, get, check, analyze) to gather evidence before performing any write operations (scale, delete, trigger, create, merge, apply).

### 4. Clarity and Justification
In your THOUGHT process, clearly state your hypothesis, the evidence supporting it, and the reason for choosing a specific tool or action. Explain **why** you are doing something, not just **what** you are doing.

### 5. Assume Nothing, Verify Everything
Do not assume a change has worked. After taking a remediation action, always use your observability tools to verify that the system has returned to a healthy state.

### 6. Recognize Limits
If you are stuck, if the issue is outside your tool's scope, or if a manual, high-risk decision is required, clearly state that you require human intervention and provide a summary of your findings.

## Incident Response Workflow

Follow this structured workflow when presented with an incident:

### 1. Initial Triage & Assessment (The "What")
- Start with the initial alert or problem description
- Use Prometheus tools to understand the immediate impact
- Determine: What services are unhealthy? What are the error rates? Are there active critical alerts?
- This establishes the blast radius and severity

### 2. Data Gathering & Correlation (The "Where" and "Why")
Form a hypothesis based on the initial triage:

- **If you suspect a service runtime issue** (e.g., crashing pods): Use Kubernetes and Loki tools to inspect the state of the affected services, check for crash loops, high restart counts, and find error patterns in aggregated logs
- **If you suspect a recent deployment is the cause**: Use Jenkins and GitHub tools to investigate the status of recent deployment pipelines and what code changes were included
- **If you suspect a performance issue** (e.g., resource exhaustion): Use Prometheus and Kubernetes tools to check CPU, memory, disk usage, and the health of underlying cluster nodes

### 3. Remediation (The "How")
- Based on your verified hypothesis, choose the most appropriate action
- State your intended action and the expected outcome before executing
- Examples: Use Jenkins to rollback a bad deployment, use Kubernetes to scale a service, or use PowerShell to apply an infrastructure fix via OpenTofu

### 4. Verification & Reporting
- After executing an action, return to your Prometheus, Kubernetes, and Loki tools
- Confirm that error rates have dropped, services are healthy, and pods are running correctly
- Provide a final summary: the initial problem, the root cause you identified, the action you took, and the final (healthy) state of the system

## Tooling Cheatsheet & Capabilities

This is your complete set of available tools. Use them to execute the workflow above.

### Observability & Monitoring

#### Prometheus (`prometheus_*`)
*Use for monitoring, performance analysis, and alert investigation.*

- `check_service_health`: Checks the health, availability, and response time of a specific service
- `analyze_performance`: Analyzes system performance metrics like 'cpu', 'memory', 'disk', and 'network'
- `analyze_errors`: Analyzes HTTP error rates (4xx, 5xx), identifies top error endpoints
- `investigate_alerts`: Views currently firing alerts, filters by name or severity
- `custom_prometheus_query`: Executes a custom PromQL query for advanced, specific investigations

#### Grafana Loki (`loki_*`)
#### Grafana Loki (`retrieve_job_logs`)
*Use for deep log analysis and searching across all services, especially for Docker containers.*

- **`retrieve_job_logs`**: Retrieves structured logs for a specific `job_name` from Loki.
  - **IMPORTANT**: This tool returns a **JSON object**, not plain text. You must inspect the JSON output to get the information you need.
  - **Key Parameters**:
    - `job_name`: (Required) The name of the service, e.g., `"fastapi-app"`.
    - `hours_back`: (Optional) How many hours of logs to search. Defaults to 1.
    - `additional_filters`: (Optional) A powerful way to narrow results. Use LogQL syntax like `|= "error"` to find errors or `|~ "DEBUG|INFO"` to match multiple patterns.
  - **How to Interpret the Output**:
    - After calling this tool, check the `status` key in the returned JSON. If it's `"error"`, read the `error` key to understand why it failed.
    - The actual log messages are in a list under the `logs` key.
    - The `log_count` key tells you how many logs were found. If it's 0, no matching logs were found.

### CI/CD & Deployments

#### Jenkins (`jenkins_*`)
*Use for managing builds, deployments, rollbacks, and checking CI/CD pipeline health.*

- `jenkins_trigger_build`: Triggers any Jenkins job, optionally with parameters
- `jenkins_job_status`: Gets the status of the last build for a specified job
- `jenkins_get_last_build_info`: Retrieves detailed information about the most recent build of a job
- `jenkins_build_info`: Gets detailed information for a specific build number of a job
- `jenkins_console_output`: Retrieves the console log for a specific build
- `jenkins_pipeline_monitor`: Monitors a specific build until it completes or times out
- `jenkins_health_check`: Performs a health check on a list of critical Jenkins pipelines
- `jenkins_emergency_deploy`: Triggers an emergency deployment pipeline with a specific branch or commit
- `jenkins_rollback`: **(Primary Remediation Tool)** Triggers a rollback pipeline to restore a previous version

### Source Code & Version Control

#### GitHub (`github_*`)
*Use for investigating code changes, managing pull requests, and interacting with repositories.*

**Repository Management:**
- `list_repositories`: Lists all accessible repositories
- `get_repository`: Gets details for a specific repository
- `list_branches`: Lists branches in a repository
- `list_commits`: **(Key Investigation Tool)** Lists recent commits to understand what has changed

**Content Management:**
- `get_file_content`: **(Key Investigation Tool)** Reads the content of a file to inspect a specific change
- `create_or_update_file`: Creates a new file or updates an existing one (for automated hotfixes)

**Pull Requests:**
- `list_pull_requests`: Lists pull requests
- `create_pull_request`: Creates a new pull request for a proposed fix
- `merge_pull_request`: Merges a pull request after approval

**Issues & Search:**
- `list_issues`, `create_issue`, `update_issue`: Manage repository issues
- `search_repositories`, `search_issues`: Search across GitHub

**Workflows & Actions:**
- `list_workflow_runs`, `trigger_workflow`, `cancel_workflow_run`: Manage GitHub Actions

### Infrastructure & Runtime

#### Kubernetes (`k8s_*`)
*Use for inspecting and managing runtime resources like pods, services, and deployments.*

**Workload Inspection:**
- `list_k8s_pods`: **(Primary Investigation Tool)** Lists pods, showing their status, IP, and namespace
- `get_k8s_pod_logs`: **(Primary Investigation Tool)** Gets logs from a specific pod to find runtime errors
- `list_k8s_deployments`: Lists deployments and shows their replica status
- `list_k8s_services`, `get_k8s_service`: Inspect service configurations and endpoints

**Workload Management:**
- `scale_k8s_deployment`: **(Remediation Tool)** Scales a deployment to a specific number of replicas
- `delete_k8s_pod`: **(Remediation Tool)** Deletes a pod, allowing the ReplicaSet to restart it

**Cluster & Config Inspection:**
- `list_k8s_nodes`: Checks the status, roles, and versions of cluster nodes
- `list_k8s_configmaps`, `list_k8s_secrets`: Lists configuration and secret objects

#### PowerShell (`powershell_*`)
*Use for interacting with local filesystems, Git, and Infrastructure-as-Code tools like OpenTofu.*

- `powershell_tofu_plan`: Runs tofu plan in a directory to preview infrastructure changes
- `powershell_tofu_apply`: **(Remediation Tool)** Runs tofu apply -auto-approve to apply infrastructure changes
- `powershell_git_status`: Runs git status to check the state of a local repository clone

```json
{
  "tool_name": "name_of_the_tool",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

---

**Remember**: You are a systematic, methodical engineer. Always justify your actions, verify your assumptions, and prioritize system stability above all else.
     """
    

def main():
    """Main function to initialize and run the DevOps agent."""
    try:
        # Setup logging
        setup_logging_from_env()
        logger.info("Starting DevOps Incident Response Agent")
        logger.info("=" * 50)
        
        logger.debug("Loading configuration settings")
        settings = get_settings()
        logger.info("Configuration settings loaded successfully")
        
        # Validate required settings
        logger.debug("Validating required settings")
        try:
            validate_settings()
            logger.info("All required settings are present")
        except ValueError as e:
            logger.critical(f"Settings validation failed: {e}")
            raise
        
        # Log configuration summary
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Available LLM providers: {settings.get_available_llm_providers()}")
        logger.info(f"Debug mode: {settings.debug}")
        
        # Get Slack user ID from environment
        slack_user_id = os.getenv("SLACK_USER_ID")
        if slack_user_id:
            logger.info(f"Slack user ID configured: {slack_user_id}")
        else:
            logger.warning("No Slack user ID found in environment variables")
        
        # Initialize LLM configuration
        logger.debug("Initializing LLM configuration")
        custom_config = ModelConfig(
            model_name="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            timeout=60,
            max_completion_tokens=4000,
        )
        
        try:
            gemini_25_provider = LLMFactory.create_provider(
                LLMArchitecture.GEMINI, config=custom_config
            )
            logger.info(f"LLM provider created successfully: {gemini_25_provider}")
        except LLMProviderError as e:
            logger.error(f"Failed to create LLM provider: {e}")
            raise
        
        # Get model instance
        logger.debug("Getting model instance")
        try:
            model = gemini_25_provider.get_model()
            logger.info("Model instance retrieved successfully")
        except Exception as e:
            logger.error(f"Failed to get model instance: {e}")
            raise
        
        # Load all tools
        logger.info("Loading tools...")
        
        logger.debug("Initializing GitHub toolset")
        try:
            github_config = get_github_config()
            github_toolset = GitHubToolset(
                github_token=github_config.github_personal_access_token
            )
            github_tools = github_toolset.tools
            logger.info(f"Successfully loaded {len(github_tools)} GitHub tools")
            logger.debug(f"Available GitHub tools: {[tool.name for tool in github_tools]}")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub toolset: {e}")
            raise
        
        k8s_tools = load_kubernetes_tools()
        prometheus_tools = load_prometheus_tools()
        jenkins_tools = load_jenkins_tools()
        powershell_tool = load_powershell_tools()
        slack_tools = load_slack_tools()
        
        logger.info("Loading Loki tool...")
        loki_tools = [retrieve_job_logs]  # <-- FIX: Wrap the tool in a list
        validate_tools_structure(loki_tools, "Loki") # Your validation will now pass
        logger.success("Successfully loaded 1 Loki tool")
        
        logger.info("Combining all tools...")
        all_tools = []
        all_tools.extend(github_tools)
        all_tools.extend(k8s_tools)
        all_tools.extend(prometheus_tools)
        all_tools.extend(jenkins_tools)
        all_tools.extend(powershell_tool)
        all_tools.extend(slack_tools)
        all_tools.extend(loki_tools)
        
        logger.success(f"Total tools loaded: {len(all_tools)}")
        
        tool_names = [getattr(tool, "name", "Unknown") for tool in all_tools]
        logger.info(f"Tool names: {tool_names}")
        
        # Create agent prompt with Slack user ID
        logger.info("Creating agent prompt...")
        prompt = create_agent_prompt(slack_user_id)
        logger.info("Agent prompt configured")
        
        # Create memory checkpointer
        checkpointer = MemorySaver()
        logger.info("Memory checkpointer created")
        
        # Create ReAct agent
        logger.info("Creating ReAct agent...")
        devops_agent = create_react_agent(
            model=model, tools=all_tools, prompt=prompt, checkpointer=checkpointer
        )
        logger.success("DevOps ReAct agent created successfully")
        
        # Configure agent
        config = {
            "configurable": {
                "thread_id": "devops-agent-main",
                "checkpoint_id": uuid.uuid4(),
                "recursion_limit": 100
            }
        }
        
        # Interactive loop
        logger.info("=" * 50)
        logger.info("DevOps Agent is ready! Type 'quit' or 'exit' to stop.")
        if slack_user_id:
            logger.info(f"Slack integration enabled for user: {slack_user_id}")
        logger.info("=" * 50)
        
        while True:
            try:
                user_input = input("\n🔧 DevOps Agent > ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    logger.info("Shutting down DevOps Agent...")
                    break
                
                if not user_input:
                    continue
                
                logger.info(f"Processing user input: {user_input}")
                
                # Get agent response
                response = devops_agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
                
                if response and "messages" in response:
                    last_message = response["messages"][-1]
                    if hasattr(last_message, "content"):
                        print(f"\n🤖 Agent: {last_message.content}")
                    else:
                        print(f"\n🤖 Agent: {last_message}")
                else:
                    print(f"\n🤖 Agent: {response}")
            
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"❌ Error: {e}")
    
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1
    
    finally:
        logger.info("DevOps Agent shutdown complete")
        logger.info("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


================================================
FILE: src/main/state.py
================================================
[Empty file]


================================================
FILE: src/utils/__init__.py
================================================
[Empty file]


================================================
FILE: src/utils/logging_config.py
================================================
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional

# Ensure src is in path to allow relative import
try:
    from src.config.settings import get_logging_config
except (ImportError, ModuleNotFoundError):
    # This fallback allows the module to be used even if settings can't be imported,
    # though configuration will rely solely on environment variables.
    print("Warning: Could not import get_logging_config. Falling back to env vars for logging.", file=sys.stderr)
    get_logging_config = None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    rotation: str = "10 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None,
) -> None:
    """
    Setup loguru logging configuration for the DevOps agent.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to logs/devops_agent.log)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        rotation: Log rotation size or time (e.g., "10 MB", "1 day", "00:00")
        retention: Log retention period (e.g., "30 days", "1 week")
        format_string: Custom format string for logs
    """

    # Remove default logger to avoid duplicate outputs
    logger.remove()

    # Default format if not provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Console logging
    if enable_console:
        logger.add(
            sys.stdout,
            format=format_string,
            level=log_level.upper(),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File logging
    if enable_file:
        if log_file is None:
            # Create logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            log_file = logs_dir / "devops_agent.log"
        else:
            # Ensure parent directory exists for custom log file paths
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_file),
            format=format_string,
            level=log_level.upper(),
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # Makes logging thread-safe
            serialize=False, # Set to True for JSON logs
        )

    logger.info(f"Logging initialized with level: {log_level}")
    if enable_file and log_file:
        logger.info(f"Log file: {Path(log_file).resolve()}")


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__ )

    Returns:
        Logger instance
    """
    return logger.bind(name=name) if name else logger


# Environment-based logging setup
def setup_logging_from_env():
    """
    Setup logging based on Pydantic settings configuration.

    This function uses the centralized settings management to configure logging
    with support for .env files and environment variables. It includes a fallback
    to environment variables if the settings module cannot be loaded.
    """
    try:
        # Use Pydantic settings if available
        if get_logging_config:
            logging_config = get_logging_config()
            setup_logging(
                log_level=logging_config.log_level,
                log_file=logging_config.log_file,
                enable_console=logging_config.log_console,
                enable_file=logging_config.log_file_enabled,
                rotation=logging_config.log_rotation,
                retention=logging_config.log_retention,
            )
            logger.debug("Logging configured successfully using Pydantic settings.")
            return
        else:
            raise ImportError("get_logging_config not available.")

    except Exception as e:
        # Fallback to environment variables if settings fail to load
        logger.warning(
            f"Could not configure logging from Pydantic settings ({e}). "
            "Falling back to environment variables."
        )

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_file = os.getenv("LOG_FILE")
        enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"
        enable_file = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
        rotation = os.getenv("LOG_ROTATION", "10 MB")
        retention = os.getenv("LOG_RETENTION", "30 days")

        setup_logging(
            log_level=log_level,
            log_file=log_file,
            enable_console=enable_console,
            enable_file=enable_file,
            rotation=rotation,
            retention=retention,
        )


================================================
FILE: src/utils/rate_limiter.py
================================================
from tenacity import (
    retry,
    retry_if_exception,
    wait_random_exponential,
    stop_after_delay,
    stop_after_attempt,
    RetryError,
    before_sleep_log,
)
import logging


logging.basicConfig(stream=logging.StreamHandler(), level=logging.INFO)
TENACITY_LOGGER = logging.getLogger(__name__)


retry_with_logging = retry(
    # Wait for an exponentially increasing random time between retries,
    # starting from 1 second, up to a maximum of 60 seconds.
    wait=wait_random_exponential(multiplier=1, max=60),
    # Stop retrying after 3 attempts.
    stop=stop_after_attempt(3),
    # Log before sleeping between retries.
    before_sleep=before_sleep_log(TENACITY_LOGGER, logging.WARNING),
    # You can specify which exceptions should trigger a retry.
    # By default, it retries on any Exception.
    # retry=retry_if_exception_type(IOError)
)


================================================
FILE: src/utils/retry.py
================================================
#!/usr/bin/env python3
"""
Tenacity-based retry logic for LLM API calls, specifically tailored for Google Gemini.

This module provides a decorator to handle transient errors commonly encountered when
interacting with LLM APIs, such as:
- Rate limit errors (HTTP 429)
- Server-side errors (HTTP 5xx)
- Timeout errors

It uses an exponential backoff with jitter to gracefully retry failed requests.
"""

import logging
from tenacity import (
    retry,
    retry_if_exception,
    wait_random_exponential,
    stop_after_delay,
    before_sleep_log,
)

# It's a good practice to anticipate potential exceptions from the underlying SDK.
# If you have google-api-core installed, you can be more specific.
try:
    from google.api_core import exceptions as google_exceptions
    # Define a tuple of specific, retryable Google API exceptions
    RETRYABLE_GOOGLE_EXCEPTIONS = (
        google_exceptions.ResourceExhausted,  # HTTP 429 Rate limit
        google_exceptions.ServiceUnavailable, # HTTP 503 Server temporarily unavailable
        google_exceptions.InternalServerError, # HTTP 500 Internal server error
        google_exceptions.Aborted,            # Often indicates a concurrency issue
        google_exceptions.DeadlineExceeded,   # Timeout on the server side
    )
    # Define a tuple of exceptions that should NOT be retried
    NON_RETRYABLE_GOOGLE_EXCEPTIONS = (
        google_exceptions.PermissionDenied,   # HTTP 403, API key issue
        google_exceptions.NotFound,           # HTTP 404
        google_exceptions.InvalidArgument,    # HTTP 400, bad request
        google_exceptions.Unauthenticated,    # HTTP 401, bad API key
    )
except ImportError:
    # If the library isn't installed, fall back to a safe default
    RETRYABLE_GOOGLE_EXCEPTIONS = ()
    NON_RETRYABLE_GOOGLE_EXCEPTIONS = ()
    logging.warning("`google-api-core` not found. Specific Google exception handling will be disabled.")


# Setup a logger for this module to see retry attempts
logger = logging.getLogger(__name__)

def is_retryable_exception(e: BaseException) -> bool:
    """
    Determines if an exception is worth retrying.

    This function checks for:
    1. Standard Python TimeoutError.
    2. Specific retryable exceptions from the Google Cloud SDK.
    3. Avoids retrying non-recoverable client errors (like authentication).
    4. As a fallback, checks exception messages for common rate limit text.

    Args:
        e: The exception instance to check.

    Returns:
        True if the exception is retryable, False otherwise.
    """
    # Do not retry non-recoverable Google API errors
    if isinstance(e, NON_RETRYABLE_GOOGLE_EXCEPTIONS):
        logger.warning(f"Encountered non-retryable Google API error: {e}")
        return False

    # Retry known transient Google API errors and standard timeouts
    if isinstance(e, RETRYABLE_GOOGLE_EXCEPTIONS) or isinstance(e, TimeoutError):
        logger.debug(f"Identified retryable exception: {type(e).__name__}")
        return True

    # Fallback for generic exceptions: check for rate limit text in the message
    error_message = str(e).lower()
    if "rate limit" in error_message or "resource has been exhausted" in error_message:
        logger.debug("Identified retryable exception based on error message content.")
        return True

    logger.error(f"Encountered non-retryable exception: {e}")
    return False


# Define the main retry decorator using our custom logic
gemini_llm_retry = retry(
    # Use our custom function to decide whether to retry
    retry=retry_if_exception(is_retryable_exception),

    # Wait for a random exponential time between retries, starting from 1s up to 60s.
    # This adds jitter and prevents thundering herd issues.
    wait=wait_random_exponential(multiplier=1, max=60),

    # Stop retrying after a total of 5 minutes (300 seconds)
    stop=stop_after_delay(300),

    # Log a warning before each retry attempt
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

  and I am usingg a mock microservices app with chaos engineering to intentionally introduce latency and error rates : # app/main.py

# Standard Library Imports
import asyncio
import logging
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List

# Third-Party Imports
import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (Boolean, Column, create_engine, DateTime, Float, Integer,
                        String)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
# Imports for Prometheus
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge


# --- Standard Logger Configuration ---
# Create a formatter
log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handler for console output (for `docker-compose logs`)
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Handler for file output (for Promtail to scrape)
file_handler = RotatingFileHandler(
    "/var/log/app.log", maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# --- <<< NEW: INTEGRATE UVICORN LOGS >>> ---
# Get the Uvicorn access logger and add our file handler to it.
# This will make uvicorn's access logs go to /var/log/app.log as well.
logging.getLogger("uvicorn.access").addHandler(file_handler)

# --- Database Setup ---
SQLITE_DATABASE_URL = "sqlite:///./chaos_backend.db"
engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ChaosEvent(Base):
    __tablename__ = "chaos_events"
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float)
    impact_level = Column(String)

class APIMetrics(Base):
    __tablename__ = "api_metrics"
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    method = Column(String)
    response_time = Column(Float)
    status_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)


# --- Pydantic Models ---
class UserCreate(BaseModel):
    username: str
    email: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool

class ChaosEventResponse(BaseModel):
    id: int
    event_type: str
    description: str
    timestamp: datetime
    duration: float
    impact_level: str

class MetricsResponse(BaseModel):
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime


# --- Custom Prometheus Metrics ---
CHAOS_EVENTS_TOTAL = Counter(
    "chaos_events_total",
    "Total number of chaos events triggered",
    ["event_type"]
)
SYSTEM_CPU_USAGE_PERCENT = Gauge(
    "system_cpu_usage_percent",
    "Current system-wide CPU utilization as a percentage"
)
SYSTEM_MEMORY_USAGE_PERCENT = Gauge(
    "system_memory_usage_percent",
    "Current system-wide memory utilization as a percentage"
)


# --- Chaos Engineering Core ---
class ChaosMonkey:
    def __init__(self):
        self.chaos_enabled = True
        self.chaos_probability = 0.1
        self.active_chaos = {}

    def should_cause_chaos(self) -> bool:
        return self.chaos_enabled and random.random() < self.chaos_probability

    async def async_latency_chaos(self, min_delay: float = 1.0, max_delay: float = 5.0):
        if self.should_cause_chaos():
            delay = random.uniform(min_delay, max_delay)
            logging.warning(f"Chaos Monkey: Introducing {delay:.2f}s async latency")
            CHAOS_EVENTS_TOTAL.labels(event_type="latency").inc()
            await asyncio.sleep(delay)
            return delay
        return 0

    def memory_chaos(self):
        if self.should_cause_chaos():
            logging.warning("Chaos Monkey: Consuming memory")
            CHAOS_EVENTS_TOTAL.labels(event_type="memory").inc()
            memory_hog = bytearray(100 * 1024 * 1024)
            time.sleep(2)
            del memory_hog
            return True
        return False

    def cpu_chaos(self, duration: float = 2.0):
        if self.should_cause_chaos():
            logging.warning(f"Chaos Monkey: Creating CPU load for {duration}s")
            CHAOS_EVENTS_TOTAL.labels(event_type="cpu_load").inc()
            end_time = time.time() + duration
            while time.time() < end_time:
                pass
            return duration
        return 0

    def exception_chaos(self):
        if self.should_cause_chaos():
            CHAOS_EVENTS_TOTAL.labels(event_type="exception").inc()
            exceptions = [
                HTTPException(status_code=500, detail="Chaos Monkey: Random server error"),
                HTTPException(status_code=503, detail="Chaos Monkey: Service temporarily unavailable"),
                HTTPException(status_code=429, detail="Chaos Monkey: Rate limit exceeded"),
            ]
            raise random.choice(exceptions)

    def database_chaos(self):
        if self.should_cause_chaos():
            logging.warning("Chaos Monkey: Simulating database issues")
            CHAOS_EVENTS_TOTAL.labels(event_type="database_issue").inc()
            time.sleep(random.uniform(0.5, 2.0))
            if random.random() < 0.3:
                raise HTTPException(status_code=503, detail="Chaos Monkey: Database connection failed")

chaos_monkey = ChaosMonkey()


# --- System Utilities ---
class SystemMetrics:
    @staticmethod
    def get_system_stats():
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.utcnow()
        }

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Application startup sequence initiated...")
    
    # Start background tasks
    metrics_task = asyncio.create_task(update_system_metrics())
    chaos_task = asyncio.create_task(background_chaos_events())
    
    yield
    
    # Shutdown
    logging.info("Shutting down application. Cleaning up tasks.")
    metrics_task.cancel()
    chaos_task.cancel()


# --- Main App Creation ---
app = FastAPI(
    title="Chaos Engineering Backend API",
    description="A FastAPI backend with comprehensive chaos engineering features",
    version="1.0.0",
    lifespan=lifespan 
)

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Background Tasks ---
async def update_system_metrics():
    """Periodically update the system metrics gauges."""
    while True:
        try:
            stats = SystemMetrics.get_system_stats()
            SYSTEM_CPU_USAGE_PERCENT.set(stats["cpu_percent"])
            SYSTEM_MEMORY_USAGE_PERCENT.set(stats["memory_percent"])
        except Exception as e:
            logging.error(f"Failed to update system metrics: {e}")
        await asyncio.sleep(5)

async def background_chaos_events():
    """Run periodic chaos events in the background"""
    while True:
        try:
            await asyncio.sleep(random.uniform(30, 120))

            if not chaos_monkey.chaos_enabled:
                continue

            event_types = ["cpu_spike", "memory_spike"]
            event_type = random.choice(event_types)

            db = SessionLocal()
            try:
                duration = 0
                if event_type == "cpu_spike":
                    duration = chaos_monkey.cpu_chaos(duration=3.0)
                    if duration > 0:
                        chaos_event = ChaosEvent(
                            event_type="background_cpu_spike",
                            description=f"Background CPU spike for {duration}s",
                            duration=duration, impact_level="medium"
                        )
                        db.add(chaos_event)
                        db.commit()
                elif event_type == "memory_spike":
                    if chaos_monkey.memory_chaos():
                        chaos_event = ChaosEvent(
                            event_type="background_memory_spike",
                            description="Background memory consumption",
                            duration=2.0, impact_level="low"
                        )
                        db.add(chaos_event)
                        db.commit()
            finally:
                db.close()
        except Exception as e:
            logging.error(f"Background chaos event failed: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    await chaos_monkey.async_latency_chaos(0.1, 1.0)
    chaos_monkey.exception_chaos()
    return {"message": "Welcome to Chaos Engineering Backend API", "version": "1.0.0", "chaos_enabled": chaos_monkey.chaos_enabled}

@app.get("/health")
async def health_check():
    original_prob = chaos_monkey.chaos_probability
    chaos_monkey.chaos_probability = 0.05
    try:
        chaos_monkey.exception_chaos()
        system_stats = SystemMetrics.get_system_stats()
        return {"status": "healthy", "timestamp": datetime.utcnow(), "system_stats": system_stats}
    finally:
        chaos_monkey.chaos_probability = original_prob

# --- User Management Endpoints ---
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.2, 2.0)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()
    chaos_monkey.memory_chaos()

    existing_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="User with that username or email already exists")

    db_user = User(username=user.username, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all users with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.1, 1.5)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()

    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get a specific user by ID with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.1, 1.0)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user by ID with chaos engineering."""
    await chaos_monkey.async_latency_chaos(0.2, 2.0)
    chaos_monkey.database_chaos()
    chaos_monkey.exception_chaos()

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

# --- Load and Stress Testing Endpoints ---
@app.get("/load-test")
async def load_test_endpoint():
    """Endpoint specifically for load testing with simulated processing."""
    processing_type = random.choice(["light", "medium", "heavy"])

    if processing_type == "light":
        await asyncio.sleep(random.uniform(0.1, 0.5))
    elif processing_type == "medium":
        await asyncio.sleep(random.uniform(0.5, 1.5))
        chaos_monkey.cpu_chaos(duration=1.0)
    else:  # heavy
        await asyncio.sleep(random.uniform(1.0, 3.0))
        chaos_monkey.cpu_chaos(duration=2.0)
        chaos_monkey.memory_chaos()

    return {
        "processing_type": processing_type,
        "timestamp": datetime.utcnow(),
        "random_data": [random.randint(1, 1000) for _ in range(100)]
    }

@app.post("/stress/cpu/{duration}")
async def stress_cpu(duration: float):
    """Manually stress CPU for a specified duration."""
    if not 0 < duration <= 30:
        raise HTTPException(status_code=400, detail="Duration must be between 0 and 30 seconds.")

    start_time = time.time()
    count = 0
    while time.time() < start_time + duration:
        count += 1

    return {
        "message": f"CPU stress test completed",
        "duration_seconds": duration,
        "iterations": count,
        "timestamp": datetime.utcnow()
    }

@app.post("/stress/memory/{megabytes}")
async def stress_memory(megabytes: int):
    """Manually stress memory by allocating a specified amount in MB."""
    if not 0 < megabytes <= 500:
        raise HTTPException(status_code=400, detail="Memory allocation must be between 0 and 500MB.")

    logging.info(f"Allocating {megabytes}MB of memory for a stress test...")
    memory_hog = bytearray(megabytes * 1024 * 1024)
    await asyncio.sleep(5)  # Hold memory for 5 seconds
    del memory_hog
    logging.info("Memory released.")

    return {
        "message": "Memory stress test completed",
        "allocated_mb": megabytes,
        "duration_seconds": 5,
        "timestamp": datetime.utcnow()
    }

# --- Chaos Control Endpoints ---
@app.post("/chaos/enable")
async def enable_chaos():
    chaos_monkey.chaos_enabled = True
    logging.info("Chaos engineering has been ENABLED.")
    return {"message": "Chaos engineering enabled", "enabled": True}

@app.post("/chaos/disable")
async def disable_chaos():
    chaos_monkey.chaos_enabled = False
    logging.info("Chaos engineering has been DISABLED.")
    return {"message": "Chaos engineering disabled", "enabled": False}

@app.post("/chaos/probability/{probability}")
async def set_chaos_probability(probability: float):
    if not 0.0 <= probability <= 1.0:
        raise HTTPException(status_code=400, detail="Probability must be between 0.0 and 1.0")

    chaos_monkey.chaos_probability = probability
    logging.info(f"Chaos probability set to {probability}")
    return {"message": f"Chaos probability set to {probability}", "probability": probability}

@app.get("/chaos/status")
async def chaos_status():
    return {
        "enabled": chaos_monkey.chaos_enabled,
        "probability": chaos_monkey.chaos_probability,
        "active_chaos": chaos_monkey.active_chaos
    }

# --- Metrics and Monitoring Endpoints ---
@app.get("/metrics/api", response_model=List[MetricsResponse])
async def get_api_metrics(limit: int = 100, db: Session = Depends(get_db)):
    """Get API performance metrics."""
    metrics = db.query(APIMetrics).order_by(APIMetrics.timestamp.desc()).limit(limit).all()
    return metrics

@app.get("/metrics/chaos", response_model=List[ChaosEventResponse])
async def get_chaos_events(limit: int = 100, db: Session = Depends(get_db)):
    """Get recorded chaos engineering events."""
    events = db.query(ChaosEvent).order_by(ChaosEvent.timestamp.desc()).limit(limit).all()
    return events

@app.get("/metrics/system")
async def get_system_metrics():
    """Get current system metrics."""
    return SystemMetrics.get_system_stats()

# --- WebSocket Endpoint ---
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            stats = SystemMetrics.get_system_stats()
            data_to_send = {
                "type": "system_metrics",
                "data": {
                    "cpu_percent": stats["cpu_percent"],
                    "memory_percent": stats["memory_percent"],
                    "disk_percent": stats["disk_percent"],
                    "timestamp": stats["timestamp"].isoformat(),
                    "chaos_enabled": chaos_monkey.chaos_enabled,
                    "chaos_probability": chaos_monkey.chaos_probability
                }
            }
            await websocket.send_json(data_to_send)
            await asyncio.sleep(5)
    except Exception as e:
        logging.warning(f"WebSocket disconnected: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) with docker-compose.yaml file as 

services:
  fastapi-app:
    build: ./app
    container_name: fastapi-app
    ports:
      - "8000:8000"
    networks:
      - monitoring
    restart: unless-stopped
    volumes: 
      - log_data:/var/log
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s
    # <<< REMOVED >>> The loki logging driver is no longer needed
    # logging:
    #   driver: loki
    #   options:
    #     loki-url: "http://loki:3100/loki/api/v1/push"
    #     loki-batch-size: "400"

  # <<< NEW >>> Add the Promtail service
  promtail:
    image: grafana/promtail:2.9.2
    container_name: promtail
    user: "0:0"
    volumes:
      - ./promtail:/etc/promtail_config
      - log_data:/var/log
    # <<< CORRECTED >>> THIS COMMAND NOW LOOKS FOR YOUR 'config.yaml' FILE
    command: -config.file=/etc/promtail_config/config.yaml 
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - loki

  loki:
    image: grafana/loki:2.9.2
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki:/etc/loki_config
      - loki_data:/data/loki
    command: -config.file=/etc/loki_config/config.yaml
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:3100/ready || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:v2.47.2
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus_config
      - prometheus_data:/prometheus
    command: --config.file=/etc/prometheus_config/prometheus.yaml
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:9090/-/ready || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5

  grafana:
    image: grafana/grafana:10.2.0
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - loki
      - prometheus
    healthcheck:
      test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:3000/api/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5

networks:
  monitoring:
    driver: bridge

volumes:
  # <<< MODIFIED >>> Add the new volume for logs
  log_data:
  loki_data:
  prometheus_data:
  grafana_data: with all the config files written for the logging and monitoring tools , I have created a Grafana_Loki_test\k8s\development.yaml and Grafana_Loki_test\k8s\service.yaml which are empty ,and I want to fill in them 