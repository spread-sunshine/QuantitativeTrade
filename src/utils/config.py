# Configuration management utility
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from config.settings import PROJECT_ROOT


class Config:
    """Configuration management class."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_file: Path to configuration file. If None, uses default.
        """
        if config_file is None:
            config_file = PROJECT_ROOT / "config" / "config.yaml"

        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Configuration dictionary.
        """
        if not self.config_file.exists():
            # Create default configuration
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                if self.config_file.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif self.config_file.suffix == ".json":
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_file.suffix}")
        except Exception as e:
            print(f"Error loading config file {self.config_file}: {e}")
            return self._get_default_config()

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file.

        Args:
            config: Configuration dictionary.
        """
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                if self.config_file.suffix in [".yaml", ".yml"]:
                    yaml.dump(config, f, default_flow_style=False)
                elif self.config_file.suffix == ".json":
                    json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config file {self.config_file}: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary.
        """
        return {
            "data_sources": {
                "yahoo": {
                    "enabled": True,
                    "rate_limit": 0.1,  # seconds between requests
                    "cache_enabled": True,
                },
                "alpha_vantage": {
                    "enabled": False,
                    "api_key": "",
                    "rate_limit": 12,  # requests per minute (free tier)
                },
            },
            "database": {
                "url": "sqlite:///data/market_data.db",
                "echo": False,
                "pool_size": 5,
                "max_overflow": 10,
            },
            "backtesting": {
                "initial_capital": 100000.0,
                "commission": 0.001,
                "slippage": 0.0001,
                "default_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"],
            },
            "risk_management": {
                "max_position_size": 0.1,
                "max_drawdown": 0.2,
                "stop_loss": 0.05,
                "take_profit": 0.1,
                "max_leverage": 1.0,
            },
            "trading": {
                "default_timeframe": "1d",
                "allowed_timeframes": ["1d", "1h", "4h", "1w"],
                "session_start": "09:30",
                "session_end": "16:00",
            },
            "logging": {
                "level": "INFO",
                "file": "logs/quant_trading.log",
                "max_size_mb": 10,
                "backup_count": 5,
            },
            "visualization": {
                "theme": "dark",
                "save_format": "png",
                "dpi": 100,
                "figure_size": [12, 8],
            },
            "performance": {
                "use_cache": True,
                "cache_expiry_days": 7,
                "parallel_processing": False,
                "max_workers": 4,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (dot notation supported).
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (dot notation supported).
            value: Value to set.
        """
        keys = key.split(".")
        config = self.config

        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

        # Save to file
        self._save_config(self.config)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values.

        Args:
            updates: Dictionary of updates.
        """
        for key, value in updates.items():
            self.set(key, value)

    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()

    def save(self) -> None:
        """Save current configuration to file."""
        self._save_config(self.config)

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary.
        """
        return self.config.copy()

    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """Get configuration for a data source.

        Args:
            source: Data source name.

        Returns:
            Data source configuration.
        """
        return self.get(f"data_sources.{source}", {})

    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration.

        Returns:
            Backtesting configuration.
        """
        return self.get("backtesting", {})

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration.

        Returns:
            Risk management configuration.
        """
        return self.get("risk_management", {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration.

        Returns:
            Trading configuration.
        """
        return self.get("trading", {})

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            # Validate data sources
            data_sources = self.get("data_sources", {})
            for source, config in data_sources.items():
                if config.get("enabled", False):
                    if source == "alpha_vantage" and not config.get("api_key"):
                        print(f"Warning: Alpha Vantage enabled but no API key provided")

            # Validate risk management
            risk_config = self.get_risk_config()
            if risk_config.get("max_position_size", 0) <= 0:
                print("Warning: max_position_size should be positive")
            if risk_config.get("max_leverage", 0) < 0:
                print("Warning: max_leverage cannot be negative")

            # Validate backtesting
            backtesting_config = self.get_backtesting_config()
            if backtesting_config.get("initial_capital", 0) <= 0:
                print("Warning: initial_capital should be positive")

            return True
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False

    def create_example_config(self, output_file: Path) -> None:
        """Create an example configuration file.

        Args:
            output_file: Path to output file.
        """
        example_config = self._get_default_config()
        
        # Add comments/descriptions
        example_config["_comment"] = "Quantitative Trading System Configuration"
        
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(example_config, f, default_flow_style=False)
        
        print(f"Example configuration created at: {output_file}")