# 配置管理工具
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from config.settings import PROJECT_ROOT


class Config:
    """配置管理类。"""

    def __init__(self, config_file: Optional[Path] = None):
        """初始化配置。

        Args:
            config_file: 配置文件路径。如果为 None，则使用默认配置。
        """
        if config_file is None:
            config_file = PROJECT_ROOT / "config" / "config.yaml"

        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """从文件加载配置。

        Returns:
            配置字典。
        """
        if not self.config_file.exists():
            # 创建默认配置
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
        """将配置保存到文件。

        Args:
            config: 配置字典。
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
        """获取默认配置。

        Returns:
            默认配置字典。
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
        """获取配置值。

        Args:
            key: 配置键（支持点号表示法）。
            default: 如果键未找到时的默认值。

        Returns:
            配置值。
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
        """设置配置值。

        Args:
            key: 配置键（支持点号表示法）。
            value: 要设置的值。
        """
        keys = key.split(".")
        config = self.config

        # 导航到嵌套字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value

        # Save to file
        self._save_config(self.config)

    def update(self, updates: Dict[str, Any]) -> None:
        """更新多个配置值。

        Args:
            updates: 更新字典。
        """
        for key, value in updates.items():
            self.set(key, value)

    def reload(self) -> None:
        """从文件重新加载配置。"""
        self.config = self._load_config()

    def save(self) -> None:
        """将当前配置保存到文件。"""
        self._save_config(self.config)

    def to_dict(self) -> Dict[str, Any]:
        """获取配置字典。

        Returns:
            配置字典。
        """
        return self.config.copy()

    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """获取数据源的配置。

        Args:
            source: 数据源名称。

        Returns:
            数据源配置。
        """
        return self.get(f"data_sources.{source}", {})

    def get_backtesting_config(self) -> Dict[str, Any]:
        """获取回测配置。

        Returns:
            回测配置。
        """
        return self.get("backtesting", {})

    def get_risk_config(self) -> Dict[str, Any]:
        """获取风险管理配置。

        Returns:
            风险管理配置。
        """
        return self.get("risk_management", {})

    def get_trading_config(self) -> Dict[str, Any]:
        """获取交易配置。

        Returns:
            交易配置。
        """
        return self.get("trading", {})

    def validate(self) -> bool:
        """验证配置。

        Returns:
            如果配置有效返回 True，否则返回 False。
        """
        try:
            # 验证数据源
            data_sources = self.get("data_sources", {})
            for source, config in data_sources.items():
                if config.get("enabled", False):
                    if source == "alpha_vantage" and not config.get("api_key"):
                        print(f"Warning: Alpha Vantage enabled but no API key provided")

            # 验证风险管理
            risk_config = self.get_risk_config()
            if risk_config.get("max_position_size", 0) <= 0:
                print("Warning: max_position_size should be positive")
            if risk_config.get("max_leverage", 0) < 0:
                print("Warning: max_leverage cannot be negative")

            # 验证回测
            backtesting_config = self.get_backtesting_config()
            if backtesting_config.get("initial_capital", 0) <= 0:
                print("Warning: initial_capital should be positive")

            return True
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False

    def create_example_config(self, output_file: Path) -> None:
        """创建示例配置文件。

        Args:
            output_file: 输出文件路径。
        """
        example_config = self._get_default_config()
        
        # 添加注释/描述
        example_config["_comment"] = "Quantitative Trading System Configuration"
        
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(example_config, f, default_flow_style=False)
        
        print(f"Example configuration created at: {output_file}")