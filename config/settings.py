# Configuration settings for Quantitative Trading System
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "backtest_results"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/market_data.db")

# Data source configuration
YAHOO_FETCH_ENABLED = os.getenv("YAHOO_FETCH_ENABLED", "true").lower() == "true"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_ENABLED = os.getenv("ALPHA_VANTAGE_ENABLED", "false").lower() == "true"

# Risk management configuration
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.2"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.05"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.1"))

# Backtesting configuration
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000.0"))
COMMISSION = float(os.getenv("COMMISSION", "0.001"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0001"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/quant_trading.log")

# Visualization configuration
PLOT_THEME = os.getenv("PLOT_THEME", "dark")
PLOT_SAVE_FORMAT = os.getenv("PLOT_SAVE_FORMAT", "png")

# Performance optimization
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))

# List of symbols to track (can be extended)
DEFAULT_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq 100 ETF
]