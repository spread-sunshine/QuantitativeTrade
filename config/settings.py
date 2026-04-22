# 量化交易系统配置设置
import os
from pathlib import Path
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "backtest_results"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 如果目录不存在则创建
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/market_data.db")

# 数据源配置
YAHOO_FETCH_ENABLED = os.getenv("YAHOO_FETCH_ENABLED", "true").lower() == "true"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_ENABLED = os.getenv("ALPHA_VANTAGE_ENABLED", "false").lower() == "true"

# 风险管理配置
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.2"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.05"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.1"))

# 回测配置
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000.0"))
COMMISSION = float(os.getenv("COMMISSION", "0.001"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0001"))

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/quant_trading.log")

# 可视化配置
PLOT_THEME = os.getenv("PLOT_THEME", "dark")
PLOT_SAVE_FORMAT = os.getenv("PLOT_SAVE_FORMAT", "png")

# 性能优化
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))

# 要跟踪的股票代码列表（可扩展）
DEFAULT_SYMBOLS = [
    "AAPL",  # 苹果
    "MSFT",  # 微软
    "GOOGL", # 谷歌
    "AMZN",  # 亚马逊
    "TSLA",  # 特斯拉
    "SPY",   # S&P 500 ETF
    "QQQ",   # 纳斯达克100 ETF
]