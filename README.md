# Quantitative Trading System

A Python-based quantitative trading system for algorithmic trading, backtesting, and strategy development.

## Features

- **Data Acquisition**: Fetch market data from various sources (Yahoo Finance, etc.)
- **Data Storage**: Store historical data in SQLite database
- **Strategy Development**: Framework for implementing trading strategies
- **Backtesting Engine**: Test strategies on historical data
- **Risk Management**: Position sizing and risk control
- **Trade Execution**: Simulated trading (extensible to real exchanges)
- **Monitoring & Reporting**: Visualize results and generate reports

## Project Structure

```
.
├── src/                    # Source code
│   ├── data/              # Data acquisition and storage
│   ├── strategies/        # Trading strategies
│   ├── backtesting/       # Backtesting engine
│   ├── risk/             # Risk management
│   ├── execution/        # Trade execution
│   ├── utils/            # Utilities and helpers
│   └── visualization/    # Visualization tools
├── tests/                 # Unit tests
├── data/                  # Market data and databases
├── config/                # Configuration files
├── backtest_results/      # Backtest results
├── reports/               # Generated reports
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd TestQuantitative
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com -r requirements.txt
   ```

## Usage

### Data Fetching
```python
from src.data.fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_yahoo("AAPL", start="2020-01-01", end="2023-12-31")
```

### Strategy Development
```python
from src.strategies.base import BaseStrategy
from src.strategies.moving_average import MovingAverageCrossover

strategy = MovingAverageCrossover(short_window=20, long_window=50)
```

### Backtesting
```python
from src.backtesting.engine import BacktestEngine

engine = BacktestEngine(strategy, data)
results = engine.run()
```

### Visualization
```python
from src.visualization.report import generate_report

generate_report(results, "backtest_results/report.html")
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your configuration:
   ```
   # Database
   DATABASE_URL=sqlite:///data/market_data.db

   # Data sources
   YAHOO_FETCH_ENABLED=true

   # Risk management
   MAX_POSITION_SIZE=0.1
   MAX_DRAWDOWN=0.2
   ```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
This project follows PEP 8 and Google Python Style Guide.

## License

MIT