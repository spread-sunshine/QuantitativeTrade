# Quantitative Trading System

A Python-based quantitative trading system for algorithmic trading, backtesting, and strategy development.

一个基于Python的量化交易系统，用于算法交易、回测和策略开发。系统采用模块化设计，覆盖数据获取、策略开发、回测引擎、风险管理、交易执行和可视化等核心功能，严格遵循PEP 8和Google Python Style Guide编码规范。

## 目录

- [项目总结](#项目总结)
- [项目结构](#项目结构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [配置](#配置)
- [开发指南](#开发指南)
- [许可证](#许可证)

## 项目总结

### 概述

本项目是一个完整的、模块化的Python量化交易系统，支持算法交易策略开发、回测和模拟。系统严格遵循PEP 8和Google Python Style Guide编码规范，采用模块化架构覆盖所有核心组件。

### 核心模块

- **数据模块 (src/data/)**: 负责市场数据的获取、存储和处理。支持从雅虎财经等源获取数据，使用SQLite进行持久化存储，并提供数据清洗、特征工程和技术指标计算功能。
- **策略模块 (src/strategies/)**: 基于抽象基类`BaseStrategy`的策略框架，便于快速实现和测试新策略（如移动平均线交叉、均值回归、突破策略等）。
- **回测模块 (src/backtesting/)**: 提供完整的回测引擎，支持单/多策略回测、滚动窗口分析、参数优化和蒙特卡洛模拟，并计算全面的性能与风险指标。
- **风险管理模块 (src/risk/)**: 实现头寸规模计算、止损策略和组合层面的风险限制（回撤、日亏损、集中度），有效控制交易风险。
- **交易执行模块 (src/execution/)**: 模拟交易执行，支持市价单、限价单、止损单，模拟佣金、滑点、延迟和部分成交，并预留与真实经纪商集成的接口。
- **工具与可视化模块 (src/utils/, src/visualization/)**: 提供日志、缓存、配置、日期处理等共享工具，以及资金曲线、回撤图等可视化图表和性能报告生成。

### 关键技术

- **数据管道**: 从外部API获取、清洗、处理和存储市场数据的结构化流程。
- **策略框架**: 面向对象设计，通过抽象基类定义统一接口，支持策略的快速开发与迭代。
- **回测引擎**: 在历史数据上模拟策略执行，评估收益率、夏普比率、最大回撤等关键指标。
- **风险管理**: 通过头寸规模算法（固定分数、凯利公式、波动率调整等）和止损策略（固定、ATR、移动止损等）控制风险。
- **技术指标**: 集成移动平均线、RSI、布林带等常用技术指标，用于信号生成。
- **性能可视化**: 生成资金曲线、回撤图、收益分布等图表，辅助策略分析与优化。

### 开发流程

1. **需求分析**: 明确系统核心功能模块与架构设计。
2. **任务拆分**: 制定详细任务清单，按模块顺序实现。
3. **模块实现**: 依次开发数据、策略、回测、风险、执行等模块，确保各模块独立且可集成。
4. **集成测试**: 通过示例脚本验证系统端到端功能，确保模块间协作正常。
5. **文档完善**: 编写README、示例代码和配置说明，便于用户理解与使用。

### 当前状态

✅ 所有核心模块已实现并集成完毕，系统功能完整。
✅ 提供详细示例代码，可运行`examples/strategy_demo.py`进行端到端测试。
✅ 代码遵循PEP 8和Google Python Style Guide规范，具备良好的可读性与可维护性。


## 项目结构

```
.
├── src/                    # 源代码
│   ├── data/              # 数据获取与存储
│   ├── strategies/        # 交易策略
│   ├── backtesting/       # 回测引擎
│   ├── risk/             # 风险管理
│   ├── execution/        # 交易执行
│   ├── utils/            # 工具函数
│   └── visualization/    # 可视化工具
├── tests/                 # 单元测试
├── config/                # 配置文件
├── examples/              # 示例脚本
├── data/                  # 市场数据存储（运行时生成）
├── backtest_results/      # 回测结果输出（运行时生成）
├── reports/               # 报告输出（运行时生成）
├── requirements.txt       # Python依赖
└── README.md              # 本文档
```

## 安装指南

1. 克隆仓库：
   ```bash
   git clone <repository-url>
   cd TestQuantitative
   ```

2. 创建虚拟环境（推荐）：
   ```bash
   python -m venv venv
   # Windows系统：
   venv\Scripts\activate
   # Unix/macOS系统：
   source venv/bin/activate
   ```

3. 安装依赖：
   ```bash
   pip install -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com -r requirements.txt
   ```

## 快速开始

完成安装后，您可以运行示例脚本快速体验系统功能：

```bash
python examples/strategy_demo.py
```

该示例演示了完整的数据获取、策略回测和结果可视化流程。

## 使用说明

### 数据获取
```python
from src.data.fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_yahoo("AAPL", start="2020-01-01", end="2023-12-31")
```

### 策略开发
```python
from src.strategies.base import BaseStrategy
from src.strategies.moving_average import MovingAverageCrossover

strategy = MovingAverageCrossover(short_window=20, long_window=50)
```

### 回测
```python
from src.backtesting.engine import BacktestEngine

engine = BacktestEngine(strategy, data)
results = engine.run()
```

### 可视化
```python
from src.visualization.report import generate_report

generate_report(results, "backtest_results/report.html")
```

## 配置

1. 复制 `.env.example` 为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，配置您的设置：
   ```
   # 数据库
   DATABASE_URL=sqlite:///data/market_data.db

   # 数据源
   YAHOO_FETCH_ENABLED=true

   # 风险管理
   MAX_POSITION_SIZE=0.1
   MAX_DRAWDOWN=0.2
   ```

## 开发指南

### 运行测试
```bash
pytest tests/
```

### 代码风格
本项目遵循 PEP 8 和 Google Python Style Guide 编码规范。

## 许可证

MIT