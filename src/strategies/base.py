# 交易策略基类
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple, TypeAlias
import pandas as pd
import numpy as np
import logging

from ..utils.logger import setup_logger

# 类型别名，提高可读性
Series: TypeAlias = pd.Series
DataFrame: TypeAlias = pd.DataFrame
Timestamp: TypeAlias = pd.Timestamp
BacktestResult = Dict[str, Any]
SignalDict = Dict[str, Union[str, float, pd.Timestamp, None]]

logger = setup_logger(__name__)


class BaseStrategy(ABC):
    """所有交易策略的基类。"""

    def __init__(
        self,
        name: str = "BaseStrategy",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
    ):
        """初始化策略。

        Args:
            name: 策略名称。
            initial_capital: 回测初始资金。
            commission: 每笔交易佣金率。
            slippage: 价格滑点比例。
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 状态变量
        self.position: int = 0  # 当前持仓 (+1 多头, -1 空头, 0 空仓)
        self.capital: float = initial_capital
        self.equity: float = initial_capital
        self.trades: List[Dict[str, Any]] = []  # 交易记录列表
        
        # 性能指标
        self.returns: List[float] = []  # Daily returns
        self.drawdown: float = 0.0  # Current drawdown
        self.max_drawdown: float = 0.0  # Maximum drawdown
        self.sharpe_ratio: float = 0.0  # Sharpe ratio
        self.sortino_ratio: float = 0.0  # Sortino ratio
        self.win_rate: float = 0.0  # Win rate
        
        # 数据
        self.data: Optional[DataFrame] = None  # 历史市场数据
        self.signals: Optional[DataFrame] = None  # 生成的交易信号

    @abstractmethod
    def generate_signals(self, data: DataFrame) -> DataFrame:
        """从市场数据生成交易信号。

        子类必须实现此方法，根据其特定逻辑生成交易信号。
        该方法应返回至少包含'signal'列的DataFrame，其值为：
        - 1: 买入/多头信号
        - -1: 卖出/空头信号
        - 0: 持有/空仓信号

        Args:
            data: 包含至少OHLCV列（'open', 'high', 'low', 'close', 'volume'）的历史市场数据DataFrame。

        Returns:
            与输入数据相同索引的DataFrame，包含：
            - 'signal': 交易信号（1, -1, 或 0）
            - 其他所需列（例如 'strength', 'reason'）

        Raises:
            ValueError: 如果输入数据缺少必需列或为空。
        """
        pass

    def calculate_returns(self, prices: Series, signals: Series) -> Series:
        """基于信号和价格计算策略收益。

        此方法计算策略的日收益，调整交易成本（佣金）和市场影响（滑点）。
        信号向后移动一个周期以避免前瞻性偏差。

        Args:
            prices: 价格序列（例如收盘价）。
            signals: 交易信号序列（1表示多头，-1表示空头，0表示空仓）。

        Returns:
            包含日策略收益（含成本）的序列。

        Raises:
            ValueError: 如果价格和信号长度不同。
            ValueError: 如果信号包含除-1、0、1以外的值。
        """
        if len(prices) != len(signals):
            raise ValueError(f"Prices ({len(prices)}) and signals ({len(signals)}) must have same length")
        
        # 验证信号值
        unique_signals = signals.unique()
        invalid_signals = set(unique_signals) - {-1, 0, 1}
        if invalid_signals:
            raise ValueError(f"Signals must be -1, 0, or 1. Found invalid values: {invalid_signals}")
        
        # 信号向后移动以避免前瞻性偏差（在下一周期开盘交易）
        signals_shifted = signals.shift(1)
        
        # 计算价格收益
        price_returns = prices.pct_change()
        
        # 计算原始策略收益（信号 * 价格收益）
        strategy_returns = signals_shifted * price_returns
        
        # 调整佣金（进出场支付佣金）
        trade_changes = signals_shifted.diff().fillna(0).abs()
        strategy_returns = strategy_returns - (trade_changes * self.commission)
        
        # 调整滑点（成本与头寸规模成比例）
        strategy_returns = strategy_returns - (abs(signals_shifted) * self.slippage)
        
        return strategy_returns

    def calculate_equity_curve(self, returns: Series) -> Series:
        """从收益计算权益曲线。

        Args:
            returns: 策略收益序列（日收益）。

        Returns:
            权益曲线序列（收益的累积乘积）。
        """
        equity_curve = (1 + returns).cumprod() * self.initial_capital
        return equity_curve

    def calculate_drawdown(self, equity_curve: Series) -> Series:
        """从权益曲线计算回撤序列。

        Args:
            equity_curve: 权益曲线序列。

        Returns:
            回撤序列（负值表示回撤）。
        """
        # 计算运行最大值
        running_max = equity_curve.expanding().max()
        
        # 计算回撤
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown

    def calculate_sharpe_ratio(
        self, returns: Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率（风险调整后收益）。

        Args:
            returns: 策略收益序列（日收益）。
            risk_free_rate: 年化无风险利率（默认0.02 = 2%）。

        Returns:
            夏普比率（年化）。数据不足时返回0.0。
        """
        if len(returns) < 2:
            return 0.0
        
        # 年化收益和标准差
        # 假设日收益（252个交易日）
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        
        if annualized_vol == 0:
            return 0.0
        
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        return sharpe

    def calculate_sortino_ratio(
        self, returns: Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算索提诺比率（下行风险调整后收益）。

        Args:
            returns: 策略收益序列（日收益）。
            risk_free_rate: 年化无风险利率（默认0.02 = 2%）。

        Returns:
            索提诺比率（年化）。数据不足时返回0.0。
        """
        if len(returns) < 2:
            return 0.0
        
        # 分离负收益
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) < 2:
            return 0.0
        
        # 年化收益和下行偏差
        annualized_return = returns.mean() * 252
        downside_vol = negative_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return 0.0
        
        sortino = (annualized_return - risk_free_rate) / downside_vol
        return sortino

    def calculate_win_rate(self, returns: Series) -> float:
        """计算胜率（盈利周期百分比）。

        Args:
            returns: 策略收益序列（日收益）。

        Returns:
            胜率介于0.0和1.0之间。数据不足时返回0.0。
        """
        if len(returns) < 2:
            return 0.0
        
        # 计算正收益数量（假设每个周期为一次\"交易\"）
        positive_trades = (returns > 0).sum()
        total_trades = len(returns)
        
        if total_trades == 0:
            return 0.0
        
        win_rate = positive_trades / total_trades
        return win_rate

    def calculate_max_drawdown(self, drawdown: Series) -> float:
        """从回撤序列计算最大回撤。

        Args:
            drawdown: 回撤序列（负值）。

        Returns:
            最大回撤（最负值，例如-0.15表示15%回撤）。
            空序列时返回0.0。
        """
        if len(drawdown) == 0:
            return 0.0
        
        max_dd = drawdown.min()
        return max_dd

    def run_backtest(self, data: DataFrame) -> BacktestResult:
        """在历史数据上运行回测。

        通过以下步骤执行完整回测：
        1. 从数据生成交易信号
        2. 计算策略收益（调整成本后）
        3. 计算性能指标（夏普、索提诺、回撤等）
        4. 将结果编译为结构化字典

        Args:
            data: 包含OHLCV列的历史市场数据DataFrame。

        Returns:
            包含全面回测结果的字典。

        Raises:
            ValueError: 如果数据缺少必需的'close'列。
            RuntimeError: 如果信号生成失败。
        """
        logger.info(f"Running backtest for {self.name}")
        
        # 存储数据
        self.data = data.copy()
        
        # 生成信号
        self.signals = self.generate_signals(data)
        
        # 计算收益
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        returns = self.calculate_returns(data['close'], self.signals['signal'])
        
        # Calculate equity curve
        equity_curve = self.calculate_equity_curve(returns)
        
        # 计算回撤
        drawdown = self.calculate_drawdown(equity_curve)
        
        # Calculate performance metrics
        self.sharpe_ratio = self.calculate_sharpe_ratio(returns)
        self.sortino_ratio = self.calculate_sortino_ratio(returns)
        self.win_rate = self.calculate_win_rate(returns)
        self.max_drawdown = self.calculate_max_drawdown(drawdown)
        
        # Store returns
        self.returns = returns
        
        # Prepare results
        results = {
            "strategy_name": self.name,
            "initial_capital": self.initial_capital,
            "final_equity": equity_curve.iloc[-1] if not equity_curve.empty else self.initial_capital,
            "total_return": (equity_curve.iloc[-1] / self.initial_capital - 1) if not equity_curve.empty else 0.0,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.signals['signal'].diff().abs().sum(),
            "returns": returns,
            "equity_curve": equity_curve,
            "drawdown": drawdown,
            "signals": self.signals,
        }
        
        # Calculate annualized return
        if len(returns) > 0:
            annualized_return = returns.mean() * 252
            results["annualized_return"] = annualized_return
        
        logger.info(f"Backtest completed for {self.name}")
        logger.info(f"  Total Return: {results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters.

        Returns:
            Dictionary containing all configurable strategy parameters.
        """
        return {
            "name": self.name,
            "initial_capital": self.initial_capital,
            "commission": self.commission,
            "slippage": self.slippage,
        }

    def set_parameters(self, **kwargs) -> None:
        """Set strategy parameters.

        Args:
            **kwargs: Parameter key-value pairs.

        Returns:
            None
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Parameter {key} not found in strategy")

    def reset(self) -> None:
        """Reset strategy state to initial conditions.

        This method clears all accumulated data and resets the strategy
        to its initial state, allowing for a fresh backtest or simulation.
        """
        self.position = 0
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.trades = []
        self.returns = []
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.win_rate = 0.0
        self.data = None
        self.signals = None
        
        logger.info(f"Strategy {self.name} reset")


class Signal:
    """表示带有时间戳、符号和元数据的交易信号。"""
    
    def __init__(
        self,
        timestamp: Timestamp,
        symbol: str,
        signal_type: str,  # 'long', 'short', 'exit'
        price: float,
        quantity: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """使用所有相关信息初始化信号。
        
        Args:
            timestamp: 信号时间戳（信号生成的日期时间）。
            symbol: 交易符号（例如 'AAPL', 'SPY'）。
            signal_type: 信号类型（'long', 'short', 'exit'）。
            price: 信号时的入场/出场价格。
            quantity: 头寸规模（正数为多头，负数为空头）。
            reason: 信号的人类可读原因（可选）。
        """
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.price = price
        self.quantity = quantity
        self.reason = reason
        
    def to_dict(self) -> SignalDict:
        """将信号转换为可序列化字典。
        
        Returns:
            适合JSON序列化的字典表示。
        """
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "price": self.price,
            "quantity": self.quantity,
            "reason": self.reason,
        }