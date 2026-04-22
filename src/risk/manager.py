"""风险管理核心模块。"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskLevel(Enum):
    """风险承受水平。"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskLimits:
    """风险限制配置。"""
    # 持仓限制
    max_position_size: float = 0.1  # 单只持仓最大占投资组合的10%
    max_sector_exposure: float = 0.3  # 单一行业最大暴露30%
    max_leverage: float = 1.0  # 最大杠杆（1.0表示无杠杆）
    
    # 风险限制
    max_drawdown_limit: float = 0.2  # 如果回撤超过20%则停止交易
    daily_loss_limit: float = 0.05  # 如果单日损失超过5%则停止交易
    weekly_loss_limit: float = 0.1  # 如果单周损失超过10%则停止交易
    
    # 集中度限制
    max_concentration: float = 0.25  # 前N大持仓最大占25%
    min_diversification: int = 5  # 最小持仓数量
    
    # 流动性限制
    min_liquidity: float = 1000000.0  # 最小日交易量
    max_slippage: float = 0.001  # 最大可接受滑点
    
    @classmethod
    def get_preset(cls, risk_level: RiskLevel) -> 'RiskLimits':
        """根据风险等级获取风险限制预设。"""
        if risk_level == RiskLevel.CONSERVATIVE:
            return cls(
                max_position_size=0.05,
                max_sector_exposure=0.2,
                max_leverage=0.5,
                max_drawdown_limit=0.15,
                daily_loss_limit=0.03,
                weekly_loss_limit=0.06,
                max_concentration=0.15,
                min_diversification=10,
                min_liquidity=5000000.0,
                max_slippage=0.0005,
            )
        elif risk_level == RiskLevel.MODERATE:
            return cls(
                max_position_size=0.1,
                max_sector_exposure=0.3,
                max_leverage=1.0,
                max_drawdown_limit=0.2,
                daily_loss_limit=0.05,
                weekly_loss_limit=0.1,
                max_concentration=0.25,
                min_diversification=5,
                min_liquidity=1000000.0,
                max_slippage=0.001,
            )
        elif risk_level == RiskLevel.AGGRESSIVE:
            return cls(
                max_position_size=0.2,
                max_sector_exposure=0.5,
                max_leverage=2.0,
                max_drawdown_limit=0.3,
                daily_loss_limit=0.1,
                weekly_loss_limit=0.2,
                max_concentration=0.4,
                min_diversification=3,
                min_liquidity=500000.0,
                max_slippage=0.002,
            )
        else:
            return cls()


class RiskManager:
    """综合性风险管理系统。
    
    功能特性：
    - 仓位规模与限制
    - 回撤控制
    - 损失限制（日度、周度）
    - 集中度风险
    - 流动性风险
    - 保证金与杠杆控制
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        custom_limits: Optional[Dict[str, float]] = None,
    ):
        """初始化风险管理器。
        
        参数：
            initial_capital: 初始投资组合资金。
            risk_level: 风险承受水平。
            custom_limits: 自定义风险限制，覆盖默认值。
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_level = risk_level
        
        # Get risk limits preset
        self.limits = RiskLimits.get_preset(risk_level)
        
        # Apply custom limits if provided
        if custom_limits:
            for key, value in custom_limits.items():
                if hasattr(self.limits, key):
                    setattr(self.limits, key, value)
        
        # State tracking
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_pnl: Dict[str, float] = {}  # date -> PNL
        self.weekly_pnl: Dict[str, float] = {}  # week -> PNL
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.daily_return = 0.0
        self.weekly_return = 0.0
        
        # Risk flags
        self.is_trading_allowed = True
        self.risk_violations: List[str] = []
        
        logger.info(
            f"RiskManager initialized with capital={initial_capital}, "
            f"risk_level={risk_level.value}"
        )
        logger.info(f"Risk limits: {self.limits}")
    
    def update_portfolio(
        self,
        positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        timestamp: datetime,
    ) -> None:
        """更新投资组合状态并计算风险指标。
        
        参数：
            positions: 当前持仓字典。
            current_prices: 持仓的当前市场价格。
            timestamp: 当前时间戳。
        """
        self.positions = positions.copy()
        
        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value(current_prices)
        self.current_capital = portfolio_value
        
        # Update risk metrics
        self._update_drawdown(portfolio_value)
        self._update_daily_pnl(portfolio_value, timestamp)
        self._update_weekly_pnl(portfolio_value, timestamp)
        
        # Check risk limits
        self._check_risk_limits(current_prices)
        
        logger.debug(
            f"Portfolio updated: value={portfolio_value:.2f}, "
            f"drawdown={self.current_drawdown:.2%}"
        )
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_per_trade: float = 0.02,
        max_position_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """基于风险计算最优仓位规模。
        
        参数：
            symbol: 交易标的。
            entry_price: 入场价格。
            stop_loss: 止损价格。
            risk_per_trade: 每笔交易最大风险（0.02 表示 2%）。
            max_position_pct: 最大仓位占投资组合的百分比。
            
        返回：
            元组 (仓位规模, 仓位价值)。
        """
        # Use default max position size if not specified
        if max_position_pct is None:
            max_position_pct = self.limits.max_position_size
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            logger.warning(f"Invalid stop loss for {symbol}: entry={entry_price}, stop={stop_loss}")
            return 0.0, 0.0
        
        # Calculate maximum risk amount
        max_risk_amount = self.current_capital * risk_per_trade
        
        # Calculate position size based on risk
        risk_based_size = max_risk_amount / risk_per_share
        
        # Calculate position size based on maximum percentage
        max_position_value = self.current_capital * max_position_pct
        percentage_based_size = max_position_value / entry_price
        
        # Use the smaller of the two
        position_size = min(risk_based_size, percentage_based_size)
        
        # Apply rounding (e.g., whole shares for stocks)
        if symbol.endswith('.US'):
            position_size = int(position_size)  # Whole shares for US stocks
        
        position_value = position_size * entry_price
        
        logger.debug(
            f"Position size for {symbol}: size={position_size:.2f}, "
            f"value={position_value:.2f}, risk={risk_per_share:.2f} per share"
        )
        
        return position_size, position_value
    
    def check_trade_allowed(
        self,
        symbol: str,
        position_size: float,
        position_value: float,
        current_prices: Dict[str, float],
        timestamp: datetime,
    ) -> Tuple[bool, List[str]]:
        """根据风险限制检查是否允许交易。
        
        参数：
            symbol: 交易标的。
            position_size: 拟建仓位规模。
            position_value: 拟建仓位价值。
            current_prices: 当前市场价格。
            timestamp: 交易时间戳。
            
        返回：
            元组 (是否允许, 违规消息列表)。
        """
        violations = []
        
        # Check if trading is globally allowed
        if not self.is_trading_allowed:
            violations.append("Trading is currently suspended due to risk limits")
        
        # Check position size limit
        position_pct = position_value / self.current_capital if self.current_capital > 0 else 0
        if position_pct > self.limits.max_position_size:
            violations.append(
                f"Position size ({position_pct:.2%}) exceeds maximum "
                f"({self.limits.max_position_size:.2%})"
            )
        
        # Check sector concentration (if sector info available)
        # This would require additional sector data
        
        # Check portfolio concentration
        if self._check_portfolio_concentration(symbol, position_value):
            violations.append(
                f"Adding position would exceed portfolio concentration limit "
                f"({self.limits.max_concentration:.2%})"
            )
        
        # Check liquidity (if volume data available)
        # This would require additional volume data
        
        # Check leverage limit
        current_leverage = self._calculate_leverage(current_prices)
        if current_leverage > self.limits.max_leverage:
            violations.append(
                f"Current leverage ({current_leverage:.2f}) exceeds maximum "
                f"({self.limits.max_leverage:.2f})"
            )
        
        # Check daily loss limit
        if abs(self.daily_return) > self.limits.daily_loss_limit:
            violations.append(
                f"Daily loss ({self.daily_return:.2%}) exceeds limit "
                f"({self.limits.daily_loss_limit:.2%})"
            )
        
        # Check weekly loss limit
        if abs(self.weekly_return) > self.limits.weekly_loss_limit:
            violations.append(
                f"Weekly loss ({self.weekly_return:.2%}) exceeds limit "
                f"({self.limits.weekly_loss_limit:.2%})"
            )
        
        # Check max drawdown limit
        if abs(self.current_drawdown) > self.limits.max_drawdown_limit:
            violations.append(
                f"Current drawdown ({self.current_drawdown:.2%}) exceeds limit "
                f"({self.limits.max_drawdown_limit:.2%})"
            )
        
        is_allowed = len(violations) == 0
        
        if not is_allowed:
            logger.warning(
                f"Trade not allowed for {symbol}. Violations: {violations}"
            )
        
        return is_allowed, violations
    
    def record_trade(
        self,
        symbol: str,
        trade_type: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """记录已完成的交易。
        
        参数：
            symbol: 交易标的。
            trade_type: 'BUY' 或 'SELL'。
            quantity: 交易数量。
            price: 交易价格。
            timestamp: 交易时间戳。
            commission: 支付的佣金。
            slippage: 滑点成本。
        """
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'type': trade_type,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'slippage': slippage,
            'value': quantity * price,
        }
        
        self.trade_history.append(trade)
        
        logger.debug(
            f"Trade recorded: {symbol} {trade_type} {quantity} @ {price:.2f}"
        )
    
    def suspend_trading(self, reason: str) -> None:
        """暂停所有交易。
        
        参数：
            reason: 暂停原因。
        """
        self.is_trading_allowed = False
        self.risk_violations.append(f"Trading suspended: {reason}")
        
        logger.warning(f"Trading suspended: {reason}")
    
    def resume_trading(self) -> None:
        """如果条件允许，恢复交易。"""
        # Check if risk limits are still violated
        if self._check_risk_limits({}):
            self.is_trading_allowed = True
            logger.info("Trading resumed")
        else:
            logger.warning("Cannot resume trading: risk limits still violated")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """生成综合风险报告。
        
        返回：
            包含风险指标和违规信息的字典。
        """
        report = {
            "portfolio": {
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "total_return": (self.current_capital / self.initial_capital) - 1,
            },
            "risk_metrics": {
                "max_drawdown": self.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "daily_return": self.daily_return,
                "weekly_return": self.weekly_return,
                "num_positions": len(self.positions),
                "leverage": self._calculate_leverage({}),
            },
            "limits": {
                "max_position_size": self.limits.max_position_size,
                "max_sector_exposure": self.limits.max_sector_exposure,
                "max_leverage": self.limits.max_leverage,
                "max_drawdown_limit": self.limits.max_drawdown_limit,
                "daily_loss_limit": self.limits.daily_loss_limit,
                "weekly_loss_limit": self.limits.weekly_loss_limit,
                "max_concentration": self.limits.max_concentration,
                "min_diversification": self.limits.min_diversification,
            },
            "status": {
                "is_trading_allowed": self.is_trading_allowed,
                "risk_violations": self.risk_violations.copy(),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        return report
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算投资组合总价值。
        
        参数：
            current_prices: 当前市场价格。
            
        返回：
            投资组合价值。
        """
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))
            quantity = position.get('quantity', 0)
            total_value += quantity * current_price
        
        return total_value
    
    def _update_drawdown(self, portfolio_value: float) -> None:
        """更新回撤指标。
        
        参数：
            portfolio_value: 当前投资组合价值。
        """
        # Update peak value
        if portfolio_value > self.initial_capital:
            self.initial_capital = portfolio_value
        
        # Calculate current drawdown
        if self.initial_capital > 0:
            self.current_drawdown = (portfolio_value - self.initial_capital) / self.initial_capital
        else:
            self.current_drawdown = 0.0
        
        # Update maximum drawdown
        if self.current_drawdown < self.max_drawdown:
            self.max_drawdown = self.current_drawdown
    
    def _update_daily_pnl(self, portfolio_value: float, timestamp: datetime) -> None:
        """更新每日盈亏跟踪。
        
        参数：
            portfolio_value: 当前投资组合价值。
            timestamp: 当前时间戳。
        """
        date_str = timestamp.strftime('%Y-%m-%d')
        
        if date_str not in self.daily_pnl:
            # New day, reset daily PNL
            self.daily_pnl[date_str] = 0.0
            self.daily_return = 0.0
        
        # Calculate daily return (simplified)
        # In practice, you would track daily opening value
        daily_pnl = portfolio_value - self.current_capital
        self.daily_pnl[date_str] += daily_pnl
        
        # Calculate daily return percentage
        if self.current_capital > 0:
            self.daily_return = daily_pnl / self.current_capital
    
    def _update_weekly_pnl(self, portfolio_value: float, timestamp: datetime) -> None:
        """更新每周盈亏跟踪。
        
        参数：
            portfolio_value: 当前投资组合价值。
            timestamp: 当前时间戳。
        """
        # Get week identifier (year-week)
        year, week, _ = timestamp.isocalendar()
        week_str = f"{year}-W{week:02d}"
        
        if week_str not in self.weekly_pnl:
            # New week, reset weekly PNL
            self.weekly_pnl[week_str] = 0.0
            self.weekly_return = 0.0
        
        # Calculate weekly return (simplified)
        # In practice, you would track weekly opening value
        weekly_pnl = portfolio_value - self.current_capital
        self.weekly_pnl[week_str] += weekly_pnl
        
        # Calculate weekly return percentage
        if self.current_capital > 0:
            self.weekly_return = weekly_pnl / self.current_capital
    
    def _check_risk_limits(self, current_prices: Dict[str, float]) -> bool:
        """检查所有风险限制。
        
        参数：
            current_prices: 当前市场价格。
            
        返回：
            如果所有限制都满足则返回 True，否则返回 False。
        """
        self.risk_violations.clear()
        
        # Check drawdown limit
        if abs(self.current_drawdown) > self.limits.max_drawdown_limit:
            self.risk_violations.append(
                f"Drawdown limit exceeded: {abs(self.current_drawdown):.2%} > "
                f"{self.limits.max_drawdown_limit:.2%}"
            )
        
        # Check daily loss limit
        if abs(self.daily_return) > self.limits.daily_loss_limit:
            self.risk_violations.append(
                f"Daily loss limit exceeded: {abs(self.daily_return):.2%} > "
                f"{self.limits.daily_loss_limit:.2%}"
            )
        
        # Check weekly loss limit
        if abs(self.weekly_return) > self.limits.weekly_loss_limit:
            self.risk_violations.append(
                f"Weekly loss limit exceeded: {abs(self.weekly_return):.2%} > "
                f"{self.limits.weekly_loss_limit:.2%}"
            )
        
        # Check leverage limit
        leverage = self._calculate_leverage(current_prices)
        if leverage > self.limits.max_leverage:
            self.risk_violations.append(
                f"Leverage limit exceeded: {leverage:.2f} > "
                f"{self.limits.max_leverage:.2f}"
            )
        
        # Check diversification
        if len(self.positions) < self.limits.min_diversification:
            self.risk_violations.append(
                f"Diversification insufficient: {len(self.positions)} positions < "
                f"minimum {self.limits.min_diversification}"
            )
        
        # Update trading suspension
        if self.risk_violations and self.is_trading_allowed:
            self.is_trading_allowed = False
            logger.warning(
                f"Trading suspended due to risk limit violations: "
                f"{self.risk_violations}"
            )
        
        return len(self.risk_violations) == 0
    
    def _check_portfolio_concentration(
        self,
        new_symbol: str,
        new_position_value: float,
    ) -> bool:
        """检查新增仓位是否超出集中度限制。
        
        参数：
            new_symbol: 新增标的。
            new_position_value: 新增仓位价值。
            
        返回：
            如果超出集中度限制则返回 True。
        """
        # Calculate total portfolio value with new position
        total_value = self.current_capital + new_position_value
        
        # Calculate position values
        position_values = []
        for symbol, position in self.positions.items():
            position_value = position.get('value', 0)
            position_values.append((symbol, position_value))
        
        # Add new position
        position_values.append((new_symbol, new_position_value))
        
        # Sort by value descending
        position_values.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate concentration of top positions
        # For simplicity, check if any single position exceeds limit
        for symbol, value in position_values:
            position_pct = value / total_value if total_value > 0 else 0
            if position_pct > self.limits.max_concentration:
                return True
        
        return False
    
    def _calculate_leverage(self, current_prices: Dict[str, float]) -> float:
        """计算投资组合杠杆。
        
        参数：
            current_prices: 当前市场价格。
            
        返回：
            杠杆比率。
        """
        # Simplified calculation: total position value / equity
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))
            quantity = position.get('quantity', 0)
            total_position_value += quantity * current_price
        
        if self.current_capital > 0:
            return total_position_value / self.current_capital
        else:
            return 0.0