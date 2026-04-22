"""用于风险管理的仓位规模算法。"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from enum import Enum
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionSizingMethod(Enum):
    """仓位规模方法。"""
    FIXED_FRACTIONAL = "fixed_fractional"  # 固定资金比例
    KELLY_CRITERION = "kelly_criterion"    # 凯利公式
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # 根据波动率调整
    OPTIMAL_F = "optimal_f"                # 最优 f (Ralph Vince)
    EQUAL_WEIGHT = "equal_weight"          # 等权重
    RISK_PARITY = "risk_parity"            # 风险平价权重
    BLACK_LITTERMAN = "black_litterman"    # 布莱克-利特曼模型


class PositionSizer:
    """多算法仓位规模计算器。
    
    实现多种仓位规模策略：
    - 固定分数规模
    - 凯利公式
    - 波动率调整规模
    - 最优 f
    - 等权重
    - 风险平价
    - 布莱克-利特曼模型
    """
    
    def __init__(
        self,
        method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL,
        max_position_pct: float = 0.1,
        max_portfolio_risk: float = 0.02,
        kelly_fraction: float = 0.5,
        volatility_lookback: int = 20,
        risk_free_rate: float = 0.02,
    ):
        """初始化仓位规模计算器。
        
        参数：
            method: 仓位规模方法。
            max_position_pct: 最大仓位占投资组合的百分比。
            max_portfolio_risk: 每笔交易最大风险占投资组合的百分比。
            kelly_fraction: 使用的凯利公式比例（0.5 表示半凯利）。
            volatility_lookback: 波动率计算的回溯期。
            risk_free_rate: 年化无风险利率。
        """
        self.method = method
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.kelly_fraction = kelly_fraction
        self.volatility_lookback = volatility_lookback
        self.risk_free_rate = risk_free_rate
        
        # Historical data for calculations
        self.returns_history: Dict[str, pd.Series] = {}
        self.volatility_history: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        logger.info(
            f"PositionSizer initialized with method={method.value}, "
            f"max_position_pct={max_position_pct}, max_portfolio_risk={max_portfolio_risk}"
        )
    
    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        stop_loss_price: Optional[float] = None,
        expected_return: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        confidence: float = 0.95,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """使用选定方法计算仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前市场价格。
            portfolio_value: 投资组合总价值。
            stop_loss_price: 止损价格（可选）。
            expected_return: 预期收益率（年化）。
            volatility: 预期波动率（年化）。
            win_rate: 历史胜率（0-1）。
            avg_win: 平均盈利百分比。
            avg_loss: 平均亏损百分比。
            confidence: 计算置信水平。
            
        返回：
            元组 (仓位规模, 仓位价值, 计算详情)。
        """
        if portfolio_value <= 0:
            return 0.0, 0.0, {"error": "Portfolio value must be positive"}
        
        # Get or calculate volatility
        if volatility is None:
            volatility = self._get_volatility(symbol)
        
        # Get or calculate expected return
        if expected_return is None:
            expected_return = self._get_expected_return(symbol)
        
        # Calculate based on method
        if self.method == PositionSizingMethod.FIXED_FRACTIONAL:
            size, value, details = self._fixed_fractional(
                symbol, current_price, portfolio_value, stop_loss_price
            )
        elif self.method == PositionSizingMethod.KELLY_CRITERION:
            size, value, details = self._kelly_criterion(
                symbol, current_price, portfolio_value,
                win_rate, avg_win, avg_loss
            )
        elif self.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            size, value, details = self._volatility_adjusted(
                symbol, current_price, portfolio_value, volatility
            )
        elif self.method == PositionSizingMethod.OPTIMAL_F:
            size, value, details = self._optimal_f(
                symbol, current_price, portfolio_value,
                win_rate, avg_win, avg_loss
            )
        elif self.method == PositionSizingMethod.EQUAL_WEIGHT:
            size, value, details = self._equal_weight(
                symbol, current_price, portfolio_value
            )
        elif self.method == PositionSizingMethod.RISK_PARITY:
            size, value, details = self._risk_parity(
                symbol, current_price, portfolio_value, volatility
            )
        elif self.method == PositionSizingMethod.BLACK_LITTERMAN:
            size, value, details = self._black_litterman(
                symbol, current_price, portfolio_value,
                expected_return, volatility, confidence
            )
        else:
            # Default to fixed fractional
            size, value, details = self._fixed_fractional(
                symbol, current_price, portfolio_value, stop_loss_price
            )
        
        # Apply maximum position size constraint
        position_pct = value / portfolio_value
        if position_pct > self.max_position_pct:
            size = (portfolio_value * self.max_position_pct) / current_price
            value = size * current_price
            details["constraint_applied"] = "max_position_pct"
            details["original_size"] = size
            details["original_value"] = value
        
        # Round position size (e.g., whole shares for stocks)
        if symbol.endswith('.US') or symbol.endswith('.HK'):
            size = int(size)  # Whole shares for stocks
            value = size * current_price
        
        details.update({
            "method": self.method.value,
            "position_size": size,
            "position_value": value,
            "position_pct": value / portfolio_value,
            "current_price": current_price,
            "portfolio_value": portfolio_value,
        })
        
        logger.debug(
            f"Position size for {symbol}: size={size:.2f}, value={value:.2f}, "
            f"pct={(value/portfolio_value):.2%}, method={self.method.value}"
        )
        
        return size, value, details
    
    def update_returns_history(
        self,
        symbol: str,
        returns: pd.Series,
    ) -> None:
        """更新标的的收益历史。
        
        参数：
            symbol: 交易标的。
            returns: 收益序列。
        """
        self.returns_history[symbol] = returns.copy()
        
        # Update volatility
        if len(returns) >= self.volatility_lookback:
            recent_returns = returns.iloc[-self.volatility_lookback:]
            self.volatility_history[symbol] = recent_returns.std() * np.sqrt(252)
        
        # Update correlation matrix periodically
        self._update_correlation_matrix()
    
    def _fixed_fractional(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        stop_loss_price: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """固定分数仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            stop_loss_price: 止损价格。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Calculate based on risk per trade
        if stop_loss_price is not None:
            # Risk-based sizing
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share > 0:
                max_risk_amount = portfolio_value * self.max_portfolio_risk
                size = max_risk_amount / risk_per_share
            else:
                size = 0.0
        else:
            # Fixed percentage sizing
            size = (portfolio_value * self.max_portfolio_risk) / current_price
        
        value = size * current_price
        
        details = {
            "risk_per_share": abs(current_price - stop_loss_price) if stop_loss_price else None,
            "max_risk_amount": portfolio_value * self.max_portfolio_risk,
            "stop_loss_price": stop_loss_price,
        }
        
        return size, value, details
    
    def _kelly_criterion(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """凯利公式仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            win_rate: 胜率（0-1）。
            avg_win: 平均盈利金额（占投注比例）。
            avg_loss: 平均亏损金额（占投注比例）。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Default values if not provided
        if win_rate is None:
            win_rate = 0.5
        
        if avg_win is None:
            avg_win = 0.1  # 10% average win
        
        if avg_loss is None:
            avg_loss = 0.05  # 5% average loss
        
        # Kelly formula: f* = p/a - q/b
        # where p = win probability, q = loss probability
        # a = avg loss, b = avg win (as fractions)
        p = win_rate
        q = 1 - win_rate
        b = avg_win
        a = avg_loss
        
        # Calculate full Kelly fraction
        if a > 0:
            kelly_fraction = p/a - q/b
        else:
            kelly_fraction = 0.0
        
        # Apply fractional Kelly
        kelly_fraction = max(0.0, kelly_fraction)  # No negative positions
        fractional_kelly = kelly_fraction * self.kelly_fraction
        
        # Calculate position size
        size = (portfolio_value * fractional_kelly) / current_price
        value = size * current_price
        
        details = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "full_kelly": kelly_fraction,
            "fractional_kelly": fractional_kelly,
            "kelly_fraction_used": self.kelly_fraction,
        }
        
        return size, value, details
    
    def _volatility_adjusted(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """波动率调整仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            volatility: 年化波动率。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Get volatility if not provided
        if volatility is None:
            volatility = self._get_volatility(symbol)
        
        if volatility is None or volatility <= 0:
            # Fallback to fixed fractional
            return self._fixed_fractional(symbol, current_price, portfolio_value)
        
        # Target volatility (e.g., 20% annualized)
        target_volatility = 0.2
        
        # Calculate position size to achieve target portfolio volatility
        # This is simplified - in practice, you would consider correlations
        position_weight = target_volatility / volatility
        
        # Apply maximum position constraint
        position_weight = min(position_weight, self.max_position_pct)
        
        size = (portfolio_value * position_weight) / current_price
        value = size * current_price
        
        details = {
            "volatility": volatility,
            "target_volatility": target_volatility,
            "position_weight": position_weight,
        }
        
        return size, value, details
    
    def _optimal_f(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """最优 f 仓位规模（Ralph Vince）。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            win_rate: 胜率。
            avg_win: 平均盈利。
            avg_loss: 平均亏损。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Default values
        if win_rate is None:
            win_rate = 0.5
        
        if avg_win is None:
            avg_win = 0.1
        
        if avg_loss is None:
            avg_loss = 0.05
        
        # Calculate geometric mean
        # Simplified calculation - real optimal f requires iterative solution
        p = win_rate
        avg_win_abs = avg_win
        avg_loss_abs = avg_loss
        
        # Calculate optimal f using simplified formula
        # f = (p * avg_win - q * avg_loss) / (avg_win * avg_loss)
        q = 1 - p
        
        numerator = p * avg_win_abs - q * avg_loss_abs
        denominator = avg_win_abs * avg_loss_abs
        
        if denominator > 0:
            optimal_f = numerator / denominator
        else:
            optimal_f = 0.0
        
        # Apply fractional optimal f
        optimal_f = max(0.0, optimal_f)
        fractional_f = optimal_f * self.kelly_fraction
        
        # Calculate position size
        size = (portfolio_value * fractional_f) / current_price
        value = size * current_price
        
        details = {
            "win_rate": win_rate,
            "avg_win": avg_win_abs,
            "avg_loss": avg_loss_abs,
            "optimal_f": optimal_f,
            "fractional_f": fractional_f,
        }
        
        return size, value, details
    
    def _equal_weight(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """等权重仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Equal weight across all positions
        # For simplicity, assume we want 10 positions
        target_num_positions = 10
        
        # Calculate equal weight per position
        weight_per_position = 1.0 / target_num_positions
        
        # Apply maximum position constraint
        weight_per_position = min(weight_per_position, self.max_position_pct)
        
        size = (portfolio_value * weight_per_position) / current_price
        value = size * current_price
        
        details = {
            "target_num_positions": target_num_positions,
            "weight_per_position": weight_per_position,
        }
        
        return size, value, details
    
    def _risk_parity(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """风险平价仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            volatility: 资产波动率。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Get volatility if not provided
        if volatility is None:
            volatility = self._get_volatility(symbol)
        
        if volatility is None or volatility <= 0:
            # Fallback to equal weight
            return self._equal_weight(symbol, current_price, portfolio_value)
        
        # In risk parity, each position contributes equally to portfolio risk
        # This requires knowing all positions and their correlations
        # Simplified version: weight inversely proportional to volatility
        
        # Get average volatility across all assets
        avg_volatility = self._get_average_volatility()
        
        if avg_volatility is None or avg_volatility <= 0:
            avg_volatility = volatility
        
        # Calculate risk parity weight
        # Weight is inversely proportional to volatility
        risk_parity_weight = (1.0 / volatility) / (1.0 / avg_volatility)
        
        # Normalize and apply constraints
        risk_parity_weight = min(risk_parity_weight, self.max_position_pct)
        
        size = (portfolio_value * risk_parity_weight) / current_price
        value = size * current_price
        
        details = {
            "volatility": volatility,
            "avg_volatility": avg_volatility,
            "risk_parity_weight": risk_parity_weight,
        }
        
        return size, value, details
    
    def _black_litterman(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        expected_return: Optional[float] = None,
        volatility: Optional[float] = None,
        confidence: float = 0.95,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """布莱克-利特曼模型仓位规模。
        
        参数：
            symbol: 交易标的。
            current_price: 当前价格。
            portfolio_value: 投资组合价值。
            expected_return: 预期收益率。
            volatility: 资产波动率。
            confidence: 观点置信度。
            
        返回：
            元组 (规模, 价值, 详情)。
        """
        # Get expected return and volatility if not provided
        if expected_return is None:
            expected_return = self._get_expected_return(symbol)
        
        if volatility is None:
            volatility = self._get_volatility(symbol)
        
        if expected_return is None or volatility is None:
            # Fallback to volatility adjusted
            return self._volatility_adjusted(symbol, current_price, portfolio_value, volatility)
        
        # Simplified Black-Litterman calculation
        # In practice, this requires market equilibrium returns and covariance matrix
        
        # Risk aversion parameter (typical value: 3.0)
        risk_aversion = 3.0
        
        # Calculate implied equilibrium return
        # pi = risk_aversion * Σ * w_market
        # For simplicity, assume market weight is 1/N
        market_weight = 0.1  # Assume 10% market weight
        
        # Calculate expected excess return
        risk_free_daily = self.risk_free_rate / 252
        excess_return = expected_return - risk_free_daily
        
        # Calculate Black-Litterman adjusted return
        # E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1Π + P'Ω^-1Q]
        # Simplified: blend equilibrium and view
        tau = 0.05  # Uncertainty in equilibrium returns
        omega = volatility * 0.1  # Uncertainty in views
        
        # Simplified blending
        equilibrium_return = risk_aversion * volatility**2 * market_weight
        view_return = excess_return
        
        # Blend with confidence
        bl_return = (
            (1 - confidence) * equilibrium_return +
            confidence * view_return
        )
        
        # Calculate optimal weight
        if volatility > 0:
            optimal_weight = bl_return / (risk_aversion * volatility**2)
        else:
            optimal_weight = 0.0
        
        # Apply constraints
        optimal_weight = min(optimal_weight, self.max_position_pct)
        
        size = (portfolio_value * optimal_weight) / current_price
        value = size * current_price
        
        details = {
            "expected_return": expected_return,
            "volatility": volatility,
            "equilibrium_return": equilibrium_return,
            "view_return": view_return,
            "bl_return": bl_return,
            "optimal_weight": optimal_weight,
            "confidence": confidence,
        }
        
        return size, value, details
    
    def _get_volatility(self, symbol: str) -> Optional[float]:
        """获取标的波动率。
        
        参数：
            symbol: 交易标的。
            
        返回：
            年化波动率或 None。
        """
        if symbol in self.volatility_history:
            return self.volatility_history[symbol]
        
        # Try to calculate from returns history
        if symbol in self.returns_history:
            returns = self.returns_history[symbol]
            if len(returns) >= self.volatility_lookback:
                recent_returns = returns.iloc[-self.volatility_lookback:]
                volatility = recent_returns.std() * np.sqrt(252)
                self.volatility_history[symbol] = volatility
                return volatility
        
        return None
    
    def _get_expected_return(self, symbol: str) -> Optional[float]:
        """获取标的预期收益率。
        
        参数：
            symbol: 交易标的。
            
        返回：
            预期年化收益率或 None。
        """
        if symbol in self.returns_history:
            returns = self.returns_history[symbol]
            if len(returns) > 0:
                # Annualized expected return
                expected_return = returns.mean() * 252
                return expected_return
        
        return None
    
    def _get_average_volatility(self) -> Optional[float]:
        """获取所有标的的平均波动率。
        
        返回：
            平均年化波动率或 None。
        """
        if not self.volatility_history:
            return None
        
        volatilities = list(self.volatility_history.values())
        return np.mean(volatilities)
    
    def _update_correlation_matrix(self) -> None:
        """根据收益历史更新相关性矩阵。"""
        if len(self.returns_history) < 2:
            return
        
        # Combine returns into DataFrame
        returns_df = pd.DataFrame(self.returns_history)
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        
        logger.debug("Correlation matrix updated")
    
    def get_method_description(self) -> str:
        """获取当前仓位规模方法的描述。
        
        返回：
            方法描述。
        """
        descriptions = {
            PositionSizingMethod.FIXED_FRACTIONAL: (
                "Fixed fractional sizing: Allocates fixed percentage of capital "
                "to each trade based on stop loss."
            ),
            PositionSizingMethod.KELLY_CRITERION: (
                "Kelly criterion: Maximizes long-term growth rate based on "
                "win probability and win/loss ratios."
            ),
            PositionSizingMethod.VOLATILITY_ADJUSTED: (
                "Volatility-adjusted sizing: Adjusts position size based on "
                "asset volatility to maintain constant portfolio risk."
            ),
            PositionSizingMethod.OPTIMAL_F: (
                "Optimal f (Ralph Vince): Maximizes geometric mean return "
                "based on historical trade outcomes."
            ),
            PositionSizingMethod.EQUAL_WEIGHT: (
                "Equal weight: Allocates equal capital to each position "
                "for maximum diversification."
            ),
            PositionSizingMethod.RISK_PARITY: (
                "Risk parity: Allocates risk equally across positions "
                "based on volatility and correlations."
            ),
            PositionSizingMethod.BLACK_LITTERMAN: (
                "Black-Litterman: Combines market equilibrium with "
                "subjective views to determine optimal weights."
            ),
        }
        
        return descriptions.get(self.method, "Unknown method")