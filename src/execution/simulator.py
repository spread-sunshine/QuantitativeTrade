"""用于回测的模拟交易执行引擎。"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
import random

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OrderType(Enum):
    """订单类型。"""
    MARKET = "market"          # 市价订单
    LIMIT = "limit"            # 限价订单
    STOP = "stop"              # 止损订单
    STOP_LIMIT = "stop_limit"  # 止损限价订单


class OrderStatus(Enum):
    """订单状态。"""
    PENDING = "pending"        # 订单已创建但未发送
    SUBMITTED = "submitted"    # 订单已发送至交易所
    FILLED = "filled"          # 订单完全成交
    PARTIALLY_FILLED = "partially_filled"  # 订单部分成交
    CANCELLED = "cancelled"    # 订单已取消
    REJECTED = "rejected"      # Order rejected
    EXPIRED = "expired"        # 订单已过期


@dataclass
class Order:
    """交易订单。"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "BUY" or "SELL"
    quantity: float
    price: Optional[float] = None  # Limit price for limit orders
    stop_price: Optional[float] = None  # Stop price for stop orders
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    parent_order_id: Optional[str] = None  # For OCO orders
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    expiration: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def is_filled(self) -> bool:
        """检查订单是否完全成交。"""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """检查订单是否处于活跃状态（待处理或已提交）。"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
    
    def remaining_quantity(self) -> float:
        """获取剩余待成交量。"""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """将订单转换为字典。"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "parent_order_id": self.parent_order_id,
            "time_in_force": self.time_in_force,
            "expiration": self.expiration.isoformat() if self.expiration else None,
        }


class SimulatedExecution:
    """模拟交易执行引擎。
    
    功能：
    - 市价、限价、止损、止损限价订单
    - 滑点和佣金建模
    - 部分成交
    - 订单过期
    - 订单匹配模拟
    - 延迟模拟
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        min_commission: float = 1.0,
        slippage_model: str = "proportional",  # 滑点模型类型："proportional"、"fixed"、"random"
        slippage_factor: float = 0.0001,  # 0.01%
        latency_ms: int = 100,
        fill_probability: float = 1.0,
        partial_fill_enabled: bool = True,
        seed: Optional[int] = None,
    ):
        """初始化模拟执行引擎。
        
        Args:
            initial_capital: 初始交易资金。
            commission_rate: 每笔交易的佣金费率。
            min_commission: 每笔交易的最低佣金。
            slippage_model: 滑点模型类型。
            slippage_factor: 滑点因子。
            latency_ms: 模拟延迟（毫秒）。
            fill_probability: 订单成交概率（0-1）。
            partial_fill_enabled: 是否允许部分成交。
            seed: 用于可重现性的随机种子。
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_model = slippage_model
        self.slippage_factor = slippage_factor
        self.latency_ms = latency_ms
        self.fill_probability = fill_probability
        self.partial_fill_enabled = partial_fill_enabled
        
        # 随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 订单管理
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # 持仓跟踪
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 市场数据缓存
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # 统计数据
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        
        logger.info(
            f"SimulatedExecution initialized with capital={initial_capital}, "
            f"commission={commission_rate}, slippage={slippage_model}"
        )
    
    def submit_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        expiration: Optional[datetime] = None,
    ) -> str:
        """提交新订单。
        
        Args:
            symbol: 交易代码。
            order_type: 订单类型。
            side: "BUY" 或 "SELL"。
            quantity: 订单数量。
            price: 限价价格（用于限价订单）。
            stop_price: 止损价格（用于止损订单）。
            time_in_force: 订单有效时间。
            expiration: 订单过期时间。
            
        Returns:
            订单 ID。
        """
        # 生成订单 ID
        order_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 验证订单
        if not self._validate_order(symbol, side, quantity, price):
            raise ValueError("Order validation failed")
        
        # 创建订单
        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=datetime.now(),
            status=OrderStatus.SUBMITTED,
            time_in_force=time_in_force,
            expiration=expiration,
        )
        
        # 存储订单
        self.orders[order_id] = order
        
        logger.info(
            f"Order submitted: {order_id}, {symbol} {side} {quantity} "
            f"{order_type.value} @ {price if price else 'market'}"
        )
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """取消现有订单。
        
        Args:
            order_id: 要取消的订单 ID。
            
        Returns:
            如果取消成功则为 True。
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active():
            logger.warning(f"Cannot cancel order in status: {order.status.value}")
            return False
        
        # 更新订单 status
        order.status = OrderStatus.CANCELLED
        
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def update_market_data(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """更新市场数据以进行订单匹配。
        
        Args:
            symbol: 交易代码。
            data: 市场数据（必须包含 'open'、'high'、'low'、'close'）。
        """
        self.market_data[symbol] = data.copy()
        logger.debug(f"Market data updated for {symbol}: {len(data)} data points")
    
    def process_orders(
        self,
        timestamp: datetime,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """处理所有待处理订单。
        
        Args:
            timestamp: 订单处理的当前时间戳。
            symbol: 可选，指定要处理订单的代码。
            
        Returns:
            已成交订单列表。
        """
        filled_orders = []
        
        # 获取待处理的订单
        orders_to_process = []
        for order_id, order in self.orders.items():
            if order.status != OrderStatus.SUBMITTED:
                continue
            
            if symbol and order.symbol != symbol:
                continue
            
            # 检查订单过期
            if order.expiration and timestamp > order.expiration:
                order.status = OrderStatus.EXPIRED
                logger.info(f"Order expired: {order_id}")
                continue
            
            orders_to_process.append(order)
        
        # 处理每个订单
        for order in orders_to_process:
            try:
                filled = self._process_order(order, timestamp)
                if filled:
                    filled_orders.append(order.to_dict())
            except Exception as e:
                logger.error(f"Error processing order {order.order_id}: {e}")
                order.status = OrderStatus.REJECTED
        
        return filled_orders
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """获取指定代码的当前持仓。
        
        Args:
            symbol: 交易代码。
            
        Returns:
            持仓字典。
        """
        return self.positions.get(symbol, {
            "symbol": symbol,
            "quantity": 0,
            "avg_price": 0,
            "market_value": 0,
            "unrealized_pnl": 0,
            "realized_pnl": 0,
        })
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要。
        
        Returns:
            投资组合摘要字典。
        """
        total_value = self.capital
        total_positions_value = 0.0
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            total_positions_value += position.get("market_value", 0)
            total_unrealized_pnl += position.get("unrealized_pnl", 0)
            total_realized_pnl += position.get("realized_pnl", 0)
        
        total_value += total_positions_value
        
        summary = {
            "capital": self.capital,
            "positions_value": total_positions_value,
            "total_value": total_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "total_commissions": self.total_commissions,
            "total_slippage": self.total_slippage,
            "total_trades": self.total_trades,
            "num_positions": len(self.positions),
            "num_orders": len(self.orders),
        }
        
        return summary
    
    def reset(self) -> None:
        """将执行引擎重置为初始状态。"""
        self.capital = self.initial_capital
        self.orders.clear()
        self.order_history.clear()
        self.trade_history.clear()
        self.positions.clear()
        self.market_data.clear()
        
        self.total_commissions = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        
        logger.info("Execution engine reset")
    
    def _validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
    ) -> bool:
        """在提交前验证订单。
        
        Args:
            symbol: 交易代码。
            side: "BUY" 或 "SELL"。
            quantity: 订单数量。
            price: 订单价格。
            
        Returns:
            如果订单有效则为 True。
        """
        if quantity <= 0:
            logger.error("Order quantity must be positive")
            return False
        
        if side not in ["BUY", "SELL"]:
            logger.error(f"Invalid order side: {side}")
            return False
        
        # 检查买单的购买力
        if side == "BUY":
            # 估算订单价值
            if price:
                order_value = quantity * price
            else:
                # 对于市价订单，需要当前价格
                # 这是一个简化的检查
                order_value = quantity * 100  # 假设每股 100 美元
            
            # 加上估算的佣金和滑点
            commission = max(order_value * self.commission_rate, self.min_commission)
            slippage = order_value * self.slippage_factor
            
            total_cost = order_value + commission + slippage
            
            if total_cost > self.capital:
                logger.error(f"Insufficient capital: need {total_cost:.2f}, have {self.capital:.2f}")
                return False
        
        return True
    
    def _process_order(
        self,
        order: Order,
        timestamp: datetime,
    ) -> bool:
        """处理单个订单。
        
        Args:
            order: 要处理的订单。
            timestamp: 当前时间戳。
            
        Returns:
            如果订单已成交则为 True。
        """
        # 获取市场数据
        if order.symbol not in self.market_data:
            logger.warning(f"No market data for {order.symbol}")
            return False
        
        market_data = self.market_data[order.symbol]
        
        # 查找与时间戳匹配的数据点
        # 这是简化的 - 实际上，您需要匹配精确的时间戳
        data_point = market_data.iloc[-1] if not market_data.empty else None
        
        if data_point is None:
            return False
        
        # 获取市场价格
        open_price = data_point.get('open', 0)
        high_price = data_point.get('high', 0)
        low_price = data_point.get('low', 0)
        close_price = data_point.get('close', 0)
        
        # 模拟延迟
        self._simulate_latency()
        
        # 检查订单是否可以成交
        fill_price, fill_quantity = self._match_order(
            order, open_price, high_price, low_price, close_price
        )
        
        if fill_quantity <= 0:
            # 订单未成交
            return False
        
        # 计算佣金和滑点
        commission = self._calculate_commission(fill_price, fill_quantity)
        slippage = self._calculate_slippage(fill_price, fill_quantity, order.order_type)
        
        # 计算有效成交价格
        effective_price = fill_price
        
        if order.side == "BUY":
            effective_price += slippage
        else:  # SELL
            effective_price -= slippage
        
        # 更新订单
        order.filled_quantity += fill_quantity
        order.avg_fill_price = (
            (order.avg_fill_price * (order.filled_quantity - fill_quantity) +
             effective_price * fill_quantity) /
            order.filled_quantity
        )
        order.commission += commission
        order.slippage += slippage
        
        # 检查订单是否完全成交
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.SUBMITTED
        
        # 更新投资组合
        self._update_portfolio(order, effective_price, fill_quantity, commission, timestamp)
        
        # 更新统计数据
        self.total_commissions += commission
        self.total_slippage += slippage
        self.total_trades += 1
        
        # 记录交易
        trade = {
            "timestamp": timestamp,
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": fill_quantity,
            "price": effective_price,
            "commission": commission,
            "slippage": slippage,
            "order_type": order.order_type.value,
        }
        self.trade_history.append(trade)
        
        # 如果订单已成交或取消，则移至历史记录
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.order_history.append(order)
            self.orders.pop(order.order_id, None)
        
        logger.info(
            f"Order filled: {order.order_id}, {order.symbol} {order.side} "
            f"{fill_quantity}/{order.quantity} @ {effective_price:.2f}, "
            f"commission={commission:.2f}, slippage={slippage:.4f}"
        )
        
        return True
    
    def _match_order(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> Tuple[float, float]:
        """Match order with market prices.

        Args:
            order: 要匹配的订单。
            open_price: 市场开盘价。
            high_price: 市场最高价。
            low_price: 市场最低价。
            close_price: 市场收盘价。

        Returns:
            元组（成交价格，成交量）。
        """
        # 检查成交概率
        if random.random() > self.fill_probability:
            return 0.0, 0.0
        
        if order.order_type == OrderType.MARKET:
            # 市价订单以当前价格成交
            fill_price = close_price
            fill_quantity = order.remaining_quantity()
            
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill if price is better than limit
            if order.side == "BUY":
                # Buy limit: fill if market price <= limit price
                if low_price <= order.price:
                    fill_price = min(order.price, close_price)
                    fill_quantity = order.remaining_quantity()
                else:
                    return 0.0, 0.0
            else:  # SELL
                # Sell limit: fill if market price >= limit price
                if high_price >= order.price:
                    fill_price = max(order.price, close_price)
                    fill_quantity = order.remaining_quantity()
                else:
                    return 0.0, 0.0
        
        elif order.order_type == OrderType.STOP:
            # Stop orders become market orders when stop price is hit
            if order.side == "BUY":
                # Buy stop: trigger if market price >= stop price
                if high_price >= order.stop_price:
                    fill_price = max(order.stop_price, close_price)
                    fill_quantity = order.remaining_quantity()
                else:
                    return 0.0, 0.0
            else:  # SELL
                # Sell stop: trigger if market price <= stop price
                if low_price <= order.stop_price:
                    fill_price = min(order.stop_price, close_price)
                    fill_quantity = order.remaining_quantity()
                else:
                    return 0.0, 0.0
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit orders become limit orders when stop price is hit
            if order.side == "BUY":
                # Buy stop-limit: trigger if market price >= stop price
                if high_price >= order.stop_price:
                    # Then fill as limit order
                    if low_price <= order.price:
                        fill_price = min(order.price, close_price)
                        fill_quantity = order.remaining_quantity()
                    else:
                        return 0.0, 0.0
                else:
                    return 0.0, 0.0
            else:  # SELL
                # Sell stop-limit: trigger if market price <= stop price
                if low_price <= order.stop_price:
                    # Then fill as limit order
                    if high_price >= order.price:
                        fill_price = max(order.price, close_price)
                        fill_quantity = order.remaining_quantity()
                    else:
                        return 0.0, 0.0
                else:
                    return 0.0, 0.0
        
        else:
            return 0.0, 0.0
        
        # 应用部分成交
        if self.partial_fill_enabled and fill_quantity > 0:
            # 随机部分成交（0-100%）
            fill_percentage = random.uniform(0.5, 1.0)
            fill_quantity = fill_quantity * fill_percentage
        
        return fill_price, fill_quantity
    
    def _calculate_commission(
        self,
        price: float,
        quantity: float,
    ) -> float:
        """Calculate commission for a trade.

        Args:
            price: 交易价格。
            quantity: 交易数量。

        Returns:
            佣金金额。
        """
        trade_value = price * quantity
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def _calculate_slippage(
        self,
        price: float,
        quantity: float,
        order_type: OrderType,
    ) -> float:
        """Calculate slippage for a trade.

        Args:
            price: 交易价格。
            quantity: 交易数量。
            order_type: 订单类型。

        Returns:
            滑点金额。
        """
        if self.slippage_model == "proportional":
            slippage = price * self.slippage_factor
        
        elif self.slippage_model == "fixed":
            slippage = self.slippage_factor
        
        elif self.slippage_model == "random":
            # 随机滑点，范围0到2*slippage_factor
            slippage = price * random.uniform(0, 2 * self.slippage_factor)
        
        else:
            slippage = 0.0

        # 根据订单类型调整
        if order_type == OrderType.MARKET:
            slippage *= 1.0  # Market orders have normal slippage
        elif order_type == OrderType.LIMIT:
            slippage *= 0.5  # Limit orders have less slippage
        elif order_type == OrderType.STOP:
            slippage *= 1.5  # Stop orders have more slippage
        elif order_type == OrderType.STOP_LIMIT:
            slippage *= 1.0  # Stop-limit orders have normal slippage
        
        return slippage
    
    def _update_portfolio(
        self,
        order: Order,
        price: float,
        quantity: float,
        commission: float,
        timestamp: datetime,
    ) -> None:
        """Update portfolio after order fill.

        Args:
            order: 已成交的订单。
            price: 成交价格。
            quantity: 成交数量。
            commission: 支付的佣金。
            timestamp: 成交时间戳。
        """
        symbol = order.symbol

        # 如果持仓不存在则初始化
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0,
                "avg_price": 0,
                "market_value": 0,
                "unrealized_pnl": 0,
                "realized_pnl": 0,
                "entry_time": timestamp,
            }
        
        position = self.positions[symbol]
        
        if order.side == "BUY":
            # 更新买入持仓
            total_cost = price * quantity + commission

            if position["quantity"] == 0:
                # 新建持仓
                position["quantity"] = quantity
                position["avg_price"] = price
                position["entry_time"] = timestamp
            else:
                # 添加到现有持仓
                old_quantity = position["quantity"]
                old_avg_price = position["avg_price"]
                old_value = old_quantity * old_avg_price
                
                new_quantity = old_quantity + quantity
                new_avg_price = (old_value + total_cost) / new_quantity
                
                position["quantity"] = new_quantity
                position["avg_price"] = new_avg_price
            
            # 更新资金
            self.capital -= total_cost

        else:  # SELL
            # 更新卖出持仓
            if position["quantity"] == 0:
                # 做空（此简化版本未实现）
                logger.warning(f"Short selling not implemented for {symbol}")
                return
            
            # 计算盈亏
            sell_value = price * quantity - commission
            cost_basis = position["avg_price"] * quantity
            pnl = sell_value - cost_basis
            
            # 更新持仓
            position["quantity"] -= quantity

            # 更新已实现盈亏
            position["realized_pnl"] += pnl

            # 更新资金
            self.capital += sell_value

            # 如果数量为零则移除持仓
            if position["quantity"] == 0:
                position["avg_price"] = 0
                position["entry_time"] = None
        
        # 更新市值和未实现盈亏
        self._update_position_value(symbol)
    
    def _update_position_value(self, symbol: str) -> None:
        """Update market value and unrealized P&L for a position.
        
        Args:
            symbol: 交易代码。
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        if symbol in self.market_data and not self.market_data[symbol].empty:
            current_price = self.market_data[symbol].iloc[-1]['close']
        else:
            current_price = position.get("avg_price", 0)
        
        quantity = position["quantity"]
        avg_price = position["avg_price"]
        
        position["market_value"] = quantity * current_price
        
        if quantity > 0:
            position["unrealized_pnl"] = (current_price - avg_price) * quantity
        else:
            position["unrealized_pnl"] = 0.0
    
    def _simulate_latency(self) -> None:
        """模拟网络和处理延迟。"""
        # In a real simulation, you might add a time delay
        # For now, just log it
        logger.debug(f"Simulated latency: {self.latency_ms}ms")
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status.

        Args:
            order_id: 订单ID。

        Returns:
            订单状态或未找到时为None。
        """
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取所有未完成订单。
        
        Args:
            symbol: 可选的代码过滤器。

        Returns:
            未完成订单列表。
        """
        open_orders = []
        for order in self.orders.values():
            if order.is_active():
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order)
        return open_orders
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """获取带过滤条件的交易历史。
        
        Args:
            symbol: 代码过滤器。
            start_date: 开始日期过滤器。
            end_date: 结束日期过滤器。

        Returns:
            过滤后的交易历史。
        """
        filtered = self.trade_history
        
        if symbol:
            filtered = [t for t in filtered if t["symbol"] == symbol]
        
        if start_date:
            filtered = [t for t in filtered if t["timestamp"] >= start_date]
        
        if end_date:
            filtered = [t for t in filtered if t["timestamp"] <= end_date]
        
        return filtered