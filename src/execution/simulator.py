"""Simulated trade execution for backtesting."""

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
    """Order types."""
    MARKET = "market"          # Market order
    LIMIT = "limit"            # Limit order
    STOP = "stop"              # Stop order
    STOP_LIMIT = "stop_limit"  # Stop-limit order


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"        # Order created but not sent
    SUBMITTED = "submitted"    # Order sent to exchange
    FILLED = "filled"          # Order completely filled
    PARTIALLY_FILLED = "partially_filled"  # Order partially filled
    CANCELLED = "cancelled"    # Order cancelled
    REJECTED = "rejected"      # Order rejected
    EXPIRED = "expired"        # Order expired


@dataclass
class Order:
    """Trade order."""
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
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """Check if order is active (pending or submitted)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
    
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
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
    """Simulated trade execution engine.
    
    Features:
    - Market, limit, stop, stop-limit orders
    - Slippage and commission modeling
    - Partial fills
    - Order expiration
    - Order matching simulation
    - Latency simulation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        min_commission: float = 1.0,
        slippage_model: str = "proportional",  # "proportional", "fixed", "random"
        slippage_factor: float = 0.0001,  # 0.01%
        latency_ms: int = 100,
        fill_probability: float = 1.0,
        partial_fill_enabled: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize simulated execution engine.
        
        Args:
            initial_capital: Initial trading capital.
            commission_rate: Commission rate per trade.
            min_commission: Minimum commission per trade.
            slippage_model: Slippage model type.
            slippage_factor: Slippage factor.
            latency_ms: Simulated latency in milliseconds.
            fill_probability: Probability of order fill (0-1).
            partial_fill_enabled: Whether to allow partial fills.
            seed: Random seed for reproducibility.
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
        
        # Random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Market data cache
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # Statistics
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
        """Submit a new order.
        
        Args:
            symbol: Trading symbol.
            order_type: Order type.
            side: "BUY" or "SELL".
            quantity: Order quantity.
            price: Limit price (for limit orders).
            stop_price: Stop price (for stop orders).
            time_in_force: Time in force.
            expiration: Order expiration time.
            
        Returns:
            Order ID.
        """
        # Generate order ID
        order_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Validate order
        if not self._validate_order(symbol, side, quantity, price):
            raise ValueError("Order validation failed")
        
        # Create order
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
        
        # Store order
        self.orders[order_id] = order
        
        logger.info(
            f"Order submitted: {order_id}, {symbol} {side} {quantity} "
            f"{order_type.value} @ {price if price else 'market'}"
        )
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel.
            
        Returns:
            True if cancelled successfully.
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active():
            logger.warning(f"Cannot cancel order in status: {order.status.value}")
            return False
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def update_market_data(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """Update market data for order matching.
        
        Args:
            symbol: Trading symbol.
            data: Market data (must contain 'open', 'high', 'low', 'close').
        """
        self.market_data[symbol] = data.copy()
        logger.debug(f"Market data updated for {symbol}: {len(data)} data points")
    
    def process_orders(
        self,
        timestamp: datetime,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Process all pending orders.
        
        Args:
            timestamp: Current timestamp for order processing.
            symbol: Optional symbol to process orders for.
            
        Returns:
            List of filled orders.
        """
        filled_orders = []
        
        # Get orders to process
        orders_to_process = []
        for order_id, order in self.orders.items():
            if order.status != OrderStatus.SUBMITTED:
                continue
            
            if symbol and order.symbol != symbol:
                continue
            
            # Check order expiration
            if order.expiration and timestamp > order.expiration:
                order.status = OrderStatus.EXPIRED
                logger.info(f"Order expired: {order_id}")
                continue
            
            orders_to_process.append(order)
        
        # Process each order
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
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Position dictionary.
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
        """Get portfolio summary.
        
        Returns:
            Portfolio summary dictionary.
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
        """Reset execution engine to initial state."""
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
        """Validate order before submission.
        
        Args:
            symbol: Trading symbol.
            side: "BUY" or "SELL".
            quantity: Order quantity.
            price: Order price.
            
        Returns:
            True if order is valid.
        """
        if quantity <= 0:
            logger.error("Order quantity must be positive")
            return False
        
        if side not in ["BUY", "SELL"]:
            logger.error(f"Invalid order side: {side}")
            return False
        
        # Check buying power for BUY orders
        if side == "BUY":
            # Estimate order value
            if price:
                order_value = quantity * price
            else:
                # For market orders, need current price
                # This is a simplified check
                order_value = quantity * 100  # Assume $100 per share
            
            # Add estimated commission and slippage
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
        """Process a single order.
        
        Args:
            order: Order to process.
            timestamp: Current timestamp.
            
        Returns:
            True if order was filled.
        """
        # Get market data
        if order.symbol not in self.market_data:
            logger.warning(f"No market data for {order.symbol}")
            return False
        
        market_data = self.market_data[order.symbol]
        
        # Find matching data point for timestamp
        # This is simplified - in practice, you'd match exact timestamp
        data_point = market_data.iloc[-1] if not market_data.empty else None
        
        if data_point is None:
            return False
        
        # Get market prices
        open_price = data_point.get('open', 0)
        high_price = data_point.get('high', 0)
        low_price = data_point.get('low', 0)
        close_price = data_point.get('close', 0)
        
        # Simulate latency
        self._simulate_latency()
        
        # Check if order can be filled
        fill_price, fill_quantity = self._match_order(
            order, open_price, high_price, low_price, close_price
        )
        
        if fill_quantity <= 0:
            # Order not filled
            return False
        
        # Calculate commission and slippage
        commission = self._calculate_commission(fill_price, fill_quantity)
        slippage = self._calculate_slippage(fill_price, fill_quantity, order.order_type)
        
        # Calculate effective fill price
        effective_price = fill_price
        
        if order.side == "BUY":
            effective_price += slippage
        else:  # SELL
            effective_price -= slippage
        
        # Update order
        order.filled_quantity += fill_quantity
        order.avg_fill_price = (
            (order.avg_fill_price * (order.filled_quantity - fill_quantity) +
             effective_price * fill_quantity) /
            order.filled_quantity
        )
        order.commission += commission
        order.slippage += slippage
        
        # Check if order is completely filled
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.SUBMITTED
        
        # Update portfolio
        self._update_portfolio(order, effective_price, fill_quantity, commission, timestamp)
        
        # Update statistics
        self.total_commissions += commission
        self.total_slippage += slippage
        self.total_trades += 1
        
        # Record trade
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
        
        # Move to history if filled or cancelled
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
            order: Order to match.
            open_price: Market open price.
            high_price: Market high price.
            low_price: Market low price.
            close_price: Market close price.
            
        Returns:
            Tuple of (fill_price, fill_quantity).
        """
        # Check fill probability
        if random.random() > self.fill_probability:
            return 0.0, 0.0
        
        if order.order_type == OrderType.MARKET:
            # Market orders fill at current price
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
        
        # Apply partial fills
        if self.partial_fill_enabled and fill_quantity > 0:
            # Random partial fill (0-100%)
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
            price: Trade price.
            quantity: Trade quantity.
            
        Returns:
            Commission amount.
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
            price: Trade price.
            quantity: Trade quantity.
            order_type: Order type.
            
        Returns:
            Slippage amount.
        """
        if self.slippage_model == "proportional":
            slippage = price * self.slippage_factor
        
        elif self.slippage_model == "fixed":
            slippage = self.slippage_factor
        
        elif self.slippage_model == "random":
            # Random slippage between 0 and 2*slippage_factor
            slippage = price * random.uniform(0, 2 * self.slippage_factor)
        
        else:
            slippage = 0.0
        
        # Adjust for order type
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
            order: Filled order.
            price: Fill price.
            quantity: Fill quantity.
            commission: Commission paid.
            timestamp: Fill timestamp.
        """
        symbol = order.symbol
        
        # Initialize position if not exists
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
            # Update position for BUY
            total_cost = price * quantity + commission
            
            if position["quantity"] == 0:
                # New position
                position["quantity"] = quantity
                position["avg_price"] = price
                position["entry_time"] = timestamp
            else:
                # Add to existing position
                old_quantity = position["quantity"]
                old_avg_price = position["avg_price"]
                old_value = old_quantity * old_avg_price
                
                new_quantity = old_quantity + quantity
                new_avg_price = (old_value + total_cost) / new_quantity
                
                position["quantity"] = new_quantity
                position["avg_price"] = new_avg_price
            
            # Update capital
            self.capital -= total_cost
        
        else:  # SELL
            # Update position for SELL
            if position["quantity"] == 0:
                # Short selling (not implemented in this simplified version)
                logger.warning(f"Short selling not implemented for {symbol}")
                return
            
            # Calculate P&L
            sell_value = price * quantity - commission
            cost_basis = position["avg_price"] * quantity
            pnl = sell_value - cost_basis
            
            # Update position
            position["quantity"] -= quantity
            
            # Update realized P&L
            position["realized_pnl"] += pnl
            
            # Update capital
            self.capital += sell_value
            
            # Remove position if quantity is zero
            if position["quantity"] == 0:
                position["avg_price"] = 0
                position["entry_time"] = None
        
        # Update market value and unrealized P&L
        self._update_position_value(symbol)
    
    def _update_position_value(self, symbol: str) -> None:
        """Update market value and unrealized P&L for a position.
        
        Args:
            symbol: Trading symbol.
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
        """Simulate network and processing latency."""
        # In a real simulation, you might add a time delay
        # For now, just log it
        logger.debug(f"Simulated latency: {self.latency_ms}ms")
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status.
        
        Args:
            order_id: Order ID.
            
        Returns:
            Order status or None if not found.
        """
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders.
        
        Args:
            symbol: Optional symbol filter.
            
        Returns:
            List of open orders.
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
        """Get trade history with filters.
        
        Args:
            symbol: Symbol filter.
            start_date: Start date filter.
            end_date: End date filter.
            
        Returns:
            Filtered trade history.
        """
        filtered = self.trade_history
        
        if symbol:
            filtered = [t for t in filtered if t["symbol"] == symbol]
        
        if start_date:
            filtered = [t for t in filtered if t["timestamp"] >= start_date]
        
        if end_date:
            filtered = [t for t in filtered if t["timestamp"] <= end_date]
        
        return filtered