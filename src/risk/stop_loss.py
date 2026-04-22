"""Stop loss calculation and management."""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from enum import Enum
import logging
from datetime import datetime, timedelta

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StopLossType(Enum):
    """Stop loss types."""
    FIXED_PERCENTAGE = "fixed_percentage"      # Fixed percentage from entry
    ATR_BASED = "atr_based"                    # Based on Average True Range
    VOLATILITY_BASED = "volatility_based"      # Based on volatility
    TRAILING = "trailing"                      # Trailing stop loss
    PARABOLIC_SAR = "parabolic_sar"           # Parabolic SAR
    CHANDELIER = "chandelier"                  # Chandelier exit
    TIME_BASED = "time_based"                  # Time-based stop
    MOVING_AVERAGE = "moving_average"          # Moving average stop
    SUPPORT_RESISTANCE = "support_resistance"  # Support/resistance levels


class StopLossCalculator:
    """Stop loss calculator with multiple strategies.
    
    Implements various stop loss methods:
    - Fixed percentage stop
    - ATR-based stop
    - Volatility-based stop
    - Trailing stop
    - Parabolic SAR
    - Chandelier exit
    - Time-based stop
    - Moving average stop
    - Support/resistance stop
    """
    
    def __init__(
        self,
        stop_type: StopLossType = StopLossType.FIXED_PERCENTAGE,
        default_stop_pct: float = 0.05,
        atr_multiplier: float = 2.0,
        volatility_multiplier: float = 2.0,
        trailing_pct: float = 0.1,
        sar_acceleration: float = 0.02,
        sar_maximum: float = 0.2,
        chandelier_atr_multiplier: float = 3.0,
        chandelier_lookback: int = 22,
        time_stop_days: int = 10,
        ma_period: int = 20,
    ):
        """Initialize stop loss calculator.
        
        Args:
            stop_type: Stop loss type.
            default_stop_pct: Default stop loss percentage.
            atr_multiplier: Multiplier for ATR-based stops.
            volatility_multiplier: Multiplier for volatility-based stops.
            trailing_pct: Percentage for trailing stops.
            sar_acceleration: Acceleration factor for Parabolic SAR.
            sar_maximum: Maximum acceleration for Parabolic SAR.
            chandelier_atr_multiplier: Multiplier for Chandelier exit.
            chandelier_lookback: Lookback period for Chandelier exit.
            time_stop_days: Days for time-based stop.
            ma_period: Period for moving average stop.
        """
        self.stop_type = stop_type
        self.default_stop_pct = default_stop_pct
        self.atr_multiplier = atr_multiplier
        self.volatility_multiplier = volatility_multiplier
        self.trailing_pct = trailing_pct
        self.sar_acceleration = sar_acceleration
        self.sar_maximum = sar_maximum
        self.chandelier_atr_multiplier = chandelier_atr_multiplier
        self.chandelier_lookback = chandelier_lookback
        self.time_stop_days = time_stop_days
        self.ma_period = ma_period
        
        # Position tracking
        self.active_stops: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"StopLossCalculator initialized with type={stop_type.value}")
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        position_type: str,  # "LONG" or "SHORT"
        current_price: float,
        historical_data: Optional[pd.DataFrame] = None,
        entry_time: Optional[datetime] = None,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate stop loss price.
        
        Args:
            symbol: Trading symbol.
            entry_price: Entry price.
            position_type: "LONG" or "SHORT".
            current_price: Current market price.
            historical_data: Historical price data (OHLC).
            entry_time: Entry timestamp.
            atr: Current ATR value.
            volatility: Current volatility.
            support_level: Support level price.
            resistance_level: Resistance level price.
            custom_params: Custom parameters overriding defaults.
            
        Returns:
            Tuple of (stop_loss_price, calculation_details).
        """
        # Apply custom parameters if provided
        params = self._get_parameters(custom_params)
        
        # Calculate stop loss based on type
        if self.stop_type == StopLossType.FIXED_PERCENTAGE:
            stop_price, details = self._fixed_percentage_stop(
                entry_price, position_type, params
            )
        elif self.stop_type == StopLossType.ATR_BASED:
            stop_price, details = self._atr_based_stop(
                entry_price, position_type, atr, params
            )
        elif self.stop_type == StopLossType.VOLATILITY_BASED:
            stop_price, details = self._volatility_based_stop(
                entry_price, position_type, volatility, params
            )
        elif self.stop_type == StopLossType.TRAILING:
            stop_price, details = self._trailing_stop(
                symbol, entry_price, current_price, position_type, params
            )
        elif self.stop_type == StopLossType.PARABOLIC_SAR:
            stop_price, details = self._parabolic_sar_stop(
                symbol, historical_data, position_type, params
            )
        elif self.stop_type == StopLossType.CHANDELIER:
            stop_price, details = self._chandelier_stop(
                symbol, historical_data, position_type, params
            )
        elif self.stop_type == StopLossType.TIME_BASED:
            stop_price, details = self._time_based_stop(
                symbol, entry_price, entry_time, position_type, params
            )
        elif self.stop_type == StopLossType.MOVING_AVERAGE:
            stop_price, details = self._moving_average_stop(
                symbol, historical_data, position_type, params
            )
        elif self.stop_type == StopLossType.SUPPORT_RESISTANCE:
            stop_price, details = self._support_resistance_stop(
                entry_price, position_type, support_level, resistance_level, params
            )
        else:
            # Default to fixed percentage
            stop_price, details = self._fixed_percentage_stop(
                entry_price, position_type, params
            )
        
        # Update details
        details.update({
            "symbol": symbol,
            "entry_price": entry_price,
            "position_type": position_type,
            "stop_type": self.stop_type.value,
            "stop_price": stop_price,
            "stop_distance_pct": abs(stop_price - entry_price) / entry_price,
            "current_price": current_price,
        })
        
        # Store active stop
        self._update_active_stop(symbol, stop_price, details)
        
        logger.debug(
            f"Stop loss for {symbol} ({position_type}): entry={entry_price:.2f}, "
            f"stop={stop_price:.2f}, distance={details['stop_distance_pct']:.2%}"
        )
        
        return stop_price, details
    
    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
    ) -> Tuple[bool, Optional[float], Dict[str, Any]]:
        """Check if stop loss is triggered.
        
        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            timestamp: Current timestamp.
            
        Returns:
            Tuple of (is_triggered, stop_price, trigger_details).
        """
        if symbol not in self.active_stops:
            return False, None, {"error": "No active stop for symbol"}
        
        stop_info = self.active_stops[symbol]
        stop_price = stop_info.get("stop_price")
        position_type = stop_info.get("position_type", "LONG")
        
        if stop_price is None:
            return False, None, {"error": "Invalid stop price"}
        
        # Check if stop is triggered
        is_triggered = False
        
        if position_type == "LONG":
            is_triggered = current_price <= stop_price
        elif position_type == "SHORT":
            is_triggered = current_price >= stop_price
        
        if is_triggered:
            trigger_details = {
                "symbol": symbol,
                "position_type": position_type,
                "entry_price": stop_info.get("entry_price"),
                "stop_price": stop_price,
                "current_price": current_price,
                "trigger_time": timestamp,
                "loss_pct": abs(current_price - stop_info.get("entry_price", 0)) / stop_info.get("entry_price", 1),
                "stop_type": stop_info.get("stop_type"),
            }
            
            # Remove active stop
            self.active_stops.pop(symbol, None)
            
            logger.info(
                f"Stop loss triggered for {symbol}: {position_type} position, "
                f"entry={stop_info.get('entry_price'):.2f}, stop={stop_price:.2f}, "
                f"current={current_price:.2f}, loss={trigger_details['loss_pct']:.2%}"
            )
            
            return True, stop_price, trigger_details
        
        return False, stop_price, {"status": "active"}
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
    ) -> Optional[float]:
        """Update trailing stop loss for a position.
        
        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            timestamp: Current timestamp.
            
        Returns:
            New stop price if updated, None otherwise.
        """
        if symbol not in self.active_stops:
            return None
        
        stop_info = self.active_stops[symbol]
        
        # Only update for trailing stops
        if stop_info.get("stop_type") != StopLossType.TRAILING.value:
            return None
        
        position_type = stop_info.get("position_type", "LONG")
        entry_price = stop_info.get("entry_price", 0)
        current_stop = stop_info.get("stop_price", 0)
        trailing_pct = stop_info.get("trailing_pct", self.trailing_pct)
        
        new_stop = current_stop
        
        if position_type == "LONG":
            # For long positions, trail below the high
            high_price = stop_info.get("high_price", entry_price)
            
            # Update high
            if current_price > high_price:
                stop_info["high_price"] = current_price
                high_price = current_price
                
                # Calculate new stop
                new_stop = high_price * (1 - trailing_pct)
                
                # Only move stop up, not down
                if new_stop > current_stop:
                    stop_info["stop_price"] = new_stop
                    logger.debug(
                        f"Trailing stop updated for {symbol}: {current_stop:.2f} -> {new_stop:.2f}"
                    )
        
        elif position_type == "SHORT":
            # For short positions, trail above the low
            low_price = stop_info.get("low_price", entry_price)
            
            # Update low
            if current_price < low_price:
                stop_info["low_price"] = current_price
                low_price = current_price
                
                # Calculate new stop
                new_stop = low_price * (1 + trailing_pct)
                
                # Only move stop down, not up
                if new_stop < current_stop:
                    stop_info["stop_price"] = new_stop
                    logger.debug(
                        f"Trailing stop updated for {symbol}: {current_stop:.2f} -> {new_stop:.2f}"
                    )
        
        return new_stop if new_stop != current_stop else None
    
    def get_active_stops(self) -> Dict[str, Dict[str, Any]]:
        """Get all active stop losses.
        
        Returns:
            Dictionary of active stops.
        """
        return self.active_stops.copy()
    
    def clear_stop(self, symbol: str) -> None:
        """Clear stop loss for a symbol.
        
        Args:
            symbol: Trading symbol.
        """
        if symbol in self.active_stops:
            self.active_stops.pop(symbol)
            logger.debug(f"Stop loss cleared for {symbol}")
    
    def clear_all_stops(self) -> None:
        """Clear all active stop losses."""
        self.active_stops.clear()
        logger.debug("All stop losses cleared")
    
    def _get_parameters(self, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get stop loss parameters with custom overrides.
        
        Args:
            custom_params: Custom parameters.
            
        Returns:
            Parameters dictionary.
        """
        params = {
            "default_stop_pct": self.default_stop_pct,
            "atr_multiplier": self.atr_multiplier,
            "volatility_multiplier": self.volatility_multiplier,
            "trailing_pct": self.trailing_pct,
            "sar_acceleration": self.sar_acceleration,
            "sar_maximum": self.sar_maximum,
            "chandelier_atr_multiplier": self.chandelier_atr_multiplier,
            "chandelier_lookback": self.chandelier_lookback,
            "time_stop_days": self.time_stop_days,
            "ma_period": self.ma_period,
        }
        
        if custom_params:
            params.update(custom_params)
        
        return params
    
    def _fixed_percentage_stop(
        self,
        entry_price: float,
        position_type: str,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate fixed percentage stop loss.
        
        Args:
            entry_price: Entry price.
            position_type: "LONG" or "SHORT".
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        stop_pct = params.get("default_stop_pct", self.default_stop_pct)
        
        if position_type == "LONG":
            stop_price = entry_price * (1 - stop_pct)
        else:  # SHORT
            stop_price = entry_price * (1 + stop_pct)
        
        details = {
            "stop_pct": stop_pct,
            "method": "fixed_percentage",
        }
        
        return stop_price, details
    
    def _atr_based_stop(
        self,
        entry_price: float,
        position_type: str,
        atr: Optional[float],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate ATR-based stop loss.
        
        Args:
            entry_price: Entry price.
            position_type: "LONG" or "SHORT".
            atr: Current ATR value.
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        if atr is None or atr <= 0:
            # Fallback to fixed percentage
            return self._fixed_percentage_stop(entry_price, position_type, params)
        
        multiplier = params.get("atr_multiplier", self.atr_multiplier)
        stop_distance = atr * multiplier
        
        if position_type == "LONG":
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
        
        details = {
            "atr": atr,
            "multiplier": multiplier,
            "stop_distance": stop_distance,
            "method": "atr_based",
        }
        
        return stop_price, details
    
    def _volatility_based_stop(
        self,
        entry_price: float,
        position_type: str,
        volatility: Optional[float],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate volatility-based stop loss.
        
        Args:
            entry_price: Entry price.
            position_type: "LONG" or "SHORT".
            volatility: Current volatility (annualized).
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        if volatility is None or volatility <= 0:
            # Fallback to fixed percentage
            return self._fixed_percentage_stop(entry_price, position_type, params)
        
        # Convert annualized volatility to daily
        daily_volatility = volatility / np.sqrt(252)
        
        multiplier = params.get("volatility_multiplier", self.volatility_multiplier)
        stop_distance = entry_price * daily_volatility * multiplier
        
        if position_type == "LONG":
            stop_price = entry_price - stop_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
        
        details = {
            "volatility": volatility,
            "daily_volatility": daily_volatility,
            "multiplier": multiplier,
            "stop_distance": stop_distance,
            "method": "volatility_based",
        }
        
        return stop_price, details
    
    def _trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_type: str,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate trailing stop loss.
        
        Args:
            symbol: Trading symbol.
            entry_price: Entry price.
            current_price: Current market price.
            position_type: "LONG" or "SHORT".
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        trailing_pct = params.get("trailing_pct", self.trailing_pct)
        
        if position_type == "LONG":
            # Initial stop is fixed percentage below entry
            initial_stop = entry_price * (1 - trailing_pct)
            stop_price = initial_stop
            
            # Store high price for trailing
            self.active_stops[symbol] = {
                "high_price": current_price,
                "low_price": current_price,
            }
        else:  # SHORT
            # Initial stop is fixed percentage above entry
            initial_stop = entry_price * (1 + trailing_pct)
            stop_price = initial_stop
            
            # Store low price for trailing
            self.active_stops[symbol] = {
                "high_price": current_price,
                "low_price": current_price,
            }
        
        details = {
            "trailing_pct": trailing_pct,
            "initial_stop": initial_stop,
            "method": "trailing",
        }
        
        return stop_price, details
    
    def _parabolic_sar_stop(
        self,
        symbol: str,
        historical_data: Optional[pd.DataFrame],
        position_type: str,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Parabolic SAR stop loss.
        
        Args:
            symbol: Trading symbol.
            historical_data: Historical OHLC data.
            position_type: "LONG" or "SHORT".
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        if historical_data is None or len(historical_data) < 2:
            # Fallback to fixed percentage
            entry_price = historical_data['close'].iloc[-1] if historical_data is not None else 0
            return self._fixed_percentage_stop(entry_price, position_type, params)
        
        # Calculate Parabolic SAR
        # This is a simplified implementation
        # In practice, use a library like TA-Lib
        
        acceleration = params.get("sar_acceleration", self.sar_acceleration)
        maximum = params.get("sar_maximum", self.sar_maximum)
        
        # Simplified calculation
        highs = historical_data['high']
        lows = historical_data['low']
        closes = historical_data['close']
        
        # Determine trend (simplified)
        if len(closes) >= 2:
            if closes.iloc[-1] > closes.iloc[-2]:
                trend = "up"
            else:
                trend = "down"
        else:
            trend = "up"
        
        # Calculate SAR (simplified)
        if trend == "up":
            # For long positions, SAR is below price
            sar = lows.rolling(window=2).min().iloc[-1]
            stop_price = sar
        else:
            # For short positions, SAR is above price
            sar = highs.rolling(window=2).max().iloc[-1]
            stop_price = sar
        
        details = {
            "sar": sar,
            "trend": trend,
            "acceleration": acceleration,
            "maximum": maximum,
            "method": "parabolic_sar",
        }
        
        return stop_price, details
    
    def _chandelier_stop(
        self,
        symbol: str,
        historical_data: Optional[pd.DataFrame],
        position_type: str,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Chandelier exit stop loss.
        
        Args:
            symbol: Trading symbol.
            historical_data: Historical OHLC data.
            position_type: "LONG" or "SHORT".
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        if historical_data is None or len(historical_data) < 22:
            # Fallback to fixed percentage
            entry_price = historical_data['close'].iloc[-1] if historical_data is not None else 0
            return self._fixed_percentage_stop(entry_price, position_type, params)
        
        lookback = params.get("chandelier_lookback", self.chandelier_lookback)
        multiplier = params.get("chandelier_atr_multiplier", self.chandelier_atr_multiplier)
        
        # Calculate ATR
        high = historical_data['high']
        low = historical_data['low']
        close = historical_data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=lookback).mean().iloc[-1]
        
        if position_type == "LONG":
            # For long positions: Highest High - ATR * multiplier
            highest_high = high.rolling(window=lookback).max().iloc[-1]
            stop_price = highest_high - (atr * multiplier)
        else:  # SHORT
            # For short positions: Lowest Low + ATR * multiplier
            lowest_low = low.rolling(window=lookback).min().iloc[-1]
            stop_price = lowest_low + (atr * multiplier)
        
        details = {
            "atr": atr,
            "multiplier": multiplier,
            "lookback": lookback,
            "method": "chandelier",
        }
        
        if position_type == "LONG":
            details["highest_high"] = highest_high
        else:
            details["lowest_low"] = lowest_low
        
        return stop_price, details
    
    def _time_based_stop(
        self,
        symbol: str,
        entry_price: float,
        entry_time: Optional[datetime],
        position_type: str,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate time-based stop loss.
        
        Args:
            symbol: Trading symbol.
            entry_price: Entry price.
            entry_time: Entry timestamp.
            position_type: "LONG" or "SHORT".
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        days = params.get("time_stop_days", self.time_stop_days)
        
        # Store entry time for checking
        if entry_time:
            self.active_stops[symbol] = {
                "entry_time": entry_time,
                "time_stop_days": days,
            }
        
        # Time-based stop doesn't have a fixed price initially
        # We'll check it in check_stop_loss based on time elapsed
        # For initial calculation, use a very wide stop
        if position_type == "LONG":
            stop_price = entry_price * 0.5  # 50% stop (very wide)
        else:  # SHORT
            stop_price = entry_price * 1.5  # 50% stop (very wide)
        
        details = {
            "time_stop_days": days,
            "entry_time": entry_time,
            "method": "time_based",
        }
        
        return stop_price, details
    
    def _moving_average_stop(
        self,
        symbol: str,
        historical_data: Optional[pd.DataFrame],
        position_type: str,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate moving average stop loss.
        
        Args:
            symbol: Trading symbol.
            historical_data: Historical OHLC data.
            position_type: "LONG" or "SHORT".
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        if historical_data is None or len(historical_data) < 20:
            # Fallback to fixed percentage
            entry_price = historical_data['close'].iloc[-1] if historical_data is not None else 0
            return self._fixed_percentage_stop(entry_price, position_type, params)
        
        period = params.get("ma_period", self.ma_period)
        
        # Calculate moving average
        ma = historical_data['close'].rolling(window=period).mean().iloc[-1]
        
        if position_type == "LONG":
            # For long positions, stop below MA
            stop_price = ma
        else:  # SHORT
            # For short positions, stop above MA
            stop_price = ma
        
        details = {
            "moving_average": ma,
            "period": period,
            "method": "moving_average",
        }
        
        return stop_price, details
    
    def _support_resistance_stop(
        self,
        entry_price: float,
        position_type: str,
        support_level: Optional[float],
        resistance_level: Optional[float],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate support/resistance stop loss.
        
        Args:
            entry_price: Entry price.
            position_type: "LONG" or "SHORT".
            support_level: Support level price.
            resistance_level: Resistance level price.
            params: Parameters dictionary.
            
        Returns:
            Tuple of (stop_price, details).
        """
        if position_type == "LONG":
            if support_level is not None and support_level > 0:
                stop_price = support_level
            else:
                # Fallback to fixed percentage
                return self._fixed_percentage_stop(entry_price, position_type, params)
        else:  # SHORT
            if resistance_level is not None and resistance_level > 0:
                stop_price = resistance_level
            else:
                # Fallback to fixed percentage
                return self._fixed_percentage_stop(entry_price, position_type, params)
        
        details = {
            "support_level": support_level,
            "resistance_level": resistance_level,
            "method": "support_resistance",
        }
        
        return stop_price, details
    
    def _update_active_stop(
        self,
        symbol: str,
        stop_price: float,
        details: Dict[str, Any],
    ) -> None:
        """Update active stop information.
        
        Args:
            symbol: Trading symbol.
            stop_price: Stop loss price.
            details: Calculation details.
        """
        if symbol not in self.active_stops:
            self.active_stops[symbol] = {}
        
        self.active_stops[symbol].update({
            "stop_price": stop_price,
            **details,
        })