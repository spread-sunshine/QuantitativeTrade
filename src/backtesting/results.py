"""回测结果存储和序列化。"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class Trade:
    """表示单笔交易。"""
    timestamp: pd.Timestamp
    type: str  # "BUY" or "SELL"
    price: float
    signal: float
    prev_signal: float
    quantity: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """将交易转换为字典。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
            "price": self.price,
            "signal": self.signal,
            "prev_signal": self.prev_signal,
            "quantity": self.quantity,
            "commission": self.commission,
            "slippage": self.slippage,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """从字典创建交易。"""
        data = data.copy()
        data['timestamp'] = pd.Timestamp(data['timestamp'])
        return cls(**data)


@dataclass
class BacktestResults:
    """具有序列化能力的回测结果容器。"""
    
    # 基本信息
    name: str
    strategy_type: str
    parameters: Dict[str, Any]
    
    # 性能指标
    metrics: Dict[str, float]
    
    # 时间序列数据
    returns: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    signals: pd.DataFrame
    
    # 交易信息
    trades: pd.DataFrame
    
    # 基准数据（可选）
    benchmark_returns: Optional[pd.Series] = None
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    data_shape: Optional[tuple] = None
    runtime_seconds: Optional[float] = None
    
    def __post_init__(self):
        """初始化后处理。"""
        # 确保returns索引为datetime类型
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)

        # 确保equity_curve索引为datetime类型
        if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
            self.equity_curve.index = pd.to_datetime(self.equity_curve.index)

        # 确保drawdown索引为datetime类型
        if not isinstance(self.drawdown.index, pd.DatetimeIndex):
            self.drawdown.index = pd.to_datetime(self.drawdown.index)

        # 存储数据形状
        self.data_shape = (
            len(self.returns),
            len(self.returns.columns) if hasattr(self.returns, 'columns') else 1,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要统计信息。

        Returns:
            包含摘要统计的字典。
        """
        summary = {
            "strategy_name": self.name,
            "strategy_type": self.strategy_type,
            "timestamp": self.timestamp.isoformat(),
            "data_points": self.data_shape[0] if self.data_shape else 0,
            "parameters": self.parameters,
            "runtime_seconds": self.runtime_seconds,
        }
        
        # 添加关键指标
        key_metrics = [
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate",
            "profit_factor", "calmar_ratio", "alpha", "beta",
        ]
        
        for metric in key_metrics:
            if metric in self.metrics:
                summary[metric] = self.metrics[metric]
        
        return summary
    
    def get_trade_analysis(self) -> Dict[str, Any]:
        """分析交易数据。
        
        Returns:
            包含交易分析的字典。
        """
        if self.trades.empty:
            return {"total_trades": 0}
        
        trades_df = self.trades.copy()
        
        # 基本交易统计
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['type'] == 'BUY'])
        sell_trades = len(trades_df[trades_df['type'] == 'SELL'])
        
        # 如果可能，计算持仓期
        holding_periods: list[float] = []
        if 'pnl' in trades_df.columns and 'timestamp' in trades_df.columns:
            # 分组连续交易
            pass

        analysis = {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_trade_duration": None,  # 如果有数据则计算
            "trade_frequency": total_trades / len(self.returns) if len(self.returns) > 0 else 0,
        }

        # 如果可用，添加盈亏统计
        if 'pnl' in trades_df.columns:
            pnl_stats = {
                "total_pnl": trades_df['pnl'].sum(),
                "avg_pnl": trades_df['pnl'].mean(),
                "std_pnl": trades_df['pnl'].std(),
                "max_pnl": trades_df['pnl'].max(),
                "min_pnl": trades_df['pnl'].min(),
                "pnl_skewness": trades_df['pnl'].skew(),
                "pnl_kurtosis": trades_df['pnl'].kurtosis(),
            }
            analysis.update(pnl_stats)

            # 胜率
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            analysis.update({
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            })
            
            # 盈利因子
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            analysis["profit_factor"] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return analysis
    
    def get_performance_metrics(self) -> pd.DataFrame:
        """获取性能指标DataFrame。

        Returns:
            包含性能指标的DataFrame。
        """
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.index = [self.name]
        return metrics_df
    
    def get_returns_analysis(self) -> Dict[str, Any]:
        """分析收益分布。

        Returns:
            包含收益分析的字典。
        """
        returns = self.returns
        
        if len(returns) == 0:
            return {}
        
        analysis = {
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "min_return": returns.min(),
            "max_return": returns.max(),
            "median_return": returns.median(),
            "var_95": returns.quantile(0.05),
            "var_99": returns.quantile(0.01),
            "cvar_95": returns[returns <= returns.quantile(0.05)].mean(),
            "cvar_99": returns[returns <= returns.quantile(0.01)].mean(),
        }
        
        # 年化指标
        analysis.update({
            "annualized_mean": returns.mean() * 252,
            "annualized_std": returns.std() * np.sqrt(252),
        })
        
        return analysis
    
    def get_equity_analysis(self) -> Dict[str, Any]:
        """分析权益曲线。

        Returns:
            包含权益分析的字典。
        """
        equity = self.equity_curve
        
        if len(equity) == 0:
            return {}
        
        analysis = {
            "initial_equity": equity.iloc[0] if not equity.empty else 0,
            "final_equity": equity.iloc[-1] if not equity.empty else 0,
            "peak_equity": equity.max(),
            "valley_equity": equity.min(),
            "avg_equity": equity.mean(),
            "std_equity": equity.std(),
        }
        
        # 计算水下期
        underwater = self.drawdown < 0
        if underwater.any():
            underwater_periods = underwater.astype(int).groupby(
                (~underwater).astype(int).cumsum()
            ).sum()
            analysis.update({
                "max_underwater_duration": underwater_periods.max(),
                "avg_underwater_duration": underwater_periods.mean(),
                "underwater_frequency": underwater.mean(),
            })
        
        return analysis
    
    def to_dict(self, include_series: bool = True) -> Dict[str, Any]:
        """将结果转换为字典。

        Args:
            include_series: 是否包含时间序列数据。

        Returns:
            字典表示。
        """
        data = {
            "name": self.name,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "data_shape": self.data_shape,
            "runtime_seconds": self.runtime_seconds,
        }

        if include_series:
            # 将pandas Series转换为字典
            data.update({
                "returns": self.returns.to_dict(),
                "equity_curve": self.equity_curve.to_dict(),
                "drawdown": self.drawdown.to_dict(),
                "signals": self.signals.to_dict(),
                "trades": self.trades.to_dict(),
            })
            
            if self.benchmark_returns is not None:
                data["benchmark_returns"] = self.benchmark_returns.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestResults':
        """从字典创建回测结果。

        Args:
            data: 字典表示。

        Returns:
            BacktestResults实例。
        """
        data = data.copy()

        # 转换时间戳
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # 如果存在，转换pandas Series
        series_fields = ['returns', 'equity_curve', 'drawdown', 'benchmark_returns']
        for field in series_fields:
            if field in data and isinstance(data[field], dict):
                data[field] = pd.Series(data[field])

        # 如果存在，转换pandas DataFrame
        df_fields = ['signals', 'trades']
        for field in df_fields:
            if field in data and isinstance(data[field], dict):
                # Handle nested dict for DataFrame
                if 'index' in data[field] and 'data' in data[field]:
                    # Complex serialization format
                    data[field] = pd.read_json(json.dumps(data[field]), orient='split')
                else:
                    # Simple dict format
                    data[field] = pd.DataFrame(data[field])

        return cls(**data)
    
    def save(self, filepath: str, format: str = "json") -> None:
        """保存结果到文件。

        Args:
            filepath: 保存文件路径。
            format: 文件格式（'json'、'pickle'）。
        """
        if format == "json":
            # 转换为JSON可序列化字典
            result_dict = self.to_dict(include_series=True)

            # 转换pandas对象
            for key, value in result_dict.items():
                if isinstance(value, pd.Series):
                    result_dict[key] = value.to_dict()
                elif isinstance(value, pd.DataFrame):
                    result_dict[key] = value.to_dict(orient='split')
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
                
        elif format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str, format: str = "json") -> 'BacktestResults':
        """从文件加载结果。

        Args:
            filepath: 加载文件路径。
            format: 文件格式（'json'、'pickle'）。

        Returns:
            BacktestResults实例。
        """
        if format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)

            # 转换回pandas对象
            for key, value in data.items():
                if key in ['returns', 'equity_curve', 'drawdown', 'benchmark_returns']:
                    if isinstance(value, dict):
                        data[key] = pd.Series(value)
                elif key in ['signals', 'trades']:
                    if isinstance(value, dict):
                        if 'index' in value and 'data' in value:
                            data[key] = pd.read_json(json.dumps(value), orient='split')
                        else:
                            data[key] = pd.DataFrame(value)
            
            return cls.from_dict(data)
            
        elif format == "pickle":
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_report(self) -> str:
        """生成文本报告。

        Returns:
            格式化的文本报告。
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"BACKTEST REPORT: {self.name}")
        lines.append("=" * 70)
        lines.append(f"Strategy Type: {self.strategy_type}")
        lines.append(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Data Points: {self.data_shape[0] if self.data_shape else 0}")
        lines.append(f"Runtime: {self.runtime_seconds:.2f} seconds" if self.runtime_seconds else "")
        lines.append("")
        
        # 参数
        lines.append("PARAMETERS:")
        for key, value in self.parameters.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # 关键指标
        lines.append("PERFORMANCE METRICS:")
        key_metrics = [
            ("Total Return", "total_return", "percentage"),
            ("Annualized Return", "annualized_return", "percentage"),
            ("Sharpe Ratio", "sharpe_ratio", "decimal"),
            ("Sortino Ratio", "sortino_ratio", "decimal"),
            ("Max Drawdown", "max_drawdown", "percentage"),
            ("Win Rate", "win_rate", "percentage"),
            ("Profit Factor", "profit_factor", "decimal"),
            ("Calmar Ratio", "calmar_ratio", "decimal"),
        ]
        
        for display_name, metric_key, fmt in key_metrics:
            if metric_key in self.metrics:
                value = self.metrics[metric_key]
                if fmt == "percentage":
                    lines.append(f"  {display_name}: {value:.2%}")
                elif fmt == "decimal":
                    lines.append(f"  {display_name}: {value:.4f}")
        
        lines.append("")
        
        # 收益分析
        returns_analysis = self.get_returns_analysis()
        if returns_analysis:
            lines.append("RETURNS ANALYSIS:")
            lines.append(f"  Mean Return: {returns_analysis['mean_return']:.4%}")
            lines.append(f"  Std Return: {returns_analysis['std_return']:.4%}")
            lines.append(f"  Skewness: {returns_analysis['skewness']:.4f}")
            lines.append(f"  Kurtosis: {returns_analysis['kurtosis']:.4f}")
            lines.append(f"  Min Return: {returns_analysis['min_return']:.4%}")
            lines.append(f"  Max Return: {returns_analysis['max_return']:.4%}")
            lines.append(f"  VaR 95%: {returns_analysis['var_95']:.4%}")
            lines.append(f"  CVaR 95%: {returns_analysis['cvar_95']:.4%}")
            lines.append("")
        
        # 交易分析
        trade_analysis = self.get_trade_analysis()
        if trade_analysis:
            lines.append("TRADE ANALYSIS:")
            lines.append(f"  Total Trades: {trade_analysis['total_trades']}")
            if 'win_rate' in trade_analysis:
                lines.append(f"  Win Rate: {trade_analysis['win_rate']:.2%}")
            if 'profit_factor' in trade_analysis:
                lines.append(f"  Profit Factor: {trade_analysis['profit_factor']:.4f}")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def plot(self, **kwargs) -> Any:
        """绘制结果图表。

        Args:
            **kwargs: 绘图的额外参数。

        Returns:
            Plot对象。
        """
        # 实际绘图将在可视化模块中实现
        from ..visualization.charts import create_performance_dashboard
        return create_performance_dashboard(self.results, **kwargs)