"""量化交易策略的回测引擎。"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ..strategies.base import BaseStrategy
from ..utils.logger import setup_logger
from .results import BacktestResults
from .metrics import calculate_metrics

logger = setup_logger(__name__)


class BacktestEngine:
    """在历史数据上运行交易策略的主回测引擎。
    
    特性：
    - 单策略回测
    - 多策略比较
    - 向前滚动分析
    - 参数优化（网格搜索）
    - 蒙特卡洛模拟
    - 并行处理支持
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
        benchmark: Optional[str] = None,
        risk_free_rate: float = 0.02,
    ):
        """初始化回测引擎。
        
        Args:
            data: 历史市场数据（必须包含'close'列）。
            initial_capital: 回测初始资金。
            commission: 每笔交易佣金率（0.001 = 0.1%）。
            slippage: 价格滑点比例。
            benchmark: 用于比较的基准符号（可选）。
            risk_free_rate: 夏普比率计算的年化无风险利率。
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        
        # Validate data
        self._validate_data()
        
        # Results storage
        self.results: Dict[str, BacktestResults] = {}
        self.comparison_results: Optional[pd.DataFrame] = None
        
        logger.info(
            f"BacktestEngine initialized with {len(data)} data points, "
            f"initial_capital={initial_capital}, commission={commission}, "
            f"slippage={slippage}"
        )
    
    def _validate_data(self) -> None:
        """Validate input data."""
        if self.data.empty:
            raise ValueError("Data cannot be empty")
        
        required_columns = ['close']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Check for NaN values
        if self.data['close'].isna().any():
            warnings.warn("Data contains NaN values in 'close' column, filling forward")
            self.data['close'] = self.data['close'].ffill()
        
        # Ensure index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")
    
    def run_strategy(
        self,
        strategy: BaseStrategy,
        name: Optional[str] = None,
        **strategy_params,
    ) -> BacktestResults:
        """运行单个策略的回测。
        
        Args:
            strategy: 策略实例（BaseStrategy的子类）。
            name: 回测的可选名称（默认：strategy.name）。
            **strategy_params: 传递给策略的额外参数。
            
        Returns:
            包含所有回测结果的BacktestResults对象。
        """
        # 如果提供了策略参数，则更新
        if strategy_params:
            strategy.set_parameters(**strategy_params)
        
        # Set backtest parameters
        strategy.initial_capital = self.initial_capital
        strategy.commission = self.commission
        strategy.slippage = self.slippage
        
        # Use provided name or strategy name
        backtest_name = name or strategy.name
        
        logger.info(f"Running backtest for strategy: {backtest_name}")
        
        # Run strategy's internal backtest
        strategy_results = strategy.run_backtest(self.data)
        
        # Calculate additional metrics
        returns = strategy_results["returns"]
        equity_curve = strategy_results["equity_curve"]
        drawdown = strategy_results["drawdown"]
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(
            returns=returns,
            equity_curve=equity_curve,
            drawdown=drawdown,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
        )
        
        # Create BacktestResults object
        results = BacktestResults(
            name=backtest_name,
            strategy_type=strategy.__class__.__name__,
            parameters=strategy.get_parameters(),
            metrics=metrics,
            returns=returns,
            equity_curve=equity_curve,
            drawdown=drawdown,
            signals=strategy_results["signals"],
            trades=self._extract_trades(strategy_results["signals"], self.data),
            benchmark_returns=self._get_benchmark_returns() if self.benchmark else None,
        )
        
        # Store results
        self.results[backtest_name] = results
        
        logger.info(f"Backtest completed for {backtest_name}")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return results
    
    def run_multiple_strategies(
        self,
        strategies: List[BaseStrategy],
        names: Optional[List[str]] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> Dict[str, BacktestResults]:
        """Run backtest for multiple strategies.
        
        Args:
            strategies: List of strategy instances.
            names: Optional list of names for each backtest.
            parallel: Whether to run strategies in parallel.
            max_workers: Maximum number of parallel workers.
            
        Returns:
            Dictionary mapping strategy names to BacktestResults.
        """
        if names and len(names) != len(strategies):
            raise ValueError("names must have same length as strategies")
        
        if not names:
            names = [strategy.name for strategy in strategies]
        
        results = {}
        
        if parallel:
            # Run strategies in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_name = {
                    executor.submit(
                        self.run_strategy,
                        strategy,
                        name,
                    ): name
                    for strategy, name in zip(strategies, names)
                }
                
                # 处理完成的结果
                for future in tqdm(
                    as_completed(future_to_name),
                    total=len(strategies),
                    desc="Running strategies in parallel",
                ):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        results[name] = result
                    except Exception as e:
                        logger.error(f"Error running strategy {name}: {e}")
                        raise
        else:
            # 顺序运行策略
            for strategy, name in tqdm(
                zip(strategies, names),
                total=len(strategies),
                desc="Running strategies sequentially",
            ):
                try:
                    result = self.run_strategy(strategy, name)
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error running strategy {name}: {e}")
                    raise
        
        # 存储所有结果
        self.results.update(results)
        
        # 生成比较表
        self._generate_comparison_table()
        
        return results
    
    def run_walk_forward(
        self,
        strategy: BaseStrategy,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
        name: str = "WalkForward",
    ) -> Dict[str, BacktestResults]:
        """运行向前滚动分析（滚动窗口回测）。
        
        Args:
            strategy: 策略实例。
            train_size: 训练窗口的天数。
            test_size: 测试窗口的天数。
            step_size: 窗口向前移动的步长。
            name: 结果的基础名称。
            
        Returns:
            映射窗口名称到BacktestResults的字典。
        """
        if len(self.data) < train_size + test_size:
            raise ValueError(
                f"Insufficient data: need at least {train_size + test_size} "
                f"data points, but only have {len(self.data)}"
            )
        
        results = {}
        total_windows = (len(self.data) - train_size - test_size) // step_size + 1
        
        logger.info(f"Starting walk-forward analysis with {total_windows} windows")
        
        for i in tqdm(range(total_windows), desc="Walk-forward analysis"):
            # 定义窗口索引
            train_start = i * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = min(test_start + test_size, len(self.data))
            
            # 提取窗口
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # 如果测试窗口太小则跳过
            if len(test_data) < 10:
                continue
            
            # 为测试窗口创建新引擎
            window_engine = BacktestEngine(
                data=test_data,
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
                benchmark=self.benchmark,
                risk_free_rate=self.risk_free_rate,
            )
            
            # 为此窗口创建新的策略实例
            strategy_cls = strategy.__class__
            window_strategy = strategy_cls(
                name=f"{strategy.name}_window_{i}",
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
            )
            
            # Run backtest on test window
            window_name = f"{name}_window_{i}"
            window_results = window_engine.run_strategy(
                window_strategy,
                window_name,
            )
            
            results[window_name] = window_results
        
        # Store walk-forward results
        self.results.update(results)
        
        # Calculate walk-forward statistics
        self._calculate_walk_forward_stats(results)
        
        return results
    
    def run_parameter_optimization(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        name: str = "Optimization",
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run parameter optimization using grid search.
        
        Args:
            strategy_class: Strategy class (not instance).
            param_grid: Dictionary mapping parameter names to list of values.
            name: Name for optimization run.
            metric: Performance metric to optimize.
            maximize: Whether to maximize the metric (True) or minimize (False).
            n_jobs: Number of parallel jobs (-1 for all cores).
            
        Returns:
            Dictionary with optimization results.
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(product(*param_grid.values()))
        
        logger.info(
            f"Starting parameter optimization with {len(param_values)} "
            f"combinations for metric: {metric}"
        )
        
        results: list[dict[str, Any]] = []
        best_score = -np.inf if maximize else np.inf
        best_params = None
        best_result = None
        
        # Run optimization
        for values in tqdm(param_values, desc="Parameter optimization"):
            # Create parameter dictionary
            params = dict(zip(param_names, values))
            
            try:
                # Create strategy with current parameters
                strategy = strategy_class(
                    name=f"{name}_{len(results)}",
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    slippage=self.slippage,
                    **params,
                )
                
                # Run backtest
                result = self.run_strategy(strategy)
                
                # Get metric value
                score = result.metrics.get(metric, 0)
                
                # Update best
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params
                    best_result = result
                
                # Store result
                results.append({
                    "params": params,
                    "score": score,
                    "result": result,
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        
        # Sort results by score
        results.sort(key=lambda x: float(x["score"]), reverse=maximize)
        
        optimization_results = {
            "name": name,
            "strategy_class": strategy_class.__name__,
            "param_grid": param_grid,
            "metric": metric,
            "maximize": maximize,
            "best_score": best_score,
            "best_params": best_params,
            "best_result": best_result,
            "all_results": results,
            "top_results": results[:10],  # Top 10 results
        }
        
        logger.info(f"Optimization completed. Best {metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return optimization_results
    
    def run_monte_carlo(
        self,
        strategy: BaseStrategy,
        n_simulations: int = 1000,
        name: str = "MonteCarlo",
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy robustness.
        
        Args:
            strategy: Strategy instance.
            n_simulations: Number of Monte Carlo simulations.
            name: Name for simulation run.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with Monte Carlo simulation results.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} iterations")
        
        # Store original returns
        original_results = self.run_strategy(strategy, f"{name}_original")
        original_returns = original_results.returns
        
        if len(original_returns) < 2:
            raise ValueError("Need at least 2 returns for Monte Carlo simulation")
        
        # Run Monte Carlo simulations
        simulated_returns = []
        simulated_metrics = []
        
        for i in tqdm(range(n_simulations), desc="Monte Carlo simulation"):
            # Bootstrap returns (random sampling with replacement)
            bootstrap_idx = np.random.choice(
                len(original_returns),
                size=len(original_returns),
                replace=True,
            )
            bootstrap_returns = original_returns.iloc[bootstrap_idx]
            
            # Calculate metrics for bootstrap sample
            equity_curve = (1 + bootstrap_returns).cumprod() * self.initial_capital
            drawdown = (equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()
            
            metrics = calculate_metrics(
                returns=bootstrap_returns,
                equity_curve=equity_curve,
                drawdown=drawdown,
                initial_capital=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
            )
            
            simulated_returns.append(bootstrap_returns)
            simulated_metrics.append(metrics)
        
        # Calculate statistics
        metrics_df = pd.DataFrame(simulated_metrics)
        
        monte_carlo_results = {
            "name": name,
            "n_simulations": n_simulations,
            "original_metrics": original_results.metrics,
            "simulated_metrics": simulated_metrics,
            "metrics_mean": metrics_df.mean().to_dict(),
            "metrics_std": metrics_df.std().to_dict(),
            "metrics_5th": metrics_df.quantile(0.05).to_dict(),
            "metrics_95th": metrics_df.quantile(0.95).to_dict(),
            "confidence_intervals": {
                metric: (
                    metrics_df[metric].quantile(0.05),
                    metrics_df[metric].quantile(0.95),
                )
                for metric in metrics_df.columns
            },
        }
        
        logger.info("Monte Carlo simulation completed")
        
        return monte_carlo_results
    
    def _extract_trades(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract trade information from signals.
        
        Args:
            signals: DataFrame with signals.
            data: Original market data.
            
        Returns:
            DataFrame with trade details.
        """
        if 'signal' not in signals.columns:
            return pd.DataFrame()
        
        signal_changes = signals['signal'].diff().fillna(0)
        trade_indices = signal_changes[signal_changes != 0].index
        
        trades = []
        for idx in trade_indices:
            if idx not in data.index:
                continue
            
            signal = signals.loc[idx, 'signal']
            prev_signal = signals.loc[signals.index[signals.index < idx][-1], 'signal'] if any(signals.index < idx) else 0
            
            # Determine trade type
            if signal > prev_signal:
                trade_type = "BUY"
            elif signal < prev_signal:
                trade_type = "SELL"
            else:
                continue
            
            trade = {
                'timestamp': idx,
                'type': trade_type,
                'price': data.loc[idx, 'close'],
                'signal': signal,
                'prev_signal': prev_signal,
            }
            trades.append(trade)
        
        return pd.DataFrame(trades)
    
    def _get_benchmark_returns(self) -> Optional[pd.Series]:
        """Get benchmark returns if benchmark is specified.
        
        Returns:
            Benchmark returns series or None.
        """
        # This is a placeholder - in a real implementation, you would
        # fetch benchmark data from a data source
        if not self.benchmark:
            return None
        
        # For now, return None - benchmark data should be provided
        # in the data or fetched separately
        return None
    
    def _generate_comparison_table(self) -> None:
        """Generate comparison table for all stored results."""
        if not self.results:
            self.comparison_results = pd.DataFrame()
            return
        
        comparison_data = []
        for name, result in self.results.items():
            row = {
                "Strategy": name,
                "Type": result.strategy_type,
                **result.metrics,
            }
            comparison_data.append(row)
        
        self.comparison_results = pd.DataFrame(comparison_data)
    
    def _calculate_walk_forward_stats(
        self,
        results: Dict[str, BacktestResults],
    ) -> None:
        """Calculate statistics for walk-forward analysis.
        
        Args:
            results: Dictionary of walk-forward results.
        """
        if not results:
            return
        
        # Calculate average metrics across windows
        metrics_list = [r.metrics for r in results.values()]
        metrics_df = pd.DataFrame(metrics_list)
        
        walk_forward_stats = {
            "mean": metrics_df.mean().to_dict(),
            "std": metrics_df.std().to_dict(),
            "min": metrics_df.min().to_dict(),
            "max": metrics_df.max().to_dict(),
            "median": metrics_df.median().to_dict(),
        }
        
        # Store stats
        self.walk_forward_stats = walk_forward_stats
        
        logger.info("Walk-forward statistics calculated")
    
    def get_results(self, name: Optional[str] = None) -> Union[BacktestResults, Dict[str, BacktestResults]]:
        """Get backtest results.
        
        Args:
            name: Specific result name, or None for all results.
            
        Returns:
            BacktestResults object or dictionary of all results.
        """
        if name:
            if name not in self.results:
                raise KeyError(f"No results found for name: {name}")
            return self.results[name]
        else:
            return self.results.copy()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table of all results.
        
        Returns:
            DataFrame with strategy comparison.
        """
        if self.comparison_results is None:
            self._generate_comparison_table()
        assert self.comparison_results is not None
        return self.comparison_results.copy()
    
    def save_results(self, filepath: str, format: str = "pickle") -> None:
        """Save backtest results to file.
        
        Args:
            filepath: Path to save file.
            format: File format ('pickle', 'csv', 'json').
        """
        import pickle
        
        if format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
        elif format == "json":
            # Convert results to JSON-serializable format
            json_results = {}
            for name, result in self.results.items():
                json_results[name] = result.to_dict()
            
            import json
            with open(filepath, 'w') as f:
                json.dump(json_results, f, default=str)
        elif format == "csv":
            # Save comparison table as CSV
            if self.comparison_results is not None:
                self.comparison_results.to_csv(filepath)
            else:
                raise ValueError("No comparison results to save as CSV")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {filepath} (format: {format})")
    
    def load_results(self, filepath: str, format: str = "pickle") -> None:
        """Load backtest results from file.
        
        Args:
            filepath: Path to load file from.
            format: File format ('pickle', 'json').
        """
        import pickle
        
        if format == "pickle":
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
        elif format == "json":
            import json
            with open(filepath, 'r') as f:
                json_results = json.load(f)
            
            # Convert back to BacktestResults objects
            self.results = {}
            for name, result_dict in json_results.items():
                self.results[name] = BacktestResults.from_dict(result_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Regenerate comparison table
        self._generate_comparison_table()
        
        logger.info(f"Results loaded from {filepath} (format: {format})")