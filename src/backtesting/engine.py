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
        
        # 验证数据
        self._validate_data()
        
        # 结果存储
        self.results: Dict[str, BacktestResults] = {}
        self.comparison_results: Optional[pd.DataFrame] = None
        
        logger.info(
            f"BacktestEngine initialized with {len(data)} data points, "
            f"initial_capital={initial_capital}, commission={commission}, "
            f"slippage={slippage}"
        )
    
    def _validate_data(self) -> None:
        """验证输入数据。"""
        if self.data.empty:
            raise ValueError("Data cannot be empty")
        
        required_columns = ['close']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # 检查NaN值
        if self.data['close'].isna().any():
            warnings.warn("Data contains NaN values in 'close' column, filling forward")
            self.data['close'] = self.data['close'].ffill()
        
        # 确保索引为datetime类型
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
        
        # 设置回测参数
        strategy.initial_capital = self.initial_capital
        strategy.commission = self.commission
        strategy.slippage = self.slippage
        
        # 使用提供的名称或策略名称
        backtest_name = name or strategy.name
        
        logger.info(f"Running backtest for strategy: {backtest_name}")
        
        # 运行策略内部回测
        strategy_results = strategy.run_backtest(self.data)
        
        # 计算额外指标
        returns = strategy_results["returns"]
        equity_curve = strategy_results["equity_curve"]
        drawdown = strategy_results["drawdown"]
        
        # 计算综合指标
        metrics = calculate_metrics(
            returns=returns,
            equity_curve=equity_curve,
            drawdown=drawdown,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
        )
        
        # 创建BacktestResults对象
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
        
        # 存储结果
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
        """运行多个策略的回测。
        
        Args:
            strategies: 策略实例列表。
            names: 每个回测的可选名称列表。
            parallel: 是否并行运行策略。
            max_workers: 最大并行工作线程数。
            
        Returns:
            映射策略名称到BacktestResults的字典。
        """
        if names and len(names) != len(strategies):
            raise ValueError("names must have same length as strategies")
        
        if not names:
            names = [strategy.name for strategy in strategies]
        
        results = {}
        
        if parallel:
            # 并行运行策略
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
            
            # 在测试窗口上运行回测
            window_name = f"{name}_window_{i}"
            window_results = window_engine.run_strategy(
                window_strategy,
                window_name,
            )
            
            results[window_name] = window_results
        
        # 存储向前滚动分析结果
        self.results.update(results)
        
        # 计算向前滚动统计信息
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
        """使用网格搜索运行参数优化。
        
        Args:
            strategy_class: 策略类（非实例）。
            param_grid: 映射参数名到值列表的字典。
            name: 优化运行的名称。
            metric: 要优化的性能指标。
            maximize: 是否最大化指标（True）或最小化（False）。
            n_jobs: 并行任务数（-1表示使用所有核心）。
            
        Returns:
            包含优化结果的字典。
        """
        from itertools import product
        
        # 生成所有参数组合
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
        
        # 运行优化
        for values in tqdm(param_values, desc="Parameter optimization"):
            # 创建参数字典
            params = dict(zip(param_names, values))
            
            try:
                # 使用当前参数创建策略
                strategy = strategy_class(
                    name=f"{name}_{len(results)}",
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    slippage=self.slippage,
                    **params,
                )
                
                # 运行回测
                result = self.run_strategy(strategy)
                
                # 获取指标值
                score = result.metrics.get(metric, 0)
                
                # 更新最佳结果
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params
                    best_result = result
                
                # 存储结果
                results.append({
                    "params": params,
                    "score": score,
                    "result": result,
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {params}: {e}")
                continue
        
        # 按分数排序结果
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
            "top_results": results[:10],  # 前10个结果
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
        """运行蒙特卡洛模拟测试策略鲁棒性。
        
        Args:
            strategy: 策略实例。
            n_simulations: 蒙特卡洛模拟次数。
            name: 模拟运行名称。
            random_seed: 随机种子，用于可重复性。
            
        Returns:
            包含蒙特卡洛模拟结果的字典。
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} iterations")
        
        # 存储原始收益
        original_results = self.run_strategy(strategy, f"{name}_original")
        original_returns = original_results.returns
        
        if len(original_returns) < 2:
            raise ValueError("Need at least 2 returns for Monte Carlo simulation")
        
        # 运行蒙特卡洛模拟
        simulated_returns = []
        simulated_metrics = []
        
        for i in tqdm(range(n_simulations), desc="Monte Carlo simulation"):
            # Bootstrap收益（有放回随机抽样）
            bootstrap_idx = np.random.choice(
                len(original_returns),
                size=len(original_returns),
                replace=True,
            )
            bootstrap_returns = original_returns.iloc[bootstrap_idx]
            
            # 计算bootstrap样本的指标
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
        
        # 计算统计信息
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
        """从信号中提取交易信息。
        
        Args:
            signals: 包含信号的DataFrame。
            data: 原始市场数据。
            
        Returns:
            包含交易详情的DataFrame。
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
            
            # 判断交易类型
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
        """如果指定了基准则获取基准收益。
        
        Returns:
            基准收益序列或None。
        """
        # 基准数据应在数据中提供或单独获取
        if not self.benchmark:
            return None
        
        # 暂时返回None - 基准数据应在数据中提供
        # 或单独获取
        return None
    
    def _generate_comparison_table(self) -> None:
        """为所有存储的结果生成比较表。"""
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
        """计算向前滚动分析的统计数据。
        
        Args:
            results: 向前滚动分析结果的字典。
        """
        if not results:
            return
        
        # 计算跨窗口的平均指标
        metrics_list = [r.metrics for r in results.values()]
        metrics_df = pd.DataFrame(metrics_list)
        
        walk_forward_stats = {
            "mean": metrics_df.mean().to_dict(),
            "std": metrics_df.std().to_dict(),
            "min": metrics_df.min().to_dict(),
            "max": metrics_df.max().to_dict(),
            "median": metrics_df.median().to_dict(),
        }
        
        # 存储统计信息
        self.walk_forward_stats = walk_forward_stats
        
        logger.info("Walk-forward statistics calculated")
    
    def get_results(self, name: Optional[str] = None) -> Union[BacktestResults, Dict[str, BacktestResults]]:
        """获取回测结果。
        
        Args:
            name: 特定结果名称，或None表示所有结果。
            
        Returns:
            BacktestResults对象或所有结果的字典。
        """
        if name:
            if name not in self.results:
                raise KeyError(f"No results found for name: {name}")
            return self.results[name]
        else:
            return self.results.copy()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """获取所有结果的比较表格。
        
        Returns:
            包含策略比较的DataFrame。
        """
        if self.comparison_results is None:
            self._generate_comparison_table()
        assert self.comparison_results is not None
        return self.comparison_results.copy()
    
    def save_results(self, filepath: str, format: str = "pickle") -> None:
        """保存回测结果到文件。
        
        Args:
            filepath: 保存文件路径。
            format: 文件格式（'pickle'、'csv'、'json'）。
        """
        import pickle
        
        if format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
        elif format == "json":
            # 将结果转换为JSON可序列化格式
            json_results = {}
            for name, result in self.results.items():
                json_results[name] = result.to_dict()
            
            import json
            with open(filepath, 'w') as f:
                json.dump(json_results, f, default=str)
        elif format == "csv":
            # 保存比较表为CSV
            if self.comparison_results is not None:
                self.comparison_results.to_csv(filepath)
            else:
                raise ValueError("No comparison results to save as CSV")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {filepath} (format: {format})")
    
    def load_results(self, filepath: str, format: str = "pickle") -> None:
        """从文件加载回测结果。
        
        Args:
            filepath: 加载文件的路径。
            format: 文件格式（'pickle'、'json'）。
        """
        import pickle
        
        if format == "pickle":
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
        elif format == "json":
            import json
            with open(filepath, 'r') as f:
                json_results = json.load(f)
            
            # 转换回BacktestResults对象
            self.results = {}
            for name, result_dict in json_results.items():
                self.results[name] = BacktestResults.from_dict(result_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # 重新生成比较表
        self._generate_comparison_table()
        
        logger.info(f"Results loaded from {filepath} (format: {format})")