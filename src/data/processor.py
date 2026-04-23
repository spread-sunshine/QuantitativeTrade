# 数据处理和特征工程模块
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    """处理市场数据并为交易策略创建特征。"""

    def __init__(self):
        """初始化数据处理器。"""
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理原始市场数据。

        Args:
            df: 包含市场数据的原始 DataFrame。

        Returns:
            清理后的 DataFrame。
        """
        if df.empty:
            return df

        df_clean = df.copy()

        # 确保索引为datetime类型
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            if "date" in df_clean.columns:
                df_clean.set_index("date", inplace=True)
            elif "Date" in df_clean.columns:
                df_clean.set_index("Date", inplace=True)
            else:
                raise ValueError("DataFrame must have a date column or index")

        # 按日期排序
        df_clean.sort_index(inplace=True)
        
        # 处理缺失值
        # OHLC前向填充，成交量填零
        ohlc_columns = ["open", "high", "low", "close", "adj_close"]
        for col in ohlc_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].ffill()

        if "volume" in df_clean.columns:
            df_clean["volume"] = df_clean["volume"].fillna(0)

        # 移除OHLC中仍有NaN的行
        df_clean = df_clean.dropna(subset=ohlc_columns)
        
        # 移除重复日期（保留最后一条）
        df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

        logger.info(f"Cleaned data: {len(df_clean)} rows after cleaning")
        return df_clean

    def add_returns(self, df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """向 DataFrame 添加收益率。

        Args:
            df: 包含价格数据的 DataFrame。
            column: 用于计算收益率的列。

        Returns:
            添加了收益率列的 DataFrame。
        """
        if df.empty:
            return df

        df_result = df.copy()

        # 简单收益率
        df_result["returns"] = df_result[column].pct_change()

        # 对数收益率
        df_result["log_returns"] = np.log(df_result[column] / df_result[column].shift(1))

        # 累计收益率
        df_result["cumulative_returns"] = (1 + df_result["returns"]).cumprod() - 1

        return df_result

    def add_moving_averages(
        self, df: pd.DataFrame, windows: List[int] = None, column: str = "close"
    ) -> pd.DataFrame:
        """向 DataFrame 添加移动平均线。

        Args:
            df: 包含价格数据的 DataFrame。
            windows: 移动平均线的窗口大小列表。
            column: 用于计算移动平均线的列。

        Returns:
            添加了移动平均线列的 DataFrame。
        """
        if df.empty:
            return df

        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]

        df_result = df.copy()

        for window in windows:
            col_name = f"ma_{window}"
            df_result[col_name] = df_result[column].rolling(window=window).mean()

        return df_result

    def add_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """向 DataFrame 添加波动率指标。

        Args:
            df: 包含收益率数据的 DataFrame。
            window: 波动率计算的滚动窗口。

        Returns:
            添加了波动率列的 DataFrame。
        """
        if df.empty:
            return df

        df_result = df.copy()

        # 确保收益率列存在
        if "returns" not in df_result.columns:
            df_result = self.add_returns(df_result)

        # 滚动标准差（历史波动率）
        df_result[f"volatility_{window}"] = (
            df_result["returns"].rolling(window=window).std() * np.sqrt(252)
        )

        # 滚动平均真实范围（ATR）
        if all(col in df_result.columns for col in ["high", "low", "close"]):
            df_result["tr"] = np.maximum(
                df_result["high"] - df_result["low"],
                np.maximum(
                    abs(df_result["high"] - df_result["close"].shift(1)),
                    abs(df_result["low"] - df_result["close"].shift(1)),
                ),
            )
            df_result[f"atr_{window}"] = df_result["tr"].rolling(window=window).mean()

        return df_result

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """向 DataFrame 添加常见的技术指标。

        Args:
            df: 包含 OHLCV 数据的 DataFrame。

        Returns:
            添加了技术指标的 DataFrame。
        """
        if df.empty:
            return df

        df_result = df.copy()

        # 确保所需列存在
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df_result.columns]
        if missing_cols:
            logger.warning(f"Missing columns for technical indicators: {missing_cols}")
            return df_result

        # 相对强弱指标（RSI）
        delta = df_result["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_result["rsi"] = 100 - (100 / (1 + rs))

        # 异同移动平均线（MACD）
        ema_12 = df_result["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_result["close"].ewm(span=26, adjust=False).mean()
        df_result["macd"] = ema_12 - ema_26
        df_result["macd_signal"] = df_result["macd"].ewm(span=9, adjust=False).mean()
        df_result["macd_hist"] = df_result["macd"] - df_result["macd_signal"]

        # 布林带
        df_result["bb_middle"] = df_result["close"].rolling(window=20).mean()
        bb_std = df_result["close"].rolling(window=20).std()
        df_result["bb_upper"] = df_result["bb_middle"] + (bb_std * 2)
        df_result["bb_lower"] = df_result["bb_middle"] - (bb_std * 2)
        df_result["bb_width"] = df_result["bb_upper"] - df_result["bb_lower"]
        df_result["bb_position"] = (df_result["close"] - df_result["bb_lower"]) / (
            df_result["bb_upper"] - df_result["bb_lower"]
        )

        # 能量潮指标（OBV）
        df_result["obv"] = 0
        df_result.loc[df_result["close"] > df_result["close"].shift(1), "obv"] = df_result["volume"]
        df_result.loc[df_result["close"] < df_result["close"].shift(1), "obv"] = -df_result["volume"]
        df_result["obv"] = df_result["obv"].cumsum()

        # 价格通道
        df_result["high_20"] = df_result["high"].rolling(window=20).max()
        df_result["low_20"] = df_result["low"].rolling(window=20).min()

        # 平均方向性指数（ADX）近似值
        df_result["tr"] = np.maximum(
            df_result["high"] - df_result["low"],
            np.maximum(
                abs(df_result["high"] - df_result["close"].shift(1)),
                abs(df_result["low"] - df_result["close"].shift(1)),
            ),
        )
        df_result["atr"] = df_result["tr"].rolling(window=14).mean()

        return df_result

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向 DataFrame 添加基于时间的特征。

        Args:
            df: 具有 datetime 索引的 DataFrame。

        Returns:
            添加了时间特征的 DataFrame。
        """
        if df.empty:
            return df

        df_result = df.copy()

        if not isinstance(df_result.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, cannot add time features")
            return df_result

        # 基础时间特征
        df_result["year"] = df_result.index.year
        df_result["month"] = df_result.index.month
        df_result["day"] = df_result.index.day
        df_result["day_of_week"] = df_result.index.dayofweek
        df_result["day_of_year"] = df_result.index.dayofyear
        df_result["week_of_year"] = df_result.index.isocalendar().week

        # 季度
        df_result["quarter"] = df_result.index.quarter

        # 是否月末/月初
        df_result["is_month_end"] = df_result.index.is_month_end.astype(int)
        df_result["is_month_start"] = df_result.index.is_month_start.astype(int)
        df_result["is_quarter_end"] = df_result.index.is_quarter_end.astype(int)
        df_result["is_quarter_start"] = df_result.index.is_quarter_start.astype(int)
        df_result["is_year_end"] = df_result.index.is_year_end.astype(int)
        df_result["is_year_start"] = df_result.index.is_year_start.astype(int)

        # 一天中的时间（用于日内数据）
        if df_result.index.hour.any():
            df_result["hour"] = df_result.index.hour
            df_result["minute"] = df_result.index.minute

        return df_result

    def prepare_features(
        self, df: pd.DataFrame, target_col: str = "returns", lookahead: int = 1
    ) -> pd.DataFrame:
        """为机器学习准备特征。

        Args:
            df: 包含市场数据的 DataFrame。
            target_col: 用作目标的列。
            lookahead: 目标向前预测的周期数。

        Returns:
            包含特征和目标的 DataFrame。
        """
        if df.empty:
            return df

        df_result = df.copy()

        # 清理数据
        df_result = self.clean_data(df_result)

        # 如果不存在则添加收益率
        if "returns" not in df_result.columns:
            df_result = self.add_returns(df_result)

        # 添加移动平均线
        df_result = self.add_moving_averages(df_result)

        # 添加波动率
        df_result = self.add_volatility(df_result)

        # 添加技术指标
        df_result = self.add_technical_indicators(df_result)

        # 添加时间特征
        df_result = self.add_time_features(df_result)

        # 创建目标
        if target_col == "returns":
            df_result["target"] = df_result["returns"].shift(-lookahead)
        elif target_col in df_result.columns:
            df_result["target"] = df_result[target_col].shift(-lookahead)
        else:
            logger.warning(f"Target column {target_col} not found, using returns")
            df_result["target"] = df_result["returns"].shift(-lookahead)

        # 删除包含NaN的行（来自滚动计算和目标偏移）
        df_result = df_result.dropna()

        logger.info(f"Prepared features: {len(df_result)} rows, {len(df_result.columns)} columns")
        return df_result

    def resample_data(self, df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
        """将数据重采样到不同的频率。

        Args:
            df: 具有 datetime 索引的 DataFrame。
            freq: 重采样频率（'D', 'W', 'M', 'Q', 'Y'）。

        Returns:
            重采样后的 DataFrame。
        """
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")

        # 定义聚合规则
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # 添加任何额外列
        for col in df.columns:
            if col not in ohlc_dict:
                ohlc_dict[col] = "last"

        # 重采样
        df_resampled = df.resample(freq).apply(ohlc_dict)

        return df_resampled

    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """归一化指定列。

        Args:
            df: 要归一化的 DataFrame。
            columns: 要归一化的列列表。如果为 None，则归一化所有数值列。

        Returns:
            包含归一化列的 DataFrame。
        """
        if df.empty:
            return df

        df_result = df.copy()

        if columns is None:
            # 选择数值列
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if col != "target"]

        for col in columns:
            if col in df_result.columns:
                # Z-score normalization
                mean = df_result[col].mean()
                std = df_result[col].std()
                if std > 0:
                    df_result[f"{col}_norm"] = (df_result[col] - mean) / std
                else:
                    df_result[f"{col}_norm"] = 0

        return df_result

    def split_train_test(
        self, df: pd.DataFrame, test_size: float = 0.2, date_cutoff: str = None
    ) -> tuple:
        """将数据分割为训练集和测试集。

        Args:
            df: 要分割的 DataFrame。
            test_size: 用于测试的数据比例（如果 date_cutoff 为 None）。
            date_cutoff: 用于分割的日期字符串（例如 '2022-01-01'）。

        Returns:
            包含 (train_df, test_df) 的元组。
        """
        if df.empty:
            return df.copy(), df.copy()

        if date_cutoff:
            # Split by date
            cutoff_date = pd.to_datetime(date_cutoff)
            train_df = df[df.index < cutoff_date].copy()
            test_df = df[df.index >= cutoff_date].copy()
        else:
            # Split by proportion
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df