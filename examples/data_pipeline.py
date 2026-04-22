#!/usr/bin/env python3
"""
量化交易系统数据管道示例脚本。

本脚本展示如何：
1. 从Yahoo Finance获取市场数据
2. 处理和清洗数据
3. 将数据存储到SQLite数据库
4. 从数据库检索数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.data.database import DatabaseManager
from src.utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger("data_pipeline")

def main():
    """主要数据管道示例。"""
    logger.info("Starting data pipeline example")
    
    # 初始化组件
    fetcher = DataFetcher(cache_enabled=True)
    processor = DataProcessor()
    db_manager = DatabaseManager()
    
    # 定义要获取的股票代码
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # 定义日期范围（例如最近30天）
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching data for symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # 为每个股票代码获取数据
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")
            
            # 从Yahoo Finance获取数据
            raw_data = fetcher.fetch_yahoo(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            
            logger.info(f"Fetched {len(raw_data)} rows for {symbol}")
            
            # 清理和处理数据
            if not raw_data.empty:
                # 将日期设置为索引以便处理
                if 'date' in raw_data.columns:
                    processed_data = raw_data.set_index('date')
                else:
                    processed_data = raw_data.copy()
                
                # 清理数据
                cleaned_data = processor.clean_data(processed_data)
                
                # 添加技术指标
                enhanced_data = processor.add_technical_indicators(cleaned_data)
                
                # 添加收益率
                final_data = processor.add_returns(enhanced_data)
                
                logger.info(f"Processed data shape: {final_data.shape}")
                
                # 存储到数据库
                db_manager.store_data(final_data.reset_index(), symbol, if_exists="replace")
                
                # 从数据库检索以验证
                retrieved_data = db_manager.get_data(symbol, start_date, end_date)
                logger.info(f"Retrieved {len(retrieved_data)} rows for {symbol} from database")
                
                # 显示基本统计信息
                if not retrieved_data.empty:
                    logger.info(f"\n{symbol} Statistics:")
                    logger.info(f"  Date range: {retrieved_data.index[0]} to {retrieved_data.index[-1]}")
                    logger.info(f"  Close price range: {retrieved_data['close'].min():.2f} - {retrieved_data['close'].max():.2f}")
                    logger.info(f"  Average volume: {retrieved_data['volume'].mean():.0f}")
                    
                    if 'returns' in retrieved_data.columns:
                        logger.info(f"  Average daily return: {retrieved_data['returns'].mean():.4%}")
                        logger.info(f"  Daily return volatility: {retrieved_data['returns'].std():.4%}")
            
            else:
                logger.warning(f"No data fetched for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    # 显示数据库信息
    db_info = db_manager.get_database_info()
    logger.info("\nDatabase Information:")
    logger.info(f"  Database URL: {db_info['database_url']}")
    logger.info(f"  Tables: {db_info['tables']}")
    
    if 'market_data_rows' in db_info:
        logger.info(f"  Total rows in market_data: {db_info['market_data_rows']}")
        logger.info(f"  Unique symbols in market_data: {db_info['market_data_symbols']}")
    
    # 从数据库获取可用的股票代码
    available_symbols = db_manager.get_available_symbols()
    logger.info(f"\nAvailable symbols in database: {available_symbols}")
    
    # 显示每个股票代码的日期范围
    for symbol in available_symbols:
        date_range = db_manager.get_date_range(symbol)
        if 'min_date' in date_range and 'max_date' in date_range:
            logger.info(f"  {symbol}: {date_range['min_date'].date()} to {date_range['max_date'].date()} ({date_range['row_count']} rows)")
    
    logger.info("\nData pipeline example completed successfully!")

if __name__ == "__main__":
    main()