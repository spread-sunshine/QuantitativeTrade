#!/usr/bin/env python3
"""
Example script demonstrating the data pipeline for quantitative trading system.

This script shows how to:
1. Fetch market data from Yahoo Finance
2. Process and clean the data
3. Store data in SQLite database
4. Retrieve data from database
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.data.database import DatabaseManager
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("data_pipeline")

def main():
    """Main data pipeline example."""
    logger.info("Starting data pipeline example")
    
    # Initialize components
    fetcher = DataFetcher(cache_enabled=True)
    processor = DataProcessor()
    db_manager = DatabaseManager()
    
    # Define symbols to fetch
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Define date range (last 30 days for example)
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching data for symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Fetch data for each symbol
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")
            
            # Fetch data from Yahoo Finance
            raw_data = fetcher.fetch_yahoo(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            
            logger.info(f"Fetched {len(raw_data)} rows for {symbol}")
            
            # Clean and process data
            if not raw_data.empty:
                # Set date as index for processing
                if 'date' in raw_data.columns:
                    processed_data = raw_data.set_index('date')
                else:
                    processed_data = raw_data.copy()
                
                # Clean data
                cleaned_data = processor.clean_data(processed_data)
                
                # Add technical indicators
                enhanced_data = processor.add_technical_indicators(cleaned_data)
                
                # Add returns
                final_data = processor.add_returns(enhanced_data)
                
                logger.info(f"Processed data shape: {final_data.shape}")
                
                # Store in database
                db_manager.store_data(final_data.reset_index(), symbol, if_exists="replace")
                
                # Retrieve from database to verify
                retrieved_data = db_manager.get_data(symbol, start_date, end_date)
                logger.info(f"Retrieved {len(retrieved_data)} rows for {symbol} from database")
                
                # Show basic statistics
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
    
    # Show database information
    db_info = db_manager.get_database_info()
    logger.info("\nDatabase Information:")
    logger.info(f"  Database URL: {db_info['database_url']}")
    logger.info(f"  Tables: {db_info['tables']}")
    
    if 'market_data_rows' in db_info:
        logger.info(f"  Total rows in market_data: {db_info['market_data_rows']}")
        logger.info(f"  Unique symbols in market_data: {db_info['market_data_symbols']}")
    
    # Get available symbols from database
    available_symbols = db_manager.get_available_symbols()
    logger.info(f"\nAvailable symbols in database: {available_symbols}")
    
    # Show date ranges for each symbol
    for symbol in available_symbols:
        date_range = db_manager.get_date_range(symbol)
        if 'min_date' in date_range and 'max_date' in date_range:
            logger.info(f"  {symbol}: {date_range['min_date'].date()} to {date_range['max_date'].date()} ({date_range['row_count']} rows)")
    
    logger.info("\nData pipeline example completed successfully!")

if __name__ == "__main__":
    main()