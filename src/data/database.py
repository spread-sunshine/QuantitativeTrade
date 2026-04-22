# Database manager for storing and retrieving market data
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, Float, Integer, UniqueConstraint
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

from ..utils.logger import setup_logger
from config.settings import DATABASE_URL

logger = setup_logger(__name__)

Base = declarative_base()


class MarketData(Base):
    """Market data table schema."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float, nullable=True)

    # Add unique constraint
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_market_data_symbol_date'),
    )


class DatabaseManager:
    """Manages database operations for market data."""

    def __init__(self, database_url: str = DATABASE_URL):
        """Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL.
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

        # Create tables if they don't exist
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database tables created/verified at {self.database_url}")

    def store_data(self, df: pd.DataFrame, symbol: str, if_exists: str = "append"):
        """Store market data in database.

        Args:
            df: DataFrame with market data.
            symbol: Stock symbol.
            if_exists: What to do if data exists ('fail', 'replace', 'append').
        """
        if df.empty:
            logger.warning(f"No data to store for {symbol}")
            return

        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Prepare DataFrame for database
        db_df = df.copy()

        # Ensure date column is datetime
        if "date" in db_df.columns:
            db_df["date"] = pd.to_datetime(db_df["date"])
        elif "Date" in db_df.columns:
            db_df["date"] = pd.to_datetime(db_df["Date"])
            db_df = db_df.drop(columns=["Date"])
        elif db_df.index.name == "Date":
            db_df = db_df.reset_index()
            db_df["date"] = pd.to_datetime(db_df["Date"])
            db_df = db_df.drop(columns=["Date"])
        else:
            raise ValueError("DataFrame must have a date column or index")

        # Add symbol if not present
        if "symbol" not in db_df.columns:
            db_df["symbol"] = symbol

        # Rename columns to match database schema
        column_mapping = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "adj close": "adj_close",
            "adj_close": "adj_close",
            "adjclose": "adj_close",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in db_df.columns and new_col not in db_df.columns:
                db_df[new_col] = db_df[old_col]

        # Select only columns that exist in the table
        table_columns = [col.name for col in MarketData.__table__.columns]
        db_df = db_df[[col for col in db_df.columns if col in table_columns]]

        # Store in database
        try:
            db_df.to_sql(
                MarketData.__tablename__,
                self.engine,
                if_exists=if_exists,
                index=False,
                method="multi",
            )
            logger.info(f"Stored {len(db_df)} rows for {symbol} in database")
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
            raise

    def get_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve market data from database.

        Args:
            symbol: Stock symbol.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            limit: Maximum number of rows to return.

        Returns:
            DataFrame with market data.
        """
        session = self.Session()

        try:
            query = session.query(MarketData).filter(MarketData.symbol == symbol)

            if start_date:
                query = query.filter(MarketData.date >= start_date)
            if end_date:
                query = query.filter(MarketData.date <= end_date)

            query = query.order_by(MarketData.date)

            if limit:
                query = query.limit(limit)

            results = query.all()

            if not results:
                logger.warning(f"No data found for {symbol} in database")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for row in results:
                data.append(
                    {
                        "symbol": row.symbol,
                        "date": row.date,
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "close": row.close,
                        "volume": row.volume,
                        "adj_close": row.adj_close,
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            logger.info(f"Retrieved {len(df)} rows for {symbol} from database")
            return df

        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            raise
        finally:
            session.close()

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve data for multiple symbols.

        Args:
            symbols: List of stock symbols.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.get_data(symbol, start_date, end_date)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to retrieve data for {symbol}: {e}")
                continue

        return data

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols available in database.

        Returns:
            List of symbols.
        """
        session = self.Session()

        try:
            results = (
                session.query(MarketData.symbol)
                .distinct()
                .order_by(MarketData.symbol)
                .all()
            )
            return [row[0] for row in results]
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
        finally:
            session.close()

    def get_date_range(self, symbol: str) -> Dict[str, Any]:
        """Get date range for a symbol in database.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with min_date, max_date, and row_count.
        """
        session = self.Session()

        try:
            # Get min date
            min_date_result = (
                session.query(MarketData.date)
                .filter(MarketData.symbol == symbol)
                .order_by(MarketData.date)
                .first()
            )

            # Get max date
            max_date_result = (
                session.query(MarketData.date)
                .filter(MarketData.symbol == symbol)
                .order_by(MarketData.date.desc())
                .first()
            )

            # Get row count
            row_count_result = (
                session.query(MarketData)
                .filter(MarketData.symbol == symbol)
                .count()
            )

            result = {
                "symbol": symbol,
                "row_count": row_count_result,
            }

            if min_date_result and max_date_result:
                result["min_date"] = min_date_result[0]
                result["max_date"] = max_date_result[0]

            return result

        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return {"symbol": symbol, "row_count": 0}
        finally:
            session.close()

    def delete_symbol(self, symbol: str) -> int:
        """Delete all data for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Number of rows deleted.
        """
        session = self.Session()

        try:
            count = session.query(MarketData).filter(MarketData.symbol == symbol).delete()
            session.commit()
            logger.info(f"Deleted {count} rows for {symbol}")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting data for {symbol}: {e}")
            raise
        finally:
            session.close()

    def cleanup_old_data(self, days_to_keep: int = 365 * 5) -> int:
        """Delete data older than specified number of days.

        Args:
            days_to_keep: Number of days of data to keep.

        Returns:
            Number of rows deleted.
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        session = self.Session()

        try:
            count = (
                session.query(MarketData)
                .filter(MarketData.date < cutoff_date)
                .delete()
            )
            session.commit()
            logger.info(f"Deleted {count} rows older than {cutoff_date}")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old data: {e}")
            raise
        finally:
            session.close()

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information.

        Returns:
            Dictionary with database information.
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        info = {
            "database_url": self.database_url,
            "tables": tables,
            "market_data_columns": [],
        }

        if "market_data" in tables:
            columns = inspector.get_columns("market_data")
            info["market_data_columns"] = [
                {"name": col["name"], "type": str(col["type"])} for col in columns
            ]

        # Get row counts
        session = self.Session()
        try:
            if "market_data" in tables:
                row_count = session.query(MarketData).count()
                symbol_count = session.query(MarketData.symbol).distinct().count()
                info["market_data_rows"] = row_count
                info["market_data_symbols"] = symbol_count
        except Exception as e:
            logger.error(f"Error getting row counts: {e}")
        finally:
            session.close()

        return info