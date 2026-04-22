# 数据库管理器，用于存储和检索市场数据
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
    """市场数据表结构。"""

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

    # 添加唯一约束
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_market_data_symbol_date'),
    )


class DatabaseManager:
    """管理市场数据的数据库操作。"""

    def __init__(self, database_url: str = DATABASE_URL):
        """初始化数据库管理器。

        Args:
            database_url: SQLAlchemy数据库URL。
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

        # 如果表不存在，则创建它们
        self._create_tables()

    def _create_tables(self):
        """如果数据库表不存在，则创建它们。"""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database tables created/verified at {self.database_url}")

    def store_data(self, df: pd.DataFrame, symbol: str, if_exists: str = "append"):
        """将市场数据存储到数据库中。

        Args:
            df: 包含市场数据的DataFrame。
            symbol: 股票代码。
            if_exists: 如果数据已存在时的处理方式（'fail'、'replace'、'append'）。
        """
        if df.empty:
            logger.warning(f"No data to store for {symbol}")
            return

        # 确保必需的列存在
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # 为数据库准备DataFrame
        db_df = df.copy()

        # 确保日期列是datetime类型
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

        # 如果不存在，则添加股票代码列
        if "symbol" not in db_df.columns:
            db_df["symbol"] = symbol

        # 重命名列以匹配数据库模式
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

        # 仅选择表中存在的列
        table_columns = [col.name for col in MarketData.__table__.columns]
        db_df = db_df[[col for col in db_df.columns if col in table_columns]]

        # 存储到数据库
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
        """从数据库检索市场数据。

        Args:
            symbol: 股票代码。
            start_date: 开始日期，格式为'YYYY-MM-DD'。
            end_date: 结束日期，格式为'YYYY-MM-DD'。
            limit: 返回的最大行数。

        Returns:
            包含市场数据的DataFrame。
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
        """检索多个股票代码的数据。

        Args:
            symbols: 股票代码列表。
            start_date: 开始日期。
            end_date: 结束日期。

        Returns:
            映射股票代码到DataFrame的字典。
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
        """获取数据库中可用的股票代码列表。

        Returns:
            股票代码列表。
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
        """获取数据库中某个股票代码的日期范围。

        Args:
            symbol: 股票代码。

        Returns:
            包含最小日期、最大日期和行数的字典。
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
        """删除某个股票代码的所有数据。

        Args:
            symbol: 股票代码。

        Returns:
            删除的行数。
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
        """删除早于指定天数的数据。

        Args:
            days_to_keep: 要保留的数据天数。

        Returns:
            删除的行数。
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
        """获取数据库信息。

        Returns:
            包含数据库信息的字典。
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