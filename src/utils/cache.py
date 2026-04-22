# 缓存管理器：用于存储和检索数据
import pickle
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import logging

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class CacheManager:
    """管理数据到磁盘的缓存。"""

    def __init__(self, cache_dir: Optional[Path] = None, expiry_days: int = 7):
        """初始化缓存管理器。

        Args:
            cache_dir: 存储缓存文件的目录。如果为 None，则使用 DATA_DIR/cache。
            expiry_days: 缓存条目过期前的天数。
        """
        if cache_dir is None:
            cache_dir = DATA_DIR / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.expiry_days = expiry_days

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存键对应的文件路径。

        Args:
            key: 缓存键。

        Returns:
            缓存文件路径。
        """
        # 为文件名创建键的哈希值
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _is_expired(self, cache_path: Path) -> bool:
        """检查缓存文件是否已过期。

        Args:
            cache_path: 缓存文件路径。

        Returns:
            如果已过期返回 True，否则返回 False。
        """
        if not cache_path.exists():
            return True

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = mtime + timedelta(days=self.expiry_days)
        return datetime.now() > expiry_time

    def get(self, key: str) -> Any:
        """从缓存中检索数据。

        Args:
            key: 缓存键。

        Returns:
            缓存的数据，如果未找到或已过期则返回 None。
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        if self._is_expired(cache_path):
            logger.debug(f"Cache expired for key: {key}")
            cache_path.unlink(missing_ok=True)
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for key: {key}")
            return data
        except Exception as e:
            logger.warning(f"Error reading cache for key {key}: {e}")
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, data: Any) -> None:
        """将数据存储到缓存中。

        Args:
            key: 缓存键。
            data: 要缓存的数据（必须可被 pickle 序列化）。
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Cached data for key: {key}")
        except Exception as e:
            logger.error(f"Error caching data for key {key}: {e}")

    def delete(self, key: str) -> bool:
        """从缓存中删除数据。

        Args:
            key: 缓存键。

        Returns:
            如果删除成功返回 True，否则返回 False。
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Deleted cache for key: {key}")
            return True
        return False

    def clear(self) -> int:
        """清除所有缓存条目。

        Returns:
            清除的缓存条目数量。
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cache entries")
        return count

    def clear_expired(self) -> int:
        """清除已过期的缓存条目。

        Returns:
            清除的过期缓存条目数量。
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting expired cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} expired cache entries")
        return count

    def get_stats(self) -> dict:
        """获取缓存统计信息。

        Returns:
            包含缓存统计信息的字典。
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())

        expired_files = [f for f in cache_files if self._is_expired(f)]

        return {
            "total_entries": len(cache_files),
            "expired_entries": len(expired_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "expiry_days": self.expiry_days,
        }