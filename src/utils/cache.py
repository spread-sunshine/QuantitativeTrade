# Cache manager for storing and retrieving data
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
    """Manages caching of data to disk."""

    def __init__(self, cache_dir: Optional[Path] = None, expiry_days: int = 7):
        """Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses DATA_DIR/cache.
            expiry_days: Number of days before cache entries expire.
        """
        if cache_dir is None:
            cache_dir = DATA_DIR / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.expiry_days = expiry_days

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key.

        Args:
            key: Cache key.

        Returns:
            Path to cache file.
        """
        # Create a hash of the key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if a cache file is expired.

        Args:
            cache_path: Path to cache file.

        Returns:
            True if expired, False otherwise.
        """
        if not cache_path.exists():
            return True

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = mtime + timedelta(days=self.expiry_days)
        return datetime.now() > expiry_time

    def get(self, key: str) -> Any:
        """Retrieve data from cache.

        Args:
            key: Cache key.

        Returns:
            Cached data, or None if not found or expired.
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
        """Store data in cache.

        Args:
            key: Cache key.
            data: Data to cache (must be pickleable).
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Cached data for key: {key}")
        except Exception as e:
            logger.error(f"Error caching data for key {key}: {e}")

    def delete(self, key: str) -> bool:
        """Delete data from cache.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False otherwise.
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Deleted cache for key: {key}")
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of cache entries cleared.
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
        """Clear expired cache entries.

        Returns:
            Number of expired cache entries cleared.
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
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
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