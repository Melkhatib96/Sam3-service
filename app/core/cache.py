"""
Simple in-memory TTL cache for API key lookups.
Identical to the pattern used by other services in this stack.
"""
from typing import Optional, Dict
from datetime import datetime, timedelta
import threading


class APIKeyCache:
    """Thread-safe in-memory cache for API key lookups with TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, tuple] = {}  # key -> (user_data, expiry)
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_seconds

    def get(self, api_key: str) -> Optional[dict]:
        with self._lock:
            if api_key not in self._cache:
                return None
            user_data, expiry = self._cache[api_key]
            if datetime.utcnow() > expiry:
                del self._cache[api_key]
                return None
            return user_data

    def set(self, api_key: str, user_data: dict):
        with self._lock:
            expiry = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
            self._cache[api_key] = (user_data, expiry)

    def invalidate(self, api_key: str):
        with self._lock:
            self._cache.pop(api_key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()


# Global singleton
api_key_cache = APIKeyCache(ttl_seconds=3600)
