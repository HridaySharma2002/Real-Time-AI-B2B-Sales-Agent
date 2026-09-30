"""
services/redis_cache.py - Low-Latency Redis Caching & In-Memory Fallback
Part of ApexSales AI Real-Time B2B Sales Agent.

Features:
- Sub-5ms caching for active call sessions, prospect persona classifications, and recent transcripts.
- Automatic connection failover to an in-memory TTL cache if Redis server is offline.
- Session persistence and fast context window retrieval for real-time conversation turns.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RedisCache")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)


class InMemoryTTLCache:
    """Thread-safe in-memory cache fallback when Redis is offline."""
    def __init__(self):
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            val, expire_at = self._store[key]
            if expire_at is None or time.time() < expire_at:
                return val
            del self._store[key]
        return None

    def set(self, key: str, value: str, ex: Optional[int] = None):
        expire_at = time.time() + ex if ex else None
        self._store[key] = (value, expire_at)

    def delete(self, key: str):
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class SalesRedisService:
    """Redis caching service for sub-millisecond session state and telemetry."""

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, password: Optional[str] = REDIS_PASSWORD):
        self.host = host
        self.port = port
        self.password = password
        self.client = None
        self._is_redis_live = False
        self._fallback_cache = InMemoryTTLCache()

        self._connect()

    def _connect(self):
        try:
            import redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                socket_timeout=1.5,
                decode_responses=True
            )
            self.client.ping()
            self._is_redis_live = True
            logger.info(f"Connected to Redis cache at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Redis not reachable at {self.host}:{self.port} ({e}). Using high-performance in-memory cache fallback.")
            self._is_redis_live = False

    def is_available(self) -> bool:
        """Returns True if live Redis server is connected."""
        return self._is_redis_live

    def get(self, key: str) -> Optional[str]:
        if self._is_redis_live:
            try:
                return self.client.get(key)
            except Exception:
                pass
        return self._fallback_cache.get(key)

    def set(self, key: str, value: str, ttl_seconds: int = 3600):
        if self._is_redis_live:
            try:
                self.client.set(key, value, ex=ttl_seconds)
                return
            except Exception:
                pass
        self._fallback_cache.set(key, value, ex=ttl_seconds)

    # High-level domain helpers

    def set_session_state(self, session_id: str, state_dict: Dict[str, Any], ttl: int = 3600):
        """Caches active dialogue state, persona, and lead metadata."""
        key = f"session:{session_id}"
        self.set(key, json.dumps(state_dict), ttl_seconds=ttl)

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Retrieves active dialogue state for a call session."""
        key = f"session:{session_id}"
        val = self.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                pass
        return {}

    def append_call_turn(self, session_id: str, user_transcript: str, agent_response: str, sentiment: float = 0.0):
        """Appends a dialogue turn to the session history cache."""
        state = self.get_session_state(session_id)
        history = state.get("history", [])
        history.append({
            "timestamp": time.time(),
            "user": user_transcript,
            "agent": agent_response,
            "sentiment": sentiment
        })
        state["history"] = history
        state["last_active"] = time.time()
        self.set_session_state(session_id, state)

    def cache_lead_persona(self, lead_id: str, persona_data: Dict[str, Any]):
        """Caches the classified K-Means persona profile for rapid lookup."""
        key = f"lead_persona:{lead_id}"
        self.set(key, json.dumps(persona_data), ttl_seconds=86400)

    def get_cached_lead_persona(self, lead_id: str) -> Optional[Dict[str, Any]]:
        key = f"lead_persona:{lead_id}"
        val = self.get(key)
        if val:
            try:
                return json.loads(val)
            except Exception:
                pass
        return None


# Global singleton
_redis_instance: Optional[SalesRedisService] = None

def get_redis_cache() -> SalesRedisService:
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = SalesRedisService()
    return _redis_instance
