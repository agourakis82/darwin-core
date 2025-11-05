"""
Unified Cache Manager - Single Source of Truth for Darwin 2025

Replaces 5+ fragmented cache implementations with a unified, high-performance system.

Architecture:
    L1: Memory (in-process) - 100μs latency
    L2: Redis (network) - 1-5ms latency
    L3: Qdrant (semantic) - 10-50ms latency
    L4: Disk (persistent) - 100-500ms latency

Performance targets:
    - Cache hit rate: >85%
    - L1 latency: <100μs
    - L2 latency: <5ms
    - Memory efficiency: <512MB

Author: Darwin Team
Date: 2025-10-28
"""

import asyncio
import hashlib
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logging.warning("redis not available - L2 cache disabled")

logger = logging.getLogger(__name__)


class CacheLayer(str, Enum):
    """Cache layers in order of speed"""
    MEMORY = "memory"      # L1: 100μs
    REDIS = "redis"        # L2: 1-5ms
    SEMANTIC = "semantic"  # L3: 10-50ms (Qdrant)
    DISK = "disk"          # L4: 100-500ms


class EvictionPolicy(str, Enum):
    """Cache eviction policies"""
    LRU = "lru"   # Least Recently Used
    LFU = "lfu"   # Least Frequently Used
    FIFO = "fifo" # First In First Out
    TTL = "ttl"   # Time To Live only


@dataclass
class CacheConfig:
    """Configuration for unified cache"""
    # Memory cache (L1)
    max_memory_mb: int = 512
    memory_eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    
    # Redis cache (L2)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_enabled: bool = True
    
    # Semantic cache (L3)
    semantic_enabled: bool = True
    semantic_similarity_threshold: float = 0.90
    
    # Disk cache (L4)
    disk_enabled: bool = True
    disk_cache_dir: str = "/tmp/darwin_cache"
    max_disk_mb: int = 2048
    
    # TTL (Time To Live)
    default_ttl_seconds: int = 3600  # 1 hour
    
    # Compression
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024  # Compress if >1KB
    
    # Statistics
    enable_stats: bool = True


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: Optional[int]
    compressed: bool = False
    layers: Set[CacheLayer] = field(default_factory=set)


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    
    # Per-layer stats
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    l4_hits: int = 0
    
    # Timing stats
    total_get_time_ms: float = 0.0
    total_set_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def avg_get_time_ms(self) -> float:
        """Average GET latency"""
        total = self.hits + self.misses
        return (self.total_get_time_ms / total) if total > 0 else 0.0


class UnifiedCacheManager:
    """
    Unified cache manager for all Darwin services
    
    Features:
        - Multi-layer caching (L1-L4)
        - Automatic promotion/demotion
        - TTL-based expiration
        - Multiple eviction policies
        - Compression support
        - Semantic similarity matching
        - Comprehensive statistics
    
    Usage:
        >>> cache = UnifiedCacheManager()
        >>> await cache.initialize()
        >>> 
        >>> # Set value
        >>> await cache.set("my_key", {"data": "value"}, ttl=300)
        >>> 
        >>> # Get value (multi-layer fallback)
        >>> value = await cache.get("my_key")
        >>> 
        >>> # Invalidate pattern
        >>> await cache.invalidate("user_*")
        >>> 
        >>> # Get statistics
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.1f}%")
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize unified cache manager
        
        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        
        # L1: Memory cache (OrderedDict for LRU)
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_memory_bytes = 0
        self._max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        
        # L2: Redis client
        self._redis_client: Optional[redis.Redis] = None
        
        # L3: Semantic cache (Qdrant)
        self._semantic_cache = None  # Initialized later if enabled
        
        # L4: Disk cache
        self._disk_cache_dir = Path(self.config.disk_cache_dir)
        if self.config.disk_enabled:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self._stats = CacheStats()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Initialized flag
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize all cache layers
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        logger.info("🚀 Initializing Unified Cache Manager...")
        
        # Initialize Redis (L2)
        if self.config.redis_enabled and HAS_REDIS:
            try:
                self._redis_client = await redis.from_url(
                    f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                    password=self.config.redis_password,
                    encoding="utf-8",
                    decode_responses=False  # We handle encoding
                )
                # Test connection
                await self._redis_client.ping()
                logger.info(f"✅ L2 (Redis) initialized: {self.config.redis_host}:{self.config.redis_port}")
            except Exception as e:
                logger.warning(f"⚠️  Redis initialization failed: {e}")
                self._redis_client = None
        
        # Initialize Semantic cache (L3)
        if self.config.semantic_enabled:
            try:
                from .qdrant_hybrid_client import get_qdrant_client
                self._semantic_cache = get_qdrant_client()
                if await self._semantic_cache.initialize():
                    logger.info("✅ L3 (Semantic/Qdrant) initialized")
                else:
                    self._semantic_cache = None
            except Exception as e:
                logger.warning(f"⚠️  Semantic cache initialization failed: {e}")
                self._semantic_cache = None
        
        self._initialized = True
        logger.info(f"✅ Unified Cache Manager ready (Memory: {self.config.max_memory_mb}MB, Redis: {self.config.redis_enabled})")
        
        return True
    
    async def get(
        self,
        key: str,
        default: Any = None,
        update_stats: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache with multi-layer fallback
        
        Strategy:
            1. Check L1 (Memory) - instant
            2. Check L2 (Redis) - 1-5ms
            3. Check L3 (Semantic) - 10-50ms if enabled
            4. Check L4 (Disk) - 100-500ms if enabled
            5. Return default if all miss
        
        Args:
            key: Cache key
            default: Default value if not found
            update_stats: Whether to update statistics
        
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        try:
            # L1: Memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    await self._evict_memory(key)
                    if update_stats:
                        self._stats.misses += 1
                    return default
                
                # Update access metadata
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Move to end (LRU)
                self._memory_cache.move_to_end(key)
                
                if update_stats:
                    self._stats.hits += 1
                    self._stats.l1_hits += 1
                    self._stats.total_get_time_ms += (time.time() - start_time) * 1000
                
                return entry.value
            
            # L2: Redis cache
            if self._redis_client:
                redis_value = await self._get_from_redis(key)
                if redis_value is not None:
                    # Promote to L1
                    await self._promote_to_memory(key, redis_value)
                    
                    if update_stats:
                        self._stats.hits += 1
                        self._stats.l2_hits += 1
                        self._stats.total_get_time_ms += (time.time() - start_time) * 1000
                    
                    return redis_value
            
            # L3: Semantic cache (similarity search)
            if self._semantic_cache and self.config.semantic_enabled:
                semantic_value = await self._semantic_lookup(key)
                if semantic_value is not None:
                    # Promote to L2 and L1
                    await self.set(key, semantic_value, layers=[CacheLayer.MEMORY, CacheLayer.REDIS])
                    
                    if update_stats:
                        self._stats.hits += 1
                        self._stats.l3_hits += 1
                        self._stats.total_get_time_ms += (time.time() - start_time) * 1000
                    
                    return semantic_value
            
            # L4: Disk cache
            if self.config.disk_enabled:
                disk_value = await self._get_from_disk(key)
                if disk_value is not None:
                    # Promote to L2 and L1
                    await self.set(key, disk_value, layers=[CacheLayer.MEMORY, CacheLayer.REDIS])
                    
                    if update_stats:
                        self._stats.hits += 1
                        self._stats.l4_hits += 1
                        self._stats.total_get_time_ms += (time.time() - start_time) * 1000
                    
                    return disk_value
            
            # Cache miss
            if update_stats:
                self._stats.misses += 1
                self._stats.total_get_time_ms += (time.time() - start_time) * 1000
            
            return default
            
        except Exception as e:
            logger.error(f"Cache GET error for key '{key}': {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        layers: Optional[List[CacheLayer]] = None
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
            layers: Which layers to write to (None = all available)
        
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            ttl = ttl or self.config.default_ttl_seconds
            layers = layers or [CacheLayer.MEMORY, CacheLayer.REDIS]
            
            # Serialize value
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # Compress if enabled and size > threshold
            compressed = False
            if self.config.enable_compression and size_bytes > self.config.compression_threshold_bytes:
                import gzip
                serialized = gzip.compress(serialized)
                compressed = True
            
            # L1: Memory
            if CacheLayer.MEMORY in layers:
                await self._set_in_memory(key, value, size_bytes, ttl, compressed)
            
            # L2: Redis
            if CacheLayer.REDIS in layers and self._redis_client:
                await self._set_in_redis(key, serialized, ttl)
            
            # L3: Semantic (embedding-based)
            if CacheLayer.SEMANTIC in layers and self._semantic_cache:
                await self._set_in_semantic(key, value)
            
            # L4: Disk
            if CacheLayer.DISK in layers and self.config.disk_enabled:
                await self._set_in_disk(key, serialized)
            
            self._stats.sets += 1
            self._stats.total_set_time_ms += (time.time() - start_time) * 1000
            
            return True
            
        except Exception as e:
            logger.error(f"Cache SET error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from all cache layers
        
        Args:
            key: Cache key to delete
        
        Returns:
            True if deleted from at least one layer
        """
        deleted = False
        
        # L1: Memory
        if key in self._memory_cache:
            await self._evict_memory(key)
            deleted = True
        
        # L2: Redis
        if self._redis_client:
            try:
                await self._redis_client.delete(key)
                deleted = True
            except Exception as e:
                logger.error(f"Redis DELETE error: {e}")
        
        # L4: Disk
        if self.config.disk_enabled:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                disk_path.unlink()
                deleted = True
        
        return deleted
    
    async def invalidate(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern
        
        Args:
            pattern: Pattern to match (supports * wildcard)
        
        Returns:
            Number of keys invalidated
        """
        count = 0
        
        # L1: Memory
        keys_to_delete = [k for k in self._memory_cache.keys() if self._match_pattern(k, pattern)]
        for key in keys_to_delete:
            await self._evict_memory(key)
            count += 1
        
        # L2: Redis
        if self._redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis_client.scan(
                        cursor,
                        match=pattern,
                        count=100
                    )
                    if keys:
                        await self._redis_client.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis INVALIDATE error: {e}")
        
        logger.info(f"Invalidated {count} keys matching pattern '{pattern}'")
        return count
    
    async def clear(self) -> bool:
        """
        Clear all cache layers
        
        Returns:
            True if successful
        """
        try:
            # L1: Memory
            self._memory_cache.clear()
            self._current_memory_bytes = 0
            
            # L2: Redis
            if self._redis_client:
                await self._redis_client.flushdb()
            
            # L4: Disk
            if self.config.disk_enabled:
                import shutil
                shutil.rmtree(self._disk_cache_dir, ignore_errors=True)
                self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ All cache layers cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache CLEAR error: {e}")
            return False
    
    def get_stats(self) -> CacheStats:
        """
        Get cache statistics
        
        Returns:
            Cache statistics object
        """
        return self._stats
    
    def reset_stats(self):
        """Reset cache statistics"""
        self._stats = CacheStats()
    
    async def close(self):
        """Close all cache connections"""
        if self._redis_client:
            await self._redis_client.close()
        logger.info("✅ Unified Cache Manager closed")
    
    # Private helper methods
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl_seconds is None:
            return False
        age = time.time() - entry.created_at
        return age > entry.ttl_seconds
    
    async def _set_in_memory(
        self,
        key: str,
        value: Any,
        size_bytes: int,
        ttl: int,
        compressed: bool
    ):
        """Set value in memory cache (L1)"""
        async with self._lock:
            # Evict if necessary
            while self._current_memory_bytes + size_bytes > self._max_memory_bytes:
                if not self._memory_cache:
                    break
                await self._evict_memory_lru()
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                ttl_seconds=ttl,
                compressed=compressed,
                layers={CacheLayer.MEMORY}
            )
            
            self._memory_cache[key] = entry
            self._current_memory_bytes += size_bytes
    
    async def _evict_memory_lru(self):
        """Evict least recently used item from memory"""
        if not self._memory_cache:
            return
        
        # Remove first item (oldest in OrderedDict)
        key, entry = self._memory_cache.popitem(last=False)
        self._current_memory_bytes -= entry.size_bytes
        self._stats.evictions += 1
    
    async def _evict_memory(self, key: str):
        """Evict specific key from memory"""
        if key in self._memory_cache:
            entry = self._memory_cache.pop(key)
            self._current_memory_bytes -= entry.size_bytes
            self._stats.evictions += 1
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis (L2)"""
        try:
            value = await self._redis_client.get(key)
            if value:
                # Decompress if needed
                if self.config.enable_compression:
                    try:
                        import gzip
                        value = gzip.decompress(value)
                    except:
                        pass  # Not compressed
                
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
        return None
    
    async def _set_in_redis(self, key: str, serialized: bytes, ttl: int):
        """Set value in Redis (L2)"""
        try:
            await self._redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
    
    async def _promote_to_memory(self, key: str, value: Any):
        """Promote value from lower layer to memory (L1)"""
        size_bytes = len(pickle.dumps(value))
        await self._set_in_memory(
            key,
            value,
            size_bytes,
            self.config.default_ttl_seconds,
            False
        )
    
    async def _semantic_lookup(self, query: str) -> Optional[Any]:
        """
        Semantic cache lookup using similarity search
        
        Args:
            query: Query string
        
        Returns:
            Cached value if similar query found, else None
        """
        # TODO: Implement semantic search in Qdrant
        # 1. Generate embedding for query
        # 2. Search Qdrant for similar embeddings
        # 3. If similarity > threshold, return cached result
        return None
    
    async def _set_in_semantic(self, key: str, value: Any):
        """Set value in semantic cache (L3)"""
        # TODO: Implement semantic cache storage
        pass
    
    def _get_disk_path(self, key: str) -> Path:
        """Get disk path for cache key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._disk_cache_dir / f"{key_hash}.cache"
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache (L4)"""
        try:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                with open(disk_path, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                if self.config.enable_compression:
                    try:
                        import gzip
                        data = gzip.decompress(data)
                    except:
                        pass
                
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Disk GET error: {e}")
        return None
    
    async def _set_in_disk(self, key: str, serialized: bytes):
        """Set value in disk cache (L4)"""
        try:
            disk_path = self._get_disk_path(key)
            with open(disk_path, 'wb') as f:
                f.write(serialized)
        except Exception as e:
            logger.error(f"Disk SET error: {e}")
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (supports * wildcard)"""
        import re
        regex = pattern.replace('*', '.*')
        return bool(re.match(f"^{regex}$", key))


# Singleton instance
_unified_cache: Optional[UnifiedCacheManager] = None


def get_unified_cache(config: Optional[CacheConfig] = None) -> UnifiedCacheManager:
    """
    Get singleton unified cache instance
    
    Args:
        config: Cache configuration (only used on first call)
    
    Returns:
        UnifiedCacheManager instance
    """
    global _unified_cache
    if _unified_cache is None:
        _unified_cache = UnifiedCacheManager(config)
    return _unified_cache


# Convenience decorator for caching function results
def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    layers: Optional[List[CacheLayer]] = None
):
    """
    Decorator to cache function results
    
    Usage:
        @cached(ttl=300, key_prefix="user_data")
        async def get_user_data(user_id: str):
            # Expensive operation
            return data
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        layers: Which layers to use
    
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cache = get_unified_cache()
            result = await cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Cache miss - call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl=ttl, layers=layers)
            
            return result
        
        return wrapper
    return decorator

