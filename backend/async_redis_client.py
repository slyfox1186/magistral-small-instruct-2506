"""🚀 ASYNC-NATIVE RESILIENT REDIS CLIENT

This module provides an async-native Redis client with built-in resilience features.
It inherits from redis.asyncio.Redis to ensure full API compatibility while adding
retry logic and circuit breaker patterns.

Key Features:
- Native async/await support (no thread pool overhead)
- Full Redis API compatibility through inheritance
- Automatic retry with exponential backoff
- Circuit breaker pattern for failover scenarios
- Zero API changes required in consuming code
"""

import asyncio
import logging
import time
from enum import Enum

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, skip Redis calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class AsyncCircuitBreaker:
    """Async-safe circuit breaker implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self._lock = asyncio.Lock()

    async def record_success(self):
        """Record successful call"""
        async with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                logger.info("🔌 Circuit breaker: Redis recovered, closing circuit")
                self.state = CircuitState.CLOSED

    async def record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.warning(
                        f"⚡ Circuit breaker: Opening after {self.failure_count} Redis failures"
                    )
                    self.state = CircuitState.OPEN

            elif self.state == CircuitState.HALF_OPEN:
                logger.warning("⚡ Circuit breaker: Recovery test failed, reopening")
                self.state = CircuitState.OPEN

    async def should_attempt_call(self) -> bool:
        """Check if we should attempt Redis call"""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info("🔌 Circuit breaker: Testing Redis recovery")
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False

            # HALF_OPEN
            return True


class AsyncResilientRedisClient(aioredis.Redis):
    """Async-native Redis client with resilience features.
    Inherits full Redis API and adds retry logic.
    """

    def __init__(
        self,
        *args,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        circuit_breaker: AsyncCircuitBreaker | None = None,
        **kwargs,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker = circuit_breaker or AsyncCircuitBreaker()
        # Initialize parent Redis client
        super().__init__(*args, **kwargs)
        logger.info(f"🚀 AsyncResilientRedisClient initialized with {max_retries} retries")

    async def execute_command(self, *args, **kwargs):
        """Override core command execution to add resilience.
        This is called by ALL Redis commands (GET, SET, HSET, etc.)
        """
        # Check circuit breaker first
        if not await self.circuit_breaker.should_attempt_call():
            raise ConnectionError("Circuit breaker is OPEN - Redis unavailable")

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Call parent's execute_command
                result = await super().execute_command(*args, **kwargs)
                await self.circuit_breaker.record_success()
                return result

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                await self.circuit_breaker.record_failure()

                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor * (2**attempt)
                    logger.warning(
                        f"Redis error: '{e}'. Retry {attempt + 1}/"
                        f"{self.max_retries} in {sleep_time}s"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Redis command failed after {self.max_retries} attempts: {e}")

        raise last_exception


async def initialize_async_redis_connection(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str | None = None,
    max_retries: int = 3,
    decode_responses: bool = True,
    **kwargs,
) -> AsyncResilientRedisClient:
    """Initialize async Redis connection with resilience features.

    Returns:
        AsyncResilientRedisClient that can be used exactly like redis.asyncio.Redis
    """
    # Create connection pool for better performance
    pool = aioredis.ConnectionPool(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=decode_responses,
        max_connections=20,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
        **kwargs,
    )

    # Create resilient client with the pool
    client = AsyncResilientRedisClient(connection_pool=pool, max_retries=max_retries)

    # Test connection
    try:
        await client.ping()
        logger.info(f"✅ Async Redis connected successfully to {host}:{port}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise

    return client
