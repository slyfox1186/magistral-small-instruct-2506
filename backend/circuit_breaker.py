#!/usr/bin/env python3
"""Circuit Breaker Pattern Implementation for External Services
Prevents cascading failures from external API timeouts and errors.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back up


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes in half-open to close circuit
    timeout: float = 30.0  # Request timeout in seconds


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation for external service calls.

    Tracks failures and prevents cascading failures by failing fast
    when a service is consistently down.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None

        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_circuit_open_time = 0.0

        logger.info(f"🔧 Circuit breaker initialized for {name}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call through the circuit breaker.

        Args:
            func: The async function to call
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            CircuitBreakerError: When circuit is open
            asyncio.TimeoutError: When call times out
        """
        self.total_requests += 1

        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"🔧 Circuit breaker {self.name}: Transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                # Circuit is open, fail fast
                self.total_failures += 1
                raise CircuitBreakerError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Service unavailable for {time.time() - self.last_failure_time:.1f}s"
                )

        # Execute the function call with timeout
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            await self._on_success()
            return result

        except TimeoutError:
            await self._on_failure(f"Timeout after {self.config.timeout}s")
            raise

        except Exception as e:
            await self._on_failure(str(e))
            raise

    async def _on_success(self):
        """Handle successful function call."""
        self.total_successes += 1
        self.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"🔧 Circuit breaker {self.name}: Success {self.success_count}/"
                f"{self.config.success_threshold} in HALF_OPEN"
            )

            if self.success_count >= self.config.success_threshold:
                logger.info(
                    f"🔧 Circuit breaker {self.name}: Transitioning to CLOSED (service recovered)"
                )
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    async def _on_failure(self, error_message: str):
        """Handle failed function call."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(
            f"🔧 Circuit breaker {self.name}: Failure {self.failure_count}/"
            f"{self.config.failure_threshold} - {error_message}"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state immediately opens circuit
            logger.warning(
                f"🔧 Circuit breaker {self.name}: Failure in HALF_OPEN, transitioning to OPEN"
            )
            self.state = CircuitState.OPEN
            self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"🔧 Circuit breaker {self.name}: Opening circuit after "
                    f"{self.failure_count} failures"
                )
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        current_time = time.time()

        # Calculate uptime percentage
        if self.total_requests > 0:
            success_rate = (self.total_successes / self.total_requests) * 100
        else:
            success_rate = 100.0

        # Calculate time in current state
        if self.state == CircuitState.OPEN and self.last_failure_time:
            time_in_state = current_time - self.last_failure_time
        elif self.last_success_time:
            time_in_state = current_time - self.last_success_time
        else:
            time_in_state = 0.0

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "success_rate": round(success_rate, 2),
            "time_in_current_state": round(time_in_state, 1),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        logger.info(f"🔧 Circuit breaker {self.name}: Manual reset to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(self, name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


# Convenience function for common external services
def get_web_scraper_breaker() -> CircuitBreaker:
    """Get circuit breaker for web scraping service."""
    config = CircuitBreakerConfig(
        failure_threshold=3,  # Open after 3 failures
        recovery_timeout=30.0,  # Try again after 30 seconds
        success_threshold=2,  # Close after 2 successes
        timeout=10.0,  # 10 second timeout for web requests
    )
    return circuit_breaker_manager.get_breaker("web_scraper", config)


def get_api_breaker(service_name: str) -> CircuitBreaker:
    """Get circuit breaker for external API service."""
    config = CircuitBreakerConfig(
        failure_threshold=5,  # Open after 5 failures
        recovery_timeout=60.0,  # Try again after 1 minute
        success_threshold=3,  # Close after 3 successes
        timeout=15.0,  # 15 second timeout for API calls
    )
    return circuit_breaker_manager.get_breaker(f"api_{service_name}", config)
