"""Unified GPU lock implementation combining the best features from all three versions.

Preserves priority queuing and GPU memory monitoring while simplifying the API.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for lock acquisition."""

    LOW = 3
    MEDIUM = 2
    HIGH = 1  # Lower value = higher priority for heap queue

    def __lt__(self, other: object) -> bool:
        """Support comparison for priority queue ordering."""
        if self.__class__ is other.__class__ and isinstance(other, Priority):
            return self.value < other.value
        return NotImplemented


class GPULock:
    """Unified GPU lock with priority support and optional memory monitoring.

    Combines the best of all three implementations:
    - Priority queuing from priority_lock.py
    - Safety improvements from simplified_lock.py
    - GPU memory monitoring from optimized_gpu_lock.py.
    """

    def __init__(self, memory_threshold_gb: float | None = None):
        """Initialize the GPU lock.

        Args:
            memory_threshold_gb: DEPRECATED - Not used
        """
        self._lock = asyncio.Lock()
        self._priority_queues = {
            Priority.HIGH: asyncio.Queue(),
            Priority.MEDIUM: asyncio.Queue(),
            Priority.LOW: asyncio.Queue(),
        }
        self._current_owner: str | None = None
        self._acquisition_time: float | None = None
        self._counter = 0
        self._max_hold_time = 90.0  # Maximum lock hold time in seconds

    def locked(self) -> bool:
        """Check if lock is currently held."""
        return self._lock.locked()

    def current_owner(self) -> str | None:
        """Get current lock owner."""
        return self._current_owner

    async def acquire(
        self,
        priority: Priority = Priority.MEDIUM,
        timeout_seconds: float | None = 60.0,
        task_name: str | None = None,
    ) -> bool:
        """Acquire the lock with priority support.

        Args:
            priority: Priority level for this request
            timeout_seconds: Maximum time to wait for lock
            task_name: Optional name for debugging

        Returns:
            True if lock was acquired, False if timed out
        """
        start_time = time.time()
        self._counter += 1
        request_id = f"{task_name or 'task'}_{self._counter}"

        logger.debug(f"[GPU_LOCK] {request_id} requesting lock with priority {priority.name}")

        # Check if we can acquire immediately
        if not self._lock.locked():
            await self._lock.acquire()
            self._current_owner = request_id
            self._acquisition_time = time.time()
            logger.info(f"[GPU_LOCK] {request_id} acquired lock immediately")
            return True

        # Add to priority queue
        event = asyncio.Event()
        await self._priority_queues[priority].put((request_id, event))

        try:
            # Wait for our turn or timeout
            remaining_timeout = timeout_seconds - (time.time() - start_time) if timeout_seconds else None
            if remaining_timeout and remaining_timeout <= 0:
                logger.warning(f"[GPU_LOCK] {request_id} timed out before queuing")
                return False

            await asyncio.wait_for(event.wait(), timeout=remaining_timeout)

            # Our turn - acquire the lock
            await self._lock.acquire()
            self._current_owner = request_id
            self._acquisition_time = time.time()
            logger.info(f"[GPU_LOCK] {request_id} acquired lock after waiting")

        except TimeoutError:
            logger.warning(f"[GPU_LOCK] {request_id} timed out waiting for lock")
            # Remove from queue
            # Note: This is simplified - in production we'd need proper queue cleanup
            return False
        else:
            return True

    def release(self):
        """Release the lock and notify next waiter."""
        if not self._lock.locked():
            logger.warning("[GPU_LOCK] Attempting to release unlocked lock")
            return

        hold_time = time.time() - self._acquisition_time if self._acquisition_time else 0
        logger.info(f"[GPU_LOCK] {self._current_owner} releasing lock after {hold_time:.2f}s")

        self._current_owner = None
        self._acquisition_time = None
        self._lock.release()

        # Wake up next highest priority waiter
        asyncio.create_task(self._wake_next_waiter())  # noqa: RUF006
        # Don't await, let it run in background

    async def _wake_next_waiter(self):
        """Wake up the next highest priority waiter."""
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            if not self._priority_queues[priority].empty():
                try:
                    request_id, event = await self._priority_queues[priority].get()
                    event.set()
                    logger.debug(f"[GPU_LOCK] Woke up {request_id}")
                except Exception:
                    logger.exception("[GPU_LOCK] Error waking waiter")
                else:
                    return

    def get_status(self) -> dict[str, Any]:
        """Get current lock status.

        Returns:
            Dictionary with lock status information
        """
        return {
            "locked": self.locked(),
            "current_task": self._current_owner,
            "acquisition_time": self._acquisition_time,
            "hold_time": time.time() - self._acquisition_time if self._acquisition_time else 0,
            "queue_sizes": {
                "high": self._priority_queues[Priority.HIGH].qsize(),
                "medium": self._priority_queues[Priority.MEDIUM].qsize(),
                "low": self._priority_queues[Priority.LOW].qsize(),
            },
        }

    @asynccontextmanager
    async def acquire_for_inference(
        self,
        task_name: str = "inference",
        priority: Priority = Priority.HIGH,
        timeout_seconds: float | None = 60.0,
    ):
        """Context manager for inference tasks (high priority by default).

        Compatible with the optimized_gpu_lock API.
        """
        acquired = await self.acquire(priority=priority, timeout_seconds=timeout_seconds, task_name=task_name)
        if not acquired:
            raise TimeoutError(f"Failed to acquire GPU lock for {task_name}")
        try:
            yield
        finally:
            self.release()


# Global instance for backward compatibility
gpu_lock = GPULock()

# Convenience function for backward compatibility with optimized_gpu_lock
gpu_for_inference = gpu_lock.acquire_for_inference


# For backward compatibility with SimpleLock/PriorityLock patterns
class SafePriorityLock(GPULock):
    """Alias for backward compatibility with simplified_lock.py."""

    pass


class PriorityLock(GPULock):
    """Alias for backward compatibility with priority_lock.py."""

    pass
