"""Memory System Metrics.

Performance tracking and monitoring for the memory system.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
SLOW_OPERATION_THRESHOLD_MS = 100


@dataclass
class OperationMetric:
    """Single operation metric."""

    operation: str
    duration_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates memory system metrics."""

    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector.

        Args:
            window_size: Number of recent operations to keep for each metric
        """
        self.window_size = window_size
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: dict[str, int] = defaultdict(int)
        self.start_time = time.time()

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: dict | None = None,
    ):
        """Record a single operation metric."""
        metric = OperationMetric(
            operation=operation, duration_ms=duration_ms, success=success, metadata=metadata or {}
        )

        self.metrics[operation].append(metric)
        self.counters[f"{operation}_total"] += 1
        if success:
            self.counters[f"{operation}_success"] += 1
        else:
            self.counters[f"{operation}_failure"] += 1

    def get_operation_stats(self, operation: str) -> dict[str, Any]:
        """Get statistics for a specific operation."""
        metrics = list(self.metrics.get(operation, []))

        if not metrics:
            return {
                "operation": operation,
                "count": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "p50_duration_ms": 0.0,
                "p90_duration_ms": 0.0,
                "p99_duration_ms": 0.0,
            }

        durations = [m.duration_ms for m in metrics if m.success]
        success_count = sum(1 for m in metrics if m.success)

        # Calculate percentiles
        durations.sort()

        return {
            "operation": operation,
            "count": len(metrics),
            "success_rate": success_count / len(metrics),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "p50_duration_ms": self._percentile(durations, 50),
            "p90_duration_ms": self._percentile(durations, 90),
            "p99_duration_ms": self._percentile(durations, 99),
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
        }

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of a sorted list."""
        if not values:
            return 0.0

        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all operations."""
        stats = {}

        # Operation-specific stats
        for operation in self.metrics:
            stats[operation] = self.get_operation_stats(operation)

        # Global stats
        stats["global"] = {
            "uptime_seconds": time.time() - self.start_time,
            "total_operations": sum(
                self.counters[k] for k in self.counters if k.endswith("_total")
            ),
            "total_successes": sum(
                self.counters[k] for k in self.counters if k.endswith("_success")
            ),
            "total_failures": sum(
                self.counters[k] for k in self.counters if k.endswith("_failure")
            ),
        }

        return stats

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        with Path(filepath).open("w") as f:
            json.dump(self.get_all_stats(), f, indent=2)


class AsyncMetricsContext:
    """Async context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, operation: str):
        """Initialize metrics context."""
        self.collector = collector
        self.operation = operation
        self.start_time = None
        self.metadata = {}

    async def __aenter__(self):
        """Enter async context."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        duration_ms = (time.time() - self.start_time) * 1000
        success = exc_type is None

        self.collector.record_operation(self.operation, duration_ms, success, self.metadata)

        # Log slow operations
        if duration_ms > SLOW_OPERATION_THRESHOLD_MS:
            logger.warning(f"Slow operation {self.operation}: {duration_ms:.2f}ms")


class MemorySystemMetrics:
    """High-level metrics for the memory system."""

    def __init__(self):
        """Initialize memory system metrics."""
        self.collector = MetricsCollector()
        self.memory_stats = {
            "stm_count": 0,
            "ltm_count": 0,
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def track_operation(self, operation: str) -> AsyncMetricsContext:
        """Create a context manager for tracking an operation."""
        return AsyncMetricsContext(self.collector, operation)

    def increment_counter(self, counter: str, amount: int = 1):
        """Increment a counter metric."""
        self.memory_stats[counter] = self.memory_stats.get(counter, 0) + amount

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        operation_stats = self.collector.get_all_stats()

        # Calculate derived metrics
        cache_total = self.memory_stats["cache_hits"] + self.memory_stats["cache_misses"]
        cache_hit_rate = self.memory_stats["cache_hits"] / cache_total if cache_total > 0 else 0

        return {
            "operations": operation_stats,
            "memory": {
                "stm_count": self.memory_stats["stm_count"],
                "ltm_count": self.memory_stats["ltm_count"],
                "total_memories": (self.memory_stats["stm_count"] + self.memory_stats["ltm_count"]),
                "total_embeddings": self.memory_stats["total_embeddings"],
            },
            "cache": {
                "hit_rate": cache_hit_rate,
                "hits": self.memory_stats["cache_hits"],
                "misses": self.memory_stats["cache_misses"],
            },
            "performance": {
                "avg_ingestion_ms": (
                    operation_stats.get("memory_ingestion", {}).get("avg_duration_ms", 0)
                ),
                "avg_retrieval_ms": (
                    operation_stats.get("memory_retrieval", {}).get("avg_duration_ms", 0)
                ),
                "avg_embedding_ms": (
                    operation_stats.get("embedding_generation", {}).get("avg_duration_ms", 0)
                ),
            },
        }

    async def log_periodic_summary(self, interval_seconds: int = 300):
        """Log metrics summary periodically."""
        while True:
            await asyncio.sleep(interval_seconds)
            summary = self.get_summary()
            logger.info(f"Memory System Metrics Summary: {json.dumps(summary, indent=2)}")


# Global metrics instance
_metrics: MemorySystemMetrics | None = None


def get_metrics() -> MemorySystemMetrics:
    """Get global metrics instance."""
    global _metrics  # noqa: PLW0603
    if _metrics is None:
        _metrics = MemorySystemMetrics()
    return _metrics
