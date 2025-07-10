"""📊 PRODUCTION MONITORING & HEALTH CHECK SYSTEM

This module provides comprehensive health monitoring, metrics collection, and
diagnostic endpoints for production deployment. It tracks system health,
performance metrics, and provides actionable insights.

Key Features:
- Health check endpoints with dependency status
- Performance metrics collection
- Memory system analytics
- Resource usage monitoring
- Prometheus-compatible metrics export
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

from gpu_lock import Priority

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a system component"""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    error_message: str | None = None
    last_check: float = 0
    metadata: dict[str, Any] = None


@dataclass
class SystemMetrics:
    """System performance metrics"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    active_threads: int
    request_count: int
    error_count: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float


class MetricsCollector:
    """Collects and aggregates system metrics"""

    def __init__(self, window_size: int = 300):  # 5-minute window
        self.window_size = window_size
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.component_checks = {}

        # Initialize process handle
        self.process = psutil.Process()

        # Metrics history
        self.metrics_history = deque(maxlen=window_size)

        # Start background metrics collection
        asyncio.create_task(self._collect_metrics_loop())

    async def _collect_metrics_loop(self):
        """Background loop to collect metrics"""
        while True:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(1)  # Collect every second
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)

    def _collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Disk usage
        disk_usage = psutil.disk_usage("/")
        disk_percent = disk_usage.percent

        # Thread count
        thread_count = self.process.num_threads()

        # Request metrics
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())

        # Response time percentiles
        if self.request_times:
            sorted_times = sorted(self.request_times)
            avg_time = sum(sorted_times) / len(sorted_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else 0
            p99_time = sorted_times[p99_index] if p99_index < len(sorted_times) else 0
        else:
            avg_time = p95_time = p99_time = 0

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_usage_percent=disk_percent,
            active_threads=thread_count,
            request_count=total_requests,
            error_count=total_errors,
            avg_response_time_ms=avg_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
        )

    def record_request(self, endpoint: str, duration_ms: float, success: bool = True):
        """Record request metrics"""
        self.request_times.append(duration_ms)
        self.request_counts[endpoint] += 1

        if not success:
            self.error_counts[endpoint] += 1

    def get_current_metrics(self) -> SystemMetrics:
        """Get latest metrics snapshot"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self._collect_current_metrics()

    def get_metrics_summary(self, window_minutes: int = 5) -> dict[str, Any]:
        """Get metrics summary for the specified window"""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"error": "No metrics available"}

        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)
        max_memory = max(m.memory_mb for m in recent_metrics)

        # Error rate
        total_requests = recent_metrics[-1].request_count - recent_metrics[0].request_count
        total_errors = recent_metrics[-1].error_count - recent_metrics[0].error_count
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        return {
            "window_minutes": window_minutes,
            "samples": len(recent_metrics),
            "avg_cpu_percent": round(avg_cpu, 2),
            "avg_memory_mb": round(avg_memory, 2),
            "max_memory_mb": round(max_memory, 2),
            "total_requests": total_requests,
            "error_rate_percent": round(error_rate, 2),
            "current_metrics": asdict(recent_metrics[-1]) if recent_metrics else None,
        }


class HealthChecker:
    """Manages health checks for all system components"""

    def __init__(self):
        self.components = {}
        self.metrics_collector = MetricsCollector()
        self.check_timeout = 5.0  # 5 second timeout for checks

    async def check_component(
        self, name: str, check_func, timeout: float | None = None
    ) -> ComponentHealth:
        """Check health of a component"""
        timeout = timeout or self.check_timeout
        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(check_func(), timeout=timeout)
            latency_ms = (time.time() - start_time) * 1000

            # Determine status based on result
            if result is True:
                status = HealthStatus.HEALTHY
                error_msg = None
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", "healthy"))
                error_msg = result.get("error")
                metadata = result.get("metadata", {})
            else:
                status = HealthStatus.DEGRADED
                error_msg = "Unexpected check result"

            health = ComponentHealth(
                name=name,
                status=status,
                latency_ms=latency_ms,
                error_message=error_msg,
                last_check=time.time(),
                metadata=metadata if "metadata" in locals() else None,
            )

        except TimeoutError:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                error_message=f"Health check timed out after {timeout}s",
                last_check=time.time(),
            )
        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                error_message=str(e),
                last_check=time.time(),
            )

        self.components[name] = health
        return health

    async def check_redis(self, redis_client) -> dict[str, Any]:
        """Check Redis health"""
        try:
            start = time.time()
            # Check if this is an async Redis client or sync
            if hasattr(redis_client, 'ping') and asyncio.iscoroutinefunction(redis_client.ping):
                # Async Redis client
                await redis_client.ping()
            else:
                # Sync Redis client - use thread executor
                await asyncio.to_thread(redis_client.ping)
            latency_ms = (time.time() - start) * 1000

            # Try to get Redis info, handle fallback mode gracefully
            try:
                if hasattr(redis_client, 'info') and asyncio.iscoroutinefunction(redis_client.info):
                    # Async Redis client
                    info = await redis_client.info()
                else:
                    # Sync Redis client
                    info = await asyncio.to_thread(redis_client.info)
                metadata = {
                    "latency_ms": round(latency_ms, 2),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                    "version": info.get("redis_version", "unknown"),
                    "mode": "connected",
                }
            except Exception:
                # Redis is in fallback mode - still functional but degraded
                metadata = {
                    "latency_ms": round(latency_ms, 2),
                    "mode": "fallback",
                    "note": "Using local cache fallback",
                }

            return {"status": "healthy", "metadata": metadata}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_llm(self, llm, model_lock) -> dict[str, Any]:
        """Check LLM health"""
        try:
            # Check if model is loaded
            if llm is None:
                return {"status": "unhealthy", "error": "LLM not initialized"}

            # Try a simple completion with timeout
            request_id = await model_lock.acquire(
                priority=Priority.HIGH, timeout=2.0, debug_name="health_check"
            )

            try:
                start = time.time()
                await asyncio.to_thread(
                    llm.create_completion,
                    prompt="Hello",
                    max_tokens=1,
                    temperature=0.2,  # Consistent with other decision calls
                    stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                )
                latency_ms = (time.time() - start) * 1000

                return {
                    "status": "healthy",
                    "metadata": {
                        "latency_ms": round(latency_ms, 2),
                        "model_loaded": True,
                        "lock_status": "operational",
                    },
                }
            finally:
                await model_lock.release(request_id)

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_memory_system(self, memory_manager, neural_memory) -> dict[str, Any]:
        """Check memory system health"""
        try:
            checks = {
                "memory_manager": memory_manager is not None,
                "neural_memory": neural_memory is not None,
                "redis_available": False,
                "embedding_model": False,
            }

            if memory_manager:
                checks["redis_available"] = memory_manager.redis_client is not None
                checks["embedding_model"] = memory_manager.init_success

            if all(checks.values()):
                status = "healthy"
            elif any(checks.values()):
                status = "degraded"
            else:
                status = "unhealthy"

            return {"status": status, "metadata": checks}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status"""
        # Check if we have recent health checks
        current_time = time.time()
        stale_threshold = 60  # 1 minute

        all_healthy = True
        any_unhealthy = False

        component_statuses = {}
        for name, health in self.components.items():
            # Check if health check is stale
            if current_time - health.last_check > stale_threshold:
                health.status = HealthStatus.DEGRADED
                health.error_message = "Health check is stale"

            component_statuses[name] = {
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "error": health.error_message,
                "last_check_ago": round(current_time - health.last_check, 1),
            }

            if health.metadata:
                component_statuses[name]["metadata"] = health.metadata

            if health.status != HealthStatus.HEALTHY:
                all_healthy = False
            if health.status == HealthStatus.UNHEALTHY:
                any_unhealthy = True

        # Determine overall status
        if all_healthy:
            overall_status = HealthStatus.HEALTHY
        elif any_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        # Get current metrics
        metrics = self.metrics_collector.get_current_metrics()

        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": component_statuses,
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_mb": round(metrics.memory_mb, 2),
                "error_rate": round(
                    (
                        (metrics.error_count / metrics.request_count * 100)
                        if metrics.request_count > 0
                        else 0
                    ),
                    2,
                ),
            },
        }


class MemoryAnalytics:
    """Analytics for the memory system"""

    def __init__(self, memory_manager, neural_memory, redis_client):
        self.memory_manager = memory_manager
        self.neural_memory = neural_memory
        self.redis_client = redis_client

    async def get_memory_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_counts": {},
            "token_usage": {},
            "performance": {},
        }

        try:
            # Count memories by type
            if user_id:
                pattern = "memory_b:*"
            else:
                pattern = "memory_b:*"

            memory_counts = {
                "total": 0,
                "compressed": 0,
                "by_category": defaultdict(int),
                "by_importance": {"high": 0, "medium": 0, "low": 0},  # > 0.8  # 0.5 - 0.8  # < 0.5
            }

            total_tokens = 0
            compressed_tokens = 0

            # Scan memories
            cursor = 0
            while True:
                cursor, keys = await asyncio.to_thread(
                    self.redis_client.scan, cursor=cursor, match=pattern, count=100
                )

                for key in keys:
                    try:
                        memory = await asyncio.to_thread(self.redis_client.json().get, key)

                        if not memory:
                            continue

                        memory_counts["total"] += 1

                        # Check if compressed
                        text = memory.get("text", "")
                        if text.startswith("[COMPRESSED]"):
                            memory_counts["compressed"] += 1
                            compressed_tokens += len(text.split())  # Rough estimate
                        else:
                            total_tokens += len(text.split())  # Rough estimate

                        # Categorize by importance
                        importance = memory.get("importance", 0.5)
                        if importance > 0.8:
                            memory_counts["by_importance"]["high"] += 1
                        elif importance >= 0.5:
                            memory_counts["by_importance"]["medium"] += 1
                        else:
                            memory_counts["by_importance"]["low"] += 1

                        # Extract category if present
                        if "[" in text and "]" in text:
                            category = text[text.find("[") + 1 : text.find("]")].lower()
                            memory_counts["by_category"][category] += 1

                    except Exception as e:
                        logger.warning(f"Error processing memory {key}: {e}")

                if cursor == 0:
                    break

            stats["memory_counts"] = dict(memory_counts)
            stats["memory_counts"]["by_category"] = dict(memory_counts["by_category"])

            # Token usage statistics
            stats["token_usage"] = {
                "total_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": (
                    round(compressed_tokens / (total_tokens + compressed_tokens) * 100, 2)
                    if (total_tokens + compressed_tokens) > 0
                    else 0
                ),
            }

            # Cache performance
            if hasattr(self.memory_manager, "essential_memory_cache"):
                cache_info = self.memory_manager.essential_memory_cache.currsize
                stats["performance"]["cache_size"] = cache_info

            # Neural memory stats if available
            if self.neural_memory:
                neural_stats = await self.neural_memory.get_network_stats()
                stats["neural_memory"] = neural_stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            stats["error"] = str(e)

        return stats


# Global instances
_health_checker: HealthChecker | None = None
_memory_analytics: MemoryAnalytics | None = None


def initialize_monitoring(memory_manager, neural_memory, redis_client):
    """Initialize monitoring systems"""
    global _health_checker, _memory_analytics

    _health_checker = HealthChecker()
    _memory_analytics = MemoryAnalytics(memory_manager, neural_memory, redis_client)

    logger.info("📊 Monitoring systems initialized")
    return _health_checker, _memory_analytics


def get_health_checker() -> HealthChecker | None:
    """Get health checker instance"""
    return _health_checker


def get_memory_analytics() -> MemoryAnalytics | None:
    """Get memory analytics instance"""
    return _memory_analytics


# Prometheus metrics format helper
def format_prometheus_metrics(metrics: SystemMetrics) -> str:
    """Format metrics in Prometheus exposition format"""
    lines = []

    # Add metric definitions and values
    lines.extend(
        [
            "# HELP aria_cpu_percent CPU usage percentage",
            "# TYPE aria_cpu_percent gauge",
            f"aria_cpu_percent {metrics.cpu_percent}",
            "",
            "# HELP aria_memory_bytes Memory usage in bytes",
            "# TYPE aria_memory_bytes gauge",
            f"aria_memory_bytes {metrics.memory_mb * 1024 * 1024}",
            "",
            "# HELP aria_request_total Total number of requests",
            "# TYPE aria_request_total counter",
            f"aria_request_total {metrics.request_count}",
            "",
            "# HELP aria_error_total Total number of errors",
            "# TYPE aria_error_total counter",
            f"aria_error_total {metrics.error_count}",
            "",
            "# HELP aria_response_time_ms Response time in milliseconds",
            "# TYPE aria_response_time_ms summary",
            f'aria_response_time_ms{{quantile="0.5"}} {metrics.avg_response_time_ms}',
            f'aria_response_time_ms{{quantile="0.95"}} {metrics.p95_response_time_ms}',
            f'aria_response_time_ms{{quantile="0.99"}} {metrics.p99_response_time_ms}',
        ]
    )

    return "\n".join(lines)
