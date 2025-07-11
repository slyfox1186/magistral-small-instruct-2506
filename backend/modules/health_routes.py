#!/usr/bin/env python3
"""Health and monitoring endpoints."""

import contextlib
import logging
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

# Import prometheus metrics if available
with contextlib.suppress(ImportError):
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Import from our modules
from memory_provider import get_memory_stats

from .globals import app_state

logger = logging.getLogger(__name__)


def setup_health_routes(app: FastAPI):
    """Setup health and monitoring routes."""
    _setup_health_endpoints(app)
    _setup_metrics_endpoints(app)
    _setup_gpu_endpoints(app)
    _setup_memory_endpoints(app)


def _setup_health_endpoints(app: FastAPI):
    """Setup health check endpoints."""
    @app.get("/health")
    @app.get("/api/health")
    @app.get("/api/mcp/health")
    async def health_check():
        """Comprehensive health check endpoint for production monitoring."""
        if not app_state.health_checker:
            return {"status": "unhealthy", "error": "Health checker not initialized"}

        try:
            await _perform_component_health_checks()
            health_status = await app_state.health_checker.get_overall_health()
            await _add_additional_health_info(health_status)
        except Exception as e:
            logger.exception("Error in health check")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        else:
            return health_status

    @app.get("/health/simple")
    async def simple_health_check():
        """Simple health check for load balancers."""
        if not app_state.health_checker:
            raise HTTPException(status_code=503, detail="Health checker not initialized")

        try:
            health_status = await app_state.health_checker.get_overall_health()
            return _process_simple_health_status(health_status)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in simple health check")
            raise HTTPException(status_code=503, detail="Health check failed") from e


def _setup_metrics_endpoints(app: FastAPI):
    """Setup metrics endpoints."""
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        if not app_state.prometheus_available:
            raise HTTPException(status_code=503, detail="Prometheus client not available")

        try:
            await _update_llm_server_metrics()
            return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            logger.exception("Error generating metrics")
            raise HTTPException(status_code=500, detail="Failed to generate metrics") from e


def _setup_gpu_endpoints(app: FastAPI):
    """Setup GPU status endpoints."""
    @app.get("/api/gpu-status")
    async def gpu_status():
        """Get real-time GPU status and memory usage."""
        try:
            from persistent_llm_server import get_llm_server
            server = await get_llm_server()
            stats = server.get_stats()
            return {
                "gpu_locked": stats.get("queue_size", 0) > 0,
                "current_owner": "persistent_server",
                "queue_size": stats.get("queue_size", 0),
                "avg_gpu_time": stats.get("avg_gpu_time", 0),
            }
        except Exception as e:
            logger.exception("Error getting GPU status from persistent server")
            return {"error": "GPU status unavailable", "details": str(e)}

    @app.get("/api/llm-stats")
    async def llm_server_stats():
        """Get persistent LLM server performance statistics."""
        try:
            from persistent_llm_server import get_llm_server
            server = await get_llm_server()
            return server.get_stats()
        except Exception as e:
            logger.exception("Error getting LLM stats")
            return {"error": str(e), "server_running": False}


def _setup_memory_endpoints(app: FastAPI):
    """Setup memory management endpoints."""
    @app.get("/api/memory-stats")
    async def memory_stats():
        """Get personal memory system statistics."""
        if not app_state.personal_memory:
            return {"error": "Memory system not initialized"}
        return get_memory_stats()

    @app.post("/api/core-memory/{key}")
    async def set_core_memory(key: str, request: dict):
        """Set a core memory (persistent fact about the user)."""
        if not app_state.personal_memory:
            return {"error": "Memory system not initialized"}

        value = request.get("value", "")
        category = request.get("category", "general")
        conversation_id = request.get("conversation_id", "global")  # Default to global if not specified

        if not value:
            raise HTTPException(status_code=400, detail="Value is required")

        await app_state.personal_memory.set_core_memory(key, value, conversation_id, category)
        return {"success": True, "key": key, "value": value, "conversation_id": conversation_id}

    @app.get("/api/core-memories")
    async def get_core_memories(conversation_id: str = "global"):
        """Get all core memories for a conversation."""
        if not app_state.personal_memory:
            return {"error": "Memory system not initialized"}

        memories = await app_state.personal_memory.get_all_core_memories(conversation_id)
        return {"core_memories": memories, "conversation_id": conversation_id}


async def _perform_component_health_checks():
    """Perform health checks on all system components."""
    await app_state.health_checker.check_component(
        "redis", lambda: app_state.health_checker.check_redis(app_state.redis_client)
    )
    await app_state.health_checker.check_component(
        "llm", lambda: app_state.health_checker.check_llm(app_state.llm, None)
    )
    await app_state.health_checker.check_component(
        "memory_system",
        lambda: {"status": "healthy" if app_state.personal_memory else "unhealthy"},
    )


async def _add_additional_health_info(health_status: dict):
    """Add additional health information to status."""
    if app_state.metacognitive_engine:
        health_status["components"]["metacognitive_engine"] = {
            "status": "healthy",
            "heuristic_evaluator": "active",
            "llm_critic": "active",
        }

    try:
        from persistent_llm_server import get_llm_server
        server = await get_llm_server()
        llm_stats = server.get_stats()
        health_status["llm_server"] = {
            "running": llm_stats.get("is_running", False),
            "queue_size": llm_stats.get("queue_size", 0),
            "cache_hit_rate": llm_stats.get("cache_hit_rate", 0),
        }
    except Exception as e:
        logger.warning(f"Failed to get LLM server status: {e}")


def _process_simple_health_status(health_status: dict) -> dict:
    """Process health status for simple endpoint."""
    if health_status["status"] == "healthy":
        return {"status": "ok"}
    elif health_status["status"] == "degraded":
        return {"status": "degraded"}
    else:
        raise HTTPException(status_code=503, detail="System unhealthy")


async def _update_llm_server_metrics():
    """Update LLM server metrics for Prometheus."""
    if app_state.gpu_queue_depth:
        try:
            from persistent_llm_server import get_llm_server
            server = await get_llm_server()
            stats = server.get_stats()
            if stats.get("queue_size", 0) > 0:
                app_state.model_lock_held.set(1)
            else:
                app_state.model_lock_held.set(0)
        except Exception as e:
            logger.warning(f"Could not update GPU metrics: {e}")
