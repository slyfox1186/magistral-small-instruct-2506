#!/usr/bin/env python3
"""Memory management and distributed component API endpoints."""

import logging
import time
from dataclasses import asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException

import monitoring
from circuit_breaker import circuit_breaker_manager

# Import from our modules
from .globals import app_state
from .models import MemoryIngestion, ScrapedContent

logger = logging.getLogger(__name__)


def setup_memory_routes(app: FastAPI):
    """Setup memory management and distributed component routes."""
    # ===================== API Endpoints for Distributed Components =====================

    @app.post("/api/v1/memory/ingest/scraped")
    async def ingest_scraped_content(item: ScrapedContent):
        """Ingest scraped content from web scraper service.
        This centralizes database writes to avoid SQLite concurrency issues.
        """
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory system not initialized")

        try:
            # Add scraped content to memory
            await app_state.personal_memory.add_memory(
                content=f"Content from {item.url}: {item.content}",
                conversation_id=f"web_scrape_{datetime.now().strftime('%Y%m%d')}",
                importance=0.6,
                metadata={"source": "web_scraper", "url": item.url, **(item.metadata or {})},
            )

            logger.info(f" Ingested scraped content from {item.url}")
            return {"status": "success", "url": item.url}

        except Exception as e:
            logger.error(f"Failed to ingest scraped content: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/memory/ingest")
    async def ingest_memory(item: MemoryIngestion):
        """General memory ingestion endpoint for distributed components.
        Allows other services to add memories without direct database access.
        """
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory system not initialized")

        try:
            memory = await app_state.personal_memory.add_memory(
                content=item.content,
                conversation_id=item.conversation_id,
                importance=item.importance,
                metadata=item.metadata,
            )

            logger.info(f" Ingested memory for conversation {item.conversation_id}")
            return {
                "status": "success",
                "memory_id": memory.id,
                "conversation_id": item.conversation_id,
            }

        except Exception as e:
            logger.error(f"Failed to ingest memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics/json")
    async def get_metrics_json():
        """Get system performance metrics in JSON format."""
        if not app_state.health_checker:
            return {"error": "Metrics collector not available"}

        try:
            # Get metrics summary
            summary = app_state.health_checker.metrics_collector.get_metrics_summary()

            # Get current metrics
            current = app_state.health_checker.metrics_collector.get_current_metrics()

            return {
                "summary": summary,
                "current": asdict(current) if current else None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}

    @app.get("/metrics/prometheus")
    async def get_prometheus_metrics():
        """Get metrics in Prometheus exposition format."""
        if not app_state.health_checker:
            return "# ERROR: Metrics collector not available\n"

        try:
            current_metrics = app_state.health_checker.metrics_collector.get_current_metrics()
            if current_metrics:
                return monitoring.format_prometheus_metrics(current_metrics)
            else:
                return "# ERROR: No metrics available\n"
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return f"# ERROR: {e!s}\n"

    @app.get("/memory/stats")
    async def memory_stats_endpoint():
        """Get comprehensive memory system statistics."""
        if not app_state.personal_memory:
            return {"error": "Personal memory system not available"}

        try:
            stats = app_state.personal_memory.get_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    @app.get("/memory/stats/{user_id}")
    async def get_user_memory_stats(user_id: str):
        """Get memory statistics for a specific user (conversation)."""
        if not personal_memory:
            return {"error": "Personal memory system not available"}

        try:
            # Get conversation memories
            memories = await app_state.personal_memory.get_conversation_context(user_id, max_messages=100)
            stats = {
                "conversation_id": user_id,
                "total_memories": len(memories),
                "recent_memories": [
                    {
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat(),
                        "importance": m.importance,
                    }
                    for m in memories[-10:]  # Last 10 memories
                ],
            }

            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats for user {user_id}: {e}")
            return {"error": str(e)}

    @app.get("/monitoring/circuit-breakers")
    async def get_circuit_breaker_stats():
        """Get circuit breaker statistics for all external services."""
        try:
            stats = circuit_breaker_manager.get_all_stats()
            return {
                "circuit_breakers": stats,
                "timestamp": time.time(),
                "summary": {
                    "total_breakers": len(stats),
                    "open_breakers": len([s for s in stats.values() if s["state"] == "open"]),
                    "half_open_breakers": len([s for s in stats.values() if s["state"] == "half_open"]),
                },
            }
        except Exception as e:
            logger.error(f"Error getting circuit breaker stats: {e}")
            return {"error": str(e)}

    @app.post("/monitoring/circuit-breakers/reset")
    async def reset_circuit_breakers():
        """Reset all circuit breakers to closed state."""
        try:
            circuit_breaker_manager.reset_all()
            logger.info(" All circuit breakers reset to closed state")
            return {"message": "All circuit breakers reset successfully"}
        except Exception as e:
            logger.error(f"Error resetting circuit breakers: {e}")
            return {"error": str(e)}

    # Memory compression endpoint removed - personal AI handles this automatically

    @app.post("/memory/optimize/{user_id}")
    async def optimize_user_memories(user_id: str, target_tokens: int = 1000):
        """Trigger comprehensive memory optimization for a user."""
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory pipeline not available")

        try:
            # Consolidate memories as optimization
            # Consolidation happens automatically in personal memory
            await app_state.personal_memory.consolidate_old_memories()
            result = {"success": True, "message": "Consolidation triggered"}
            result["target_tokens"] = target_tokens
            return result
        except Exception as e:
            logger.error(f"Error optimizing memories for user {user_id}: {e}")
            return {"success": False, "error": str(e)}

    @app.put("/memory/correct/{user_id}/{memory_id}")
    async def correct_memory(user_id: str, memory_id: str, correction: dict[str, str]):
        """Correct a specific memory.

        Args:
            user_id: User who owns the memory
            memory_id: ID of memory to correct
            correction: Dict with 'new_content' and optional 'reason'
        """
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory pipeline not available")

        new_content = correction.get("new_content")
        if not new_content:
            raise HTTPException(status_code=400, detail="new_content is required")

        try:
            # Memory correction not implemented in personal system
            result = {
                "success": False,
                "error": "Memory correction not supported in personal memory system",
            }

            if not result["success"]:
                raise HTTPException(status_code=400, detail=result.get("error", "Correction failed"))

            return result
        except Exception as e:
            logger.error(f"Error correcting memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/memory/{user_id}/{memory_id}")
    async def delete_memory(user_id: str, memory_id: str, reason: str = "User requested deletion"):
        """Delete a specific memory."""
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory pipeline not available")

        try:
            # Memory deletion not implemented in personal system
            result = {
                "success": False,
                "error": "Direct memory deletion not supported in personal memory system",
            }

            if not result["success"]:
                raise HTTPException(status_code=404, detail=result.get("error", "Memory not found"))

            return result
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/bulk-correct/{user_id}")
    async def bulk_correct_memories(user_id: str, bulk_correction: dict[str, str]):
        """Search and correct multiple memories matching a pattern.

        Args:
            user_id: User who owns the memories
            bulk_correction: Dict with 'search_query', 'correction_pattern', and 'replacement'
        """
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory pipeline not available")

        required_fields = ["search_query", "correction_pattern", "replacement"]
        for field in required_fields:
            if field not in bulk_correction:
                raise HTTPException(status_code=400, detail=f"{field} is required")

        try:
            # Search and correct not implemented in personal system
            result = {
                "success": False,
                "error": "Bulk memory correction not supported in personal memory system",
            }

            return result
        except Exception as e:
            logger.error(f"Error in bulk memory correction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/memory/consolidate/{user_id}/schedule")
    async def schedule_consolidation(user_id: str):
        """Schedule memory consolidation for a user.

        This enqueues a background job using ARQ if available,
        otherwise runs it inline (not recommended for production).
        """
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory pipeline not available")

        try:
            # Try to use ARQ if available
            try:
                from memory.consolidation_worker import enqueue_consolidation

                job_id = await enqueue_consolidation(user_id)
                return {
                    "status": "scheduled",
                    "job_id": job_id,
                    "user_id": user_id,
                    "message": "Consolidation job enqueued for background processing",
                }
            except ImportError:
                # ARQ not available, run inline (not recommended)
                logger.warning("ARQ not available, running consolidation inline")
                # Consolidation happens automatically in personal memory
                await app_state.personal_memory.consolidate_old_memories()
                result = {"success": True, "message": "Consolidation triggered"}
                return {
                    "status": "completed",
                    "user_id": user_id,
                    "result": result,
                    "message": "Consolidation completed inline (consider setting up ARQ)",
                }
        except Exception as e:
            logger.error(f"Error scheduling consolidation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/memory/search/{user_id}")
    async def search_memories(user_id: str, query: str, limit: int = 10):
        """Search memories for a user.

        Args:
            user_id: User to search memories for
            query: Search query
            limit: Maximum results to return
        """
        if not app_state.personal_memory:
            raise HTTPException(status_code=503, detail="Memory pipeline not available")

        try:
            # Use personal memory retrieval
            memories = await app_state.personal_memory.get_relevant_memories(query=query, limit=limit)
            results = memories

            # Convert to simpler format for API response
            memories = []
            for memory in results:
                memories.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "category": "general",  # Personal memory doesn't have categories
                        "importance": memory.importance,
                        "relevance_score": 1.0,  # No relevance scoring in personal memory
                        "created_at": memory.timestamp.isoformat(),
                        "is_summary": memory.summary is not None,
                    }
                )

            return {"user_id": user_id, "query": query, "count": len(memories), "memories": memories}
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/clear-vital-memories")
    @app.delete("/clear-vital-memories")  # Support both paths for compatibility
    async def clear_vital_memories():
        """Clear all vital memories stored for the assistant."""
        try:
            logger.info("🧹 Clearing all memories from personal memory system...")

            deleted_count = 0

            # Clear from personal memory system (SQLite)
            if app_state.personal_memory:
                try:
                    # Use the proper clear method
                    deleted_count = app_state.personal_memory.clear_all_memories()
                    logger.info(f"🗑️ Cleared {deleted_count} memories from SQLite database")

                except Exception as e:
                    logger.error(f"Failed to clear personal memory system: {e}")

            # Also clear from Redis if available (for compatibility)
            if app_state.redis_client:
                patterns = [
                    "memory:*",
                    "vital_memory:*",
                    "user_memory:*",
                    "neural_memory:*",
                    "redis_memory:*",
                    "conversation:*",
                    "session:*",
                    "msg:*",  # Clear message cache
                    "*session*",  # Clear any session-related keys
                    "*memory*",  # Clear any memory-related keys
                ]

                for pattern in patterns:
                    try:
                        keys = await app_state.redis_client.keys(pattern)
                        if keys:
                            for key in keys:
                                await app_state.redis_client.delete(key)
                            deleted_count += len(keys)
                            logger.info(f"Deleted {len(keys)} Redis keys matching pattern '{pattern}'")
                    except Exception as e:
                        logger.warning(f"Failed to clear Redis pattern '{pattern}': {e}")

            logger.info(f"✅ Successfully cleared {deleted_count} total memory entries")

            # Import VitalMemoryResponse for proper return type
            from .models import VitalMemoryResponse
            return VitalMemoryResponse(
                success=True,
                message=f"Successfully cleared {deleted_count} memories",
                deleted_count=deleted_count,
            )

        except Exception as e:
            logger.error(f"Error clearing memories: {e}", exc_info=True)
            from .models import VitalMemoryResponse
            return VitalMemoryResponse(
                success=False,
                message=f"Failed to clear memories: {e!s}",
                deleted_count=0,
            )
