#!/usr/bin/env python3
"""Main chat streaming endpoint and related routes."""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import utils
from memory.importance_scorer import get_importance_scorer

from .chat_helpers import (
    check_service_availability,
    classify_query_with_llm,
    lightweight_memory_processing,
)
from .globals import (
    app_state,
    bg_state,
)
from .intent_handlers import (
    handle_conversation_history_intent,
    handle_conversation_intent,
    handle_crypto_query_intent,
    handle_personal_info_recall_intent,
    handle_personal_info_storage_intent,
    handle_stock_query_intent,
    handle_weather_query_intent,
    handle_web_search_intent,
)

# Import models and globals
from .models import ChatStreamRequest

logger = logging.getLogger(__name__)

# Initialize importance scorer
importance_scorer = get_importance_scorer()


async def calculate_message_importance(
    content: str, role: str, session_id: str, messages: list | None = None
) -> float:
    """Calculate importance score for a message using the sophisticated scorer."""
    try:
        # Build conversation history from recent messages if available
        conversation_history = []
        if messages:
            for msg in messages[-10:]:  # Last 10 messages for context
                # Handle both dict and Message objects
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    # It's a Message object
                    conversation_history.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    # It's already a dict
                    conversation_history.append(
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    )

        # Calculate importance
        importance = importance_scorer.calculate_importance(
            text=content, role=role, conversation_history=conversation_history
        )

        logger.info(
            f"🧠 IMPORTANCE SCORER: Calculated importance={importance:.3f} for {role} message in session {session_id}"
        )
    except Exception:
        logger.exception("Error calculating importance")
        # Fallback to default values
        return 0.7 if role == "user" else 0.8
    else:
        return importance


async def get_memory_context(user_prompt: str, session_id: str) -> str:
    """Get memory context from personal memory."""
    memory_context = ""
    logger.info(f"🧠 MEMORY DEBUG: Attempting to get memory context for session {session_id}")

    if app_state.personal_memory:
        logger.info(f"🧠 MEMORY DEBUG: personal_memory is available: {type(app_state.personal_memory)}")
        try:
            # Quick memory retrieval for context
            logger.info(f"🧠 MEMORY DEBUG: Calling get_relevant_memories with query: '{user_prompt[:50]}...'")
            memories = await app_state.personal_memory.get_relevant_memories(query=user_prompt, limit=5)
            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(memories) if memories else 0} memories")

            # Get core memories (user facts) for this conversation
            logger.info(f"🧠 MEMORY DEBUG: Getting core memories for conversation {session_id}...")
            core_memories = await app_state.personal_memory.get_all_core_memories(session_id)
            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(core_memories) if core_memories else 0} core memories")

            # Build memory context from both sources
            memory_parts = []

            # Add core memories first (most important user facts)
            if core_memories:
                core_facts = []
                for key, value in core_memories.items():
                    core_facts.append(f"{key}: {value}")
                if core_facts:
                    memory_parts.append("User Facts:\n" + "\n".join(core_facts))
                    logger.info(f"🧠 MEMORY DEBUG: Added {len(core_facts)} core memory facts")

            # Add regular memories
            if memories:
                regular_memories = [m.content for m in memories[:3]]
                memory_parts.extend(regular_memories)
                logger.info(f"🧠 MEMORY DEBUG: Added {len(regular_memories)} regular memories")

            if memory_parts:
                memory_context = "\n\n".join(memory_parts)
                logger.info(f"🧠 MEMORY DEBUG: ✅ Using combined memory context ({len(memory_context)} chars)")
                logger.debug(f"🧠 MEMORY DEBUG: Memory context preview: {memory_context[:200]}...")
            else:
                logger.info("🧠 MEMORY DEBUG: No memories found")
        except Exception as e:
            logger.error(f"🧠 MEMORY DEBUG: ❌ Memory retrieval failed: {e}", exc_info=True)
    else:
        logger.warning("🧠 MEMORY DEBUG: personal_memory is None - no memory context available")

    return memory_context


async def create_memory_processing_task(user_prompt: str, full_response: str, session_id: str):
    """Create background memory processing task."""
    if full_response and app_state.personal_memory:
        try:
            task = asyncio.create_task(lightweight_memory_processing(user_prompt, full_response, session_id))
            # Keep reference to prevent garbage collection
            task.add_done_callback(lambda t: None)
        except Exception:
            logger.exception("🧠 MEMORY DEBUG: ❌ Failed to create memory processing task")


async def stream_conversation_response(user_prompt: str, session_id: str, system_prompt: str):
    """Stream conversation response using LLM server."""
    from persistent_llm_server import get_llm_server

    server = await get_llm_server()
    formatted_prompt = utils.format_prompt(system_prompt, user_prompt)

    full_response = ""
    async for token in server.generate_stream(
        prompt=formatted_prompt,
        max_tokens=40960,
        temperature=0.7,
        top_p=0.95,
        session_id=session_id,
        priority=0,
    ):
        if token:
            full_response += token
            yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"

    # Background memory processing
    await create_memory_processing_task(user_prompt, full_response, session_id)


def update_chat_activity():
    """Update global chat activity timestamp."""
    bg_state.last_chat_activity = time.time()


def _create_chat_stream_route(app: FastAPI):
    """Create the main chat stream route."""
    @app.post("/api/chat-stream")
    async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        logger.info(
            f"  CHAT REQUEST RECEIVED for session {request.session_id} at {datetime.now(UTC).isoformat()}"
        )
        logger.info(
            f"🧠 MEMORY DEBUG: Chat request details - Messages: {len(request.messages)}, Session: {request.session_id}"
        )
        logger.info(f"🧠 MEMORY DEBUG: Request object type: {type(request)}")

        #  Update chat activity for intelligent background processing
        update_chat_activity()

        # Phase 3A: Start request timing for Prometheus metrics
        time.time()
        endpoint = "chat_stream"

        # Increment request counter
        if app_state.request_total:
            app_state.request_total.labels(endpoint=endpoint, status="started").inc()

        logger.info("🧠 MEMORY DEBUG: Performing service availability check...")
        try:
            check_service_availability()
            logger.info("🧠 MEMORY DEBUG: ✅ Service availability check passed")
        except Exception as service_error:
            logger.error(
                f"🧠 MEMORY DEBUG: ❌ Service availability check failed: {service_error}",
                exc_info=True,
            )
            raise

        session_id = request.session_id
        
        # Set conversation context for Redis compatibility layer
        if app_state.personal_memory and hasattr(app_state.personal_memory, 'set_conversation_context'):
            app_state.personal_memory.set_conversation_context(session_id)
            logger.info(f"🧠 MEMORY DEBUG: Set Redis compatibility conversation context to {session_id}")
        
        user_prompt = request.messages[-1].content if request.messages else ""
        logger.info(
            f"🧠 MEMORY DEBUG: Extracted user prompt: '{user_prompt[:100]}...' (length: {len(user_prompt)})"
        )
        logger.info(f"🧠 MEMORY DEBUG: Session ID: {session_id}")

        if not user_prompt:
            logger.warning("🧠 MEMORY DEBUG: Rejected empty prompt request")
            raise HTTPException(status_code=400, detail="Empty prompt")

        #  PII REDACTION DISABLED - User wants AI to remember personal information
        # Add constant for prompt preview length
        prompt_preview_length = 100
        logger.debug(
            f"Processing prompt: {user_prompt[:prompt_preview_length]}{'...' if len(user_prompt) > prompt_preview_length else ''}"
        )

        # Always use the LLM for classification - it's more accurate than regex
        logger.info("🔄 FULL PIPELINE: Processing request")

        # Step 1: Classify query intent using LLM
        classification = await classify_query_with_llm(user_prompt)

        primary_intent = classification.get("primary_intent", "conversation")
        logger.debug(f"Query classified as: {primary_intent}")

        # Step 2: Handle based on intent
        async def generate_response():
            # Capture primary_intent in local scope
            intent = primary_intent
            logger.info(f"🧠 MEMORY DEBUG: Starting generate_response for intent: {intent}")

            # Intent handler mapping
            intent_handlers = {
                "conversation": handle_conversation_intent,
                "web_search": handle_web_search_intent,
                "personal_info_storage": handle_personal_info_storage_intent,
                "personal_info_recall": handle_personal_info_recall_intent,
                "conversation_history": handle_conversation_history_intent,
                "stock_query": handle_stock_query_intent,
                "weather_query": handle_weather_query_intent,
                "crypto_query": handle_crypto_query_intent,
            }

            # Get appropriate handler or default to conversation
            handler = intent_handlers.get(intent, handle_conversation_intent)

            try:
                # Some handlers need the request parameter
                if intent in ["web_search", "stock_query", "weather_query", "crypto_query"]:
                    async for chunk in handler(user_prompt, session_id, request):
                        yield chunk
                else:
                    async for chunk in handler(user_prompt, session_id):
                        yield chunk

            except Exception as e:
                error_msg = f"Error in response generation: {e!s}"
                logger.error(f"🧠 MEMORY DEBUG: ❌ Error in generate_response: {error_msg}", exc_info=True)
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )


def setup_chat_routes(app: FastAPI):
    """Setup chat streaming routes."""
    _create_chat_stream_route(app)
