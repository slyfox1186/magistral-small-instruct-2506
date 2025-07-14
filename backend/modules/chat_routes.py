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
    calculate_message_importance,
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
from pydantic import BaseModel

class TitleGenerationRequest(BaseModel):
    """Request model for title generation."""
    user_message: str

logger = logging.getLogger(__name__)

# Configuration constants
MAX_MESSAGE_LENGTH = 50000  # 50KB limit per message
MAX_TOTAL_MESSAGES = 100    # Maximum number of messages in conversation
MAX_SESSION_ID_LENGTH = 128 # Reasonable session ID length limit

def validate_chat_input(content: str) -> str:
    """Basic chat input validation - just length and empty check.
    
    Args:
        content: Raw user input
        
    Returns:
        Validated content
        
    Raises:
        ValueError: If input is invalid
    """
    if not content or not content.strip():
        raise ValueError("Message cannot be empty")
    
    if len(content) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message too long (max {MAX_MESSAGE_LENGTH} characters)")
    
    # No content sanitization - let the LLM handle content processing naturally
    return content.strip()

def validate_session_id(session_id: str) -> str:
    """Basic session ID validation - just length and empty check.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Validated session ID
        
    Raises:
        ValueError: If session ID is invalid
    """
    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty")
    
    if len(session_id) > MAX_SESSION_ID_LENGTH:
        raise ValueError(f"Session ID too long (max {MAX_SESSION_ID_LENGTH} characters)")
    
    # No regex validation - let the LLM handle content processing
    return session_id.strip()



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


async def generate_conversation_title(user_message: str) -> str:
    """Generate a concise, meaningful title for a conversation based on the user's first message.
    
    Args:
        user_message: The user's first message in the conversation
        
    Returns:
        A short, descriptive title (max 60 characters)
    """
    from persistent_llm_server import get_llm_server
    
    # Create a focused system prompt for title generation
    system_prompt = """You are an expert at creating concise, descriptive titles for conversations. Generate a title for a conversation that starts with the given user message. The title should:
- Be 2-8 words maximum
- Capture the main topic or intent
- Be professional and clear
- Not include quotes or special formatting
- Be under 60 characters

Return only the title, nothing else."""

    user_prompt = f'Generate a title for this conversation starter: "{user_message.strip()}"'

    try:
        server = await get_llm_server()
        
        # Use proper prompt formatting from utils.py
        formatted_prompt = utils.format_prompt(system_prompt, user_prompt)
        
        # Generate title with focused parameters
        full_response = ""
        async for token in server.generate_stream(
            prompt=formatted_prompt,
            max_tokens=20,  # Keep it short
            temperature=0.3,  # Lower temperature for consistency
            top_p=0.8,
            session_id="title_generation",
            priority=1,  # Higher priority for quick response
        ):
            if token:
                full_response += token
        
        # Clean up the response
        title = full_response.strip()
        
        # Remove common prefixes/suffixes that the LLM might add
        prefixes_to_remove = ["Title:", "title:", "**", "*", '"', "'", "- "]
        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Remove quotes and ensure reasonable length
        title = title.strip('"\'*').strip()
        
        # Fallback if title is too long or empty
        if len(title) > 60:
            title = title[:57] + "..."
        elif not title or len(title) < 3:
            # Fallback to first few words of the message
            words = user_message.strip().split()[:5]
            title = " ".join(words)
            if len(title) > 60:
                title = title[:57] + "..."
        
        logger.info(f"Generated title: '{title}' for message: '{user_message[:50]}...'")
        return title
        
    except Exception as e:
        logger.error(f"Failed to generate title via LLM: {e}", exc_info=True)
        
        # Fallback to simple word extraction
        words = user_message.strip().split()[:5]
        fallback_title = " ".join(words)
        if len(fallback_title) > 60:
            fallback_title = fallback_title[:57] + "..."
        
        logger.info(f"Using fallback title: '{fallback_title}'")
        return fallback_title


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

        # Validate session ID
        try:
            session_id = validate_session_id(request.session_id)
        except ValueError as e:
            logger.warning(f"Invalid session ID: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Validate messages count
        if len(request.messages) > MAX_TOTAL_MESSAGES:
            logger.warning(f"Too many messages: {len(request.messages)}")
            raise HTTPException(status_code=400, detail=f"Too many messages (max {MAX_TOTAL_MESSAGES})")
        
        # Set conversation context for Redis compatibility layer
        if app_state.personal_memory and hasattr(app_state.personal_memory, 'set_conversation_context'):
            app_state.personal_memory.set_conversation_context(session_id)
            logger.info(f"🧠 MEMORY DEBUG: Set Redis compatibility conversation context to {session_id}")
        
        # Validate and sanitize user prompt
        raw_user_prompt = request.messages[-1].content if request.messages else ""
        try:
            user_prompt = validate_chat_input(raw_user_prompt)
        except ValueError as e:
            logger.warning(f"Invalid user input: {e}")
            raise HTTPException(status_code=400, detail=str(e))
            
        logger.info(
            f"🧠 MEMORY DEBUG: Validated user prompt: '{user_prompt[:100]}...' (length: {len(user_prompt)})"
        )
        logger.info(f"🧠 MEMORY DEBUG: Session ID: {session_id}")

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
                "perform_web_search": handle_web_search_intent,  # Fixed: was "web_search"
                "store_personal_info": handle_personal_info_storage_intent,  # Fixed: was "personal_info_storage"
                "recall_personal_info": handle_personal_info_recall_intent,  # Fixed: was "personal_info_recall"
                "query_conversation_history": handle_conversation_history_intent,  # Fixed: was "conversation_history"
                "query_stocks": handle_stock_query_intent,  # Fixed: was "stock_query"
                "query_weather": handle_weather_query_intent,  # Fixed: was "weather_query"
                "query_cryptocurrency": handle_crypto_query_intent,  # Fixed: was "crypto_query"
            }

            # Get appropriate handler or default to conversation
            handler = intent_handlers.get(intent, handle_conversation_intent)

            try:
                # Some handlers need the request parameter
                if intent in ["perform_web_search", "query_stocks", "query_weather", "query_cryptocurrency"]:
                    async for chunk in handler(user_prompt, session_id, request):
                        yield chunk
                elif intent in ["store_personal_info", "recall_personal_info"]:
                    # Personal info handlers only need user_prompt and session_id
                    async for chunk in handler(user_prompt, session_id):
                        yield chunk
                else:
                    # Pass conversation history to conversation intent handler
                    async for chunk in handler(user_prompt, session_id, request.messages[:-1] if len(request.messages) > 1 else []):
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


def _create_title_generation_route(app: FastAPI):
    """Create the title generation route."""
    @app.post("/api/generate-title")
    async def generate_title(request: TitleGenerationRequest) -> dict[str, str]:
        """Generate a conversation title from a user message."""
        logger.info(f"Title generation request for message: '{request.user_message[:100]}...'")
        
        try:
            # Validate input
            user_message = validate_chat_input(request.user_message)
            
            # Generate title using LLM
            title = await generate_conversation_title(user_message)
            
            return {"title": title}
            
        except ValueError as e:
            logger.warning(f"Invalid input for title generation: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Title generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate title")


def setup_chat_routes(app: FastAPI):
    """Setup chat streaming routes."""
    _create_chat_stream_route(app)
    _create_title_generation_route(app)
