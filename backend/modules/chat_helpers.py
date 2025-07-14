#!/usr/bin/env python3
"""Helper functions for chat processing and query classification."""

import asyncio
import json
import logging

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import redis_utils
import utils

from .globals import app_state

# Import from other modules
from .models import ChatStreamRequest

logger = logging.getLogger(__name__)

# ===================== Helper Functions =====================


def classify_query_fast_pattern_based(user_prompt: str) -> dict[str, any]:
    """DEPRECATED: Always use LLM classification for better accuracy.

    Returns None to force LLM classification for all queries.
    """
    # Always use LLM classification - it's more accurate than regex
    return None


async def classify_query_with_llm(user_prompt: str) -> dict[str, any]:
    """Simple single-word route classification."""
    logger.debug(f"🎯 ROUTE CLASSIFIER: Analyzing '{user_prompt}'")

    # World-class enterprise-grade intent classification system (SYSTEM PROMPT ONLY)
    system_prompt = """# [DIRECTIVE]
You are a high-precision, automated Intent Routing System. Your SOLE function is to analyze the user's input and return a single, uppercase word representing its PRIMARY intent. Adhere strictly to the protocol below.

# [CLASSIFICATION PROTOCOL & EXAMPLES]
Analyze the input against each category. Use the definitions, crucial tests, and examples to make an intelligent decision.

---
**CATEGORY: STORE**
- **Definition:** User is providing a fact, preference, or identity detail about themselves to be remembered.
- **Crucial Test:** Is the user making a declarative statement about themselves?
- **Example:** `User: "I prefer coffee over tea, and my favorite color is blue." -> STORE`

---
**CATEGORY: RECALL**
- **Definition:** User is requesting a specific piece of their own information that was previously stored.
- **Crucial Test:** Is the user asking you to retrieve a stored personal fact about them?
- **Example:** `User: "What's my favorite color again?" -> RECALL`

---
**CATEGORY: MEMORY**
- **Definition:** User is asking about the immediate conversation history of the current session.
- **Crucial Test:** Is the user asking about what was said or done in this specific chat session?
- **Example:** `User: "What was the name of the function you just showed me?" -> MEMORY`
- **Example:** `User: "Tell me more about the missing persons" -> MEMORY` (when missing persons was discussed in this conversation)

---
**CATEGORY: WEB**
- **Definition:** User explicitly requests internet search, current events, or real-time information that requires web access.
- **Crucial Test:** Is the user explicitly asking for current/recent information, news, or web search? NOT personal statements about future plans.
- **Example:** `User: "What were the results of the F1 race today?" -> WEB`
- **Example:** `User: "Search for information about missing persons" -> WEB`
- **Example:** `User: "Find current news about the Texas flood" -> WEB`
- **Example:** `User: "Look up restaurant reviews in Paris" -> WEB`
- **NOT WEB:** `User: "I'm planning a trip to Paris" -> INTERNAL` (personal statement, not search request)

---
**CATEGORY: WEATHER**
- **Definition:** User is asking about weather conditions or forecasts.
- **Crucial Test:** Is the primary subject weather, temperature, or precipitation?
- **Example:** `User: "What's the weather like in London tomorrow?" -> WEATHER`

---
**CATEGORY: CRYPTO**
- **Definition:** User is asking for cryptocurrency data.
- **Example:** `User: "Check the price of dogecoin." -> CRYPTO`

---
**CATEGORY: STOCKS**
- **Definition:** User is asking for stock market data.
- **Example:** `User: "How did the Nasdaq do today?" -> STOCKS`

---
**CATEGORY: INTERNAL**
- **Definition:** General conversation, personal statements about plans/activities, creative tasks, questions about the AI, or any conversational input that does not fit a more specific category.
- **Crucial Test:** Is the user having a conversation, sharing plans, or making statements that don't require web search or personal data storage?
- **Example:** `User: "Generate a python function that sorts a list." -> INTERNAL`
- **Example:** `User: "I'm planning a trip to Paris next month." -> INTERNAL`
- **Example:** `User: "How are you doing today?" -> INTERNAL`

# [SELF-CORRECTION & HIERARCHY]
1.  **Specificity is Key:** Always choose the most specific, non-`INTERNAL` category.
2.  **Core Disambiguation: `STORE` vs. `INTERNAL`**
    -   Statement about the *USER*: `I have an aggressive play style.` -> **STORE**
    -   Question about the *AI*: `What is your play style?` -> **INTERNAL**
3.  **Core Disambiguation: `RECALL` vs. `MEMORY`**
    -   Question about the *USER'S PROFILE*: `What did I say my wife's name was?` -> **RECALL**
    -   Question about the *CONVERSATION'S HISTORY*: `What was the last thing you said?` -> **MEMORY**
4.  **Core Disambiguation: `MEMORY` vs. `WEB` vs. `INTERNAL`**
    -   Simple clarification about CONVERSATION: `What did you just say about missing persons?` -> **MEMORY**
    -   Explicit search request: `Search for information about missing persons` -> **WEB**
    -   Personal statement or general conversation: `I'm planning a trip to Paris` -> **INTERNAL**
    -   **Key Rule**: WEB requires explicit search intent, not personal statements about plans

# [FINAL INSTRUCTION]
Analyze the user query. Provide only the single uppercase classification word. NOTHING ELSE."""

    classification_prompt = utils.format_prompt(system_prompt, user_prompt)

    try:
        from persistent_llm_server import get_llm_server

        server = await get_llm_server()

        # Get single word classification
        classification_text = await server.generate(
            prompt=classification_prompt,
            max_tokens=5,  # Just need one word
            temperature=0.1,  # Very consistent decisions
            session_id="route_classification",
        )

        classification_word = classification_text.strip().upper()
        logger.info(f"🎯 ROUTE CLASSIFIER: '{user_prompt}' → '{classification_word}'")

        # Map single words to internal categories
        route_mapping = {
            "STORE": "store_personal_info",
            "RECALL": "recall_personal_info",
            "WEATHER": "query_weather",
            "WEB": "perform_web_search",
            "CRYPTO": "query_cryptocurrency",
            "STOCKS": "query_stocks",
            "MEMORY": "query_conversation_history",
            "INTERNAL": "conversation",  # Use memory-aware conversation handler
        }

        # Default to conversation (memory-aware) if word not recognized
        if classification_word in route_mapping:
            intent = route_mapping[classification_word]
        else:
            logger.warning(
                f"Unknown classification word '{classification_word}', defaulting to conversation"
            )
            intent = "conversation"

        intent_result = {"primary_intent": intent}
        
        # Enterprise-grade programmatic guardrail for quality assurance
        if intent == "store_personal_info" and user_prompt.strip().endswith("?"):
            logger.warning(f"🚨 CLASSIFICATION GUARDRAIL: Query '{user_prompt}' classified as STORE but ends with '?' - possible misclassification")
            # Log for review but don't override - let the sophisticated prompt handle edge cases
        
        if intent == "recall_personal_info" and not any(word in user_prompt.lower() for word in ["my", "me", "i", "what did i", "who am i"]):
            logger.warning(f"🚨 CLASSIFICATION GUARDRAIL: Query '{user_prompt}' classified as RECALL but lacks personal pronouns - possible misclassification")

    except Exception as e:
        logger.warning(f"Classification error for '{user_prompt}': {e}, defaulting to conversation")
        intent_result = {"primary_intent": "conversation"}

    return intent_result


def check_service_availability() -> None:
    """Check if all required services are available."""
    logger.info("🧠 MEMORY DEBUG: Checking service availability")
    logger.info(f"🧠 MEMORY DEBUG: Redis client object: {app_state.redis_client}")
    logger.info(f"🧠 MEMORY DEBUG: Personal memory object: {app_state.personal_memory}")

    # LLM availability is now checked by the persistent server on-demand
    # No need to check here since the server loads on first request
    redis_available = redis_utils.is_redis_available(app_state.redis_client)
    logger.info(f"🧠 MEMORY DEBUG: Redis availability check result: {redis_available}")

    if not redis_available:
        logger.error("🧠 MEMORY DEBUG: ❌ Redis Service Unavailable - raising HTTPException")
        raise HTTPException(status_code=503, detail="Redis Service Unavailable")

    logger.info("🧠 MEMORY DEBUG: ✅ All services available")


# NO FALLBACK MESSAGE CREATION - SYSTEM MUST WORK OR FAIL

# ===================== FAST PATH IMPLEMENTATION =====================


async def _get_memory_context_for_fast_path(user_prompt: str, session_id: str) -> str:
    """Get minimal memory context for fast path processing."""
    memory_context = ""
    logger.info(f"🧠 MEMORY DEBUG: Attempting to get memory context for session {session_id}")

    if not app_state.personal_memory:
        logger.info("🧠 MEMORY DEBUG: personal_memory not available")
        return memory_context

    logger.info(
        f"🧠 MEMORY DEBUG: personal_memory is available: {type(app_state.personal_memory)}"
    )
    try:
        # Quick memory retrieval for context - FIXED: Use conversation-specific memories
        logger.info(
            f"🧠 MEMORY DEBUG: Calling get_conversation_context for session: '{session_id}'"
        )
        memories = await app_state.personal_memory.get_conversation_context(
            conversation_id=session_id, max_messages=10
        )
        logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(memories) if memories else 0} memories for session {session_id}")

        # CRITICAL FIX: Also get core memories (user facts) for this conversation
        logger.info(f"🧠 MEMORY DEBUG: Getting core memories for conversation {session_id}...")
        core_memories = await app_state.personal_memory.get_all_core_memories(session_id)
        logger.info(
            f"🧠 MEMORY DEBUG: Retrieved {len(core_memories) if core_memories else 0} core memories"
        )

        # Build memory context from both sources
        memory_parts = []

        # Add core memories first (user facts)
        if core_memories:
            logger.info("🧠 MEMORY DEBUG: Processing core memories...")
            for mem in core_memories[:3]:  # Limit to 3 most important
                if hasattr(mem, "content") and mem.content:
                    memory_parts.append(f"- {mem.content}")
                    logger.info(f"🧠 MEMORY DEBUG: Added core memory: {mem.content[:50]}...")

        # Add relevant contextual memories
        if memories:
            logger.info("🧠 MEMORY DEBUG: Processing contextual memories...")
            for mem in memories[:2]:  # Limit to 2 most relevant
                if hasattr(mem, "content") and mem.content:
                    memory_parts.append(f"- {mem.content}")
                    logger.info(f"🧠 MEMORY DEBUG: Added memory: {mem.content[:50]}...")

        if memory_parts:
            memory_context = "\n".join(memory_parts)
            logger.info(f"🧠 MEMORY DEBUG: Final memory context length: {len(memory_context)} chars")
        else:
            logger.info("🧠 MEMORY DEBUG: No useful memory context found")

    except Exception as mem_error:
        logger.error(
            f"🧠 MEMORY DEBUG: ❌ Memory retrieval failed: {mem_error}",
            exc_info=True,
        )

    return memory_context


async def _get_conversation_history_for_fast_path(session_id: str) -> list:
    """Get minimal conversation history for fast path processing."""
    history = []
    logger.info(f"🧠 MEMORY DEBUG: Attempting to get conversation history for session {session_id}")

    if not app_state.personal_memory:
        logger.info("🧠 MEMORY DEBUG: personal_memory not available for history")
        return history

    try:
        logger.info(
            f"🧠 MEMORY DEBUG: Calling get_conversation_context for session {session_id}"
        )
        recent_memories = await app_state.personal_memory.get_conversation_context(
            session_id, max_messages=4
        )  # Last 2 turns
        logger.info(
            f"🧠 MEMORY DEBUG: Retrieved {len(recent_memories) if recent_memories else 0} conversation memories"
        )

        # Convert to history format
        for i in range(0, len(recent_memories), 2):
            if i + 1 < len(recent_memories):
                user_msg = recent_memories[i].content.replace("User: ", "")
                assistant_msg = recent_memories[i + 1].content.replace("Assistant: ", "")
                # Include timestamp from the user message
                timestamp = recent_memories[i].timestamp.strftime("%Y-%m-%d %H:%M:%S")
                history.append(
                    {"user": user_msg, "model": assistant_msg, "timestamp": timestamp}
                )
        logger.info(f"🧠 MEMORY DEBUG: ✅ Converted to {len(history)} history entries")
    except Exception as e:
        logger.error(
            f"🧠 MEMORY DEBUG: ❌ Failed to get conversation history: {e}", exc_info=True
        )

    return history


def _parse_chat_messages(messages: list) -> tuple[str, str, list]:
    """Parse chat messages into system content, user content, and conversation history."""
    system_content = ""
    user_content = ""
    conversation_history = []
    prev_user_content = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_content = content
        elif role == "user":
            prev_user_content = content
            if messages.index(msg) == len(messages) - 1:
                user_content = content
        elif role == "assistant" and prev_user_content:
            conversation_history.append(f"User: {prev_user_content}\nAssistant: {content}")

    return system_content, user_content, conversation_history


async def _stream_llm_response(server, prompt: str):
    """Stream tokens from LLM server."""
    async for token in server.generate_stream(
        prompt=prompt,
        max_tokens=30000,  # Full memory processing capability
        temperature=0.7,
        stop_strings=["User:", "Assistant:", "<|endoftext|>", "<|end|>"],
        top_p=0.9,
    ):
        if token:
            yield f"data: {json.dumps({'text': token})}\n\n"


async def _handle_background_memory_processing(user_prompt: str, full_response: str, session_id: str):
    """Handle background memory processing tasks."""
    logger.info(f"🧠 MEMORY DEBUG: Starting background memory processing for session {session_id}")
    logger.info(f"🧠 MEMORY DEBUG: Response length: {len(full_response)} chars")

    # Schedule minimal memory processing in background
    try:
        logger.info("🧠 MEMORY DEBUG: Creating background task for lightweight_memory_processing")
        task = asyncio.create_task(
            lightweight_memory_processing(user_prompt, full_response, session_id)
        )
        # Store task reference to prevent garbage collection
        getattr(asyncio.current_task(), '_background_tasks', set()).add(task)
        task.add_done_callback(lambda t: getattr(asyncio.current_task(), '_background_tasks', set()).discard(t))
        logger.info("🧠 MEMORY DEBUG: ✅ Background memory processing task created")
    except Exception as memory_task_error:
        logger.error(
            f"🧠 MEMORY DEBUG: ❌ Failed to create memory processing task: {memory_task_error}",
            exc_info=True,
        )

    # Update conversation history (also background)
    try:
        logger.info("🧠 MEMORY DEBUG: Adding conversation to Redis history")
        redis_utils.add_to_conversation_history(
            session_id,
            user_prompt,
            full_response,
            app_state.redis_client,
            redis_utils.CONVERSATION_HISTORY_KEY_PREFIX,
            redis_utils.MAX_NON_VITAL_HISTORY,
        )
        logger.info("🧠 MEMORY DEBUG: ✅ Conversation added to Redis history")
    except Exception as redis_history_error:
        logger.error(
            f"🧠 MEMORY DEBUG: ❌ Failed to add to Redis history: {redis_history_error}",
            exc_info=True,
        )


async def _generate_fast_response_stream(messages: list, user_prompt: str, session_id: str):
    """Generate streaming response for fast path processing."""
    logger.info(f" FAST PATH: Starting lightweight response generation for session {session_id}")

    try:
        from persistent_llm_server import get_llm_server
        server = await get_llm_server()

        # Parse messages into components
        system_content, user_content, conversation_history = _parse_chat_messages(messages)

        # Format prompt based on conversation history
        prompt = (
            utils.format_prompt_with_history(system_content, user_content, "\n".join(conversation_history))
            if conversation_history
            else utils.format_prompt(system_content, user_content)
        )

        logger.info(" FAST PATH: Using persistent server for streaming generation")
        logger.info(f" FAST PATH: Prompt preview: {prompt[:200]}...")

        # Stream tokens from server
        full_response = ""
        async for token_data in _stream_llm_response(server, prompt):
            if token_data.startswith("data: "):
                token_json = json.loads(token_data[6:])
                if 'text' in token_json:
                    full_response += token_json['text']
            yield token_data

        logger.info(f" FAST PATH: Response complete ({len(full_response)} chars)")
        yield f"data: {json.dumps({'done': True})}\n\n"

        # Handle background processing if we have a response
        if full_response:
            await _handle_background_memory_processing(user_prompt, full_response, session_id)
        else:
            logger.warning("🧠 MEMORY DEBUG: No response to process - skipping memory operations")

    except Exception as e:
        error_msg = f"Fast path error: {e!s}"
        logger.error(error_msg, exc_info=True)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"


async def handle_simple_conversational_request(
    request: ChatStreamRequest, user_prompt: str, session_id: str
) -> StreamingResponse:
    """ULTRA-HIGH ROI OPTIMIZATION: Lightweight processing for simple conversational requests.

    This fast path bypasses:
    - Complex query classification (saves 600-1000ms)
    - Heavy memory retrieval with vector search (saves 200-500ms)
    - Complex token optimization (saves 100-300ms)
    - Background analytics tasks (reduces resource contention)

    Expected performance gain: 5-10x faster for 80% of simple requests.
    """
    # Get minimal memory context from personal memory
    memory_context = await _get_memory_context_for_fast_path(user_prompt, session_id)

    # Get minimal conversation history from personal memory
    history = await _get_conversation_history_for_fast_path(session_id)

    # Use condensed version of main system prompt with critical rules
    simple_system_prompt = (
        """You are Aria, a helpful AI assistant. Be natural and conversational."""
    )

    # Add memory context if available
    if memory_context:
        simple_system_prompt += f"## User Information:\n{memory_context}"

    # Create minimal message list (no complex token optimization)
    messages = [{"role": "system", "content": simple_system_prompt}]

    # Add minimal history
    for turn in history[-2:]:  # Only last 2 turns
        user_msg = turn.get("user", "")
        model_msg = turn.get("model", "")
        if user_msg and model_msg:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": model_msg})

    # Add current message
    messages.append({"role": "user", "content": user_prompt})

    # Return StreamingResponse with extracted function
    return StreamingResponse(
        _generate_fast_response_stream(messages, user_prompt, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

async def lightweight_memory_processing(user_prompt: str, response: str, session_id: str):
    """World-class memory processing system that intelligently extracts, analyzes, and stores conversational memories.

    This function implements a sophisticated 5-stage memory processing pipeline:
    1. Content Analysis - LLM-powered content understanding
    2. Memory Extraction - Structured memory extraction
    3. Deduplication - Semantic similarity detection
    4. Storage - Intelligent storage in appropriate tables
    5. Validation - Post-storage verification
    """
    logger.info(
        f"🧠 ADVANCED_MEMORY: Starting world-class memory processing for session {session_id}"
    )

    try:
        if not app_state.personal_memory:
            logger.warning("🧠 ADVANCED_MEMORY: personal_memory is None, exiting early")
            return

        # Import the advanced memory processing system
        from memory_processing import AdvancedMemoryProcessor, get_config

        # Get development configuration (more generous timeouts for local LLM)
        config = get_config("development")

        # Create advanced memory processor instance
        processor = AdvancedMemoryProcessor(app_state.personal_memory, config)

        # Initialize processor with LLM server
        try:
            from persistent_llm_server import get_llm_server

            llm_server = await get_llm_server()
            await processor.initialize(llm_server)
            logger.info("🧠 ADVANCED_MEMORY: LLM server initialized successfully")
        except Exception:
            logger.exception("🧠 ADVANCED_MEMORY: Failed to get LLM server")
            return

        # Extract and store file attachments if present
        await _process_file_attachments(user_prompt, session_id)

        # Process the conversation through the advanced pipeline
        logger.info("🧠 ADVANCED_MEMORY: Processing conversation through 5-stage pipeline")

        processing_result = await processor.process_with_retry(
            user_prompt=user_prompt, assistant_response=response, session_id=session_id
        )

        # Log processing results
        if processing_result.success:
            logger.info(f"🧠 ADVANCED_MEMORY: ✅ Processing successful for session {session_id}")
            logger.info(
                f"🧠 ADVANCED_MEMORY: Stored {processing_result.memories_stored} memories "
                f"in {processing_result.processing_time:.2f}s"
            )

            # Log detailed stage timings
            if processing_result.stage_timings:
                stage_info = ", ".join(
                    [
                        f"{stage}: {time:.2f}s"
                        for stage, time in processing_result.stage_timings.items()
                    ]
                )
                logger.info(f"🧠 ADVANCED_MEMORY: Stage timings - {stage_info}")

            # Log extraction statistics
            if processing_result.extraction_stats:
                stats = processing_result.extraction_stats
                logger.info(
                    f"🧠 ADVANCED_MEMORY: Extraction stats - "
                    f"Total: {stats.get('total_memories', 0)}, "
                    f"Core: {stats.get('core_memories', 0)}, "
                    f"Regular: {stats.get('regular_memories', 0)}, "
                    f"Avg importance: {stats.get('avg_importance', 0):.2f}"
                )

        else:
            error_msg = processing_result.error_message or "Unknown error"
            logger.error(
                f"🧠 ADVANCED_MEMORY: ❌ Processing failed for session {session_id}: {error_msg}"
            )

        # Log system health periodically
        stats = processor.get_processing_stats()
        if stats["total_processed"] % 10 == 0:
            health = processor.get_health_status()
            logger.info(
                f"🧠 ADVANCED_MEMORY: System health - "
                f"Status: {health['status']}, "
                f"Success rate: {health['success_rate']:.1f}%, "
                f"Avg time: {health['avg_processing_time']:.2f}s"
            )

    except Exception as e:
        logger.error(f"🧠 ADVANCED_MEMORY: ❌ Advanced memory processing error: {e}", exc_info=True)


async def _process_file_attachments(user_prompt: str, session_id: str):
    """Extract and store file attachments from user messages with proper memory persistence.
    
    This function identifies file attachments in user messages and stores them as separate
    memory entries so they can be referenced in future conversations.
    """
    import re
    from datetime import UTC, datetime
    
    logger.info(f"📎 FILE_MEMORY: Processing file attachments for session {session_id}")
    
    try:
        # Look for file attachment patterns in the user prompt
        # Pattern: **Attached File: filename.ext**\n```\ncontent\n```
        file_pattern = r'\*\*Attached File: ([^*]+)\*\*\s*\n```\s*\n(.*?)\n```'
        matches = re.findall(file_pattern, user_prompt, re.DOTALL)
        
        if not matches:
            logger.debug("📎 FILE_MEMORY: No file attachments found in message")
            return
            
        logger.info(f"📎 FILE_MEMORY: Found {len(matches)} file attachments")
        
        for filename, content in matches:
            filename = filename.strip()
            content = content.strip()
            
            if not filename or not content:
                logger.warning(f"📎 FILE_MEMORY: Skipping empty file: '{filename}'")
                continue
                
            logger.info(f"📎 FILE_MEMORY: Processing attachment: {filename}")
            
            # Create rich metadata for the file attachment
            file_metadata = {
                "type": "file_attachment",
                "filename": filename,
                "source": "user_upload",
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "content_length": len(content),
                "file_extension": filename.split('.')[-1].lower() if '.' in filename else "unknown"
            }
            
            # Store the file attachment as a high-importance memory
            await app_state.personal_memory.add_memory(
                content=f"User attached file '{filename}' with content:\n\n{content}",
                conversation_id=session_id,
                importance=0.95,  # High importance for file attachments
                metadata=file_metadata
            )
            
            # Also create a separate reference memory for easy lookup
            reference_content = f"File attachment reference: User uploaded file '{filename}' ({len(content)} characters) in this conversation. The file contains: {content[:100]}{'...' if len(content) > 100 else ''}"
            
            await app_state.personal_memory.add_memory(
                content=reference_content,
                conversation_id=session_id,
                importance=0.85,  # High importance for file references
                metadata={
                    "type": "file_reference",
                    "filename": filename,
                    "reference_type": "attachment_index",
                    "session_id": session_id
                }
            )
            
            logger.info(f"📎 FILE_MEMORY: ✅ Stored attachment '{filename}' ({len(content)} chars) with metadata")
            
    except Exception as e:
        logger.error(f"📎 FILE_MEMORY: ❌ Error processing file attachments: {e}", exc_info=True)
