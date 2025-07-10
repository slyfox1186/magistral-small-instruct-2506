#!/usr/bin/env python3
"""Main chat streaming endpoint and related routes."""

import asyncio
import json
import logging
import time
from datetime import datetime

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

# Import models and globals
from .models import ChatStreamRequest

logger = logging.getLogger(__name__)

# Initialize importance scorer
importance_scorer = get_importance_scorer()


async def calculate_message_importance(
    content: str,
    role: str,
    session_id: str,
    messages: list = None
) -> float:
    """Calculate importance score for a message using the sophisticated scorer."""
    try:
        # Build conversation history from recent messages if available
        conversation_history = []
        if messages:
            for msg in messages[-10:]:  # Last 10 messages for context
                # Handle both dict and Message objects
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # It's a Message object
                    conversation_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                elif isinstance(msg, dict):
                    # It's already a dict
                    conversation_history.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

        # Calculate importance
        importance = importance_scorer.calculate_importance(
            text=content,
            role=role,
            conversation_history=conversation_history
        )

        logger.info(f"🧠 IMPORTANCE SCORER: Calculated importance={importance:.3f} for {role} message in session {session_id}")
        return importance
    except Exception as e:
        logger.error(f"Error calculating importance: {e}")
        # Fallback to default values
        return 0.7 if role == "user" else 0.8


def update_chat_activity():
    """Update global chat activity timestamp."""
    bg_state.last_chat_activity = time.time()


def setup_chat_routes(app: FastAPI):
    """Setup chat streaming routes."""

    # ===================== API Endpoints =====================
    @app.post("/api/chat-stream")
    async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        logger.info(
            f"  CHAT REQUEST RECEIVED for session {request.session_id} at {datetime.now().isoformat()}"
        )
        logger.info(f"🧠 MEMORY DEBUG: Chat request details - Messages: {len(request.messages)}, Session: {request.session_id}")
        logger.info(f"🧠 MEMORY DEBUG: Request object type: {type(request)}")

        #  Update chat activity for intelligent background processing
        update_chat_activity()

        # Phase 3A: Start request timing for Prometheus metrics
        start_time = time.time()
        endpoint = "chat_stream"

        # Increment request counter
        if app_state.request_total:
            app_state.request_total.labels(endpoint=endpoint, status="started").inc()

        logger.info("🧠 MEMORY DEBUG: Performing service availability check...")
        try:
            check_service_availability()
            logger.info("🧠 MEMORY DEBUG: ✅ Service availability check passed")
        except Exception as service_error:
            logger.error(f"🧠 MEMORY DEBUG: ❌ Service availability check failed: {service_error}", exc_info=True)
            raise

        session_id = request.session_id
        user_prompt = request.messages[-1].content if request.messages else ""
        logger.info(f"🧠 MEMORY DEBUG: Extracted user prompt: '{user_prompt[:100]}...' (length: {len(user_prompt)})")
        logger.info(f"🧠 MEMORY DEBUG: Session ID: {session_id}")

        if not user_prompt:
            logger.warning("🧠 MEMORY DEBUG: Rejected empty prompt request")
            raise HTTPException(status_code=400, detail="Empty prompt")

        #  PII REDACTION DISABLED - User wants AI to remember personal information
        logger.debug(f"Processing prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")

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
            try:
                if intent == "conversation":
                    logger.info("🧠 MEMORY DEBUG: Processing conversation intent - using inline conversation handling")

                    # Use the same logic as handle_simple_conversational_request but inline
                    # to avoid nested StreamingResponse issues

                    # Get memory context from personal memory
                    memory_context = ""
                    logger.info(f"🧠 MEMORY DEBUG: Attempting to get memory context for session {session_id}")

                    if app_state.personal_memory:
                        logger.info(f"🧠 MEMORY DEBUG: personal_memory is available: {type(app_state.personal_memory)}")
                        try:
                            # Quick memory retrieval for context
                            logger.info(f"🧠 MEMORY DEBUG: Calling get_relevant_memories with query: '{user_prompt[:50]}...'")
                            memories = await app_state.personal_memory.get_relevant_memories(query=user_prompt, limit=5)
                            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(memories) if memories else 0} memories")

                            # CRITICAL FIX: Also get core memories (user facts)
                            logger.info("🧠 MEMORY DEBUG: Getting core memories...")
                            core_memories = await app_state.personal_memory.get_all_core_memories()
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

                    # Use condensed system prompt with memory context
                    simple_system_prompt = """You are Jane, a helpful AI assistant. Be natural and conversational.

🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)"""

                    # Add memory context if available
                    if memory_context:
                        simple_system_prompt += f"\n\n## User Information:\n{memory_context}"

                    # Use persistent LLM server for conversation
                    from persistent_llm_server import get_llm_server
                    server = await get_llm_server()

                    # Format prompt properly
                    formatted_prompt = utils.format_prompt(simple_system_prompt, user_prompt)

                    # Stream tokens directly
                    full_response = ""
                    async for token in server.generate_stream(
                        prompt=formatted_prompt,
                        max_tokens=40960,
                        temperature=0.7,
                        top_p=0.95,
                        session_id=session_id,
                        priority=0
                    ):
                        if token:
                            full_response += token
                            yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                    yield f"data: {json.dumps({'done': True})}\n\n"

                    # Background memory processing
                    if full_response and app_state.personal_memory:
                        try:
                            asyncio.create_task(
                                lightweight_memory_processing(user_prompt, full_response, session_id)
                            )
                        except Exception as memory_error:
                            logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to create memory processing task: {memory_error}")

                    return
                elif intent == "generate_from_knowledge":
                    # DEPRECATED: This intent should use memory-aware conversation instead
                    # Redirect to conversation logic for memory awareness
                    logger.info("🧠 REDIRECTING: generate_from_knowledge → conversation (memory-aware)")
                    intent = "conversation"

                    # Use the same memory-aware logic as conversation intent
                    # Get memory context from personal memory
                    memory_context = ""
                    logger.info(f"🧠 MEMORY DEBUG: Attempting to get memory context for session {session_id}")

                    if app_state.personal_memory:
                        logger.info(f"🧠 MEMORY DEBUG: personal_memory is available: {type(app_state.personal_memory)}")
                        try:
                            # Quick memory retrieval for context
                            logger.info(f"🧠 MEMORY DEBUG: Calling get_relevant_memories with query: '{user_prompt[:50]}...'")
                            memories = await app_state.personal_memory.get_relevant_memories(query=user_prompt, limit=5)
                            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(memories) if memories else 0} memories")

                            # CRITICAL FIX: Also get core memories (user facts)
                            logger.info("🧠 MEMORY DEBUG: Getting core memories...")
                            core_memories = await app_state.personal_memory.get_all_core_memories()
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

                    # Use memory-aware system prompt
                    system_prompt = """You are Jane, a helpful and honest AI assistant. You are always natural and conversational with the user.

### Rules:
- You are NEVER allowed to use [REF] tags in ANY form ([REF]numbers[/REF], [REF]literally_anything[/REF]) and must ONLY use proper markdown links: [Website Title](URL)"""

                    # Add memory context if available
                    if memory_context:
                        system_prompt += f"\n\n## User Information:\n{memory_context}"

                    # Use persistent LLM server for conversation
                    from persistent_llm_server import get_llm_server
                    server = await get_llm_server()

                    # Format prompt properly
                    formatted_prompt = utils.format_prompt(system_prompt, user_prompt)

                    # Stream tokens directly
                    full_response = ""
                    async for token in server.generate_stream(
                        prompt=formatted_prompt,
                        max_tokens=40960,
                        temperature=0.7,
                        top_p=0.95,
                        session_id=session_id,
                        priority=0
                    ):
                        if token:
                            full_response += token
                            yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                    yield f"data: {json.dumps({'done': True})}\n\n"

                    # Background memory processing
                    if full_response and app_state.personal_memory:
                        try:
                            asyncio.create_task(
                                lightweight_memory_processing(user_prompt, full_response, session_id)
                            )
                        except Exception as memory_error:
                            logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to create memory processing task: {memory_error}")

                    return

                elif intent == "perform_web_search":
                    # Handle web search requests with LLM synthesis
                    logger.info("🧠 MEMORY DEBUG: Processing perform_web_search intent")

                    yield f"data: {json.dumps({'token': {'text': '🔍 Searching the web...'}})}\n\n"

                    try:
                        from web_scraper import perform_web_search_async

                        search_results = await perform_web_search_async(
                            query=user_prompt,
                            num_results=8
                        )

                        if search_results:
                            yield f"data: {json.dumps({'token': {'text': '\n\n📊 Found information, generating comprehensive response...\n\n---\n\n'}})}\n\n"

                            # Use persistent LLM server to generate response with search results
                            from persistent_llm_server import get_llm_server
                            llm_server = await get_llm_server()

                            # Create system prompt for knowledge synthesis
                            system_prompt = """You are Jane, a helpful, logical, and honest AI assistant. Use the search results to answer the user's question comprehensively.

🛑 STOP: Before you write ANYTHING, remember these rules:
- NEVER write the characters [ R E F ] followed by anything followed by [ / R E F ]
- NEVER write [REF] in any combination
- ALWAYS write links as [Text](URL) format only

🚨 CRITICAL FORMATTING RULES - VIOLATION IS ABSOLUTELY FORBIDDEN 🚨

**CRITICAL LINK FORMATTING RULE - ABSOLUTE REQUIREMENT:**
You are NEVER allowed to use [REF]URL[/REF] tags for ANY reason and MUST ONLY use proper markdown links: [Text Here](URL)

**LINK FORMATTING RULE**:
❌ ABSOLUTELY FORBIDDEN: [REF]1[/REF], [REF]2[/REF], [REF]source[/REF], [REF]anything[/REF] - NEVER USE THESE
✅ REQUIRED: [Description](https://actual-url.com) - ALWAYS USE THESE INSTEAD

Examples:
❌ BAD: "Donald Trump is president [REF]1,2,3[/REF]"
✅ GOOD: "Donald Trump is president according to [Wikipedia](https://en.wikipedia.org/wiki/Donald_Trump)"

**MANDATORY TABLE USAGE**: When presenting any structured data, comparisons, lists of items with attributes, or multiple data points, YOU MUST use markdown tables:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |

**STRUCTURE EVERYTHING**: Convert any structured information into organized sections with headers, tables, and lists. If data can be structured, it MUST be in a table format.

REMEMBER: NEVER use [REF] tags in any form. Always use proper markdown links and tables for structured data.

🚨🚨🚨 FINAL WARNING 🚨🚨🚨
DO NOT WRITE [REF] FOLLOWED BY ANY TEXT FOLLOWED BY [/REF]
DO NOT WRITE [REF]1[/REF] OR [REF]2[/REF] OR [REF]URL[/REF] OR ANY VJaneNT
ONLY USE: [Description](URL) format for ALL links
🚨🚨🚨 FINAL WARNING 🚨🚨🚨"""

                            user_content = f"User Query: {user_prompt}\n\nSearch Results:\n{search_results}\n\nPlease provide a comprehensive response to the user's query using the search results above. CRITICAL: Use ONLY [Description](URL) format for links, NEVER use [REF] tags."

                            formatted_prompt = utils.format_prompt(system_prompt, user_content)

                            # Stream the response token by token
                            full_response = ""
                            async for token in llm_server.generate_stream(
                                prompt=formatted_prompt,
                                max_tokens=40960,
                                temperature=0.7,
                                top_p=0.95,
                                session_id=session_id,
                                priority=1
                            ):
                                if token:
                                    print(f"🚀 PERFORM_WEB_SEARCH RECEIVED TOKEN: '{token}' (len={len(token)})")
                                    full_response += token
                                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                            # Store in memory for future reference
                            if app_state.personal_memory and full_response:
                                try:
                                    logger.info(f"🧠 MEMORY STORAGE: Storing web search results in memory for session {session_id}")

                                    # Calculate dynamic importance for user message
                                    user_importance = await calculate_message_importance(
                                        content=user_prompt,
                                        role="user",
                                        session_id=session_id,
                                        messages=request.messages
                                    )

                                    await app_state.personal_memory.add_memory(
                                        content=f"User: {user_prompt}",
                                        conversation_id=session_id,
                                        importance=user_importance,
                                    )

                                    # Calculate dynamic importance for assistant response (with web search context)
                                    assistant_importance = await calculate_message_importance(
                                        content=full_response,
                                        role="assistant",
                                        session_id=session_id,
                                        messages=request.messages + [{"role": "user", "content": user_prompt}]
                                    )

                                    await app_state.personal_memory.add_memory(
                                        content=f"Assistant: {full_response}",
                                        conversation_id=session_id,
                                        importance=assistant_importance,
                                    )
                                    logger.info("🧠 MEMORY STORAGE: ✅ Successfully stored web search memories")
                                except Exception as memory_error:
                                    logger.error(f"🧠 MEMORY STORAGE: ❌ Failed to store in memory: {memory_error}", exc_info=True)
                        else:
                            yield f"data: {json.dumps({'token': {'text': '\n\n❌ No search results found.\n\n'}})}\n\n"

                    except Exception as search_error:
                        error_msg = f"Search error: {search_error}"
                        logger.error(f"🧠 MEMORY DEBUG: ❌ {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

                    yield f"data: {json.dumps({'done': True})}\n\n"

                elif intent == "store_personal_info":
                    # Handle storing personal information
                    logger.info("🧠 MEMORY STORE: Processing store_personal_info intent")

                    yield f"data: {json.dumps({'token': {'text': '🧠 Storing your information...\n\n---\n\n'}})}\n\n"

                    try:
                        if app_state.personal_memory:
                            # Extract and store personal information from the user's message
                            logger.info(f"🧠 MEMORY STORE: Extracting personal info from: '{user_prompt}'")

                            # Store with high importance since user explicitly provided it
                            await app_state.personal_memory.add_memory(
                                content=f"User provided personal information: {user_prompt}",
                                conversation_id=session_id,
                                importance=0.9,  # High importance for explicit personal info
                            )

                            logger.info("🧠 MEMORY STORE: ✅ Successfully stored personal information")

                            # Generate natural confirmation response using LLM
                            from persistent_llm_server import get_llm_server
                            llm_server = await get_llm_server()

                            system_prompt = """You are Jane, a helpful AI assistant with a warm, caring personality. The user has just shared personal information with you, and you have successfully stored it in your memory. 

You must return your responses using proper markdown formatting and use markdown tables for structured data.

When creating tables, use this format:
| Category | Details |
|----------|---------|
| Field1 | Value1 |
| Field2 | Value2 |

Be genuinely warm and appreciative that they shared personal details with you. Acknowledge what they shared, express that you'll remember it, and show how this helps you understand them better. Keep it conversational and natural - like a friend would respond. Be brief but meaningful."""

                            formatted_prompt = utils.format_prompt(system_prompt, f"I just told you: {user_prompt}")

                            # Stream the natural response
                            async for token in llm_server.generate_stream(
                                prompt=formatted_prompt,
                                max_tokens=300,  # Increased to allow for better formatted responses
                                temperature=0.7,  # Slightly higher temperature for natural responses
                                top_p=0.95,
                                session_id=session_id,
                                priority=1
                            ):
                                if token:
                                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"
                        else:
                            logger.warning("🧠 MEMORY STORE: personal_memory not available")
                            yield f"data: {json.dumps({'token': {'text': '❌ Memory system not available to store information.'}})}\n\n"

                    except Exception as store_error:
                        error_msg = f"Failed to store information: {store_error}"
                        logger.error(f"🧠 MEMORY STORE: ❌ {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'❌ {error_msg}'}})}\n\n"

                    # CRITICAL: Use world-class memory processing for personal info storage
                    if app_state.personal_memory:
                        try:
                            asyncio.create_task(
                                lightweight_memory_processing(user_prompt, "Information successfully stored in memory", session_id)
                            )
                        except Exception as memory_error:
                            logger.error(f"🧠 MEMORY STORE: ❌ Failed to create memory processing task: {memory_error}")

                    yield f"data: {json.dumps({'done': True})}\n\n"

                elif intent == "recall_personal_info":
                    # Handle recalling personal information
                    logger.info("🧠 MEMORY RECALL: Processing recall_personal_info intent")

                    yield f"data: {json.dumps({'token': {'text': '🧠 Searching my memory...\n\n'}})}\n\n"

                    try:
                        if app_state.personal_memory:
                            # Search for relevant personal information
                            logger.info(f"🧠 MEMORY RECALL: Searching for: '{user_prompt}'")

                            # Get relevant memories
                            memories = await app_state.personal_memory.get_relevant_memories(query=user_prompt, limit=10)

                            # Also get core memories (user facts)
                            core_memories = await app_state.personal_memory.get_all_core_memories()

                            # Build response with found information
                            if memories or core_memories:
                                from persistent_llm_server import get_llm_server
                                llm_server = await get_llm_server()

                                # Prepare memory context
                                memory_parts = []

                                if core_memories:
                                    core_facts = []
                                    for key, value in core_memories.items():
                                        core_facts.append(f"{key}: {value}")
                                    if core_facts:
                                        memory_parts.append("Personal Facts:\n" + "\n".join(core_facts))

                                if memories:
                                    memory_content = [m.content for m in memories[:5]]
                                    memory_parts.extend(memory_content)

                                memory_context = "\n\n".join(memory_parts)

                                # Generate response using memory context
                                system_prompt = f"""You are Jane, a helpful, logical, and honest AI assistant recalling information about the user. Based on the memories below, answer the user's question directly and naturally.

🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)

## Available Information:
{memory_context}

Answer the user's question based on this information. If the information isn't available, say so clearly."""

                                formatted_prompt = utils.format_prompt(system_prompt, user_prompt)

                                # Stream the response
                                async for token in llm_server.generate_stream(
                                    prompt=formatted_prompt,
                                    max_tokens=1024,
                                    temperature=0.3,  # Lower temperature for factual recall
                                    top_p=0.95,
                                    session_id=session_id,
                                    priority=1
                                ):
                                    if token:
                                        yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                                logger.info("🧠 MEMORY RECALL: ✅ Successfully recalled information")
                            else:
                                logger.info("🧠 MEMORY RECALL: No relevant memories found")
                                yield f"data: {json.dumps({'token': {'text': 'I do not have any information about that in my memory.'}})}\n\n"
                        else:
                            logger.warning("🧠 MEMORY RECALL: personal_memory not available")
                            yield f"data: {json.dumps({'token': {'text': '❌ Memory system not available to recall information.'}})}\n\n"

                    except Exception as recall_error:
                        error_msg = f"Failed to recall information: {recall_error}"
                        logger.error(f"🧠 MEMORY RECALL: ❌ {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'❌ {error_msg}'}})}\n\n"

                    # Background memory processing for recall queries - store the fact that user asked about this topic
                    if app_state.personal_memory:
                        try:
                            asyncio.create_task(
                                lightweight_memory_processing(user_prompt, "Assistant recalled information from memory", session_id)
                            )
                        except Exception as memory_error:
                            logger.error(f"🧠 MEMORY RECALL: ❌ Failed to create memory processing task: {memory_error}")

                    yield f"data: {json.dumps({'done': True})}\n\n"

                elif intent == "query_conversation_history":
                    # Handle memory retrieval requests
                    logger.info("🧠 MEMORY DEBUG: Processing query_conversation_history intent")

                    yield f"data: {json.dumps({'token': {'text': '🧠 Searching memory...'}})}\n\n"

                    try:
                        # Get memory context from personal memory
                        memory_context = ""
                        if app_state.personal_memory:
                            try:
                                logger.info(f"🧠 MEMORY RETRIEVAL: Searching for memories with query: '{user_prompt}'")
                                logger.info(f"🧠 MEMORY RETRIEVAL: Session ID: {session_id}")

                                memories = await app_state.personal_memory.get_relevant_memories(
                                    query=user_prompt,
                                    limit=10
                                )

                                logger.info(f"🧠 MEMORY RETRIEVAL: Raw memories retrieved: {len(memories) if memories else 0}")

                                if memories:
                                    logger.info("🧠 MEMORY RETRIEVAL: === DETAILED MEMORY CONTENTS ===")
                                    for i, memory in enumerate(memories):
                                        logger.info(f"🧠 MEMORY RETRIEVAL: Memory {i+1}:")
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   ID: {memory.id}")
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   Conversation ID: {memory.conversation_id}")
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   Timestamp: {memory.timestamp}")
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   Importance: {memory.importance}")
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   Content (first 200 chars): '{memory.content[:200]}...'")
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   Full content length: {len(memory.content)}")
                                        if hasattr(memory, 'summary') and memory.summary:
                                            logger.info(f"🧠 MEMORY RETRIEVAL:   Summary: {memory.summary}")
                                        logger.info("🧠 MEMORY RETRIEVAL:   ---")

                                    memory_context = "\n".join([
                                        f"Memory: {memory.content}" for memory in memories
                                    ])
                                    logger.info(f"🧠 MEMORY RETRIEVAL: ✅ Successfully retrieved {len(memories)} memories")
                                    logger.info(f"🧠 MEMORY RETRIEVAL: Combined memory context length: {len(memory_context)}")
                                    logger.info(f"🧠 MEMORY RETRIEVAL: Combined context preview (first 500 chars): '{memory_context[:500]}...'")
                                else:
                                    logger.info("🧠 MEMORY RETRIEVAL: No relevant memories found in database")
                            except Exception as memory_error:
                                logger.error(f"🧠 MEMORY RETRIEVAL: ❌ Memory retrieval failed: {memory_error}", exc_info=True)

                        if memory_context:
                            yield f"data: {json.dumps({'token': {'text': '\n\n📋 Found relevant information from our conversation...\n\n'}})}\n\n"

                            # Use LLM to synthesize memory content
                            from persistent_llm_server import get_llm_server
                            llm_server = await get_llm_server()

                            system_prompt = """You are Jane, a helpful, logical, and honest AI assistant retrieving information from conversation memory.

🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)
                            
## Core Rules:
1. Use proper markdown formatting: headers, lists, bold text, etc.
2. Use memory content: Base your response on the provided memory context
3. Be accurate: Only reference what's actually in the memory
4. Be helpful: Present the information clearly and completely
5. Use standard markdown links: [Description](URL) format"""

                            user_content = f"User Query: {user_prompt}\n\nRelevant Memory Context:\n{memory_context}\n\nPlease provide a helpful response based on the memory context above."

                            formatted_prompt = utils.format_prompt(system_prompt, user_content)

                            # Stream the response token by token
                            full_response = ""
                            async for token in llm_server.generate_stream(
                                prompt=formatted_prompt,
                                max_tokens=40960,
                                temperature=0.7,
                                top_p=0.95,
                                session_id=session_id,
                                priority=1
                            ):
                                if token:
                                    full_response += token
                                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"
                        else:
                            yield f"data: {json.dumps({'token': {'text': '\n\n🤔 I don\'t have any relevant information about that in our conversation history. Would you like me to search for new information instead?\n\n'}})}\n\n"

                    except Exception as memory_error:
                        error_msg = f"Memory retrieval error: {memory_error}"
                        logger.error(f"🧠 MEMORY DEBUG: ❌ {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

                    # Background memory processing for conversation history queries
                    if 'full_response' in locals() and full_response and app_state.personal_memory:
                        try:
                            asyncio.create_task(
                                lightweight_memory_processing(user_prompt, full_response, session_id)
                            )
                        except Exception as memory_error:
                            logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to create memory processing task: {memory_error}")

                    yield f"data: {json.dumps({'done': True})}\n\n"

                elif intent == "query_stocks":
                    # Handle stock market queries
                    logger.info(f"📈 STOCKS: Processing stock query: '{user_prompt}'")

                    yield f"data: {json.dumps({'token': {'text': '📈 Fetching stock data...'}})}\n\n"

                    try:
                        # Let the LLM handle the complexity of extracting stock symbols with improved prompt
                        from persistent_llm_server import get_llm_server
                        llm_server = await get_llm_server()

                        # Enhanced system prompt for accurate symbol extraction
                        system_prompt = """You are a stock market expert. Extract valid US stock ticker symbols from the user's query.

CRITICAL INSTRUCTIONS:
1. Return ONLY a JSON array of valid US stock ticker symbols (uppercase)
2. Use standard NYSE/NASDAQ symbols: AAPL (not APPL), MSFT, GOOGL, AMZN, TSLA, META, NVDA, etc.
3. Be precise with symbol spelling - AAPL for Apple, not APPL
4. If company names are mentioned, convert to correct ticker symbols
5. Maximum 10 stocks
6. NO explanations, NO other text, ONLY the JSON array

Examples:
- "apple stock" → ["AAPL"]
- "microsoft and google" → ["MSFT", "GOOGL"]
- "tech stocks" → ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
- "tesla vs apple" → ["TSLA", "AAPL"]"""

                        formatted_prompt = utils.format_prompt(system_prompt, user_prompt)

                        symbol_response = await llm_server.generate(
                            prompt=formatted_prompt,
                            max_tokens=100,
                            temperature=0.1,
                            session_id=f"{session_id}_symbol_extraction"
                        )

                        logger.info(f"📈 STOCKS: LLM extracted symbols: '{symbol_response.strip()}'")

                        # Parse JSON response from LLM (no regex patterns)
                        tickers = []
                        if symbol_response.strip():
                            try:
                                # Try to find JSON array in the response
                                start_idx = symbol_response.find("[")
                                end_idx = symbol_response.rfind("]")
                                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                    symbols_json = symbol_response[start_idx:end_idx+1]
                                    potential_tickers = json.loads(symbols_json)
                                    if isinstance(potential_tickers, list):
                                        potential_tickers = [str(t).upper().strip() for t in potential_tickers[:10]]
                                        logger.info(f"📈 STOCKS: Potential tickers to validate: {potential_tickers}")

                                        # Validate each ticker with yfinance
                                        if app_state.stock_searcher and potential_tickers:
                                            valid_results = app_state.stock_searcher.validate_symbols(potential_tickers)
                                            tickers = [t for t, is_valid in valid_results.items() if is_valid]
                                            logger.info(f"📈 STOCKS: Validated tickers: {tickers}")
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.warning(f"📈 STOCKS: Failed to parse LLM JSON response: {e}")
                                logger.warning(f"📈 STOCKS: Raw response was: '{symbol_response.strip()}'")
                                # Fallback: if JSON parsing fails, don't process anything
                                pass

                        # If no tickers found, try to be helpful
                        if not tickers:
                            yield f"data: {json.dumps({'token': {'text': '\n\n❓ No stock symbols detected in your query. Please specify stock ticker symbols (e.g., AAPL, MSFT, GOOGL) or company names.\n\n'}})}\n\n"
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            return

                        # Get stock data
                        if app_state.stock_searcher:
                            yield f"data: {json.dumps({'token': {'text': f' Getting quotes for {", ".join(tickers)}...\n\n'}})}\n\n"

                            # Get raw stock data instead of pre-formatted strings
                            stock_quotes = await asyncio.to_thread(
                                app_state.stock_searcher.get_multiple_quotes, tickers
                            )

                            if stock_quotes:
                                # Convert raw stock data to JSON for AI processing
                                stock_data_json = []
                                for symbol, quote in stock_quotes.items():
                                    if quote:
                                        stock_data_json.append({
                                            "symbol": symbol,
                                            "name": quote.name,
                                            "price": quote.price,
                                            "change": quote.change,
                                            "change_percent": quote.change_percent,
                                            "volume": quote.volume,
                                            "market_cap": quote.market_cap
                                        })

                                # Use LLM to create a natural response with raw JSON data
                                from persistent_llm_server import get_llm_server
                                llm_server = await get_llm_server()

                                system_prompt = """You are a knowledgeable financial assistant with access to real-time stock market data.

CRITICAL FORMATTING RULES:
1. **MANDATORY**: Always format stock data in clean markdown tables like this:

| Symbol | Name | Price | Change | % Change | Volume | Market Cap |
|--------|------|-------|--------|----------|--------|------------|
| AAPL | Apple Inc. | $175.84 | +$2.41 | +1.39% | 50,334,500 | $2.75T |

2. Use proper formatting:
   - Green indicators for positive changes: ✅ or 🟢
   - Red indicators for negative changes: ❌ or 🔴  
   - Format large numbers appropriately (T for trillion, B for billion, M for million)
   - Include currency symbols ($) for prices
   - Show + or - signs for changes

3. Provide brief analysis of the data after the table."""

                                user_content = f"User Query: {user_prompt}\n\nStock Market Data (JSON):\n{json.dumps(stock_data_json, indent=2)}\n\nCreate a clean markdown table and respond naturally to the user's query using this data."

                                formatted_prompt = utils.format_prompt(system_prompt, user_content)

                                # Stream the response
                                full_response = ""
                                async for token in llm_server.generate_stream(
                                    prompt=formatted_prompt,
                                    max_tokens=2048,
                                    temperature=0.7,
                                    top_p=0.95,
                                    session_id=session_id,
                                    priority=1
                                ):
                                    if token:
                                        full_response += token
                                        yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                                # Add sources
                                sources_text = "\n\n📊 **Data Sources:**\n"
                                for symbol in tickers[:3]:  # Limit to 3 sources
                                    sources_text += f"- [Yahoo Finance - {symbol}](https://finance.yahoo.com/quote/{symbol})\n"
                                yield f"data: {json.dumps({'token': {'text': sources_text}})}\n\n"

                                # Store in memory
                                if app_state.personal_memory and full_response:
                                    try:
                                        # Calculate importance
                                        user_importance = await calculate_message_importance(
                                            content=user_prompt,
                                            role="user",
                                            session_id=session_id,
                                            messages=request.messages
                                        )

                                        await app_state.personal_memory.add_memory(
                                            content=f"User: {user_prompt}",
                                            conversation_id=session_id,
                                            importance=user_importance,
                                        )

                                        assistant_importance = await calculate_message_importance(
                                            content=full_response,
                                            role="assistant",
                                            session_id=session_id,
                                            messages=request.messages + [{"role": "user", "content": user_prompt}]
                                        )

                                        await app_state.personal_memory.add_memory(
                                            content=f"Assistant: {full_response}",
                                            conversation_id=session_id,
                                            importance=assistant_importance,
                                        )
                                        logger.info("📈 STOCKS: Stored stock query conversation in memory")
                                    except Exception as memory_error:
                                        logger.error(f"📈 STOCKS: Failed to store in memory: {memory_error}")
                            else:
                                yield f"data: {json.dumps({'token': {'text': '\n\n❌ Unable to fetch stock data. The symbols may be invalid or the market data service is unavailable.\n\n'}})}\n\n"
                        else:
                            yield f"data: {json.dumps({'token': {'text': '\n\n❌ Stock market data service is not available.\n\n'}})}\n\n"

                    except Exception as stock_error:
                        error_msg = f"Error fetching stock data: {stock_error}"
                        logger.error(f"📈 STOCKS: {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

                    yield f"data: {json.dumps({'done': True})}\n\n"

                elif intent == "query_weather":
                    # Handle weather queries
                    logger.info(f"🌤️ WEATHER: Processing weather query: '{user_prompt}'")

                    yield f"data: {json.dumps({'token': {'text': '🌤️ Fetching weather data...\n\n'}})}\n\n"

                    try:
                        # Use the weather module directly
                        # Extract city from user query using LLM
                        from persistent_llm_server import get_llm_server
                        from weather import (
                            format_weather_response,
                            get_weather_for_city,
                        )
                        llm_server = await get_llm_server()

                        # System prompt for city extraction
                        system_prompt = """You are a location extraction expert. Extract the city name from the user's weather query.
                        
Return ONLY the city name (e.g., "New York", "London", "Tokyo"). If no specific city is mentioned, return "current location".
If multiple cities are mentioned, return the first one mentioned.

Examples:
"What's the weather in Paris?" → "Paris"
"How hot is it in New York City?" → "New York City"
"Is it raining in London?" → "London"
"What's the weather like?" → "current location"
"Tell me about the weather" → "current location"
"""

                        formatted_prompt = utils.format_prompt(system_prompt, user_prompt)

                        # Extract city name
                        city_response = ""
                        async for token in llm_server.generate_stream(
                            prompt=formatted_prompt,
                            max_tokens=50,
                            temperature=0.1,
                            top_p=0.9,
                            session_id=session_id,
                            priority=1
                        ):
                            if token:
                                city_response += token

                        city_name = city_response.strip()
                        logger.info(f"🌤️ WEATHER: Extracted city: '{city_name}'")

                        # Handle "current location" case
                        if city_name.lower() == "current location":
                            yield f"data: {json.dumps({'token': {'text': '📍 Please specify a city name for weather information.\n\n'}})}\n\n"
                        else:
                            # Convert state names to abbreviations using LLM and add country code
                            state_convert_prompt = f"""You are a helpful and honest AI assistant. Your only job is to convert any USA state names in the following location query to their standard 2-letter abbreviations.

### Rules:
- Append the country code for US locations ',US'
- Keep city names unchanged.
- Use no spaces after commas in the output format.
- Return only the converted location and nothing else.

### Examples:
- "Midway Georgia" -> "Midway,GA,US"
- "Los Angeles California" -> "Los Angeles,CA,US" 
- "New York" -> "New York,NY,US"
- "Georgia" -> "GA,US"
- "Paris France" -> "Paris,FR" (not a US state)
- "London" -> "London,GB"

### Location(s):
{city_name}

### Final Instructions:
- The full output must look exactly like this every time: 'CITY,ABBREVIATED_STATE,US'

Converted:"""

                            state_convert_response = ""
                            async for token in llm_server.generate_stream(
                                prompt=utils.format_prompt("You are a precise location converter.", state_convert_prompt),
                                max_tokens=50,
                                temperature=0.1,
                                top_p=0.9,
                                session_id=session_id,
                                priority=1
                            ):
                                if token:
                                    state_convert_response += token

                            converted_city = state_convert_response.strip()
                            logger.info(f"🌤️ WEATHER: State conversion: '{city_name}' -> '{converted_city}'")

                            # Get weather data with converted city name
                            weather_data = await get_weather_for_city(converted_city)

                            if weather_data and "error" not in weather_data:
                                yield f"data: {json.dumps({'token': {'text': '\n\n📊 Found weather data, generating comprehensive response...\n\n---\n\n'}})}\n\n"

                                # Use LLM to generate a comprehensive weather response
                                weather_system_prompt = """You are Jane, a helpful weather assistant. Use the weather data to provide a comprehensive and natural weather report.

You must return your responses using proper markdown formatting and use markdown tables for structured data when appropriate.

When creating tables, use this format:
| Category | Details |
|----------|---------|
| Field1 | Value1 |
| Field2 | Value2 |

Format the weather information in a natural, conversational way. Include all the important details like temperature (show both Fahrenheit and Celsius), conditions, humidity, wind speed, pressure, and UV index if available. Be helpful and engaging."""

                                # Format weather data for LLM
                                formatted_weather = format_weather_response(weather_data)
                                user_content = f"User Query: {user_prompt}\n\nWeather Data:\n{formatted_weather}\n\nPlease provide a comprehensive, natural weather report using the weather data above."

                                formatted_prompt = utils.format_prompt(weather_system_prompt, user_content)

                                # Stream the response token by token
                                full_response = ""
                                async for token in llm_server.generate_stream(
                                    prompt=formatted_prompt,
                                    max_tokens=2048,
                                    temperature=0.7,
                                    top_p=0.95,
                                    session_id=session_id,
                                    priority=1
                                ):
                                    if token:
                                        full_response += token
                                        yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                                # Add data source
                                yield f"data: {json.dumps({'token': {'text': '\n\n📊 **Data Source:** [OpenWeatherMap](https://openweathermap.org/)\n\n'}})}\n\n"

                                # Store in memory
                                if app_state.personal_memory and full_response:
                                    try:
                                        # Calculate importance
                                        user_importance = await calculate_message_importance(
                                            content=user_prompt,
                                            role="user",
                                            session_id=session_id,
                                            messages=request.messages
                                        )

                                        await app_state.personal_memory.add_memory(
                                            content=f"User: {user_prompt}",
                                            conversation_id=session_id,
                                            importance=user_importance,
                                        )

                                        assistant_importance = await calculate_message_importance(
                                            content=full_response,
                                            role="assistant",
                                            session_id=session_id,
                                            messages=request.messages + [{"role": "user", "content": user_prompt}]
                                        )

                                        await app_state.personal_memory.add_memory(
                                            content=f"Assistant: {full_response}",
                                            conversation_id=session_id,
                                            importance=assistant_importance,
                                        )
                                        logger.info("🌤️ WEATHER: Stored weather query conversation in memory")
                                    except Exception as memory_error:
                                        logger.error(f"🌤️ WEATHER: Failed to store in memory: {memory_error}")
                            else:
                                error_msg = weather_data.get("error", "Unable to fetch weather data") if weather_data else "Weather service unavailable"
                                yield f"data: {json.dumps({'token': {'text': f'❌ {error_msg}\n\n'}})}\n\n"

                    except Exception as weather_error:
                        error_msg = f"Weather error: {weather_error}"
                        logger.error(f"🌤️ WEATHER: {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

                    yield f"data: {json.dumps({'done': True})}\n\n"

                elif intent == "query_cryptocurrency":
                    # Handle cryptocurrency queries
                    logger.info(f"₿ CRYPTO: Processing cryptocurrency query: '{user_prompt}'")

                    yield f"data: {json.dumps({'token': {'text': '₿ Fetching cryptocurrency data...\n\n'}})}\n\n"

                    try:
                        # Use the crypto trading module directly
                        from crypto_trading import CryptoTrading
                        crypto_trader = CryptoTrading()

                        # Extract cryptocurrency mentions from user query
                        query_lower = user_prompt.lower()
                        common_cryptos = ['bitcoin', 'ethereum', 'cardano', 'solana', 'binancecoin', 'ripple', 'dogecoin']
                        requested_cryptos = []

                        # Check for specific cryptocurrency mentions
                        for crypto in common_cryptos:
                            if crypto in query_lower or crypto[:3] in query_lower:
                                requested_cryptos.append(crypto)

                        # If no specific crypto mentioned, default to top cryptocurrencies
                        if not requested_cryptos:
                            requested_cryptos = ['bitcoin', 'ethereum', 'cardano']

                        # Get cryptocurrency price data
                        crypto_data = crypto_trader.get_multiple_crypto_quotes(requested_cryptos[:5])

                        if crypto_data:
                            logger.info(f"₿ CRYPTO: Retrieved {len(crypto_data)} cryptocurrency quotes")

                            # Format the price data
                            formatted_data, sources = crypto_trader.format_crypto_data_with_sources(requested_cryptos[:5])

                            # Get market sentiment
                            sentiment = crypto_trader.get_market_sentiment()

                            # Use LLM to create a natural response with the price data
                            from persistent_llm_server import get_llm_server
                            llm_server = await get_llm_server()

                            system_prompt = """You are Jane, a helpful, knowledgeable, and honest AI assistant. You are a cryptocurrency expert with access to real-time market data from CoinGecko.
### Instructions:
Return the highest quality responses possible to the user's query in order to fully satisfies their needs.

### Response Rules:
1. ONLY return your reponse using proper markdown formatting
2. Return ALL structured data in markdown tables
3. Use standard markdown links: [Description](URL) format
4. 🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)
5. Use the market sentiment data to give context
6. Be helpful and informative about cryptocurrency trends
7. NEVER make up information or hallucinate and ONLY return data that you have had direct access to. If real data was not presented to you then you must inform the user of this.

### Example Markdown Table Format:
| Coin | Price | 24h Change | Market Cap |
|------|-------|------------|------------|
| Bitcoin | $XX,XXX | +X.XX% | $X.XX B |"""

                            sentiment_text = ""
                            if sentiment:
                                sentiment_text = f"\n\nMarket Sentiment Analysis:\n- Overall sentiment: {sentiment.get('overall_sentiment', 'Unknown')}\n- Positive coins: {sentiment.get('positive_sentiment', 0)}\n- Negative coins: {sentiment.get('negative_sentiment', 0)}\n- Neutral coins: {sentiment.get('neutral_sentiment', 0)}"

                            user_content = f"User Query: {user_prompt}\n\nCryptocurrency Price Data:\n{formatted_data}{sentiment_text}\n\nRespond naturally to the user's query using this market data."

                            formatted_prompt = utils.format_prompt(system_prompt, user_content)

                            # Stream the response
                            full_response = ""
                            async for token in llm_server.generate_stream(
                                prompt=formatted_prompt,
                                max_tokens=None,
                                temperature=0.15,
                                top_p=0.95,
                                session_id=session_id,
                                priority=1
                            ):
                                if token:
                                    full_response += token
                                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

                            # Add sources if available
                            if sources:
                                sources_text = "\n\n📊 **Data Sources:**\n"
                                for source in sources[:3]:  # Limit to 3 sources
                                    sources_text += f"- [{source['title'][:50]}...]({source['url']})\n"
                                yield f"data: {json.dumps({'token': {'text': sources_text}})}\n\n"

                            # Store in memory
                            if app_state.personal_memory and full_response:
                                try:
                                    # Calculate importance for user query
                                    user_importance = await calculate_message_importance(
                                        content=user_prompt,
                                        role="user",
                                        session_id=session_id,
                                        messages=request.messages
                                    )

                                    await app_state.personal_memory.add_memory(
                                        content=f"User asked about cryptocurrency: {user_prompt}",
                                        conversation_id=session_id,
                                        importance=user_importance,
                                    )

                                    # Calculate importance for assistant response
                                    assistant_importance = await calculate_message_importance(
                                        content=full_response,
                                        role="assistant",
                                        session_id=session_id,
                                        messages=request.messages + [{"role": "user", "content": user_prompt}]
                                    )

                                    await app_state.personal_memory.add_memory(
                                        content=f"Assistant provided crypto price data: {full_response[:200]}...",
                                        conversation_id=session_id,
                                        importance=assistant_importance,
                                    )
                                    logger.info("₿ CRYPTO: Stored crypto query conversation in memory")
                                except Exception as memory_error:
                                    logger.error(f"₿ CRYPTO: Failed to store in memory: {memory_error}")
                        else:
                            yield f"data: {json.dumps({'token': {'text': '\n\n❌ Unable to fetch cryptocurrency price data. The CoinGecko API may be unavailable.\n\n'}})}\n\n"

                    except Exception as crypto_error:
                        error_msg = f"Error fetching cryptocurrency data: {crypto_error}"
                        logger.error(f"₿ CRYPTO: {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

                    yield f"data: {json.dumps({'done': True})}\n\n"

                else:
                    # Handle other intent types with placeholder
                    logger.info(f"🧠 MEMORY DEBUG: Processing unimplemented intent: {intent}")
                    yield f"data: {json.dumps({'token': {'text': f'🚧 Handler for {intent} not yet implemented. This would typically involve specialized processing.'}})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"

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
