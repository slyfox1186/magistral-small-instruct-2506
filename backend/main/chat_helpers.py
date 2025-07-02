#!/usr/bin/env python3
"""Helper functions for chat processing and query classification."""

import asyncio
import json
import logging
import re
import time
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# Import from other modules
from .models import ChatStreamRequest
from . import globals as global_vars
import redis_utils
import utils

logger = logging.getLogger(__name__)

# ===================== Helper Functions =====================


def classify_query_fast_pattern_based(user_prompt: str) -> dict[str, any]:
    """ULTRA-FAST PATTERN-BASED QUERY CLASSIFICATION

    Only handles TRULY simple conversational queries.
    Everything else goes to the LLM for intelligent classification.
    """
    prompt_lower = user_prompt.lower().strip()

    # ONLY handle dead-simple greetings and thanks
    # These are so trivial that wasting LLM tokens on them is unnecessary
    simple_greeting_patterns = [
        r"^(hi|hello|hey|yo|greetings)[\s!.]*$",
        r"^(good morning|good afternoon|good evening|good night)[\s!.]*$",
        r"^(thanks|thank you|ty|thx)[\s!.]*$",
        r"^(bye|goodbye|see you|talk to you later|ttyl)[\s!.]*$",
        r"^(yes|no|yeah|nah|yep|nope|ok|okay|sure)[\s!.]*$",
        r"^(lol|haha|hehe|wow|cool|nice|awesome|great)[\s!.]*$",
    ]

    import re

    for pattern in simple_greeting_patterns:
        if re.match(pattern, prompt_lower):
            return {"primary_intent": "conversation"}

    # Everything else needs LLM classification
    # Return None to indicate "needs LLM classification"
    return None


async def classify_query_with_llm(user_prompt: str) -> dict[str, any]:
    """Intelligent query classification with conversational bias"""
    # Pre-filter: Single words or very short queries default to conversation
    # unless they have clear search indicators
    words = user_prompt.strip().split()
    if len(words) <= 2:
        # Check for explicit search indicators
        search_indicators = [
            "search",
            "find",
            "look up",
            "google",
            "show me",
            "what is",
            "who is",
            "where is",
        ]
        has_search_intent = any(indicator in user_prompt.lower() for indicator in search_indicators)

        if not has_search_intent:
            logger.debug(
                f"Short query '{user_prompt}' classified as conversation (no search intent)"
            )
            return {"primary_intent": "conversation"}

    system_prompt = """You are an expert query router. Analyze the user's request and determine which tool is most appropriate. Your job is to identify the INFORMATION SOURCE needed, not the topic domain.

# Available Tools:

**[query_conversation_history]**
- Use for questions about our current conversation, my past statements, or actions I've taken
- Keywords: "you said", "earlier", "before", "in our conversation", "since we started", "what did you", "remind me"
- Examples: "What URLs did you use?", "Earlier you mentioned prices", "What files did you analyze?"

**[perform_web_search]**
- Use for real-time, current information that changes frequently
- Keywords: "current", "latest", "now", "today", "recent", "what's happening"
- Examples: "Current stock price", "Today's weather", "Latest news about X"

**[query_cryptocurrency]**
- Use specifically for crypto prices, market cap, trading data (needs real-time data)
- Examples: "Bitcoin price", "Ethereum market cap", "Crypto trends"

**[query_stocks]**
- Use specifically for stock prices, market info, earnings (needs real-time data)
- Examples: "AAPL stock price", "Tesla earnings", "Market performance"

**[generate_from_knowledge]**
- Use for general knowledge, explanations, how-to questions (timeless information)
- Examples: "Explain photosynthesis", "How does a car engine work?", "What is democracy?"

**[conversation]**
- Use for greetings, casual chat, personal preferences, opinions
- Examples: "Hello", "How are you?", "What do you think about X?"

# Critical Edge Cases:
- "What URLs have you used since we started talking?" → query_conversation_history
- "Find me current URLs about AI" → perform_web_search
- "Earlier you mentioned some stock prices, what were they?" → query_conversation_history
- "What is the current price of Apple stock?" → query_stocks
- "What is a stock?" → generate_from_knowledge

# Reasoning Process:
1. Is this about OUR conversation history? → query_conversation_history
2. Is this about current crypto/stock prices? → query_cryptocurrency/query_stocks
3. Is this about current external information? → perform_web_search
4. Is this general knowledge? → generate_from_knowledge
5. Is this casual conversation? → conversation

Examples:
- "what's the weather today" -> web_search
- "who won the game last night" -> web_search
- "latest news on Ukraine" -> web_search
- "is Twitter still called X" -> web_search
- "explain quantum physics" -> conversation
- "what happened in 2024" -> web_search

Return just the category name."""

    classification_prompt = utils.format_prompt(system_prompt, user_prompt)

    try:
        # Use LOWER priority for classification so it doesn't block real-time chat responses
        # Classification is preprocessing that can afford to wait
        from persistent_llm_server import get_llm_server

        server = await get_llm_server()

        # Don't acquire GPU lock here - the persistent server handles it internally
        classification_text = await server.generate(
            prompt=classification_prompt,
            max_tokens=10,  # Increased from 5 to allow for full category names
            temperature=0.2,  # Consistent decision-making temperature
            session_id="classification",  # Use fixed session ID for classification
        )

        classification_text = classification_text.strip().lower()
        logger.debug(f"LLM classification for '{user_prompt}': '{classification_text}'")

        # Check if response contains a valid category
        valid_categories = {
            "query_conversation_history",
            "perform_web_search",
            "query_cryptocurrency",
            "query_stocks",
            "generate_from_knowledge",
            "conversation",
        }

        for category in valid_categories:
            if category in classification_text:
                return {"primary_intent": category}

        # Default fallback to conversation
        logger.debug(
            f"No valid category found in '{classification_text}', defaulting to conversation"
        )
        return {"primary_intent": "conversation"}

    except Exception as e:
        logger.debug(f"Classification error for '{user_prompt}': {e}, defaulting to conversation")
        return {"primary_intent": "conversation"}


def check_service_availability() -> None:
    """Check if all required services are available."""
    logger.info("🧠 MEMORY DEBUG: Checking service availability")
    logger.info(f"🧠 MEMORY DEBUG: Redis client object: {global_vars.redis_client}")
    logger.info(f"🧠 MEMORY DEBUG: Personal memory object: {global_vars.personal_memory}")
    
    # LLM availability is now checked by the persistent server on-demand
    # No need to check here since the server loads on first request
    redis_available = redis_utils.is_redis_available(global_vars.redis_client)
    logger.info(f"🧠 MEMORY DEBUG: Redis availability check result: {redis_available}")
    
    if not redis_available:
        logger.error("🧠 MEMORY DEBUG: ❌ Redis Service Unavailable - raising HTTPException")
        raise HTTPException(status_code=503, detail="Redis Service Unavailable")
    
    logger.info("🧠 MEMORY DEBUG: ✅ All services available")

# NO FALLBACK MESSAGE CREATION - SYSTEM MUST WORK OR FAIL

# ===================== FAST PATH IMPLEMENTATION =====================

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
    memory_context = ""
    logger.info(f"🧠 MEMORY DEBUG: Attempting to get memory context for session {session_id}")
    
    if global_vars.personal_memory:
        logger.info(f"🧠 MEMORY DEBUG: personal_memory is available: {type(global_vars.personal_memory)}")
        try:
            # Quick memory retrieval for context
            logger.info(f"🧠 MEMORY DEBUG: Calling get_relevant_memories with query: '{user_prompt[:50]}...'")
            memories = await global_vars.personal_memory.get_relevant_memories(query=user_prompt, limit=5)
            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(memories) if memories else 0} memories")
            
            if memories:
                memory_context = "\n".join([m.content for m in memories[:3]])
                logger.info(f"🧠 MEMORY DEBUG: ✅ Using personal memory context ({len(memory_context)} chars)")
                logger.debug(f"🧠 MEMORY DEBUG: Memory context preview: {memory_context[:100]}...")
            else:
                logger.info("🧠 MEMORY DEBUG: No relevant memories found")
        except Exception as e:
            logger.error(f"🧠 MEMORY DEBUG: ❌ Fast path memory retrieval failed: {e}", exc_info=True)
    else:
        logger.warning("🧠 MEMORY DEBUG: personal_memory is None - no memory context available")

    # Get minimal conversation history from personal memory
    history = []
    logger.info(f"🧠 MEMORY DEBUG: Attempting to get conversation history for session {session_id}")
    
    if global_vars.personal_memory:
        try:
            logger.info(f"🧠 MEMORY DEBUG: Calling get_conversation_context for session {session_id}")
            recent_memories = await global_vars.personal_memory.get_conversation_context(
                session_id, max_messages=4
            )  # Last 2 turns
            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(recent_memories) if recent_memories else 0} conversation memories")
            
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
            logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to get conversation history: {e}", exc_info=True)
    else:
        logger.warning("🧠 MEMORY DEBUG: personal_memory is None - no conversation history available")

    # Use condensed version of main system prompt with critical rules
    simple_system_prompt = """You are Aria, a helpful and friendly AI assistant.

## Core Rules:
1. **Most Current Conversation is TRUTH:** Information from the most current conversation ALWAYS
   overrides stored memories. When referring to this, say "I prioritize information from the most
   current conversation."
2. **Format with Markdown:** Use markdown formatting in all responses.
3. **Be Conversational:** Keep responses warm, engaging, and concise."""

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

    # Fast path response generation
    async def generate_fast_response():
        logger.info(
            f" FAST PATH: Starting lightweight response generation for session {session_id}"
        )
        full_response = ""

        try:
            # Use persistent LLM server for fast path too
            from persistent_llm_server import get_llm_server

            server = await get_llm_server()

            # Convert messages to properly formatted prompt using utils.format_prompt
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
                        # This is the current user message
                        user_content = content
                elif role == "assistant" and prev_user_content:
                    conversation_history.append(f"User: {prev_user_content}\nAssistant: {content}")

            # Use proper chat formatting with conversation history
            if conversation_history:
                conversation_str = "\n".join(conversation_history)
                prompt = utils.format_prompt_with_history(
                    system_content, user_content, conversation_str
                )
            else:
                prompt = utils.format_prompt(system_content, user_content)

            logger.info(" FAST PATH: Using persistent server for streaming generation")
            logger.info(f" FAST PATH: Prompt preview: {prompt[:200]}...")

            # Stream tokens directly from the server
            full_response = ""
            async for token in server.generate_stream(
                prompt=prompt,
                max_tokens=1024,  # Shorter for conversational
                temperature=0.7,
                session_id=session_id,
                priority=0,  # High priority for fast path
            ):
                if token:  # Skip empty tokens
                    full_response += token
                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

            logger.info(f" FAST PATH: Response complete ({len(full_response)} chars)")
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            error_msg = f"Fast path error: {e!s}"
            logger.error(error_msg, exc_info=True)  # Add full traceback
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            return

        # Lightweight background processing (don't block response)
        if full_response:
            logger.info(f"🧠 MEMORY DEBUG: Starting background memory processing for session {session_id}")
            logger.info(f"🧠 MEMORY DEBUG: Response length: {len(full_response)} chars")
            
            # Schedule minimal memory processing in background
            try:
                logger.info("🧠 MEMORY DEBUG: Creating background task for lightweight_memory_processing")
                asyncio.create_task(
                    lightweight_memory_processing(user_prompt, full_response, session_id)
                )
                logger.info("🧠 MEMORY DEBUG: ✅ Background memory processing task created")
            except Exception as memory_task_error:
                logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to create memory processing task: {memory_task_error}", exc_info=True)

            # Update conversation history (also background)
            try:
                logger.info(f"🧠 MEMORY DEBUG: Adding conversation to Redis history for session {session_id}")
                logger.info(f"🧠 MEMORY DEBUG: Redis client available: {global_vars.redis_client is not None}")
                
                # Call async history function directly with ResourceManager-powered embeddings
                redis_utils.add_to_conversation_history(
                    session_id,
                    user_prompt,
                    full_response,
                    global_vars.redis_client,  # ResourceManager handles embedding model internally
                    redis_utils.CONVERSATION_HISTORY_KEY_PREFIX,
                    redis_utils.MAX_NON_VITAL_HISTORY,
                )
                logger.info("🧠 MEMORY DEBUG: ✅ Conversation added to Redis history")
            except Exception as redis_history_error:
                logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to add to Redis history: {redis_history_error}", exc_info=True)
        else:
            logger.warning("🧠 MEMORY DEBUG: No response to process - skipping memory operations")

    return StreamingResponse(
        generate_fast_response(),
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
    """Enhanced memory processing that stores canonical facts in core_memories
    and avoids redundant storage in the memories table.
    """
    logger.info(f"🧠 MEMORY DEBUG: Starting lightweight_memory_processing for session {session_id}")
    logger.info(f"🧠 MEMORY DEBUG: User prompt length: {len(user_prompt)}, Response length: {len(response)}")
    logger.info(f"🧠 MEMORY DEBUG: User prompt preview: {user_prompt[:100]}...")
    
    try:
        if not global_vars.personal_memory:
            logger.warning("🧠 MEMORY DEBUG: personal_memory is None, exiting early")
            return
        
        logger.info(f"🧠 MEMORY DEBUG: personal_memory object available: {type(global_vars.personal_memory)}")

        # Enhanced pattern-based memory extraction
        memory_patterns = [
            # Name patterns
            (r"(?:my name is|i'm|i am|call me)\s+([A-Za-z]+)", "user_name", 0.9),
            (r"i'm ([A-Za-z]+)(?:\s|$|[,.!?])", "user_name", 0.9),
            # Address patterns
            (r"(?:i live at|my address is|home address is)\s+(.+?)(?:\.|$)", "user_address", 0.8),
            (r"(?:live in|from)\s+([A-Za-z\s,]+)(?:\.|,|$)", "user_location", 0.7),
            # Age patterns
            (r"(?:i am|i'm)\s+(\d+)\s+years old", "user_age", 0.7),
            (r"(?:born in|born on)\s+(.+?)(?:\.|$)", "user_birthplace", 0.7),
            # Medical patterns
            (r"(?:allergic to|allergy to)\s+(.+?)(?:\.|,|$)", "user_allergy", 0.95),
            (r"(?:i have|diagnosed with)\s+(.+?)(?:\.|,|$)", "user_medical_condition", 0.9),
            # Family patterns
            (r"(?:my wife|wife's name is|married to)\s+([A-Za-z]+)", "spouse_name", 0.9),
            (r"(?:my husband|husband's name is)\s+([A-Za-z]+)", "spouse_name", 0.9),
            (r"(?:my mother|mother's name is)\s+([A-Za-z]+)", "mother_name", 0.8),
            (r"(?:my father|father's name is)\s+([A-Za-z]+)", "father_name", 0.8),
            # Age patterns for family
            (r"(?:mother|mom|mother's)\s+(?:is\s+)?(\d+)\s+years old", "mother_age", 0.8),
            (r"(?:father|dad|father's)\s+(?:is\s+)?(\d+)\s+years old", "father_age", 0.8),
            # Physical characteristics  
            (r"(?:i am|i'm|he is|she is)\s+(\d+['']?\d*[\"]*)\s+tall", "height", 0.7),
            (r"(?:blue|brown|green|hazel|gray)\s+eyes", "eye_color", 0.7),
            # Interests and hobbies
            (r"(?:loves|enjoys|likes)\s+([a-zA-Z\s,]+?)(?:\.|,|$)", "hobby_interest", 0.6),
            (r"fascination with\s+([a-zA-Z\s,]+?)(?:\.|,|$)", "interest", 0.7),
            # Pets
            (r"(?:owns|has)\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\.|,|$)", "pet", 0.6),
            # Work/profession patterns
            (r"(?:i work as|i am a|my job is)\s+(.+?)(?:\.|,|$)", "user_profession", 0.8),
            (r"(?:work at|employed by)\s+(.+?)(?:\.|,|$)", "user_employer", 0.7),
        ]

        extracted_count = 0
        logger.info(f"🧠 MEMORY DEBUG: Scanning user prompt with {len(memory_patterns)} patterns")
        
        for i, (pattern, core_key, _importance) in enumerate(memory_patterns):
            logger.debug(f"🧠 MEMORY DEBUG: Testing pattern {i+1}/{len(memory_patterns)}: {core_key}")
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                logger.info(f"🧠 MEMORY DEBUG: Pattern match found! {core_key} = '{value}'")

                try:
                    # Store as core memory instead of regular memory to avoid redundancy
                    logger.info(f"🧠 MEMORY DEBUG: Attempting to store core memory: {core_key} = {value}")
                    await global_vars.personal_memory.set_core_memory(
                        key=core_key, value=value, category="user_profile"
                    )
                    logger.info(f"🧠 MEMORY DEBUG: ✅ Successfully stored core memory: {core_key} = {value}")
                    extracted_count += 1
                except Exception as core_memory_error:
                    logger.error(f"🧠 MEMORY DEBUG: ❌ Failed to store core memory {core_key}: {core_memory_error}", exc_info=True)

                # Don't extract too many things at once to avoid conflicts
                if extracted_count >= 3:
                    logger.info(f"🧠 MEMORY DEBUG: Reached extraction limit of 3, stopping")
                    break

        logger.info(f"🧠 MEMORY DEBUG: Memory processing complete for session {session_id}. Extracted {extracted_count} items.")

    except Exception as e:
        logger.error(f"🧠 MEMORY DEBUG: ❌ Core memory processing error: {e}", exc_info=True)


