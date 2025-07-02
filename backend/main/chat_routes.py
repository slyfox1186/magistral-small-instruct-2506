#!/usr/bin/env python3
"""Main chat streaming endpoint and related routes."""

import asyncio
import json
import logging
import re
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# Import models and globals
from .models import ChatStreamRequest
from .globals import (
    last_chat_activity,
    request_total,
    personal_memory,
    redis_client,
)
from .chat_helpers import (
    classify_query_fast_pattern_based,
    classify_query_with_llm,
    handle_simple_conversational_request,
    check_service_availability,
)

import redis_utils
import utils

logger = logging.getLogger(__name__)


def update_chat_activity():
    """Update global chat activity timestamp."""
    global last_chat_activity
    last_chat_activity = time.time()


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
        if request_total:
            request_total.labels(endpoint=endpoint, status="started").inc()

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

        #  ULTRA-HIGH ROI OPTIMIZATION: LIGHTWEIGHT FAST PATH FOR SIMPLE REQUESTS
        # Detect simple conversational requests that don't need heavy processing
        def is_simple_conversational_request(prompt: str) -> bool:
            """Fast pattern-based detection of simple conversational requests.
            Returns True if this is a simple request that can use the fast path.
            """
            prompt_lower = prompt.lower().strip()

            # Very short requests (1-3 words) are usually conversational
            word_count = len(prompt_lower.split())
            if word_count <= 3:
                # Check for obvious non-conversational short requests
                non_conversational_short = [
                    "search",
                    "find",
                    "google",
                    "lookup",
                    "directions",
                    "route",
                    "weather",
                    "stock",
                    "crypto",
                    "bitcoin",
                    "price",
                    "market",
                ]
                if not any(term in prompt_lower for term in non_conversational_short):
                    return True

            # Common conversational patterns
            simple_patterns = [
                r"^(hi|hello|hey|good morning|good afternoon|good evening)$",
                r"^(how are you|how\'s it going|what\'s up|wassup)[\?]*$",
                r"^(thanks|thank you|ty|thx)[\!]*$",
                r"^(bye|goodbye|see you|talk to you later|ttyl)[\!]*$",
                r"^(yes|no|yeah|nah|yep|nope|ok|okay|sure)[\!]*$",
                r"^(lol|haha|hehe|wow|cool|nice|awesome|great)[\!]*$",
                r"^(i see|i understand|got it|makes sense)[\!]*$",
            ]

            for pattern in simple_patterns:
                if re.match(pattern, prompt_lower):
                    return True

            # Longer conversational requests (but still simple)
            if word_count <= 8:
                conversational_indicators = [
                    "how are you",
                    "what are you",
                    "tell me about yourself",
                    "what can you do",
                    "who are you",
                    "nice to meet",
                    "how was your day",
                    "what do you think",
                    "do you like",
                    "have you ever",
                    "i feel",
                    "i think",
                    "i like",
                    "i love",
                ]
                if any(indicator in prompt_lower for indicator in conversational_indicators):
                    return True

            # Personal information sharing patterns (these should use fast path for memory extraction)
            personal_info_patterns = [
                "my name is", "i'm ", "i am ", "call me",
                "my wife", "my husband", "my mother", "my father", 
                "my age", "years old", "born in", "born on",
                "i live", "my address", "allergic to", "allergy to",
                "i work", "my job", "employed by", "profession",
                "i love", "i enjoy", "hobby", "hobbies",
                "things about", "about me", "personal info"
            ]
            
            if any(pattern in prompt_lower for pattern in personal_info_patterns):
                logger.info(f"🧠 MEMORY DEBUG: Detected personal info sharing - routing to fast path: '{prompt[:50]}...'")
                return True

            return False

        # Check if this is a simple conversational request that can use fast path
        logger.info(f"🧠 MEMORY DEBUG: Checking if request is simple conversational for user prompt: '{user_prompt[:50]}...'")
        is_simple = is_simple_conversational_request(user_prompt)
        logger.info(f"🧠 MEMORY DEBUG: Simple conversational check result: {is_simple}")
        
        if is_simple:
            logger.info("🚀 FAST PATH: Simple conversational request detected")
            logger.info("🧠 MEMORY DEBUG: Entering handle_simple_conversational_request")
            return await handle_simple_conversational_request(request, user_prompt, session_id)

        # For complex requests, use the full pipeline
        logger.info("🔄 FULL PIPELINE: Complex request detected")

        # Step 1: Classify query intent
        classification = classify_query_fast_pattern_based(user_prompt)
        
        if classification is None:
            # Need LLM classification
            classification = await classify_query_with_llm(user_prompt)

        primary_intent = classification.get("primary_intent", "conversation")
        logger.debug(f"Query classified as: {primary_intent}")

        # Step 2: Handle based on intent
        async def generate_response():
            logger.info(f"🧠 MEMORY DEBUG: Starting generate_response for intent: {primary_intent}")
            try:
                if primary_intent == "conversation":
                    logger.info("🧠 MEMORY DEBUG: Processing conversation intent - delegating to handle_simple_conversational_request")
                    # Use the working fast path implementation for conversation
                    # This handles memory retrieval, LLM generation, and memory storage properly
                    
                    logger.info("🧠 MEMORY DEBUG: Calling handle_simple_conversational_request for full pipeline conversation")
                    
                    # Use the existing handle_simple_conversational_request which has full implementation
                    # This properly handles memory extraction, LLM generation, and background processing
                    async for chunk in handle_simple_conversational_request(request, user_prompt, session_id).body_iterator:
                        yield chunk
                    return
                else:
                    # Handle other intent types
                    logger.info(f"🧠 MEMORY DEBUG: Processing non-conversation intent: {primary_intent}")
                    yield f"data: {json.dumps({'token': {'text': f'Processing {primary_intent} request...'}})}\n\n"
                    # Add minimal response completion for non-conversation intents
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