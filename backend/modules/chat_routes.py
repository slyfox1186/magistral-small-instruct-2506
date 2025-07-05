#!/usr/bin/env python3
"""Main chat streaming endpoint and related routes."""

import asyncio
import json
import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# Import models and globals
from .models import ChatStreamRequest
from .globals import (
    app_state,
    bg_state,
)
from .chat_helpers import (
    classify_query_fast_pattern_based,
    classify_query_with_llm,
    handle_simple_conversational_request,
    check_service_availability,
    lightweight_memory_processing,
)
from memory.importance_scorer import get_importance_scorer

import redis_utils
import utils

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
            logger.info(f"🧠 MEMORY DEBUG: Starting generate_response for intent: {primary_intent}")
            try:
                if primary_intent == "conversation":
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
                            logger.info(f"🧠 MEMORY DEBUG: Getting core memories...")
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
                    simple_system_prompt = """You are Aria, a helpful AI assistant. Be natural and conversational."""

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
                elif primary_intent == "generate_from_knowledge":
                    # Handle knowledge generation using LLM's internal training data
                    logger.info(f"🧠 INTERNAL KNOWLEDGE: Processing generate_from_knowledge intent")
                    
                    try:
                        # Use LLM's internal knowledge directly - NO web search
                        from persistent_llm_server import get_llm_server
                        llm_server = await get_llm_server()
                        
                        # System prompt for internal knowledge generation
                        system_prompt = """You are a helpful AI assistant with broad knowledge. Provide comprehensive, accurate responses using your training data.

## Core Rules:
1. Use proper markdown formatting: headers, lists, bold text, etc.
2. Be comprehensive: Provide detailed, useful information
3. Be accurate: Only provide information you're confident about
4. Structure clearly: Use headers, lists, and formatting for readability
5. Use standard markdown links: [Description](URL) format
6. For code generation: Provide complete, working examples with explanations"""
                        
                        formatted_prompt = utils.format_prompt(system_prompt, user_prompt)
                        
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
                        
                        logger.info(f"🧠 INTERNAL KNOWLEDGE: Response complete ({len(full_response)} chars)")
                        
                        # Store in memory for future reference
                        if app_state.personal_memory and full_response:
                            try:
                                logger.info(f"🧠 MEMORY STORAGE: Storing internal knowledge response in memory for session {session_id}")
                                
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
                                logger.info(f"🧠 MEMORY STORAGE: ✅ Stored USER prompt with importance={user_importance:.3f}")
                                
                                # Calculate dynamic importance for assistant response
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
                                logger.info(f"🧠 MEMORY STORAGE: ✅ Stored ASSISTANT response with importance={assistant_importance:.3f}")
                            except Exception as memory_error:
                                logger.error(f"🧠 MEMORY STORAGE: ❌ Failed to store in memory: {memory_error}", exc_info=True)
                    
                    except Exception as knowledge_error:
                        error_msg = f"Error in knowledge generation: {knowledge_error}"
                        logger.error(f"🧠 INTERNAL KNOWLEDGE: ❌ {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    
                elif primary_intent == "perform_web_search":
                    # Handle web search requests with LLM synthesis 
                    logger.info(f"🧠 MEMORY DEBUG: Processing perform_web_search intent")
                    
                    yield f"data: {json.dumps({'token': {'text': '🔍 Searching the web...'}})}\n\n"
                    
                    try:
                        from web_scraper import perform_web_search_async
                        
                        search_results = await perform_web_search_async(
                            query=user_prompt,
                            num_results=8
                        )
                        
                        if search_results:
                            yield f"data: {json.dumps({'token': {'text': '\n\n📊 Found information, generating comprehensive response...\n\n'}})}\n\n"
                            
                            # Use persistent LLM server to generate response with search results
                            from persistent_llm_server import get_llm_server
                            llm_server = await get_llm_server()
                            
                            # Create system prompt for knowledge synthesis
                            system_prompt = """You are a helpful assistant. Use the search results to answer the user's question comprehensively."""

                            user_content = f"User Query: {user_prompt}\n\nSearch Results:\n{search_results}\n\nPlease provide a comprehensive response to the user's query using the search results above."
                            
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
                                    logger.info(f"🧠 MEMORY STORAGE: ✅ Successfully stored web search memories")
                                except Exception as memory_error:
                                    logger.error(f"🧠 MEMORY STORAGE: ❌ Failed to store in memory: {memory_error}", exc_info=True)
                        else:
                            yield f"data: {json.dumps({'token': {'text': '\n\n❌ No search results found.\n\n'}})}\n\n"
                    
                    except Exception as search_error:
                        error_msg = f"Search error: {search_error}"
                        logger.error(f"🧠 MEMORY DEBUG: ❌ {error_msg}", exc_info=True)
                        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    
                elif primary_intent == "query_conversation_history":
                    # Handle memory retrieval requests
                    logger.info(f"🧠 MEMORY DEBUG: Processing query_conversation_history intent")
                    
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
                                    logger.info(f"🧠 MEMORY RETRIEVAL: === DETAILED MEMORY CONTENTS ===")
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
                                        logger.info(f"🧠 MEMORY RETRIEVAL:   ---")
                                    
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
                            
                            system_prompt = """You are a helpful AI assistant retrieving information from conversation memory.
                            
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
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    
                elif primary_intent == "query_stocks":
                    # Handle stock market queries
                    logger.info(f"📈 STOCKS: Processing stock query: '{user_prompt}'")
                    
                    yield f"data: {json.dumps({'token': {'text': '📈 Fetching stock data...'}})}\n\n"
                    
                    try:
                        # Let the LLM handle the complexity of extracting stock symbols
                        from persistent_llm_server import get_llm_server
                        llm_server = await get_llm_server()
                        
                        # Use proper system/user prompt separation
                        system_prompt = "Extract stock ticker symbols from the user's query. Return only the ticker symbols, nothing else."
                        formatted_prompt = utils.format_prompt(system_prompt, user_prompt)
                        
                        symbol_response = await llm_server.generate(
                            prompt=formatted_prompt,
                            max_tokens=50,
                            temperature=0.1,
                            session_id=f"{session_id}_symbol_extraction"
                        )
                        
                        logger.info(f"📈 STOCKS: LLM extracted symbols: '{symbol_response.strip()}'")
                        
                        # Parse the response to get ticker symbols
                        import re
                        tickers = []
                        if symbol_response.strip():
                            # Split by commas or spaces and clean up
                            raw_tickers = re.split(r'[,\s]+', symbol_response.strip().upper())
                            potential_tickers = [t.strip() for t in raw_tickers if t.strip() and 1 <= len(t.strip()) <= 5]
                            
                            logger.info(f"📈 STOCKS: Potential tickers to validate: {potential_tickers}")
                            
                            # Validate each ticker with yfinance
                            if app_state.stock_searcher and potential_tickers:
                                valid_results = app_state.stock_searcher.validate_symbols(potential_tickers)
                                tickers = [t for t, is_valid in valid_results.items() if is_valid]
                                logger.info(f"📈 STOCKS: Validated tickers: {tickers}")
                        
                        # If no tickers found, try to be helpful
                        if not tickers:
                            yield f"data: {json.dumps({'token': {'text': '\n\n❓ No stock symbols detected in your query. Please specify stock ticker symbols (e.g., AAPL, MSFT, GOOGL) or company names.\n\n'}})}\n\n"
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            return
                        
                        # Get stock data
                        if app_state.stock_searcher:
                            yield f"data: {json.dumps({'token': {'text': f' Getting quotes for {", ".join(tickers)}...\n\n'}})}\n\n"
                            
                            formatted_data, sources = await asyncio.to_thread(
                                app_state.stock_searcher.format_stock_data_with_sources, tickers
                            )
                            
                            if formatted_data and "No stock data available" not in formatted_data:
                                # Use LLM to create a natural response with the data
                                from persistent_llm_server import get_llm_server
                                llm_server = await get_llm_server()
                                
                                system_prompt = """You are a knowledgeable financial assistant with access to real-time stock market data."""
                                
                                user_content = f"User Query: {user_prompt}\n\nStock Market Data:\n{formatted_data}\n\nRespond naturally to the user's query using this data."
                                
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
                                
                                # Add sources if available
                                if sources:
                                    sources_text = "\n\n📊 **Data Sources:**\n"
                                    for source in sources[:3]:  # Limit to 3 sources
                                        sources_text += f"- [{source['name']}]({source['url']})\n"
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
                                        logger.info(f"📈 STOCKS: Stored stock query conversation in memory")
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
                    
                else:
                    # Handle other intent types with placeholder
                    logger.info(f"🧠 MEMORY DEBUG: Processing unimplemented intent: {primary_intent}")
                    yield f"data: {json.dumps({'token': {'text': f'🚧 Handler for {primary_intent} not yet implemented. This would typically involve specialized processing.'}})}\n\n"
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
