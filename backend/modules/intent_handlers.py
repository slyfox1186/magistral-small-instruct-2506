#!/usr/bin/env python3
"""Intent handlers for different types of chat requests."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from fnmatch import fnmatch

import utils

from .chat_helpers import lightweight_memory_processing
from .globals import app_state

logger = logging.getLogger(__name__)

# Load URL exclusion config once at module level
try:
    _config_path = os.path.join(os.path.dirname(__file__), '..', 'url_exclude_list.json')
    with open(_config_path, 'r') as f:
        _url_config = json.load(f)
    BLOCKED_GLOBS = _url_config.get('blocked_globs', [])
    BLOCKED_PATTERNS = _url_config.get('blocked_patterns', [])
    logger.info(f"Loaded {len(BLOCKED_GLOBS)} URL globs and {len(BLOCKED_PATTERNS)} patterns from config")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load URL exclude list: {e}")
    logger.error(f"Config file should be at: {_config_path}")
    raise RuntimeError("URL exclusion config required but not found")

# Track failed URLs during scraping sessions for learning
_FAILED_URLS = []

# Track session state for web search/scraping decisions
# Key: session_id, Value: {'web_search_performed': bool, 'last_search_timestamp': float}
_SESSION_WEB_STATE = {}


def is_blocked_url(url: str) -> bool:
    """Efficient glob and pattern matching for URL blocking."""
    url_lower = url.lower()
    return (any(fnmatch(url_lower, glob.lower()) for glob in BLOCKED_GLOBS) or
            any(pattern in url_lower for pattern in BLOCKED_PATTERNS))


def track_failed_url(url: str, error_reason: str):
    """Track a URL that failed to scrape for learning purposes."""
    _FAILED_URLS.append({
        'url': url,
        'error': error_reason,
        'timestamp_ms': int(time.time() * 1000)  # Millisecond precision for ordering
    })
    logger.info(f"📝 LEARNING: Tracked failed URL: {url} - {error_reason}")


def mark_web_search_performed(session_id: str):
    """Mark that a web search has been performed for this session."""
    _SESSION_WEB_STATE[session_id] = {
        'web_search_performed': True,
        'last_search_timestamp': time.time()
    }
    logger.info(f"🔍 STATE: Marked web search performed for session {session_id}")
    
    # Cleanup old sessions to prevent memory leaks (keep only last 1000 sessions)
    if len(_SESSION_WEB_STATE) > 1000:
        # Remove oldest sessions based on timestamp
        current_time = time.time()
        sessions_to_remove = []
        for sid, state in _SESSION_WEB_STATE.items():
            timestamp = state.get('last_search_timestamp', 0)
            if not isinstance(timestamp, (int, float)) or current_time - timestamp > 7200:  # 2 hours
                sessions_to_remove.append(sid)
        
        for sid in sessions_to_remove:
            _SESSION_WEB_STATE.pop(sid, None)
        
        if sessions_to_remove:
            logger.info(f"🔍 STATE: Cleaned up {len(sessions_to_remove)} old session states")


def check_web_search_performed(session_id: str) -> bool:
    """Check if a web search has been performed for this session recently."""
    try:
        session_state = _SESSION_WEB_STATE.get(session_id, {})
        performed = session_state.get('web_search_performed', False)
        
        # Clear state if it's too old (more than 1 hour)
        if performed:
            last_search = session_state.get('last_search_timestamp', 0)
            # Handle corrupted timestamp data
            if not isinstance(last_search, (int, float)):
                logger.warning(f"🔍 STATE: Invalid timestamp for session {session_id}, clearing state")
                _SESSION_WEB_STATE.pop(session_id, None)
                return False
                
            if time.time() - last_search > 3600:  # 1 hour
                logger.info(f"🔍 STATE: Clearing old web search state for session {session_id}")
                _SESSION_WEB_STATE.pop(session_id, None)
                return False
        
        logger.info(f"🔍 STATE: Web search performed for session {session_id}: {performed}")
        return performed
    except Exception as e:
        logger.warning(f"🔍 STATE: Error checking web search state for session {session_id}: {e}")
        return False


def clear_web_search_state(session_id: str):
    """Clear web search state after scraping is completed."""
    if session_id in _SESSION_WEB_STATE:
        _SESSION_WEB_STATE.pop(session_id)
        logger.info(f"🔍 STATE: Cleared web search state for session {session_id}")


def reset_web_search_state(session_id: str):
    """Reset web search state to allow new search cycles."""
    clear_web_search_state(session_id)
    logger.info(f"🔍 STATE: Reset web search state for session {session_id}")


async def _update_block_list_from_failures():
    """Use LLM to analyze failed URLs and update the block list intelligently."""
    global BLOCKED_GLOBS, BLOCKED_PATTERNS
    
    if not _FAILED_URLS:
        return
    
    logger.info(f"🧠 LEARNING: Analyzing {len(_FAILED_URLS)} failed URLs for block list updates")
    
    try:
        from persistent_llm_server import get_llm_server
        
        # Step 1: Load and provide COMPLETE existing block configuration
        with open(_config_path, 'r') as f:
            complete_config = json.load(f)
        
        # Step 2: Analyze failure patterns and prepare intelligent summary
        from urllib.parse import urlparse
        from collections import defaultdict
        
        # Group failures by domain to identify patterns
        domain_failures = defaultdict(list)
        for fail in _FAILED_URLS:
            try:
                domain = urlparse(fail['url']).netloc
                domain_failures[domain].append(fail)
            except Exception:
                # Handle malformed URLs
                domain_failures['unknown'].append(fail)
        
        # Build intelligent failure summary with pattern analysis
        failure_parts = []
        for domain, failures in domain_failures.items():
            failure_count = len(failures)
            error_types = [f['error'] for f in failures]
            
            if failure_count > 1:
                failure_parts.append(f"- DOMAIN: {domain} ({failure_count} failures)")
                failure_parts.append(f"  Errors: {', '.join(error_types[:3])}{'...' if len(error_types) > 3 else ''}")
            else:
                failure_parts.append(f"- SINGLE FAILURE: {failures[0]['url']} (Error: {failures[0]['error']})")
        
        failure_summary = "\n".join(failure_parts)
        
        # Step 3: Comprehensive analysis with full context
        analysis_prompt = f"""COMPREHENSIVE URL BLOCKING ANALYSIS

You must analyze the COMPLETE context before making any decisions:

STEP 1 - EXISTING BLOCK CONFIGURATION (FULL FILE):
{json.dumps(complete_config, indent=2)}

STEP 2 - INTELLIGENT FAILURE PATTERN ANALYSIS:
{failure_summary}

PATTERN RECOGNITION NOTES:
- Domains with multiple failures may indicate systematic issues
- Single failures are likely temporary and should NOT be blocked
- Consider error types: auth errors vs 404s vs server errors

STEP 3 - INTELLIGENT DECISION MAKING:
Using the COMPLETE existing block configuration AND the recent scraping results:

1. **DO NOT BLOCK** for temporary errors:
   - HTTP 404 (page not found) - content may have moved
   - HTTP 500/502/503 (server errors) - temporary issues
   - Timeout errors - network/performance issues
   - Single occurrence failures

2. **CONSIDER BLOCKING** only for systematic issues:
   - Authentication/login required (401/403 + auth content)
   - Paywall/subscription content
   - Multiple failures from same domain over time
   - Content explicitly stating access restrictions

3. **ANALYSIS REQUIREMENTS**:
   - Check if failed URLs are already covered by existing patterns
   - Only suggest new blocks for persistent systematic issues
   - Avoid blocking entire domains for isolated failures

4. **REMOVAL AUTHORITY**:
   You have FULL PERMISSION to remove incorrectly blocked patterns:
   - Remove blocks that were added for temporary issues (404s, server errors)
   - Remove overly broad blocks that prevent legitimate scraping
   - Remove outdated blocks that no longer apply
   - Clean up duplicate or redundant patterns

CRITICAL: Be conservative with blocking. Only block when there's clear evidence of systematic access restrictions, not temporary failures.

Return ONLY a JSON object:
{{
    "analysis_summary": "Brief analysis of the complete context",
    "existing_coverage": "What current patterns already cover",
    "new_globs": ["*.example.com*"],
    "new_patterns": ["/login*", "/paywall*"],
    "remove_globs": ["*.unnecessarily-blocked.com*"],
    "remove_patterns": ["/outdated-pattern*"],
    "reasoning": "Detailed explanation of changes needed"
}}

If no changes are needed, return: {{"analysis_summary": "...", "existing_coverage": "...", "new_globs": [], "new_patterns": [], "remove_globs": [], "remove_patterns": [], "reasoning": "No changes needed - current configuration is optimal"}}"""

        llm_server = await get_llm_server()
        response = await llm_server.generate(
            prompt=utils.format_prompt("You are a web scraping optimization assistant. Analyze failed URLs to improve blocking patterns.", analysis_prompt),
            max_tokens=25000,  # Full learning analysis capability
            temperature=0.2,
            session_id="url_learning",
        )
        
        logger.info(f"🧠 LEARNING: LLM analysis response: {response}")
        
        # Parse LLM comprehensive analysis
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1:
            analysis = json.loads(response[start_idx:end_idx + 1])
            
            analysis_summary = analysis.get('analysis_summary', 'No analysis provided')
            existing_coverage = analysis.get('existing_coverage', 'No coverage analysis')
            new_globs = analysis.get('new_globs', [])
            new_patterns = analysis.get('new_patterns', [])
            remove_globs = analysis.get('remove_globs', [])
            remove_patterns = analysis.get('remove_patterns', [])
            reasoning = analysis.get('reasoning', 'No reasoning provided')
            
            logger.info(f"🧠 LEARNING: Analysis Summary: {analysis_summary}")
            logger.info(f"🧠 LEARNING: Existing Coverage: {existing_coverage}")
            
            if new_globs or new_patterns or remove_globs or remove_patterns:
                # Update the config file with comprehensive data
                current_time_ms = int(time.time() * 1000)
                current_datetime = datetime.now().isoformat()
                
                config_data = complete_config.copy()
                
                # Add new patterns
                config_data['blocked_globs'].extend(new_globs)
                config_data['blocked_patterns'].extend(new_patterns)
                
                # Remove patterns as suggested by LLM
                for pattern in remove_globs:
                    if pattern in config_data['blocked_globs']:
                        config_data['blocked_globs'].remove(pattern)
                        logger.info(f"🧠 LEARNING: Removed glob pattern: {pattern}")
                
                for pattern in remove_patterns:
                    if pattern in config_data['blocked_patterns']:
                        config_data['blocked_patterns'].remove(pattern)
                        logger.info(f"🧠 LEARNING: Removed pattern: {pattern}")
                
                config_data['last_updated'] = current_datetime
                config_data['version'] = str(float(config_data.get('version', '1.0')) + 0.1)
                
                # Update metadata timestamps
                if 'metadata' not in config_data:
                    config_data['metadata'] = {}
                config_data['metadata']['last_updated_timestamp_ms'] = current_time_ms
                
                # Add learning metadata
                if 'learning_history' not in config_data:
                    config_data['learning_history'] = []
                config_data['learning_history'].append({
                    'timestamp': current_datetime,
                    'timestamp_ms': current_time_ms,
                    'failed_urls_count': len(_FAILED_URLS),
                    'new_globs': new_globs,
                    'new_patterns': new_patterns,
                    'removed_globs': remove_globs,
                    'removed_patterns': remove_patterns,
                    'reasoning': reasoning
                })
                
                # Write updated config
                with open(_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Update module-level variables
                BLOCKED_GLOBS.extend(new_globs)
                BLOCKED_PATTERNS.extend(new_patterns)
                
                # Remove patterns from module-level variables
                for pattern in remove_globs:
                    if pattern in BLOCKED_GLOBS:
                        BLOCKED_GLOBS.remove(pattern)
                        
                for pattern in remove_patterns:
                    if pattern in BLOCKED_PATTERNS:
                        BLOCKED_PATTERNS.remove(pattern)
                
                logger.info(f"🧠 LEARNING: Updated block list - Added {len(new_globs)} globs, {len(new_patterns)} patterns, Removed {len(remove_globs)} globs, {len(remove_patterns)} patterns")
                logger.info(f"🧠 LEARNING: Configuration saved to {_config_path}")
                logger.info(f"🧠 LEARNING: Detailed Reasoning: {reasoning}")
            else:
                logger.info(f"🧠 LEARNING: No new blocks needed - {reasoning}")
        
        # Clear the failed URLs list after processing
        _FAILED_URLS.clear()
        
    except Exception as e:
        logger.error(f"🧠 LEARNING: Failed to update block list: {e}", exc_info=True)


async def calculate_message_importance(
    content: str, role: str, session_id: str, messages: list | None = None
) -> float:
    """Calculate importance score for a message using the sophisticated scorer."""
    try:
        from memory.importance_scorer import get_importance_scorer
        importance_scorer = get_importance_scorer()

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

        # Calculate importance using async method
        importance = await importance_scorer.calculate_importance_async(
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
            # Quick memory retrieval for context - use conversation-specific context
            logger.info(f"🧠 MEMORY DEBUG: Calling get_conversation_context for session: '{session_id}'")
            memories = await app_state.personal_memory.get_conversation_context(conversation_id=session_id, max_messages=10)
            logger.info(f"🧠 MEMORY DEBUG: Retrieved {len(memories) if memories else 0} memories")

            # Get core memories (user facts) for this specific conversation
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

            # Get file attachments for this conversation
            file_attachments = await _get_file_attachments_for_conversation(session_id)
            if file_attachments:
                file_context = "Previously Attached Files:\n" + "\n".join(file_attachments)
                memory_parts.append(file_context)
                logger.info(f"📎 MEMORY DEBUG: Added {len(file_attachments)} file attachment references")

            # Add recent conversation memories (prioritize most recent)
            if memories:
                # Get the most recent memories to understand conversation flow
                recent_memories = [m.content for m in memories[:5]]  # Increased from 3 to 5
                
                # Add a recent conversation context header
                if recent_memories:
                    memory_parts.append("Recent Conversation Context:\n" + "\n\n".join(recent_memories))
                    logger.info(f"🧠 MEMORY DEBUG: Added {len(recent_memories)} recent conversation memories")

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


async def handle_conversation_intent(user_prompt: str, session_id: str, conversation_history: list = None):
    """Handle conversation intent with memory-aware processing."""
    logger.info("🧠 MEMORY DEBUG: Processing conversation intent - using inline conversation handling")

    # Get user settings to determine language preference
    user_settings = None
    preferred_language = "en"  # Default to English
    try:
        from .crud_routes import crud_service
        user_settings = await crud_service.get_user_settings(session_id)
        if user_settings and hasattr(user_settings, 'preferred_language'):
            preferred_language = user_settings.preferred_language
            logger.info(f"🌍 LANGUAGE: Using user's preferred language: {preferred_language}")
        else:
            logger.info("🌍 LANGUAGE: No user settings found, using default English")
    except Exception as e:
        logger.warning(f"🌍 LANGUAGE: Failed to fetch user settings: {e}, using default English")

    # Get memory context from personal memory
    memory_context = await get_memory_context(user_prompt, session_id)

    # Create language instruction based on preference
    language_instruction = ""
    if preferred_language != "en":
        language_map = {
            "es": "Spanish (Español)",
            "fr": "French (Français)", 
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "zh": "Chinese (中文)",
            "ja": "Japanese (日本語)"
        }
        language_name = language_map.get(preferred_language, preferred_language)
        language_instruction = f"\n\n🌍 LANGUAGE REQUIREMENT: You MUST respond ONLY in {language_name}. Do not use English unless specifically asked. All your responses should be in {language_name}."

    # Use condensed system prompt with memory context and language preference
    simple_system_prompt = f"""You are Jane, a helpful AI assistant. Be natural and conversational.

🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)

🧠 MEMORY INSTRUCTION: If I provide User Information below, you MUST use that information to answer questions about the user. This information comes from our previous conversations and is completely accurate.

🔄 CONVERSATION FLOW: 
- If the user gives a one-word answer or short phrase after you asked for clarification, they are answering your clarification question
- Look at the "Recent Conversation Context" to understand what specific topic they're choosing when they clarify
- When they clarify a topic, continue with that specific topic instead of starting a general conversation about it{language_instruction}"""

    # Add memory context if available
    if memory_context:
        simple_system_prompt += f"\n\n## User Information:\n{memory_context}"
        logger.info(f"🧠 MEMORY DEBUG: Full system prompt with memory context: {simple_system_prompt[:500]}...")

    # Use persistent LLM server for conversation
    from persistent_llm_server import get_llm_server

    server = await get_llm_server()

    # Format conversation history for the LLM
    conversation_context = ""
    if conversation_history:
        logger.info(f"🧠 CONVERSATION: Processing {len(conversation_history)} previous messages")
        
        # Build conversation context from message history
        conversation_messages = []
        for msg in conversation_history:
            role = msg.role if hasattr(msg, 'role') else 'unknown'
            content = msg.content if hasattr(msg, 'content') else str(msg)
            
            # Format each message properly
            if role == 'user':
                conversation_messages.append(f"User: {content}")
            elif role == 'assistant':
                conversation_messages.append(f"Assistant: {content}")
        
        if conversation_messages:
            conversation_context = "\n".join(conversation_messages)
            logger.info(f"🧠 CONVERSATION: Built conversation context with {len(conversation_messages)} messages")
    else:
        logger.info("🧠 CONVERSATION: No conversation history provided")

    # Format prompt with conversation history using the appropriate utility function
    if conversation_context:
        # Use format_prompt_with_history to include conversation context
        formatted_prompt = utils.format_prompt_with_history(
            simple_system_prompt, 
            user_prompt, 
            f"## Recent Conversation Context:\n{conversation_context}"
        )
        logger.info("🧠 CONVERSATION: Using prompt with conversation history")
    else:
        # Use regular format_prompt for first message or when no history
        formatted_prompt = utils.format_prompt(simple_system_prompt, user_prompt)
        logger.info("🧠 CONVERSATION: Using prompt without conversation history")

    # Show memory saving indicator before starting response
    yield f"data: {json.dumps({'token': {'text': '💾 Saving to memory...\n\n'}})}\n\n"

    # Stream tokens directly
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
    if full_response and app_state.personal_memory:
        try:
            task = asyncio.create_task(
                lightweight_memory_processing(user_prompt, full_response, session_id)
            )
            # Store reference to prevent garbage collection
            if not hasattr(app_state, '_background_tasks'):
                app_state._background_tasks = set()
            app_state._background_tasks.add(task)
            task.add_done_callback(lambda t: app_state._background_tasks.discard(t))
        except Exception:
            logger.exception("🧠 MEMORY DEBUG: ❌ Failed to create memory processing task")


async def _handle_direct_scraping(urls_to_scrape: list, user_prompt: str, session_id: str):
    """Handle direct URL scraping using web_scraper functionality."""
    import asyncio
    logger.info(f"🕷️ SCRAPING: Processing {len(urls_to_scrape)} URLs for session {session_id}")
    
    try:
        if not urls_to_scrape:
            yield f"data: {json.dumps({'token': {'text': '\n\n❓ No URLs found to scrape. Please provide specific URLs.\n\n'}})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
            
        # Import web scraping functionality
        from web_scraper import scrape_website
        
        # Single-pass filter with limit using module-level config
        urls_to_process = [
            url for url in urls_to_scrape[:5] 
            if url and url.strip() and not is_blocked_url(url)
        ]
        
        blocked_count = len(urls_to_scrape[:5]) - len(urls_to_process)
        
        if blocked_count > 0:
            logger.info(f"🚫 SCRAPING: Filtered {blocked_count} blocked URLs")
            yield f"data: {json.dumps({'token': {'text': f'🚫 Filtered {blocked_count} Facebook URLs (not scrapeable)\n\n'}})}\n\n"
        
        if not urls_to_process:
            yield f"data: {json.dumps({'token': {'text': '❌ No scrapeable URLs available\n\n'}})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        
        yield f"data: {json.dumps({'token': {'text': f'\n\n🎯 Scraping {len(urls_to_process)} URLs:\n\n'}})}\n\n"
        
        # Display URLs
        for i, url in enumerate(urls_to_process, 1):
            yield f"data: {json.dumps({'token': {'text': f'{i}. {url}\n'}})}\n\n"
            
        yield f"data: {json.dumps({'token': {'text': '\n\n🕷️ Starting scraping process...\n\n---\n\n'}})}\n\n"
        
        scraped_contents = []
        success_count = 0
        
        # Scrape each URL
        for i, url in enumerate(urls_to_process, 1):
            try:
                yield f"data: {json.dumps({'token': {'text': f'**Scraping {i}/{len(urls_to_process)}:** {url[:60]}...\n\n'}})}\n\n"
                
                content = await scrape_website(url, timeout_seconds=8, max_retries=2)
                logger.info(f"🕷️ SCRAPING: Got content length: {len(content) if content else 0} for {url}")
                
                # Use LLM to classify if this is an authentication message
                is_auth_message = False
                if content and len(content.strip()) > 10:
                    try:
                        # Get LLM server for classification
                        from persistent_llm_server import get_llm_server
                        llm_server = await get_llm_server()
                        
                        classification_prompt = f"""Analyze this web page content and determine if it indicates authentication is required.

Content to analyze:
{content[:500]}

Respond with only "AUTH_REQUIRED" if this content indicates authentication/login is needed, or "CONTENT_AVAILABLE" if this appears to be accessible content."""

                        classification_response = await llm_server.generate(
                            prompt=classification_prompt,
                            max_tokens=100,
                            temperature=0.1,
                            session_id="auth_classification"
                        )
                        
                        is_auth_message = "AUTH_REQUIRED" in classification_response.upper()
                        logger.info(f"🤖 LLM CLASSIFICATION: {url[:40]}... -> {'AUTH_REQUIRED' if is_auth_message else 'CONTENT_AVAILABLE'}")
                        
                    except Exception as classification_error:
                        logger.warning(f"🤖 LLM CLASSIFICATION: Failed for {url[:40]}... -> {classification_error}")
                        # Fallback: treat very short content as potentially auth-related
                        is_auth_message = len(content.strip()) < 100
                
                if content and (len(content.strip()) > 100 or is_auth_message):
                    if is_auth_message:
                        # Handle authentication messages specifically
                        yield f"data: {json.dumps({'token': {'text': f'🔒 {url[:60]}... requires authentication/login\n\n'}})}\n\n"
                        track_failed_url(url, f"Authentication required: {content[:50]}")
                    else:
                        # Regular successful content
                        # Truncate very long content
                        if len(content) > 3000:
                            content = content[:3000] + "\n\n[Content truncated for brevity...]"
                        
                        scraped_contents.append({
                            'url': url,
                            'content': content.strip()
                        })
                        success_count += 1
                        yield f"data: {json.dumps({'token': {'text': f'✅ Successfully scraped content ({len(content)} chars)\n\n'}})}\n\n"
                        
                        # Process content through LLM instead of raw dumping
                        yield f"data: {json.dumps({'token': {'text': f'🤖 Processing content through AI...\n\n'}})}\n\n"
                else:
                    # Track failed URL for learning
                    content_preview = content[:50] if content else "No content returned"
                    track_failed_url(url, f"Insufficient content: {content_preview}")
                    yield f"data: {json.dumps({'token': {'text': f'❌ Insufficient content from {url[:40]}...\n\n'}})}\n\n"
                    
            except Exception as scrape_error:
                # Track failed URL for learning
                track_failed_url(url, str(scrape_error))
                logger.warning(f"🕷️ SCRAPING: Error scraping {url}: {scrape_error}")
                yield f"data: {json.dumps({'token': {'text': f'❌ Error: {str(scrape_error)[:100]}\n\n'}})}\n\n"
        
        if scraped_contents:
            yield f"data: {json.dumps({'token': {'text': f'\n\n🤖 Analyzing {success_count} scraped sources with AI...\n\n---\n\n'}})}\n\n"
            
            # Use LLM to process and synthesize the scraped content
            from persistent_llm_server import get_llm_server
            
            llm_server = await get_llm_server()
            
            # Create system prompt for content synthesis  
            system_prompt = """You are Jane, a helpful AI assistant with access to scraped web content. Your task is to analyze and synthesize the scraped content to answer the user's question comprehensively.

🚨 CRITICAL FORMATTING RULES - ABSOLUTE REQUIREMENTS 🚨

**COMPLETELY FORBIDDEN - NEVER USE THESE:**
- [REF]1[/REF] - FORBIDDEN
- [REF]2[/REF] - FORBIDDEN  
- [REF]anything[/REF] - FORBIDDEN
- ANY variation of [REF]...[/REF] - COMPLETELY FORBIDDEN

**REQUIRED LINK FORMAT - ALWAYS USE THIS:**
- [Specific Website Name](URL) - REQUIRED
- [CNN](URL) - GOOD
- [Reuters](URL) - GOOD  
- [Associated Press](URL) - GOOD
- [USA Today](URL) - GOOD
- [Texas Tribune](URL) - GOOD

**LINK TEXT REQUIREMENTS:**
- Use the actual website/publication name: [CNN](URL), [BBC News](URL), [Reuters](URL)
- For news articles: [CNN flood coverage](URL), [Reuters breaking news](URL)
- For official sources: [FEMA updates](URL), [Texas Governor's Office](URL)
- NEVER use generic text like "Source", "Link", "Here", "Article"

**MANDATORY TABLE USAGE**: When presenting any structured data, comparisons, lists of items with attributes, or multiple data points, YOU MUST use markdown tables:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |

**STRUCTURE EVERYTHING**: Convert any structured information into organized sections with headers, tables, and lists. If data can be structured, it MUST be in a table format.

CRITICAL: Use ONLY [Specific Website Name](URL) format for links, NEVER use [REF] tags or generic "Source" text."""

            # Format all scraped content for LLM
            all_content = ""
            for item in scraped_contents:
                all_content += f"### Content from {item['url']}:\n{item['content']}\n\n---\n\n"
            
            user_content = f"User Query: {user_prompt}\n\nScraped Web Content:\n{all_content}\n\nPlease analyze and synthesize this information to comprehensively answer the user's question. Use proper markdown formatting, create tables for structured data, and use specific website names for links (never generic 'Source' text)."
            
            formatted_prompt = utils.format_prompt(system_prompt, user_content)
            
            # Stream the LLM response token by token
            async for token in llm_server.generate_stream(
                prompt=formatted_prompt,
                max_tokens=40960,
                temperature=0.7,
                top_p=0.95,
                session_id=session_id,
                priority=1,
            ):
                if token:
                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"
        else:
            yield f"data: {json.dumps({'token': {'text': '\n\n❌ Unable to scrape any content. The websites may be blocking access or require authentication.\n\n'}})}\n\n"
            
    except Exception as e:
        error_msg = f"Scraping error: {e}"
        logger.error(f"🕷️ SCRAPING: ❌ {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"
    
    # Learn from scraping failures and update block list
    await _update_block_list_from_failures()
    
    # Clear web search state since scraping is now complete
    clear_web_search_state(session_id)
    
    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_web_search_intent(user_prompt: str, session_id: str, request):
    """Handle web search requests with LLM synthesis."""
    logger.info("🧠 MEMORY DEBUG: Processing perform_web_search intent")

    try:
        # Get conversation memory context to provide the LLM with URLs from previous searches
        memory_context = await get_memory_context(user_prompt, session_id)
        
        # Ask the LLM to intelligently decide: new search vs scraping existing URLs
        from persistent_llm_server import get_llm_server
        import utils
        
        system_prompt = """You are an intelligent web assistant. Analyze the user's request and conversation context to determine the best action.

DECISION RULES:
1. If the user wants to search for NEW information not covered in the conversation context, choose "SEARCH"
2. If the user wants to learn more about topics already discussed AND there are URLs in the conversation context, choose "SCRAPE"
3. Look for URLs in the conversation context - if the user's request relates to existing topics with URLs, prioritize SCRAPE

CRITICAL: When choosing SCRAPE, you MUST extract ALL relevant URLs from the conversation context.

Return ONLY a JSON object:
{
    "action": "SEARCH" | "SCRAPE",
    "reasoning": "brief explanation",
    "urls_to_scrape": ["url1", "url2", "url3"] // REQUIRED if action is SCRAPE - extract ALL URLs from context that relate to the user's request
}"""

        user_content = f"""User Request: {user_prompt}

Conversation Context (contains any previous search results with URLs):
{memory_context}

TASK: Analyze if the user's request relates to topics already discussed. If there are relevant URLs in the context above, extract them for scraping. If not, choose to search for new information."""

        formatted_prompt = utils.format_prompt(system_prompt, user_content)
        llm_server = await get_llm_server()
        
        decision_response = await llm_server.generate(
            prompt=formatted_prompt,
            max_tokens=8000,
            temperature=0.2,
            session_id=f"{session_id}_web_decision",
        )
        
        logger.info(f"🤖 LLM web decision: {decision_response.strip()}")
        
        # Parse the LLM decision with robust error handling
        import json
        try:
            # Extract JSON from response with multiple fallback strategies
            start_idx = decision_response.find("{")
            end_idx = decision_response.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = decision_response[start_idx:end_idx + 1]
                logger.debug(f"🔍 PARSING DEBUG: Extracted JSON: {json_text[:200]}...")
                decision_json = json.loads(json_text)
            else:
                # Fallback: try to find incomplete JSON and fix it
                logger.warning(f"🔍 PARSING DEBUG: Malformed JSON detected, attempting repair...")
                logger.warning(f"🔍 PARSING DEBUG: Raw response: {decision_response}")
                
                # Use LLM to extract URLs from malformed response
                logger.info("🔍 PARSING DEBUG: Using LLM to extract URLs from malformed JSON")
                
                url_extraction_prompt = f"""The following LLM response was truncated and contains malformed JSON. Please extract any valid URLs from this text and return them as a clean JSON array.

Malformed Response:
{decision_response}

Memory Context for Reference:
{memory_context}

Return ONLY a JSON array of URLs like: ["url1", "url2", "url3"]
If no URLs found, return: []"""

                try:
                    url_extraction_response = await llm_server.generate(
                        prompt=utils.format_prompt("You are a URL extraction assistant. Extract only valid URLs from malformed text.", url_extraction_prompt),
                        max_tokens=500,
                        temperature=0.1,
                        session_id=f"{session_id}_url_extraction",
                    )
                    
                    logger.info(f"🔍 LLM URL extraction response: {url_extraction_response}")
                    
                    # Parse the LLM's URL extraction
                    start_bracket = url_extraction_response.find("[")
                    end_bracket = url_extraction_response.rfind("]")
                    if start_bracket != -1 and end_bracket != -1:
                        urls_json = url_extraction_response[start_bracket:end_bracket + 1]
                        urls = json.loads(urls_json)
                        
                        if urls:
                            logger.info(f"🔍 PARSING DEBUG: LLM extracted {len(urls)} URLs")
                            for idx, url in enumerate(urls):
                                logger.info(f"🕷️ SCRAPING DEBUG: URL {idx + 1}: {url}")
                            
                            yield f"data: {json.dumps({'token': {'text': f'🕷️ Scraping {len(urls)} URLs from conversation (LLM extracted from truncated response)...'}})}\n\n"
                            async for chunk in _handle_direct_scraping(urls, user_prompt, session_id):
                                yield chunk
                            return
                
                except Exception as extract_error:
                    logger.warning(f"🔍 LLM URL extraction failed: {extract_error}")
                
                # Complete fallback - create default decision
                decision_json = {"action": "SEARCH", "reasoning": "JSON parsing failed"}
            
            action = decision_json.get("action", "SEARCH")
            reasoning = decision_json.get("reasoning", "")
            
            if action == "SCRAPE":
                urls_to_scrape = decision_json.get("urls_to_scrape", [])
                logger.info(f"🕷️ SCRAPING DEBUG: LLM provided {len(urls_to_scrape)} URLs to scrape")
                for idx, url in enumerate(urls_to_scrape):
                    logger.info(f"🕷️ SCRAPING DEBUG: URL {idx + 1}: {url}")
                
                if urls_to_scrape:
                    yield f"data: {json.dumps({'token': {'text': f'🕷️ Scraping {len(urls_to_scrape)} URLs from conversation ({reasoning})...'}})}\n\n"
                    async for chunk in _handle_direct_scraping(urls_to_scrape, user_prompt, session_id):
                        yield chunk
                    return
                else:
                    logger.warning("🕷️ SCRAPING DEBUG: No URLs found in LLM decision, falling back to search")
                        
        except json.JSONDecodeError as e:
            logger.warning(f"🔍 PARSING DEBUG: JSON decode failed: {e}")
            logger.warning(f"🔍 PARSING DEBUG: Failed to parse LLM decision: {decision_response[:500]}...")
        except Exception as e:
            logger.warning(f"🔍 PARSING DEBUG: LLM decision error: {e}", exc_info=True)
            
        # Only reach here if SCRAPE decision failed or wasn't made - perform search as fallback
        logger.info("🔍 SEARCH FALLBACK: No valid URLs to scrape, performing web search")
        
        # Mark that we're performing a web search for this session
        mark_web_search_performed(session_id)
        
        yield f"data: {json.dumps({'token': {'text': '🔍 Searching the web...'}})}\n\n"

        from web_scraper import perform_web_search_async
        search_results = await perform_web_search_async(query=user_prompt, num_results=8, session_id=session_id)

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
❌ FORBIDDEN: [Source](URL) - Generic "Source" text is forbidden
✅ REQUIRED: [Descriptive Website Name](URL) - ALWAYS USE SPECIFIC WEBSITE NAMES

**LINK TEXT REQUIREMENTS:**
- Use the actual website name/title as link text: [CNN](URL), [BBC News](URL), [Reuters](URL)
- For news articles: [CNN article on floods](URL), [BBC breaking news](URL)
- For organizations: [FEMA disaster response](URL), [Red Cross updates](URL)
- For official sources: [Texas Governor's Office](URL), [White House statement](URL)
- NEVER use generic words like "Source", "Link", "Here", "Article"

Examples:
❌ BAD: "Flooding reported [Source](URL)" 
❌ BAD: "According to reports [Source](URL)"
✅ GOOD: "Flooding reported by [CNN](URL)"
✅ GOOD: "According to [Reuters](URL) and [Associated Press](URL)"

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

            user_content = f"User Query: {user_prompt}\n\nSearch Results:\n{search_results}\n\nPlease provide a comprehensive response to the user's query using the search results above. CRITICAL: Use ONLY [Specific Website Name](URL) format for links, NEVER use [REF] tags or generic 'Source' text."

            formatted_prompt = utils.format_prompt(system_prompt, user_content)

            # Stream the response token by token
            full_response = ""
            async for token in llm_server.generate_stream(
                prompt=formatted_prompt,
                max_tokens=40960,
                temperature=0.7,
                top_p=0.95,
                session_id=session_id,
                priority=1,
            ):
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

            # Store in memory for future reference
            if app_state.personal_memory and full_response:
                await _store_web_search_memory(user_prompt, full_response, session_id, request)
        else:
            yield f"data: {json.dumps({'token': {'text': '\n\n❌ No search results found.\n\n'}})}\n\n"

    except Exception as search_error:
        error_msg = f"Search error: {search_error}"
        logger.error(f"🧠 MEMORY DEBUG: ❌ {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

    # Learn from scraping failures and update block list
    await _update_block_list_from_failures()

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_personal_info_storage_intent(user_prompt: str, session_id: str):
    """Handle storing personal information."""
    logger.info("🧠 MEMORY STORE: Processing store_personal_info intent")

    yield f"data: {json.dumps({'token': {'text': '🧠 Storing your personal information...\n\n---\n\n'}})}\n\n"

    try:
        if app_state.personal_memory:
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

            system_prompt = """You are Jane, an advanced AI assistant with exceptional emotional intelligence and conversational skills. The user has just shared personal information with you, and you have successfully stored it in your long-term memory system.

CORE INSTRUCTIONS:
1. ACKNOWLEDGE SPECIFICALLY what they shared - don't be generic
2. EXPRESS genuine appreciation for their trust in sharing personal details
3. EXPLAIN how this information will help you assist them better in future conversations
4. DEMONSTRATE understanding by connecting their information to potential future help
5. BE conversational, warm, and personable - like a close friend would respond
6. SHOW excitement about getting to know them better

RESPONSE STRUCTURE:
- Start with genuine appreciation
- Acknowledge the specific information they shared
- Explain how you'll use this to help them better
- End with a warm, forward-looking statement

FORMATTING REQUIREMENTS:
- Use proper markdown formatting throughout
- Create structured data using markdown tables when appropriate:
  | Category | Details |
  |----------|---------|
  | Field1 | Value1 |
- NEVER use placeholder text like [Your Color] or [Field1] - ALWAYS use the actual information they shared
- Extract and use the specific details from what they told you
- Use **bold** for emphasis on key points
- Use *italics* for emotional emphasis

TONE: Warm, intelligent, caring, excited to learn about them. Show genuine interest in who they are as a person. Be substantive but not overly verbose."""

            formatted_prompt = utils.format_prompt(system_prompt, f"I just told you: {user_prompt}")

            # Stream the natural response
            # Calculate max_tokens dynamically based on config
            from config import MODEL_CONFIG
            context_window = MODEL_CONFIG.get("n_ctx", 12288)
            # Reserve ~30% of context for prompt, use 70% for response
            max_response_tokens = int(context_window * 0.7)
            
            async for token in llm_server.generate_stream(
                prompt=formatted_prompt,
                max_tokens=max_response_tokens,
                temperature=0.7,
                top_p=0.95,
                session_id=session_id,
                priority=1,
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
            task = asyncio.create_task(
                lightweight_memory_processing(user_prompt, "Information successfully stored in memory", session_id)
            )
            # Store reference to prevent garbage collection
            if not hasattr(app_state, '_background_tasks'):
                app_state._background_tasks = set()
            app_state._background_tasks.add(task)
            task.add_done_callback(lambda t: app_state._background_tasks.discard(t))
        except Exception:
            logger.exception("🧠 MEMORY STORE: ❌ Failed to create memory processing task")

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_personal_info_recall_intent(user_prompt: str, session_id: str):
    """Handle recalling personal information with comprehensive search."""
    logger.info("🧠 MEMORY RECALL: Processing recall_personal_info intent")

    yield f"data: {json.dumps({'token': {'text': '🧠 Searching my memory...\n\n'}})}\n\n"

    try:
        if app_state.personal_memory:
            # Search for relevant personal information
            logger.info(f"🧠 MEMORY RECALL: Searching for: '{user_prompt}'")

            # ENHANCED SEARCH: Try multiple search strategies for revolutionary recall
            all_memories = []
            
            # Strategy 1: Semantic search with original query - CONVERSATION SPECIFIC
            memories = await app_state.personal_memory.get_relevant_memories(query=user_prompt, limit=20, conversation_id=session_id)
            logger.info(f"🧠 MEMORY DEBUG: Conversation-specific semantic search found {len(memories)} memories")
            all_memories.extend(memories)
            
            # Strategy 2: Keyword-based search for experiences
            experience_keywords = ["went", "visited", "bought", "did", "experienced", "happened", "store", "shop", "yesterday", "today", "last"]
            if any(keyword in user_prompt.lower() for keyword in experience_keywords):
                # Get conversation context which includes experiences
                conversation_memories = await app_state.personal_memory.get_conversation_context(session_id, max_messages=50)
                logger.info(f"🧠 MEMORY DEBUG: Conversation context found {len(conversation_memories)} memories")
                
                # Filter for experience-related memories
                for mem in conversation_memories:
                    if any(keyword in mem.content.lower() for keyword in experience_keywords):
                        if mem not in all_memories:  # Avoid duplicates
                            all_memories.append(mem)
                            logger.info(f"🧠 MEMORY DEBUG: Added experience memory: {mem.content[:100]}...")
            
            # Strategy 3: Search for high-importance memories within this conversation only
            try:
                # CRITICAL FIX: Only search within the current conversation to prevent cross-contamination
                conversation_specific_memories = await app_state.personal_memory.get_conversation_context(session_id, max_messages=100)
                for mem in conversation_specific_memories:
                    if getattr(mem, 'importance', 0) >= 0.8:  # High importance threshold
                        if mem not in all_memories:
                            all_memories.append(mem)
                            logger.info(f"🧠 MEMORY DEBUG: Added high-importance conversation memory: {mem.content[:100]}...")
            except Exception as e:
                logger.warning(f"🧠 MEMORY DEBUG: Conversation-specific search failed: {e}")
            
            # Remove duplicates while preserving order
            unique_memories = []
            seen_ids = set()
            for mem in all_memories:
                if hasattr(mem, 'id') and mem.id not in seen_ids:
                    unique_memories.append(mem)
                    seen_ids.add(mem.id)
            
            logger.info(f"🧠 MEMORY DEBUG: Final search found {len(unique_memories)} unique memories for query: '{user_prompt}'")
            
            # DEBUG: Log what memories were found
            for i, memory in enumerate(unique_memories[:10]):
                logger.info(f"🧠 MEMORY DEBUG: Memory {i+1}: {memory.content[:100]}... (importance: {getattr(memory, 'importance', 'unknown')})")

            # Also get core memories (user facts) for this conversation
            core_memories = await app_state.personal_memory.get_all_core_memories(session_id)

            # Build response with found information
            if unique_memories or core_memories:
                async for chunk in _generate_memory_recall_response(unique_memories, core_memories, user_prompt, session_id):
                    yield chunk
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

    # Background memory processing for recall queries
    if app_state.personal_memory:
        try:
            task = asyncio.create_task(
                lightweight_memory_processing(user_prompt, "Assistant recalled information from memory", session_id)
            )
            # Store reference to prevent garbage collection
            if not hasattr(app_state, '_background_tasks'):
                app_state._background_tasks = set()
            app_state._background_tasks.add(task)
            task.add_done_callback(lambda t: app_state._background_tasks.discard(t))
        except Exception:
            logger.exception("🧠 MEMORY RECALL: ❌ Failed to create memory processing task")

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_conversation_history_intent(user_prompt: str, session_id: str):
    """Handle memory retrieval requests."""
    logger.info("🧠 MEMORY DEBUG: Processing query_conversation_history intent")

    yield f"data: {json.dumps({'token': {'text': '🧠 Searching memory...'}})}\n\n"

    try:
        # Get memory context from personal memory
        memory_context = ""
        if app_state.personal_memory:
            memory_context = await _get_conversation_history_context(user_prompt, session_id)

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
                priority=1,
            ):
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

            # Background memory processing
            if full_response and app_state.personal_memory:
                await _store_conversation_history_memory(user_prompt, full_response, session_id)
        else:
            message = "\n\n🤔 I don't have any relevant information about that in our conversation history. Would you like me to search for new information instead?\n\n"
            yield f"data: {json.dumps({'token': {'text': message}})}\n\n"

    except Exception as memory_error:
        error_msg = f"Memory retrieval error: {memory_error}"
        logger.error(f"🧠 MEMORY DEBUG: ❌ {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_stock_query_intent(user_prompt: str, session_id: str, request):
    """Handle stock market queries."""
    logger.info(f"📈 STOCKS: Processing stock query: '{user_prompt}'")

    yield f"data: {json.dumps({'token': {'text': '📈 Fetching stock data...'}})}\n\n"

    try:
        # Let the LLM handle the complexity of extracting stock symbols
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
            session_id=f"{session_id}_symbol_extraction",
        )

        logger.info(f"📈 STOCKS: LLM extracted symbols: '{symbol_response.strip()}'")

        # Parse and validate tickers
        tickers = await _parse_and_validate_tickers(symbol_response)

        if not tickers:
            yield f"data: {json.dumps({'token': {'text': '\n\n❓ No stock symbols detected in your query. Please specify stock ticker symbols (e.g., AAPL, MSFT, GOOGL) or company names.\n\n'}})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Get and process stock data
        if app_state.stock_searcher:
            async for data_chunk in _process_stock_data(tickers, user_prompt, session_id, request):
                yield data_chunk
        else:
            yield f"data: {json.dumps({'token': {'text': '\n\n❌ Stock market data service is not available.\n\n'}})}\n\n"

    except Exception as stock_error:
        error_msg = f"Error fetching stock data: {stock_error}"
        logger.error(f"📈 STOCKS: {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_weather_query_intent(user_prompt: str, session_id: str, request):
    """Handle weather queries."""
    logger.info(f"🌤️ WEATHER: Processing weather query: '{user_prompt}'")

    yield f"data: {json.dumps({'token': {'text': '🌤️ Fetching weather data...\n\n'}})}\n\n"

    try:
        # Extract city from user query using LLM
        from persistent_llm_server import get_llm_server
        from weather import get_weather_for_city

        llm_server = await get_llm_server()

        # Extract city name
        city_name = await _extract_city_from_query(user_prompt, session_id, llm_server)

        if city_name.lower() == "current location":
            yield f"data: {json.dumps({'token': {'text': '📍 Please specify a city name for weather information.\n\n'}})}\n\n"
        else:
            # Convert state names and get weather data
            converted_city = await _convert_city_with_state(city_name, session_id, llm_server)
            weather_data = await get_weather_for_city(converted_city)

            if weather_data and "error" not in weather_data:
                async for data_chunk in _process_weather_data(weather_data, user_prompt, session_id, request, llm_server):
                    yield data_chunk
            else:
                error_msg = (
                    weather_data.get("error", "Unable to fetch weather data")
                    if weather_data
                    else "Weather service unavailable"
                )
                yield f"data: {json.dumps({'token': {'text': f'❌ {error_msg}\n\n'}})}\n\n"

    except Exception as weather_error:
        error_msg = f"Weather error: {weather_error}"
        logger.error(f"🌤️ WEATHER: {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_crypto_query_intent(user_prompt: str, session_id: str, request):
    """Handle cryptocurrency queries."""
    logger.info(f"₿ CRYPTO: Processing cryptocurrency query: '{user_prompt}'")

    yield f"data: {json.dumps({'token': {'text': '₿ Fetching cryptocurrency data...\n\n'}})}\n\n"

    try:
        # Use the crypto trading module directly
        from crypto_trading import CryptoTrading

        crypto_trader = CryptoTrading()

        # Extract cryptocurrency mentions from user query
        requested_cryptos = _extract_crypto_mentions(user_prompt)

        # Get cryptocurrency price data
        crypto_data = crypto_trader.get_multiple_crypto_quotes(requested_cryptos[:5])

        if crypto_data:
            async for data_chunk in _process_crypto_data(crypto_trader, requested_cryptos, user_prompt, session_id, request):
                yield data_chunk
        else:
            yield f"data: {json.dumps({'token': {'text': '\n\n❌ Unable to fetch cryptocurrency price data. The CoinGecko API may be unavailable.\n\n'}})}\n\n"

    except Exception as crypto_error:
        error_msg = f"Error fetching cryptocurrency data: {crypto_error}"
        logger.error(f"₿ CRYPTO: {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'token': {'text': f'\n\n❌ {error_msg}\n\n'}})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


# Helper functions
async def _store_web_search_memory(user_prompt: str, full_response: str, session_id: str, request):
    """Store web search results in memory."""
    try:
        logger.info(f"🧠 MEMORY STORAGE: Storing web search results in memory for session {session_id}")

        # Calculate dynamic importance for user message
        user_importance = await calculate_message_importance(
            content=user_prompt,
            role="user",
            session_id=session_id,
            messages=request.messages,
        )

        await app_state.personal_memory.add_memory(
            content=f"User: {user_prompt}",
            conversation_id=session_id,
            importance=user_importance,
        )

        # Calculate dynamic importance for assistant response
        assistant_importance = await calculate_message_importance(
            content=full_response,
            role="assistant",
            session_id=session_id,
            messages=[*request.messages, {"role": "user", "content": user_prompt}],
        )

        await app_state.personal_memory.add_memory(
            content=f"Assistant: {full_response}",
            conversation_id=session_id,
            importance=assistant_importance,
        )
        logger.info("🧠 MEMORY STORAGE: ✅ Successfully stored web search memories")
    except Exception:
        logger.exception("🧠 MEMORY STORAGE: ❌ Failed to store in memory")


async def _generate_memory_recall_response(memories, core_memories, user_prompt: str, session_id: str):
    """Generate response using memory context."""
    from persistent_llm_server import get_llm_server

    llm_server = await get_llm_server()

    # Prepare memory context
    memory_parts = []

    if core_memories:
        core_facts = [f"{key}: {value}" for key, value in core_memories.items()]
        if core_facts:
            memory_parts.append("Personal Facts:\n" + "\n".join(core_facts))

    if memories:
        # Include timestamps for temporal context
        memory_content = []
        for i, m in enumerate(memories[:5], 1):
            timestamp_str = getattr(m, 'timestamp', 'Unknown time')
            importance = getattr(m, 'importance', 'Unknown')
            memory_content.append(f"[Memory {i} - {timestamp_str} - Importance: {importance}]\n{m.content}")
        memory_parts.extend(memory_content)

    memory_context = "\n\n".join(memory_parts)

    # Generate response using memory context
    system_prompt = f"""You are Jane, a helpful, logical, and honest AI assistant recalling information about the user. Based on the memories below, answer the user's question directly and naturally.

🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)

⭐ PRIORITY GUIDANCE: Give MUCH GREATER WEIGHT to information that relates to the user's MOST RECENT queries and current conversation context. Pay close attention to the timestamps - prioritize newer memories over older ones when answering questions. If the user asks about "what we just discussed" or uses recent context, focus on the most recent memories.

## Available Information:
{memory_context}

Answer the user's question based on this information, prioritizing the most recent and contextually relevant memories. If the information isn't available, say so clearly."""

    formatted_prompt = utils.format_prompt(system_prompt, user_prompt)

    # Stream the response
    # Calculate max_tokens dynamically based on config
    from config import MODEL_CONFIG
    context_window = MODEL_CONFIG.get("n_ctx", 12288)
    # Reserve ~40% of context for prompt, use 60% for response
    max_response_tokens = int(context_window * 0.6)
    
    async for token in llm_server.generate_stream(
        prompt=formatted_prompt,
        max_tokens=max_response_tokens,
        temperature=0.3,
        top_p=0.95,
        session_id=session_id,
        priority=1,
    ):
        if token:
            yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

    logger.info("🧠 MEMORY RECALL: ✅ Successfully recalled information")


async def _get_conversation_history_context(user_prompt: str, session_id: str) -> str:
    """Get conversation history context from memory."""
    try:
        logger.info(f"🧠 MEMORY RETRIEVAL: Searching conversation {session_id} for recent memories")
        
        # Get conversation-specific context from this session
        memories = await app_state.personal_memory.get_conversation_context(
            conversation_id=session_id, max_messages=20
        )
        
        if memories:
            # Format memories with timestamps for better context
            memory_lines = []
            for memory in memories:
                if hasattr(memory, 'content') and memory.content:
                    memory_lines.append(f"Previous discussion: {memory.content}")
            
            if memory_lines:
                memory_context = "\n".join(memory_lines)
                logger.info(f"🧠 MEMORY RETRIEVAL: ✅ Successfully retrieved {len(memory_lines)} conversation memories")
                return memory_context
        
        logger.info("🧠 MEMORY RETRIEVAL: No conversation history found for this session")
        return ""
    except Exception:
        logger.exception("🧠 MEMORY RETRIEVAL: ❌ Memory retrieval failed")
        return ""


async def _store_conversation_history_memory(user_prompt: str, full_response: str, session_id: str):
    """Store conversation history query in memory."""
    try:
        # Store reference to prevent garbage collection
        if not hasattr(app_state, "_background_tasks"):
            app_state._background_tasks = set()
        task = asyncio.create_task(
            lightweight_memory_processing(user_prompt, full_response, session_id)
        )
        app_state._background_tasks.add(task)
        task.add_done_callback(app_state._background_tasks.discard)
    except Exception:
        logger.exception("🧠 MEMORY DEBUG: ❌ Failed to create memory processing task")


async def _parse_and_validate_tickers(symbol_response: str) -> list:
    """Parse and validate stock ticker symbols."""
    tickers = []
    if symbol_response.strip():
        try:
            # Try to find JSON array in the response
            start_idx = symbol_response.find("[")
            end_idx = symbol_response.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                symbols_json = symbol_response[start_idx : end_idx + 1]
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

    return tickers


async def _process_stock_data(tickers: list, user_prompt: str, session_id: str, request):
    """Process and stream stock data."""
    yield f"data: {json.dumps({'token': {'text': f' Getting quotes for {', '.join(tickers)}...\n\n'}})}\n\n"

    # Get raw stock data instead of pre-formatted strings
    import asyncio
    stock_quotes = await asyncio.to_thread(app_state.stock_searcher.get_multiple_quotes, tickers)

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
                    "market_cap": quote.market_cap,
                })

        # Use LLM to create a natural response with raw JSON data
        from persistent_llm_server import get_llm_server

        llm_server = await get_llm_server()

        system_prompt = """You are a knowledgeable financial assistant with access to real-time stock market data.

You must always return your responses using proper markdown formatting and must always use markdown tables to display structured data.

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
            priority=1,
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
            await _store_stock_memory(user_prompt, full_response, session_id, request)
    else:
        yield f"data: {json.dumps({'token': {'text': '\n\n❌ Unable to fetch stock data. The symbols may be invalid or the market data service is unavailable.\n\n'}})}\n\n"


async def _store_stock_memory(user_prompt: str, full_response: str, session_id: str, request):
    """Store stock query in memory."""
    try:
        # Calculate importance
        user_importance = await calculate_message_importance(
            content=user_prompt,
            role="user",
            session_id=session_id,
            messages=request.messages,
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
            messages=[*request.messages, {"role": "user", "content": user_prompt}],
        )

        await app_state.personal_memory.add_memory(
            content=f"Assistant: {full_response}",
            conversation_id=session_id,
            importance=assistant_importance,
        )
        logger.info("📈 STOCKS: Stored stock query conversation in memory")
    except Exception:
        logger.exception("📈 STOCKS: Failed to store in memory")


async def _extract_city_from_query(user_prompt: str, session_id: str, llm_server) -> str:
    """Extract city name from weather query."""
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
        priority=1,
    ):
        if token:
            city_response += token

    city_name = city_response.strip()
    logger.info(f"🌤️ WEATHER: Extracted city: '{city_name}'")
    return city_name


async def _convert_city_with_state(city_name: str, session_id: str, llm_server) -> str:
    """Convert state names to abbreviations using LLM."""
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
        priority=1,
    ):
        if token:
            state_convert_response += token

    converted_city = state_convert_response.strip()
    logger.info(f"🌤️ WEATHER: State conversion: '{city_name}' -> '{converted_city}'")
    return converted_city


async def _process_weather_data(weather_data, user_prompt: str, session_id: str, request, llm_server):
    """Process and stream weather data."""
    yield f"data: {json.dumps({'token': {'text': '\n\n📊 Found weather data, generating comprehensive response...\n\n---\n\n'}})}\n\n"

    # Use LLM to generate a comprehensive weather response
    from weather import format_weather_response

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
        priority=1,
    ):
        if token:
            full_response += token
            yield f"data: {json.dumps({'token': {'text': token}})}\n\n"

    # Add data source
    yield f"data: {json.dumps({'token': {'text': '\n\n📊 **Data Source:** [OpenWeatherMap](https://openweathermap.org/)\n\n'}})}\n\n"

    # Store in memory
    if app_state.personal_memory and full_response:
        await _store_weather_memory(user_prompt, full_response, session_id, request)


async def _store_weather_memory(user_prompt: str, full_response: str, session_id: str, request):
    """Store weather query in memory."""
    try:
        # Calculate importance
        user_importance = await calculate_message_importance(
            content=user_prompt,
            role="user",
            session_id=session_id,
            messages=request.messages,
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
            messages=[*request.messages, {"role": "user", "content": user_prompt}],
        )

        await app_state.personal_memory.add_memory(
            content=f"Assistant: {full_response}",
            conversation_id=session_id,
            importance=assistant_importance,
        )
        logger.info("🌤️ WEATHER: Stored weather query conversation in memory")
    except Exception:
        logger.exception("🌤️ WEATHER: Failed to store in memory")


def _extract_crypto_mentions(user_prompt: str) -> list:
    """Extract cryptocurrency mentions from user query."""
    query_lower = user_prompt.lower()
    common_cryptos = [
        "bitcoin", "ethereum", "cardano", "solana", "binancecoin", "ripple", "dogecoin"
    ]

    # Check for specific cryptocurrency mentions
    requested_cryptos = [
        crypto for crypto in common_cryptos
        if crypto in query_lower or crypto[:3] in query_lower
    ]

    # If no specific crypto mentioned, default to top cryptocurrencies
    if not requested_cryptos:
        requested_cryptos = ["bitcoin", "ethereum", "cardano"]

    return requested_cryptos


async def _process_crypto_data(crypto_trader, requested_cryptos: list, user_prompt: str, session_id: str, request):
    """Process and stream cryptocurrency data."""
    logger.info(f"₿ CRYPTO: Retrieved {len(crypto_trader.get_multiple_crypto_quotes(requested_cryptos[:5]))} cryptocurrency quotes")

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
        priority=1,
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
        await _store_crypto_memory(user_prompt, full_response, session_id, request)


async def _store_crypto_memory(user_prompt: str, full_response: str, session_id: str, request):
    """Store cryptocurrency query in memory."""
    try:
        # Calculate importance for user query
        user_importance = await calculate_message_importance(
            content=user_prompt,
            role="user",
            session_id=session_id,
            messages=request.messages,
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
            messages=[*request.messages, {"role": "user", "content": user_prompt}],
        )

        await app_state.personal_memory.add_memory(
            content=f"Assistant provided crypto price data: {full_response[:200]}...",
            conversation_id=session_id,
            importance=assistant_importance,
        )
        logger.info("₿ CRYPTO: Stored crypto query conversation in memory")
    except Exception:
        logger.exception("₿ CRYPTO: Failed to store in memory")


async def _get_file_attachments_for_conversation(session_id: str) -> list[str]:
    """Get file attachment references for a conversation session.
    
    Returns a list of formatted file attachment descriptions for context.
    """
    try:
        logger.info(f"📎 FILE_RETRIEVAL: Getting file attachments for session {session_id}")
        
        if not app_state.personal_memory:
            logger.warning("📎 FILE_RETRIEVAL: personal_memory not available")
            return []
            
        # Search for file attachments in this conversation
        # We'll search for both types: actual file content and file references
        file_memories = await app_state.personal_memory.get_relevant_memories(
            query="file attachment filename user uploaded", 
            limit=20,
            conversation_id=session_id
        )
        
        if not file_memories:
            logger.debug(f"📎 FILE_RETRIEVAL: No file memories found for session {session_id}")
            return []
            
        file_attachments = []
        processed_files = set()  # Avoid duplicates
        
        for memory in file_memories:
            # Check if this memory is from the current conversation
            if hasattr(memory, 'conversation_id') and memory.conversation_id != session_id:
                continue
                
            # Check if this is a file-related memory
            if hasattr(memory, 'metadata') and memory.metadata:
                metadata = memory.metadata
                if metadata.get('type') == 'file_reference' and metadata.get('filename'):
                    filename = metadata['filename']
                    if filename not in processed_files:
                        file_attachments.append(f"- {filename}: {memory.content}")
                        processed_files.add(filename)
                        logger.debug(f"📎 FILE_RETRIEVAL: Found reference for {filename}")
                        
                elif metadata.get('type') == 'file_attachment' and metadata.get('filename'):
                    filename = metadata['filename']
                    if filename not in processed_files:
                        content_preview = metadata.get('content_preview', 'No preview available')
                        file_attachments.append(f"- {filename}: {content_preview}")
                        processed_files.add(filename)
                        logger.debug(f"📎 FILE_RETRIEVAL: Found attachment for {filename}")
            else:
                # Fallback: look for file patterns in content
                if 'attached file' in memory.content.lower() or 'file attachment' in memory.content.lower():
                    file_attachments.append(f"- {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}")
                    
        logger.info(f"📎 FILE_RETRIEVAL: Found {len(file_attachments)} file attachments for session {session_id}")
        return file_attachments[:5]  # Limit to 5 most relevant files
        
    except Exception as e:
        logger.error(f"📎 FILE_RETRIEVAL: ❌ Error retrieving file attachments: {e}", exc_info=True)
        return []

