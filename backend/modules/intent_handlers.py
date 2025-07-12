#!/usr/bin/env python3
"""Intent handlers for different types of chat requests."""

import asyncio
import json
import logging

import utils

from .chat_helpers import lightweight_memory_processing
from .globals import app_state

logger = logging.getLogger(__name__)


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


async def handle_conversation_intent(user_prompt: str, session_id: str):
    """Handle conversation intent with memory-aware processing."""
    logger.info("🧠 MEMORY DEBUG: Processing conversation intent - using inline conversation handling")

    # Get memory context from personal memory
    memory_context = await get_memory_context(user_prompt, session_id)

    # Use condensed system prompt with memory context
    simple_system_prompt = """You are Jane, a helpful AI assistant. Be natural and conversational.

🚨 CRITICAL: You are NEVER allowed to use [REF] tags in ANY form ([REF]1[/REF], [REF]2[/REF], [REF]anything[/REF]) and must ONLY use proper markdown links: [Text Here](URL)

🧠 MEMORY INSTRUCTION: If I provide User Information below, you MUST use that information to answer questions about the user. This information comes from our previous conversations and is completely accurate."""

    # Add memory context if available
    if memory_context:
        simple_system_prompt += f"\n\n## User Information:\n{memory_context}"
        logger.info(f"🧠 MEMORY DEBUG: Full system prompt with memory context: {simple_system_prompt[:500]}...")

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
        
        # Limit to reasonable number of URLs
        urls_to_process = urls_to_scrape[:5]
        
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
                
                if content and len(content.strip()) > 100:
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
                    yield f"data: {json.dumps({'token': {'text': f'❌ No content found or access denied\n\n'}})}\n\n"
                    
            except Exception as scrape_error:
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
1. If the user wants to search for NEW information, choose "SEARCH"
2. If the user wants to scrape/read more from URLs mentioned in the conversation context, choose "SCRAPE"
3. Consider the intent behind the user's words

Return ONLY a JSON object:
{
    "action": "SEARCH" | "SCRAPE",
    "reasoning": "brief explanation",
    "urls_to_scrape": ["url1", "url2"] // Only if action is SCRAPE, extract URLs from context
}"""

        user_content = f"""User Request: {user_prompt}

Conversation Context (contains any previous search results with URLs):
{memory_context[:2000]}

What should I do: search for new information or scrape existing URLs from our conversation?"""

        formatted_prompt = utils.format_prompt(system_prompt, user_content)
        llm_server = await get_llm_server()
        
        decision_response = await llm_server.generate(
            prompt=formatted_prompt,
            max_tokens=300,
            temperature=0.2,
            session_id=f"{session_id}_web_decision",
        )
        
        logger.info(f"🤖 LLM web decision: {decision_response.strip()}")
        
        # Parse the LLM decision
        import json
        try:
            # Extract JSON from response
            start_idx = decision_response.find("{")
            end_idx = decision_response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                decision_json = json.loads(decision_response[start_idx:end_idx + 1])
                action = decision_json.get("action", "SEARCH")
                reasoning = decision_json.get("reasoning", "")
                
                if action == "SCRAPE":
                    urls_to_scrape = decision_json.get("urls_to_scrape", [])
                    yield f"data: {json.dumps({'token': {'text': f'🕷️ Scraping URLs from conversation ({reasoning})...'}})}\n\n"
                    async for chunk in _handle_direct_scraping(urls_to_scrape, user_prompt, session_id):
                        yield chunk
                    return
                        
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM decision: {decision_response}")
        except Exception as e:
            logger.warning(f"LLM decision error: {e}")
            
        # Default to search
        yield f"data: {json.dumps({'token': {'text': '🔍 Searching the web...'}})}\n\n"

        from web_scraper import perform_web_search_async
        search_results = await perform_web_search_async(query=user_prompt, num_results=8)

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

    yield f"data: {json.dumps({'done': True})}\n\n"


async def handle_personal_info_storage_intent(user_prompt: str, session_id: str):
    """Handle storing personal information."""
    logger.info("🧠 MEMORY STORE: Processing store_personal_info intent")

    yield f"data: {json.dumps({'token': {'text': '🧠 Storing your information...\n\n---\n\n'}})}\n\n"

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
                max_tokens=300,
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
    """Handle recalling personal information."""
    logger.info("🧠 MEMORY RECALL: Processing recall_personal_info intent")

    yield f"data: {json.dumps({'token': {'text': '🧠 Searching my memory...\n\n'}})}\n\n"

    try:
        if app_state.personal_memory:
            # Search for relevant personal information
            logger.info(f"🧠 MEMORY RECALL: Searching for: '{user_prompt}'")

            # Get relevant memories
            memories = await app_state.personal_memory.get_relevant_memories(query=user_prompt, limit=10)

            # Also get core memories (user facts) for this conversation
            core_memories = await app_state.personal_memory.get_all_core_memories(session_id)

            # Build response with found information
            if memories or core_memories:
                await _generate_memory_recall_response(memories, core_memories, user_prompt, session_id)
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
            limit=20
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

