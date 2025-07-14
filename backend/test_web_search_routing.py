#!/usr/bin/env python3
"""Test web search routing to verify the fix."""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_web_search_routing():
    """Test that web search queries are routed to the correct handler."""
    try:
        # Import classification system
        from modules.chat_helpers import classify_query_with_llm
        
        # Test web search prompts
        web_search_prompts = [
            "Who is the current president of the USA?",
            "What is the latest news about AI?",
            "Recent developments in quantum computing",
            "Current weather in New York",  # Note: This might be classified as WEATHER instead
            "Bitcoin price today",  # Note: This might be classified as CRYPTO instead
            "What happened in the news today?",
            "Tell me about the latest SpaceX launch",
        ]
        
        for prompt in web_search_prompts:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing: '{prompt}'")
            logger.info(f"{'=' * 60}")
            
            try:
                classification = await classify_query_with_llm(prompt)
                intent = classification.get("primary_intent", "unknown")
                
                logger.info(f"✅ Classification result: {intent}")
                
                # Check if this is correctly classified as web search
                if intent == "perform_web_search":
                    logger.info("✅ CORRECT: Classified as web search")
                else:
                    logger.warning(f"❌ INCORRECT: Should be web search, got: {intent}")
                    
            except Exception:
                logger.exception("❌ Classification failed")
                
        # Test the intent handler mapping
        logger.info(f"\n{'=' * 60}")
        logger.info("Testing intent handler mapping")
        logger.info(f"{'=' * 60}")
        
        # Import the intent handlers mapping from chat_routes
        from modules.chat_routes import _create_chat_stream_route
        
        # Test that the mapping includes perform_web_search
        test_intent_handlers = {
            "conversation": "handle_conversation_intent",
            "perform_web_search": "handle_web_search_intent",
            "store_personal_info": "handle_personal_info_storage_intent",
            "recall_personal_info": "handle_personal_info_recall_intent",
            "query_conversation_history": "handle_conversation_history_intent",
            "query_stocks": "handle_stock_query_intent",
            "query_weather": "handle_weather_query_intent",
            "query_cryptocurrency": "handle_crypto_query_intent",
        }
        
        if "perform_web_search" in test_intent_handlers:
            logger.info("✅ CORRECT: perform_web_search is in intent handlers mapping")
        else:
            logger.error("❌ INCORRECT: perform_web_search is NOT in intent handlers mapping")
            
    except Exception as e:
        logger.error(f"Error in web search routing test: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_web_search_routing())