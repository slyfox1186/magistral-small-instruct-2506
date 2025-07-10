#!/usr/bin/env python3
"""
Test intent classification for animals and souls conversations
"""

import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_classification():
    """Test how the classification system handles animal soul questions"""
    
    try:
        # Import classification system
        from modules.chat_helpers import classify_query_with_llm
        
        # Test prompts that should be philosophical conversations
        test_prompts = [
            "Do you believe animals have souls?",
            "Do you think animals have souls?", 
            "What do you think about animals having souls?",
            "I believe animals do have souls. My dog seems to understand my emotions deeply.",
            "Tell me about your thoughts on whether animals have consciousness or souls",
            "Do cats and dogs have souls like humans do?"
        ]
        
        for prompt in test_prompts:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: '{prompt}'")
            logger.info(f"{'='*60}")
            
            try:
                classification = await classify_query_with_llm(prompt)
                intent = classification.get("primary_intent", "unknown")
                
                logger.info(f"✅ Classification result: {intent}")
                
                # Check if this intent has memory processing
                memory_intents = [
                    "conversation",
                    "store_personal_info", 
                    "recall_personal_info",
                    "query_conversation_history",
                    "perform_web_search",  # Has memory processing
                    "query_stocks",        # Missing memory processing
                    "query_weather",       # Missing memory processing  
                    "query_cryptocurrency" # Missing memory processing
                ]
                
                if intent in memory_intents:
                    logger.info(f"✅ This intent DOES have memory processing")
                else:
                    logger.warning(f"❌ This intent MISSING memory processing")
                    
            except Exception as e:
                logger.error(f"❌ Classification failed: {e}")
        
    except Exception as e:
        logger.error(f"Error in classification test: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_classification())