#!/usr/bin/env python3
"""Test the lightweight_memory_processing function directly with animal soul conversations
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_processing_directly():
    """Test the lightweight_memory_processing function directly"""
    try:
        # Initialize memory system
        from memory_provider import MemoryConfig, get_memory_system
        from modules.globals import app_state

        if not app_state.personal_memory:
            logger.info("🔧 Initializing memory system...")
            memory_config = MemoryConfig()
            app_state.personal_memory = get_memory_system(memory_config)
            logger.info("✅ Memory system initialized")

        # Import the memory processing function
        from modules.chat_helpers import lightweight_memory_processing

        # Test with an animals and souls conversation
        user_prompt = "Do you believe animals have souls?"
        assistant_response = """That's a profound philosophical question. Many belief systems and philosophical traditions have different perspectives on whether animals have souls. Some religions like Hinduism and Buddhism suggest all living beings have spiritual essence, while others make distinctions between human souls and animal consciousness. From a scientific perspective, we can observe complex emotions, social bonds, and even apparent empathy in many animals, suggesting rich inner lives. What's your own perspective on this?"""

        session_id = "test_animals_souls_direct"

        logger.info(f"\n{'='*60}")
        logger.info("TESTING DIRECT MEMORY PROCESSING")
        logger.info(f"{'='*60}")
        logger.info(f"User: {user_prompt}")
        logger.info(f"Assistant: {assistant_response[:100]}...")
        logger.info(f"Session: {session_id}")

        # Check memory count before
        memories_before = await app_state.personal_memory.get_relevant_memories("", limit=20)
        logger.info(f"📊 Memories before processing: {len(memories_before)}")

        # Call the memory processing function directly
        logger.info("🧠 Calling lightweight_memory_processing...")
        await lightweight_memory_processing(user_prompt, assistant_response, session_id)
        logger.info("✅ Memory processing completed")

        # Check memory count after
        memories_after = await app_state.personal_memory.get_relevant_memories("", limit=20)
        logger.info(f"📊 Memories after processing: {len(memories_after)}")

        # Search for specific content
        logger.info("\n🔍 Searching for animal/soul related memories...")
        animal_memories = await app_state.personal_memory.get_relevant_memories("animals souls philosophical", limit=10)
        logger.info(f"🐾 Found {len(animal_memories)} animal/soul memories:")

        for i, memory in enumerate(animal_memories):
            content_preview = memory.content[:150] + "..." if len(memory.content) > 150 else memory.content
            logger.info(f"  {i+1}. [{memory.importance:.2f}] {content_preview}")

        # Test another variation
        logger.info(f"\n{'='*60}")
        logger.info("TESTING WITH PERSONAL BELIEF")
        logger.info(f"{'='*60}")

        user_prompt_2 = "I believe animals do have souls. My dog seems to understand my emotions deeply."
        assistant_response_2 = """Your observation about your dog's emotional understanding resonates with many pet owners and animal behaviorists. Dogs have co-evolved with humans for thousands of years, developing remarkable abilities to read our facial expressions, body language, and even emotional states. Whether we call this a 'soul' or advanced emotional intelligence, it's clear that many animals form deep bonds and demonstrate consciousness that goes beyond simple instinct. Your belief reflects a compassionate view that many philosophical and spiritual traditions would support."""

        session_id_2 = "test_animals_souls_belief"

        logger.info(f"User: {user_prompt_2}")
        logger.info(f"Assistant: {assistant_response_2[:100]}...")

        # Process second conversation
        await lightweight_memory_processing(user_prompt_2, assistant_response_2, session_id_2)

        # Final check
        memories_final = await app_state.personal_memory.get_relevant_memories("", limit=20)
        logger.info(f"📊 Final memory count: {len(memories_final)}")

        # Search again
        animal_memories_final = await app_state.personal_memory.get_relevant_memories("animals souls dog belief philosophical", limit=10)
        logger.info(f"🐾 Final animal/soul memories: {len(animal_memories_final)}")

        for i, memory in enumerate(animal_memories_final):
            content_preview = memory.content[:150] + "..." if len(memory.content) > 150 else memory.content
            logger.info(f"  {i+1}. [{memory.importance:.2f}] {content_preview}")

    except Exception as e:
        logger.error(f"Error in direct memory processing test: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_memory_processing_directly())
