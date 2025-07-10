#!/usr/bin/env python3
"""
Debug script to test the complete memory processing pipeline
"""

import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def debug_full_pipeline():
    """Test the complete memory processing pipeline"""
    
    try:
        # Import required components
        from memory_processing import AdvancedMemoryProcessor, get_config
        from persistent_llm_server import get_llm_server
        from modules.globals import app_state
        
        # Check if personal memory system is available
        logger.info(f"Personal memory system: {app_state.personal_memory}")
        
        if not app_state.personal_memory:
            logger.error("❌ Personal memory system not available!")
            return
        
        # Initialize the advanced memory processor
        config = get_config('development')  # Use development for more detailed logging
        processor = AdvancedMemoryProcessor(app_state.personal_memory, config)
        
        # Initialize with LLM server
        llm_server = await get_llm_server()
        await processor.initialize(llm_server)
        
        logger.info("✅ Advanced memory processor initialized")
        
        # Test with animals and souls conversation
        user_prompt = "Do you think animals have souls?"
        assistant_response = """That's a profound philosophical question. Many belief systems and philosophical traditions have different perspectives on whether animals have souls. Some religions like Hinduism and Buddhism suggest all living beings have spiritual essence, while others make distinctions between human souls and animal consciousness. From a scientific perspective, we can observe complex emotions, social bonds, and even apparent empathy in many animals, suggesting rich inner lives. What's your own perspective on this?"""
        
        session_id = "animals_souls_debug_test"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing full pipeline with: {user_prompt}")
        logger.info(f"{'='*60}")
        
        # Process through complete pipeline
        result = await processor.process_with_retry(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            session_id=session_id
        )
        
        # Log complete results
        logger.info(f"\n📊 PIPELINE RESULTS:")
        logger.info(f"  - Success: {result.success}")
        logger.info(f"  - Memories stored: {result.memories_stored}")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Error message: {result.error_message}")
        
        if result.stage_timings:
            logger.info(f"  - Stage timings:")
            for stage, timing in result.stage_timings.items():
                logger.info(f"    * {stage}: {timing:.2f}s")
        
        if result.content_analysis:
            logger.info(f"  - Content analysis:")
            logger.info(f"    * Memory worthy: {result.content_analysis.memory_worthy}")
            logger.info(f"    * Importance: {result.content_analysis.importance_score:.2f}")
            logger.info(f"    * Categories: {result.content_analysis.categories}")
            logger.info(f"    * Key facts: {len(result.content_analysis.key_facts)} facts")
        
        if result.extraction_stats:
            logger.info(f"  - Extraction stats: {result.extraction_stats}")
        
        # Check what's actually in the database
        logger.info(f"\n🔍 CHECKING DATABASE:")
        
        # Get recent memories
        recent_memories = await app_state.personal_memory.get_relevant_memories("", limit=10)
        logger.info(f"  - Recent memories count: {len(recent_memories)}")
        
        for i, memory in enumerate(recent_memories[:3]):
            if hasattr(memory, 'content'):
                content_preview = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content
                logger.info(f"    {i+1}. {content_preview}")
        
        # Get core memories
        core_memories = await app_state.personal_memory.get_all_core_memories()
        logger.info(f"  - Core memories count: {len(core_memories) if core_memories else 0}")
        
        if core_memories:
            for key, value in list(core_memories.items())[:3]:
                value_preview = value[:100] + "..." if len(str(value)) > 100 else str(value)
                logger.info(f"    {key}: {value_preview}")
        
        # Test with a second conversation to see accumulation
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing second conversation...")
        logger.info(f"{'='*60}")
        
        user_prompt_2 = "I believe animals do have souls. My dog seems to understand my emotions deeply."
        assistant_response_2 = """Your observation about your dog's emotional understanding resonates with many pet owners and animal behaviorists. Dogs have co-evolved with humans for thousands of years, developing remarkable abilities to read our facial expressions, body language, and even emotional states. Whether we call this a 'soul' or advanced emotional intelligence, it's clear that many animals form deep bonds and demonstrate consciousness that goes beyond simple instinct. Your belief reflects a compassionate view that many philosophical and spiritual traditions would support."""
        
        result_2 = await processor.process_with_retry(
            user_prompt=user_prompt_2,
            assistant_response=assistant_response_2,
            session_id=session_id + "_2"
        )
        
        logger.info(f"Second conversation results:")
        logger.info(f"  - Success: {result_2.success}")
        logger.info(f"  - Memories stored: {result_2.memories_stored}")
        
        # Final database check
        logger.info(f"\n🔍 FINAL DATABASE CHECK:")
        recent_memories_final = await app_state.personal_memory.get_relevant_memories("", limit=15)
        logger.info(f"  - Total recent memories: {len(recent_memories_final)}")
        
        # Look for animal/soul related memories
        animal_memories = []
        for memory in recent_memories_final:
            content = memory.content if hasattr(memory, 'content') else str(memory)
            if any(keyword in content.lower() for keyword in ['animal', 'soul', 'dog', 'philosophical']):
                animal_memories.append(content)
        
        logger.info(f"  - Animal/soul related memories found: {len(animal_memories)}")
        for i, content in enumerate(animal_memories):
            content_preview = content[:150] + "..." if len(content) > 150 else content
            logger.info(f"    {i+1}. {content_preview}")
        
    except Exception as e:
        logger.error(f"Error in debug pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(debug_full_pipeline())