#!/usr/bin/env python3
"""Debug script to test the complete memory processing pipeline."""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for content preview lengths
CONTENT_PREVIEW_LENGTH = 100
CORE_VALUE_PREVIEW_LENGTH = 100
ANIMAL_MEMORY_PREVIEW_LENGTH = 150


async def initialize_processor():
    """Initialize the memory processor with dependencies."""
    from memory_processing import AdvancedMemoryProcessor, get_config
    from modules.globals import app_state
    from persistent_llm_server import get_llm_server

    logger.info(f"Personal memory system: {app_state.personal_memory}")

    if not app_state.personal_memory:
        logger.error("❌ Personal memory system not available!")
        return None, None

    config = get_config("development")
    processor = AdvancedMemoryProcessor(app_state.personal_memory, config)

    llm_server = await get_llm_server()
    await processor.initialize(llm_server)

    logger.info("✅ Advanced memory processor initialized")
    return processor, llm_server


def get_test_conversations():
    """Get test conversations for pipeline testing."""
    return [
        {
            "user_prompt": "Do you think animals have souls?",
            "assistant_response": """That's a profound philosophical question. Many belief systems and philosophical traditions have different perspectives on whether animals have souls. Some religions like Hinduism and Buddhism suggest all living beings have spiritual essence, while others make distinctions between human souls and animal consciousness. From a scientific perspective, we can observe complex emotions, social bonds, and even apparent empathy in many animals, suggesting rich inner lives. What's your own perspective on this?""",
            "session_id": "animals_souls_debug_test"
        },
        {
            "user_prompt": "I believe animals do have souls. My dog seems to understand my emotions deeply.",
            "assistant_response": """Your observation about your dog's emotional understanding resonates with many pet owners and animal behaviorists. Dogs have co-evolved with humans for thousands of years, developing remarkable abilities to read our facial expressions, body language, and even emotional states. Whether we call this a 'soul' or advanced emotional intelligence, it's clear that many animals form deep bonds and demonstrate consciousness that goes beyond simple instinct. Your belief reflects a compassionate view that many philosophical and spiritual traditions would support.""",
            "session_id": "animals_souls_debug_test_2"
        }
    ]


def log_pipeline_results(result):
    """Log the results from pipeline processing."""
    logger.info("\n📊 PIPELINE RESULTS:")
    logger.info(f"  - Success: {result.success}")
    logger.info(f"  - Memories stored: {result.memories_stored}")
    logger.info(f"  - Processing time: {result.processing_time:.2f}s")
    logger.info(f"  - Error message: {result.error_message}")

    if result.stage_timings:
        logger.info("  - Stage timings:")
        for stage, timing in result.stage_timings.items():
            logger.info(f"    * {stage}: {timing:.2f}s")

    if result.content_analysis:
        logger.info("  - Content analysis:")
        logger.info(f"    * Memory worthy: {result.content_analysis.memory_worthy}")
        logger.info(f"    * Importance: {result.content_analysis.importance_score:.2f}")
        logger.info(f"    * Categories: {result.content_analysis.categories}")
        logger.info(f"    * Key facts: {len(result.content_analysis.key_facts)} facts")

    if result.extraction_stats:
        logger.info(f"  - Extraction stats: {result.extraction_stats}")


async def check_database_contents():
    """Check what memories are stored in the database."""
    from modules.globals import app_state

    logger.info("\n🔍 CHECKING DATABASE:")

    recent_memories = await app_state.personal_memory.get_relevant_memories("", limit=10)
    logger.info(f"  - Recent memories count: {len(recent_memories)}")

    for i, memory in enumerate(recent_memories[:3]):
        if hasattr(memory, "content"):
            content_preview = (
                memory.content[:CONTENT_PREVIEW_LENGTH] + "..."
                if len(memory.content) > CONTENT_PREVIEW_LENGTH
                else memory.content
            )
            logger.info(f"    {i + 1}. {content_preview}")

    core_memories = await app_state.personal_memory.get_all_core_memories("debug")
    logger.info(f"  - Core memories count: {len(core_memories) if core_memories else 0}")

    if core_memories:
        for key, value in list(core_memories.items())[:3]:
            value_preview = (
                value[:CORE_VALUE_PREVIEW_LENGTH] + "..."
                if len(str(value)) > CORE_VALUE_PREVIEW_LENGTH
                else str(value)
            )
            logger.info(f"    {key}: {value_preview}")


async def check_final_database():
    """Check final database state after all processing."""
    from modules.globals import app_state

    logger.info("\n🔍 FINAL DATABASE CHECK:")
    recent_memories_final = await app_state.personal_memory.get_relevant_memories("", limit=15)
    logger.info(f"  - Total recent memories: {len(recent_memories_final)}")

    animal_memories = []
    for memory in recent_memories_final:
        content = memory.content if hasattr(memory, "content") else str(memory)
        if any(keyword in content.lower() for keyword in ["animal", "soul", "dog", "philosophical"]):
            animal_memories.append(content)

    logger.info(f"  - Animal/soul related memories found: {len(animal_memories)}")
    for i, content in enumerate(animal_memories):
        content_preview = (
            content[:ANIMAL_MEMORY_PREVIEW_LENGTH] + "..."
            if len(content) > ANIMAL_MEMORY_PREVIEW_LENGTH
            else content
        )
        logger.info(f"    {i + 1}. {content_preview}")


async def process_conversation(processor, conversation, conversation_num):
    """Process a single conversation through the pipeline."""
    logger.info(f"\n{'=' * 60}")
    if conversation_num == 1:
        logger.info(f"Testing full pipeline with: {conversation['user_prompt']}")
    else:
        logger.info("Testing second conversation...")
    logger.info(f"{'=' * 60}")

    result = await processor.process_with_retry(
        user_prompt=conversation["user_prompt"],
        assistant_response=conversation["assistant_response"],
        session_id=conversation["session_id"]
    )

    if conversation_num == 1:
        log_pipeline_results(result)
    else:
        logger.info("Second conversation results:")
        logger.info(f"  - Success: {result.success}")
        logger.info(f"  - Memories stored: {result.memories_stored}")

    return result


async def debug_full_pipeline():
    """Test the complete memory processing pipeline."""
    try:
        processor, llm_server = await initialize_processor()
        if not processor:
            return

        conversations = get_test_conversations()

        # Process first conversation
        await process_conversation(processor, conversations[0], 1)
        await check_database_contents()

        # Process second conversation
        await process_conversation(processor, conversations[1], 2)
        await check_final_database()

    except Exception as e:
        logger.error(f"Error in debug pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(debug_full_pipeline())
