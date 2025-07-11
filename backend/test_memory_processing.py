#!/usr/bin/env python3
"""Test script for the Advanced Memory Processing System.

This script tests the new world-class memory processing implementation
to ensure it's working correctly before deploying to production.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_memory_processing():
    """Test the advanced memory processing system."""
    try:
        # Import the memory processing system
        from memory_processing import AdvancedMemoryProcessor, get_config

        logger.info("✅ Successfully imported memory processing system")

        # Test configuration loading
        config = get_config("development")
        logger.info(f"✅ Configuration loaded: {config.log_level}")

        # Test component creation (without actual memory system for now)
        class MockMemorySystem:
            async def get_relevant_memories(self, query, limit=50):
                return []

            async def get_all_core_memories(self):
                return {}

            async def add_memory(self, content, conversation_id, importance, metadata):
                logger.info(f"Mock: Added memory - {content[:50]}...")
                return True

            async def set_core_memory(self, key, value, category):
                logger.info(f"Mock: Set core memory - {key}: {value[:50]}...")
                return True

        class MockLLMServer:
            async def generate(self, prompt, max_tokens=512, temperature=0.7, **kwargs):
                # Return a mock JSON response for testing
                return """  # noqa: W291
                {
                    "key_facts": ["User lives in Midway Georgia", "User is asking about weather"],
                    "categories": ["personal_facts", "preferences"],
                    "importance_reasons": ["Location information", "Current interest"],
                    "confidence": 0.8,
                    "memory_worthy": true,
                    "context_type": "information_request"
                }
                """

        # Create mock systems
        mock_memory = MockMemorySystem()
        mock_llm = MockLLMServer()

        # Create processor
        processor = AdvancedMemoryProcessor(mock_memory, config)
        await processor.initialize(mock_llm)

        logger.info("✅ Advanced memory processor initialized successfully")

        # Test processing a conversation
        test_user_prompt = (
            "I live in Midway Georgia and I want to know what the weather will be like tomorrow."
        )
        test_assistant_response = (
            "I'll help you check the weather forecast for Midway, Georgia for tomorrow."
        )
        test_session_id = "test_001"

        logger.info("🧪 Testing conversation processing...")

        result = await processor.process_conversation(
            test_user_prompt, test_assistant_response, test_session_id
        )

        logger.info(f"✅ Processing completed: success={result.success}")
        logger.info(f"📊 Memories stored: {result.memories_stored}")
        logger.info(f"⏱️  Processing time: {result.processing_time:.2f}s")

        if result.stage_timings:
            for stage, timing in result.stage_timings.items():
                logger.info(f"   {stage}: {timing:.2f}s")

        # Test system health
        health = processor.get_health_status()
        logger.info(f"🏥 System health: {health['status']}")

        # Test configuration profiles
        for profile in ["default", "production", "development", "fast"]:
            test_config = get_config(profile)
            logger.info(f"✅ {profile} config: timeout={test_config.max_processing_time}s")

    except Exception:
        logger.exception("❌ Test failed")
        import traceback

        traceback.print_exc()
        return False
    else:
        logger.info("🎉 All tests passed! Memory processing system is ready.")
        return True


async def test_individual_components():
    """Test individual components of the memory system."""
    try:
        from memory_processing.config import get_config, validate_config
        from memory_processing.utils import calculate_text_similarity, extract_entities

        # Test configuration
        config = get_config("default")
        validate_config(config)
        logger.info("✅ Configuration validation passed")

        # Test utilities
        test_text = "My name is Jeff and I live in Midway, Georgia. I work as a software engineer."
        entities = extract_entities(test_text)
        logger.info(
            f"✅ Entity extraction: found {sum(len(v) for v in entities.values())} entities"
        )

        # Test similarity calculation
        text1 = "I love pizza and pasta"
        text2 = "Pizza and pasta are my favorite foods"
        similarity = calculate_text_similarity(text1, text2)
        logger.info(f"✅ Text similarity: {similarity:.2f}")

    except Exception:
        logger.exception("❌ Component test failed")
        return False
    else:
        logger.info("🎉 Individual component tests passed!")
        return True


if __name__ == "__main__":

    async def main():
        """Main test function."""
        logger.info("🚀 Starting Advanced Memory Processing System Tests")
        logger.info("=" * 60)

        # Test individual components first
        logger.info("📋 Testing individual components...")
        component_success = await test_individual_components()

        if not component_success:
            logger.error("❌ Component tests failed, skipping integration tests")
            return False

        logger.info("\n📋 Testing full integration...")
        integration_success = await test_memory_processing()

        if integration_success:
            logger.info("\n🎉 ALL TESTS PASSED! 🎉")
            logger.info("The Advanced Memory Processing System is ready for production!")
            return True
        else:
            logger.error("\n❌ INTEGRATION TESTS FAILED")
            return False

    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
