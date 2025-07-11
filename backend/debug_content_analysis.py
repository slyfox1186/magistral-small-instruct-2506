#!/usr/bin/env python3
"""Debug script to test content analysis with philosophical conversations."""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_test_conversations():
    """Get test conversations for analysis."""
    return [
        {
            "user_prompt": "Do you think animals have souls?",
            "assistant_response": "That's a profound philosophical question. Many belief systems and philosophical traditions have different perspectives on whether animals have souls. Some religions like Hinduism and Buddhism suggest all living beings have spiritual essence, while others make distinctions between human souls and animal consciousness. From a scientific perspective, we can observe complex emotions, social bonds, and even apparent empathy in many animals, suggesting rich inner lives. What's your own perspective on this?",
        },
        {
            "user_prompt": "I believe animals do have souls. My dog seems to understand my emotions deeply.",
            "assistant_response": "Your observation about your dog's emotional understanding resonates with many pet owners and animal behaviorists. Dogs have co-evolved with humans for thousands of years, developing remarkable abilities to read our facial expressions, body language, and even emotional states. Whether we call this a 'soul' or advanced emotional intelligence, it's clear that many animals form deep bonds and demonstrate consciousness that goes beyond simple instinct. Your belief reflects a compassionate view that many philosophical and spiritual traditions would support.",
        },
    ]


def log_config_info(config, config_name="production"):
    """Log configuration information."""
    logger.info(f"Using {config_name} config with thresholds:")
    logger.info(f"  - min_confidence_threshold: {config.min_confidence_threshold}")
    logger.info(f"  - high_confidence_threshold: {config.high_confidence_threshold}")
    if hasattr(config, 'personal_info_boost'):
        logger.info(f"  - personal_info_boost: {config.personal_info_boost}")


def log_analysis_results(analysis):
    """Log detailed analysis results."""
    logger.info("Analysis Results:")
    logger.info(f"  - Memory worthy: {analysis.memory_worthy}")
    logger.info(f"  - Importance score: {analysis.importance_score:.3f}")
    logger.info(f"  - Confidence: {analysis.confidence:.3f}")
    logger.info(f"  - Categories: {analysis.categories}")
    logger.info(f"  - Key facts: {analysis.key_facts}")
    logger.info(f"  - Context type: {analysis.context_type}")
    logger.info(f"  - Processing time: {analysis.processing_time:.3f}s")


def log_memory_worthiness_analysis(analysis, config):
    """Log why content might not be memory worthy."""
    if not analysis.memory_worthy:
        logger.warning("❌ NOT MEMORY WORTHY because:")
        if analysis.importance_score < config.min_confidence_threshold:
            logger.warning(
                f"  - Importance score {analysis.importance_score:.3f} < min threshold {config.min_confidence_threshold}"
            )
        if not analysis.key_facts:
            logger.warning("  - No key facts extracted")

        high_value_categories = ["personal_facts", "relationships", "goals"]
        has_high_value = any(cat in high_value_categories for cat in analysis.categories)
        if not has_high_value and analysis.importance_score < config.high_confidence_threshold:
            logger.warning(
                f"  - No high-value categories AND importance score {analysis.importance_score:.3f} < high threshold {config.high_confidence_threshold}"
            )
    else:
        logger.info("✅ MEMORY WORTHY!")


async def test_with_production_config(analyzer, test_conversations, config):
    """Test conversations with production configuration."""
    for i, conv in enumerate(test_conversations):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing conversation {i + 1}: {conv['user_prompt'][:50]}...")
        logger.info(f"{'=' * 60}")

        analysis = await analyzer.analyze_content(
            conv["user_prompt"], conv["assistant_response"], f"debug_session_{i + 1}"
        )

        log_analysis_results(analysis)
        log_memory_worthiness_analysis(analysis, config)


async def test_with_dev_config(llm_server, test_conversations):
    """Test with development configuration."""
    from memory_processing import get_config
    from memory_processing.content_analyzer import ContentAnalyzer

    logger.info(f"\n{'=' * 60}")
    logger.info("Testing with DEVELOPMENT config (more lenient)...")
    logger.info(f"{'=' * 60}")

    dev_config = get_config("development")
    log_config_info(dev_config, "development")

    dev_analyzer = ContentAnalyzer(llm_server, dev_config)
    conv = test_conversations[0]
    analysis = await dev_analyzer.analyze_content(
        conv["user_prompt"], conv["assistant_response"], "dev_test_session"
    )

    logger.info("Development Config Results:")
    logger.info(f"  - Memory worthy: {analysis.memory_worthy}")
    logger.info(f"  - Importance score: {analysis.importance_score:.3f}")
    logger.info(f"  - Categories: {analysis.categories}")
    logger.info(f"  - Key facts: {analysis.key_facts}")


async def debug_content_analysis():
    """Test content analysis with philosophical conversations."""
    try:
        from memory_processing import get_config
        from memory_processing.content_analyzer import ContentAnalyzer
        from persistent_llm_server import get_llm_server

        test_conversations = get_test_conversations()
        config = get_config("production")
        log_config_info(config)

        logger.info("Initializing LLM server...")
        llm_server = await get_llm_server()

        logger.info("Initializing content analyzer...")
        analyzer = ContentAnalyzer(llm_server, config)

        await test_with_production_config(analyzer, test_conversations, config)
        await test_with_dev_config(llm_server, test_conversations)

    except Exception as e:
        logger.error(f"Error in debug analysis: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(debug_content_analysis())
