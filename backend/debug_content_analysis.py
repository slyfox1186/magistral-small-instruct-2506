#!/usr/bin/env python3
"""
Debug script to test content analysis with philosophical conversations
"""

import asyncio
import logging
import sys
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def debug_content_analysis():
    """Test content analysis with philosophical conversations"""
    
    try:
        # Import memory processing components
        from memory_processing import get_config
        from memory_processing.content_analyzer import ContentAnalyzer
        from persistent_llm_server import get_llm_server
        
        # Test conversations about animals and souls
        test_conversations = [
            {
                "user_prompt": "Do you think animals have souls?",
                "assistant_response": "That's a profound philosophical question. Many belief systems and philosophical traditions have different perspectives on whether animals have souls. Some religions like Hinduism and Buddhism suggest all living beings have spiritual essence, while others make distinctions between human souls and animal consciousness. From a scientific perspective, we can observe complex emotions, social bonds, and even apparent empathy in many animals, suggesting rich inner lives. What's your own perspective on this?"
            },
            {
                "user_prompt": "I believe animals do have souls. My dog seems to understand my emotions deeply.",
                "assistant_response": "Your observation about your dog's emotional understanding resonates with many pet owners and animal behaviorists. Dogs have co-evolved with humans for thousands of years, developing remarkable abilities to read our facial expressions, body language, and even emotional states. Whether we call this a 'soul' or advanced emotional intelligence, it's clear that many animals form deep bonds and demonstrate consciousness that goes beyond simple instinct. Your belief reflects a compassionate view that many philosophical and spiritual traditions would support."
            }
        ]
        
        # Get production config to see current thresholds
        config = get_config('production')
        logger.info(f"Using production config with thresholds:")
        logger.info(f"  - min_confidence_threshold: {config.min_confidence_threshold}")
        logger.info(f"  - high_confidence_threshold: {config.high_confidence_threshold}")
        logger.info(f"  - personal_info_boost: {config.personal_info_boost}")
        
        # Initialize LLM server and content analyzer
        logger.info("Initializing LLM server...")
        llm_server = await get_llm_server()
        
        logger.info("Initializing content analyzer...")
        analyzer = ContentAnalyzer(llm_server, config)
        
        # Test each conversation
        for i, conv in enumerate(test_conversations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing conversation {i+1}: {conv['user_prompt'][:50]}...")
            logger.info(f"{'='*60}")
            
            # Analyze content
            analysis = await analyzer.analyze_content(
                conv['user_prompt'], 
                conv['assistant_response'], 
                f"debug_session_{i+1}"
            )
            
            # Log detailed results
            logger.info(f"Analysis Results:")
            logger.info(f"  - Memory worthy: {analysis.memory_worthy}")
            logger.info(f"  - Importance score: {analysis.importance_score:.3f}")
            logger.info(f"  - Confidence: {analysis.confidence:.3f}")
            logger.info(f"  - Categories: {analysis.categories}")
            logger.info(f"  - Key facts: {analysis.key_facts}")
            logger.info(f"  - Context type: {analysis.context_type}")
            logger.info(f"  - Processing time: {analysis.processing_time:.3f}s")
            
            # Show why it might not be memory worthy
            if not analysis.memory_worthy:
                logger.warning(f"❌ NOT MEMORY WORTHY because:")
                if analysis.importance_score < config.min_confidence_threshold:
                    logger.warning(f"  - Importance score {analysis.importance_score:.3f} < min threshold {config.min_confidence_threshold}")
                if not analysis.key_facts:
                    logger.warning(f"  - No key facts extracted")
                high_value_categories = ['personal_facts', 'relationships', 'goals']
                has_high_value = any(cat in high_value_categories for cat in analysis.categories)
                if not has_high_value and analysis.importance_score < config.high_confidence_threshold:
                    logger.warning(f"  - No high-value categories AND importance score {analysis.importance_score:.3f} < high threshold {config.high_confidence_threshold}")
            else:
                logger.info(f"✅ MEMORY WORTHY!")
        
        # Test with development config (more lenient)
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with DEVELOPMENT config (more lenient)...")
        logger.info(f"{'='*60}")
        
        dev_config = get_config('development')
        logger.info(f"Development config thresholds:")
        logger.info(f"  - min_confidence_threshold: {dev_config.min_confidence_threshold}")
        logger.info(f"  - high_confidence_threshold: {dev_config.high_confidence_threshold}")
        
        dev_analyzer = ContentAnalyzer(llm_server, dev_config)
        
        # Test first conversation with dev config
        conv = test_conversations[0]
        analysis = await dev_analyzer.analyze_content(
            conv['user_prompt'], 
            conv['assistant_response'], 
            "dev_test_session"
        )
        
        logger.info(f"Development Config Results:")
        logger.info(f"  - Memory worthy: {analysis.memory_worthy}")
        logger.info(f"  - Importance score: {analysis.importance_score:.3f}")
        logger.info(f"  - Categories: {analysis.categories}")
        logger.info(f"  - Key facts: {analysis.key_facts}")
        
    except Exception as e:
        logger.error(f"Error in debug analysis: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(debug_content_analysis())