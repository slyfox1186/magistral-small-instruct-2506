#!/usr/bin/env python3
"""
Example Usage of the Advanced Memory Processing System

This file demonstrates various ways to use the memory processing system
with different configurations and scenarios.
"""

import asyncio
from memory_processing import AdvancedMemoryProcessor
from memory_processing.config import get_config, merge_config

async def basic_usage_example():
    """Basic usage example with default configuration."""
    print("=== Basic Usage Example ===")
    
    # Assume you have a personal memory system instance
    # personal_memory_system = PersonalMemorySystem("path/to/db")
    
    # Create processor with default configuration
    processor = AdvancedMemoryProcessor(personal_memory_system)
    
    # Process a conversation
    result = await processor.process_conversation(
        user_prompt="Hi, my name is Alice and I'm a data scientist at Microsoft",
        assistant_response="Nice to meet you, Alice! Data science at Microsoft sounds exciting. What kind of projects do you work on?",
        session_id="conversation_001"
    )
    
    # Check results
    if result["success"]:
        metrics = result["metrics"]
        print(f"✅ Processed successfully")
        print(f"   Stored {metrics['stored_memories']} memories")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Analysis confidence: {metrics.get('analysis_confidence', 0):.2f}")
    else:
        print(f"❌ Processing failed: {result['errors']}")

async def production_configuration_example():
    """Example using production configuration with custom overrides."""
    print("\n=== Production Configuration Example ===")
    
    # Start with production config and customize
    config = merge_config(
        get_config("production"),
        {
            "min_confidence_threshold": 0.4,     # Slightly lower threshold
            "max_processing_time": 15.0,         # Faster timeout
            "llm_analysis": {
                "temperature": 0.05,             # More deterministic
                "max_tokens": 1024               # Shorter responses
            },
            "logging": {
                "log_level": "INFO",             # Moderate logging
                "metric_logging_interval": 5     # Log every 5 sessions
            }
        }
    )
    
    processor = AdvancedMemoryProcessor(personal_memory_system, config)
    
    # Process multiple conversations
    conversations = [
        {
            "user": "I prefer tea over coffee, especially green tea",
            "assistant": "That's great! Green tea has many health benefits. Do you have a favorite brand?",
            "session": "session_tea"
        },
        {
            "user": "My birthday is coming up on June 15th",
            "assistant": "That's exciting! Are you planning anything special for your birthday?",
            "session": "session_birthday"
        }
    ]
    
    for conv in conversations:
        result = await processor.process_conversation(
            user_prompt=conv["user"],
            assistant_response=conv["assistant"],
            session_id=conv["session"]
        )
        print(f"Session {conv['session']}: {'✅' if result['success'] else '❌'}")

async def fast_processing_example():
    """Example using fast configuration for high-volume scenarios."""
    print("\n=== Fast Processing Example ===")
    
    # Use fast configuration for high-volume processing
    config = get_config("fast")
    processor = AdvancedMemoryProcessor(personal_memory_system, config)
    
    # Simulate high-volume processing
    conversations = [
        ("Hello", "Hi there!", "fast_001"),
        ("What's my name?", "I don't have that information stored yet.", "fast_002"),
        ("I work at Google", "That's interesting! What do you do at Google?", "fast_003"),
        ("I like pizza", "Pizza is delicious! What's your favorite topping?", "fast_004"),
        ("Thanks for the help", "You're welcome! Happy to help.", "fast_005")
    ]
    
    results = []
    start_time = asyncio.get_event_loop().time()
    
    for user_msg, assistant_msg, session in conversations:
        result = await processor.process_conversation(
            user_prompt=user_msg,
            assistant_response=assistant_msg,
            session_id=session
        )
        results.append(result["success"])
    
    end_time = asyncio.get_event_loop().time()
    processing_time = end_time - start_time
    
    success_rate = sum(results) / len(results)
    print(f"Processed {len(conversations)} conversations in {processing_time:.2f}s")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average time per conversation: {processing_time/len(conversations):.3f}s")

async def monitoring_example():
    """Example of monitoring and health checking."""
    print("\n=== Monitoring Example ===")
    
    processor = AdvancedMemoryProcessor(personal_memory_system)
    
    # Process some conversations first
    await processor.process_conversation(
        user_prompt="I live in New York City",
        assistant_response="NYC is an amazing city! Which borough do you live in?",
        session_id="monitor_001"
    )
    
    # Get processing metrics
    metrics = processor.get_processing_metrics()
    print("📊 Processing Metrics:")
    print(f"   Total processed: {metrics['total_processed']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Average processing time: {metrics['average_processing_time']:.2f}s")
    print(f"   Memories stored: {metrics['memories_stored']}")
    print(f"   Duplicates removed: {metrics['duplicates_removed']}")
    
    # Perform health check
    health = await processor.health_check()
    print(f"\n🏥 Health Check:")
    print(f"   Overall status: {health['status']}")
    print(f"   Components:")
    for component, info in health['components'].items():
        status = info.get('status', 'unknown')
        print(f"      {component}: {status}")

async def error_handling_example():
    """Example of error handling and recovery."""
    print("\n=== Error Handling Example ===")
    
    # Create processor with very strict timeouts to trigger errors
    config = merge_config(
        get_config("default"),
        {
            "max_processing_time": 0.001,        # Very short timeout
            "min_confidence_threshold": 0.99     # Very high confidence requirement
        }
    )
    
    processor = AdvancedMemoryProcessor(personal_memory_system, config)
    
    result = await processor.process_conversation(
        user_prompt="This will probably timeout due to the short processing time limit",
        assistant_response="Yes, this is likely to cause a timeout error for demonstration",
        session_id="error_example"
    )
    
    if not result["success"]:
        print("❌ Expected error occurred:")
        for error in result["errors"]:
            print(f"   - {error}")
        print(f"   Processing still attempted for: {result['processing_time']:.3f}s")
    
    # Show that the system continues working with normal config
    processor = AdvancedMemoryProcessor(personal_memory_system)
    result2 = await processor.process_conversation(
        user_prompt="This should work fine with normal settings",
        assistant_response="Yes, this should process successfully",
        session_id="recovery_example"
    )
    
    print(f"✅ Recovery successful: {result2['success']}")

async def custom_configuration_example():
    """Example of creating a custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create a custom configuration for a specific use case
    custom_config = {
        "max_processing_time": 25.0,
        "min_confidence_threshold": 0.35,
        "similarity_thresholds": {
            "personal_fact": 0.98,    # Very strict for facts
            "preference": 0.80,       # Moderate for preferences
            "experience": 0.70,       # Relaxed for experiences
            "relationship": 0.95,     # Very strict for relationships
            "question_answer": 0.75,  # Moderate for Q&A
            "conversation": 0.65      # Relaxed for general conversation
        },
        "llm_analysis": {
            "max_tokens": 1500,
            "temperature": 0.1,
            "max_retries": 2,
            "retry_delay": 1.5
        },
        "importance_scoring": {
            "boosters": {
                "high_confidence": 0.25,      # Boost high confidence more
                "personal_identifier": 0.4,   # Boost personal info significantly
                "emotional_significance": 0.2,
                "temporal_relevance": 0.15,
                "relationship_context": 0.15
            }
        },
        "logging": {
            "log_level": "DEBUG",
            "log_analysis_details": True,
            "log_extraction_details": True,
            "metric_logging_interval": 1
        }
    }
    
    processor = AdvancedMemoryProcessor(personal_memory_system, custom_config)
    
    # Test with a rich conversation that should trigger multiple memory types
    result = await processor.process_conversation(
        user_prompt="Hi! My name is Dr. Sarah Johnson, I'm a cardiologist at Stanford Hospital. "
                   "I love hiking and classical music, especially Bach. My husband Mark and I "
                   "have been married for 8 years and we live in Palo Alto.",
        assistant_response="It's wonderful to meet you, Dr. Johnson! That's quite impressive - "
                          "cardiology at Stanford is such important work. I'd love to hear about "
                          "your hiking adventures, and Bach is such beautiful music. How did you "
                          "and Mark meet?",
        session_id="rich_conversation"
    )
    
    if result["success"]:
        analysis = result["analysis"]
        print("🧠 Rich conversation analysis:")
        print(f"   Personal facts: {analysis.get('personal_facts', 0)}")
        print(f"   Preferences: {analysis.get('preferences', 0)}")
        print(f"   Relationships: {analysis.get('relationships', 0)}")
        print(f"   Topics: {analysis.get('topics_discussed', [])}")
        print(f"   Confidence: {analysis.get('confidence_score', 0):.3f}")

async def main():
    """Run all examples."""
    # Note: These examples assume you have a PersonalMemorySystem instance
    # In practice, you would initialize it like this:
    # from personal_memory_system import PersonalMemorySystem
    # personal_memory_system = PersonalMemorySystem("path/to/your/database.db")
    
    print("Advanced Memory Processing System - Usage Examples")
    print("=" * 60)
    print("Note: These examples assume you have initialized personal_memory_system")
    print("=" * 60)
    
    # Uncomment these to run with actual personal memory system
    # await basic_usage_example()
    # await production_configuration_example()
    # await fast_processing_example()
    # await monitoring_example()
    # await error_handling_example()
    # await custom_configuration_example()
    
    print("\n✅ All examples completed!")
    print("\nTo run these examples with a real memory system:")
    print("1. Initialize PersonalMemorySystem with your database")
    print("2. Uncomment the example function calls above")
    print("3. Run this script: python EXAMPLE_USAGE.py")

if __name__ == "__main__":
    asyncio.run(main())