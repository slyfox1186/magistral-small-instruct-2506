#!/usr/bin/env python3
"""Test script for the memory importance scorer."""

import sys

sys.path.append('.')

from memory.importance_scorer import MemoryImportanceScorer


def test_importance_scorer():
    """Test various messages to validate the scoring algorithm."""
    scorer = MemoryImportanceScorer()

    # Test messages with expected importance ranges
    test_cases = [
        # (message, role, expected_min, expected_max, description)
        ("Hi", "user", 0.0, 0.3, "Simple greeting"),
        ("Hello, how are you?", "user", 0.0, 0.3, "Casual greeting"),
        ("Thanks!", "user", 0.0, 0.3, "Simple thanks"),

        ("What's the weather like today?", "user", 0.3, 0.6, "Basic question"),
        ("Can you explain how photosynthesis works?", "user", 0.4, 0.7, "Educational question"),

        ("Remember that my birthday is on July 15th", "user", 0.7, 1.0, "Personal info with keyword"),
        ("My email is john.doe@example.com", "user", 0.8, 1.0, "PII - email"),
        ("My phone number is 555-123-4567", "user", 0.8, 1.0, "PII - phone"),
        ("Create a Python script to analyze sales data", "user", 0.6, 0.9, "Instruction with verb"),
        ("CRITICAL: The server is down and needs immediate attention", "user", 0.7, 1.0, "Critical keyword"),

        ("The weather today is sunny with a high of 75°F", "assistant", 0.3, 0.6, "Basic response"),
        ("I'll remember that your birthday is July 15th", "assistant", 0.7, 1.0, "Confirmation of personal info"),
    ]

    # Test with conversation history
    conversation_history = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    print("=" * 80)
    print("MEMORY IMPORTANCE SCORER TEST RESULTS")
    print("=" * 80)
    print()

    # Test individual messages
    print("Individual Message Tests:")
    print("-" * 80)

    for message, role, min_expected, max_expected, description in test_cases:
        score = scorer.calculate_importance(message, role)
        status = "✓ PASS" if min_expected <= score <= max_expected else "✗ FAIL"
        print(f"{status} | {description:40} | Score: {score:.3f} | Expected: {min_expected:.1f}-{max_expected:.1f}")
        print(f"     Message: '{message[:60]}{'...' if len(message) > 60 else ''}'")
        print()

    # Test with conversation context
    print("\nConversation Context Tests:")
    print("-" * 80)

    # Test answer boost
    answer_score = scorer.calculate_importance(
        "Paris is the capital and largest city of France.",
        "assistant",
        conversation_history
    )
    print(f"Answer to question score: {answer_score:.3f} (should be boosted)")

    # Test topic shift
    history_with_shift = conversation_history + [
        {"role": "user", "content": "Now let's talk about machine learning algorithms"}
    ]
    shift_score = scorer.calculate_importance(
        "What is the difference between supervised and unsupervised learning?",
        "user",
        history_with_shift
    )
    print(f"Topic shift score: {shift_score:.3f} (should be higher due to topic change)")

    print("\n" + "=" * 80)
    print("Test completed!")

    # Check if spaCy is available
    if scorer.nlp:
        print("\n✓ Advanced NLP features (spaCy) are available")
    else:
        print("\n⚠ Running in heuristic-only mode (spaCy not available)")


if __name__ == "__main__":
    test_importance_scorer()
