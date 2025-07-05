# Memory Importance Scoring System

This document describes the sophisticated algorithm for calculating memory importance scores in the AI assistant's memory system.

## Overview

The memory importance scoring system dynamically calculates importance scores for each message based on multiple factors including content analysis, conversation context, NLP features, and PII detection. This replaces the previous hardcoded values (0.7 for user messages, 0.8 for assistant responses).

## Score Ranges

- **0.0-0.3**: Low importance (greetings, fillers, acknowledgments)
- **0.3-0.6**: Medium importance (general conversation, basic questions)
- **0.6-0.8**: High importance (questions, answers, facts, instructions)
- **0.8-1.0**: Critical importance (personal info, PII, key decisions)

## Scoring Components

### 1. Base Score
- User messages: 0.2
- Assistant messages: 0.25

### 2. Content Analysis Features

#### Message Length Score (weight: 0.05)
- Messages < 20 chars: 0.0
- Messages 20-200 chars: Linear scaling
- Messages > 200 chars: 1.0

#### Question Detection (weight: 0.15)
- Detects questions by:
  - Question mark at end
  - Question words at start (who, what, where, when, why, how, etc.)
- Score: 1.0 if question, 0.0 otherwise

#### Keyword Matching (weight: 0.25)
- High importance keywords: "remember", "important", "critical", "my name is", etc.
- Low importance keywords: "hello", "thanks", "ok", "bye", etc.
- Multiple keywords increase/decrease score proportionally

#### Instructional Verbs (weight: 0.20)
- Detects commands: "create", "generate", "analyze", "build", etc.
- Multiple verbs increase score

#### PII Detection (weight: 0.40)
- Regex patterns for:
  - Email addresses
  - Phone numbers
  - Social Security Numbers
  - Credit card numbers
  - Dates
- Score: 1.0 if any PII detected

### 3. NLP Features (when spaCy is available)

#### Named Entity Recognition (weight: 0.30)
- Important entities: PERSON, ORG, GPE, LOC, DATE, MONEY, TIME, PERCENT
- Score based on number of entities found

#### Sentiment Analysis (weight: 0.15)
- Uses sentiment polarity magnitude (not direction)
- Higher emotional content = higher importance

#### Topic Shift Detection (weight: 0.20)
- Compares semantic similarity to previous message
- Low similarity (<0.6) indicates topic shift = 1.0 score
- Medium similarity (0.6-0.8) = 0.5 score

### 4. Contextual Features

#### Answer Boost (weight: 0.10)
- Assistant responses to user questions get bonus score
- Helps preserve Q&A pairs

## Implementation

### Installation

```bash
# Install required dependencies
pip install spacy spacytextblob

# Download spaCy model
python -m spacy download en_core_web_md

# Download TextBlob corpora
python -m textblob.download_corpora
```

### Usage

```python
from memory.importance_scorer import get_importance_scorer

# Get singleton instance
scorer = get_importance_scorer()

# Calculate importance
importance = scorer.calculate_importance(
    text="Remember that my email is john@example.com",
    role="user",
    conversation_history=[
        {"role": "assistant", "content": "How can I help you today?"},
        {"role": "user", "content": "I need to update my contact info"}
    ]
)
# Result: ~0.9 (high importance due to PII + keyword)
```

### Integration with Memory System

The scorer is integrated into the chat routes to automatically calculate importance for every message:

```python
# In chat_routes.py
user_importance = await calculate_message_importance(
    content=user_prompt,
    role="user",
    session_id=session_id,
    messages=request.messages
)

await app_state.personal_memory.add_memory(
    content=f"User: {user_prompt}",
    conversation_id=session_id,
    importance=user_importance,
)
```

## Dynamic Importance Evolution

The system also includes functionality to update importance scores based on access patterns:

1. When memories are retrieved, their importance increases slightly
2. Boost formula: `new_importance = current + (1.0 - current) * 0.05`
3. This creates a feedback loop where frequently accessed memories stay relevant

## Configuration

The scorer can be configured with custom weights:

```python
config = {
    "length_weight": 0.05,
    "question_weight": 0.15,
    "keyword_weight": 0.25,
    "instruction_weight": 0.20,
    "entity_weight": 0.30,
    "sentiment_weight": 0.15,
    "pii_weight": 0.40,
    "topic_shift_weight": 0.20,
    "answer_boost": 0.10,
    "base_score_user": 0.20,
    "base_score_assistant": 0.25
}

scorer = MemoryImportanceScorer(config)
```

## Performance Considerations

- **Without spaCy**: ~1-2ms per message (heuristics only)
- **With spaCy**: ~50-200ms per message (includes NLP processing)
- The scorer uses a singleton pattern to avoid reloading models
- NLP processing runs only once per message
- Background importance updates don't block retrieval

## Future Enhancements

1. **Machine Learning Model**: Train a classifier on human-labeled importance scores
2. **User Feedback Loop**: Allow users to mark messages as important/unimportant
3. **Domain-Specific Rules**: Add specialized scoring for specific use cases
4. **Multi-Language Support**: Extend to support languages beyond English
5. **Conversation-Level Scoring**: Score entire conversation threads
6. **Time Decay Functions**: Implement sophisticated temporal importance decay