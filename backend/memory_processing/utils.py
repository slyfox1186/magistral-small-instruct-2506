"""Utility functions for memory processing system

Provides helper functions for text processing, similarity calculation,
entity extraction, and memory formatting.
"""

import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract basic entities from text using pattern matching
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with entity types as keys and lists of entities as values
    """
    entities = {
        'names': [],
        'locations': [],
        'dates': [],
        'emails': [],
        'phones': [],
        'urls': [],
        'numbers': []
    }

    # Extract names (capitalized words, common patterns)
    name_patterns = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',  # Title Name
        r'\bmy name is\s+([A-Z][a-z]+)\b',  # "my name is John"
        r'\bI\'m\s+([A-Z][a-z]+)\b',  # "I'm John"
        r'\bcall me\s+([A-Z][a-z]+)\b'  # "call me John"
    ]

    for pattern in name_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['names'].extend(matches)

    # Extract locations
    location_patterns = [
        r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, State
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City Name, State
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # "in CityName"
        r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # "from CityName"
        r'\blive\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'  # "live in CityName"
    ]

    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['locations'].extend(matches)

    # Extract dates
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['dates'].extend(matches)

    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)

    # Extract phone numbers
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # XXX.XXX.XXXX
        r'\b\d{10}\b'  # XXXXXXXXXX
    ]

    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        entities['phones'].extend(matches)

    # Extract URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+|www\.[^\s<>"{}|\\^`[\]]+\.[a-z]{2,}'
    entities['urls'] = re.findall(url_pattern, text, re.IGNORECASE)

    # Extract numbers
    number_pattern = r'\b\d+(?:\.\d+)?\b'
    entities['numbers'] = re.findall(number_pattern, text)

    # Clean up and deduplicate
    for entity_type, entity_list in entities.items():
        entities[entity_type] = list(set(entity_list))  # Remove duplicates

    return entities

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using multiple methods
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Normalize texts
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()

    if text1_norm == text2_norm:
        return 1.0

    # Calculate different similarity measures
    scores = []

    # 1. Sequence matcher (character-level similarity)
    seq_similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
    scores.append(seq_similarity)

    # 2. Word-level similarity
    words1 = set(text1_norm.split())
    words2 = set(text2_norm.split())

    if words1 or words2:
        word_similarity = len(words1 & words2) / len(words1 | words2)
        scores.append(word_similarity)

    # 3. Jaccard similarity (character n-grams)
    def get_ngrams(text, n=3):
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(text1_norm)
    ngrams2 = get_ngrams(text2_norm)

    if ngrams1 or ngrams2:
        ngram_similarity = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
        scores.append(ngram_similarity)

    # 4. Length-weighted similarity
    max_len = max(len(text1_norm), len(text2_norm))
    min_len = min(len(text1_norm), len(text2_norm))

    if max_len > 0:
        length_similarity = min_len / max_len
        scores.append(length_similarity)

    # Return weighted average
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0.0

def format_memory_content(content: str, max_length: int = 500) -> str:
    """Format memory content for storage
    
    Args:
        content: Raw memory content
        max_length: Maximum length for formatted content
        
    Returns:
        Formatted memory content
    """
    if not content:
        return ""

    # Clean up whitespace
    content = ' '.join(content.split())

    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length-3] + "..."

    return content

def extract_personal_info_patterns(text: str) -> dict[str, str]:
    """Extract personal information using pattern matching
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with personal information types and values
    """
    info = {}

    # Age patterns
    age_patterns = [
        r'\bi am\s+(\d+)\s+years?\s+old\b',
        r'\bmy age is\s+(\d+)\b',
        r'\bi\'m\s+(\d+)\b',
        r'\bturned\s+(\d+)\b'
    ]

    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info['age'] = match.group(1)
            break

    # Job/profession patterns
    job_patterns = [
        r'\bi work as\s+(?:a|an)?\s*([a-zA-Z\s]+)\b',
        r'\bi am\s+(?:a|an)?\s*([a-zA-Z\s]+)\b',
        r'\bmy job is\s+([a-zA-Z\s]+)\b',
        r'\bi\'m\s+(?:a|an)?\s*([a-zA-Z\s]+)\b'
    ]

    for pattern in job_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            job = match.group(1).strip()
            # Filter out common false positives
            if job not in ['good', 'fine', 'ok', 'here', 'there', 'new', 'old'] and len(job) > 2:
                info['job'] = job
                break

    # Relationship patterns
    relationship_patterns = [
        r'\bmy (?:wife|husband|spouse)\s+(?:is\s+)?([A-Z][a-z]+)\b',
        r'\bmy (?:girlfriend|boyfriend|partner)\s+(?:is\s+)?([A-Z][a-z]+)\b',
        r'\bmarried to\s+([A-Z][a-z]+)\b',
        r'\bmy (?:mom|mother|dad|father|parent)\s+(?:is\s+)?([A-Z][a-z]+)\b'
    ]

    for pattern in relationship_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'relationships' not in info:
                info['relationships'] = []
            info['relationships'].append(match.group(1))

    return info

def detect_emotional_content(text: str) -> dict[str, Any]:
    """Detect emotional content in text
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with emotion analysis results
    """
    emotions = {
        'positive': ['happy', 'joy', 'excited', 'love', 'amazing', 'wonderful', 'great', 'fantastic', 'awesome'],
        'negative': ['sad', 'angry', 'frustrated', 'hate', 'terrible', 'awful', 'disappointed', 'worried', 'scared'],
        'neutral': ['think', 'believe', 'consider', 'suppose', 'maybe', 'perhaps']
    }

    text_lower = text.lower()
    emotion_scores = {}

    for emotion_type, keywords in emotions.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion_type] = score

    # Detect intensity markers
    intensity_markers = ['very', 'extremely', 'really', 'so', 'quite', 'absolutely', 'totally']
    intensity_score = sum(1 for marker in intensity_markers if marker in text_lower)

    return {
        'emotions': emotion_scores,
        'intensity': intensity_score,
        'has_emotional_content': bool(emotion_scores),
        'dominant_emotion': max(emotion_scores, key=emotion_scores.get) if emotion_scores else None
    }

def validate_memory_content(content: str, min_length: int = 10, max_length: int = 2000) -> tuple[bool, str]:
    """Validate memory content before storage
    
    Args:
        content: Memory content to validate
        min_length: Minimum content length
        max_length: Maximum content length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not content or not content.strip():
        return False, "Content is empty or whitespace only"

    content = content.strip()

    if len(content) < min_length:
        return False, f"Content too short (minimum {min_length} characters)"

    if len(content) > max_length:
        return False, f"Content too long (maximum {max_length} characters)"

    # Check for potentially problematic content
    if re.search(r'^[^\w\s]*$', content):
        return False, "Content contains only special characters"

    return True, ""

def create_memory_summary(memories: list[dict[str, Any]], max_length: int = 200) -> str:
    """Create a summary of multiple memories
    
    Args:
        memories: List of memory dictionaries
        max_length: Maximum summary length
        
    Returns:
        Summary string
    """
    if not memories:
        return ""

    # Extract key information from memories
    topics = []
    entities = []

    for memory in memories:
        content = memory.get('content', '')
        if content:
            # Extract topics (simple keyword extraction)
            words = content.lower().split()
            topics.extend([word for word in words if len(word) > 3])

            # Extract entities
            memory_entities = extract_entities(content)
            for entity_type, entity_list in memory_entities.items():
                entities.extend(entity_list)

    # Create summary
    summary_parts = []

    if entities:
        unique_entities = list(set(entities))[:5]  # Top 5 entities
        summary_parts.append(f"Entities: {', '.join(unique_entities)}")

    if topics:
        # Get most common topics
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_names = [topic for topic, count in common_topics]
        summary_parts.append(f"Topics: {', '.join(topic_names)}")

    summary = "; ".join(summary_parts)

    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."

    return summary

def log_memory_metrics(operation: str, duration: float, success: bool, **kwargs):
    """Log memory processing metrics
    
    Args:
        operation: Operation name
        duration: Operation duration in seconds
        success: Whether operation succeeded
        **kwargs: Additional metrics
    """
    status = "SUCCESS" if success else "FAILED"

    logger.info(f"[MEMORY_METRICS] {operation} {status} in {duration:.2f}s", extra={
        'operation': operation,
        'duration': duration,
        'success': success,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    })

def sanitize_content(content: str) -> str:
    """Sanitize content for safe storage
    
    Args:
        content: Content to sanitize
        
    Returns:
        Sanitized content
    """
    if not content:
        return ""

    # Remove potential SQL injection attempts
    content = re.sub(r'[;\'"`]', '', content)

    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)

    # Remove control characters
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)

    return content.strip()
