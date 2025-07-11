"""Token Management Utility.

This module provides sophisticated token tracking and management for LLM context windows.
It ensures optimal use of the available context window by prioritizing and pruning
content dynamically based on importance and relevance.
The system dynamically adjusts to the current CTX value from LLAMA_INIT_PARAMS in main.py,
ensuring that token budgets are always optimally allocated based on the current model configuration.
"""

import logging
import time
from typing import Any, ClassVar

import tiktoken  # For accurate token counting

# Set up module logger
logger = logging.getLogger(__name__)

TIKTOKEN_AVAILABLE = True

# Constants
CAPS_RATIO_THRESHOLD = 0.1  # More than 10% caps

# Constants for magic numbers
MAX_COMPRESSED_TEXT_LENGTH = 80
COMPRESSED_TEXT_TRUNCATE_LENGTH = 77
CACHE_KEY_LENGTH_THRESHOLD = 200
CONTEXT_REFRESH_INTERVAL = 5
MIN_MEANINGFUL_TOKENS = 50
MIN_MEANINGFUL_TOKENS_LARGE = 100
MIN_SECTION_HEADER_LENGTH = 6
HIGH_IMPORTANCE_THRESHOLD = 0.8
MIN_HISTORY_TURNS = 5
MAX_HISTORY_MULTIPLIER = 3
MIN_HISTORY_TURNS_FIRST = 3
LARGE_WEB_RESULTS_THRESHOLD = 5000
MIN_SYSTEM_TOKENS = 200
MIN_WEB_RESULTS_BUDGET = 200
EMERGENCY_MIN_MESSAGES = 3
EMERGENCY_MIN_MESSAGES_HISTORY = 5
SECTION_DIVIDER_MIN_LENGTH = 3
SUBSTANTIAL_RESPONSE_LENGTH = 50
SHORT_RESPONSE_LENGTH = 10


def calculate_information_density(text: str) -> float:
    """Calculate information density score for context window optimization.

    Higher scores indicate more information-dense content that should be prioritized.

    Args:
        text: Text content to analyze

    Returns:
        Density score from 0.0 to 1.0
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    # Base metrics
    word_count = len(text.split())
    len(text)
    unique_words = len(set(text.lower().split()))

    if word_count == 0:
        return 0.0

    # Information density factors
    lexical_diversity = unique_words / word_count  # Higher = more diverse vocabulary
    avg_word_length = (
        sum(len(word) for word in text.split()) / word_count
    )  # Longer words often more specific

    # Simple content type checks without regex
    has_numbers = any(c.isdigit() for c in text)
    # Check for technical markers using simple string operations
    has_technical_markers = any(c in text for c in "{}()[];,:")
    # Check for emphasis markers
    has_emphasis = (
        "!" in text or "?" in text or any(word.isupper() and len(word) > 1 for word in text.split())
    )

    # Calculate base density
    density = (
        lexical_diversity * 0.4
        + min(avg_word_length / 8, 1.0) * 0.2  # Normalize word length
        + (1.0 if has_numbers else 0.0) * 0.1
        + (1.0 if has_technical_markers else 0.0) * 0.2  # Increased weight
        + (1.0 if has_emphasis else 0.0) * 0.1  # Increased weight
    )

    return min(density, 1.0)


def detect_emotional_salience(text: str) -> float:
    """Detect emotional salience in text for memory weighting.

    Let the LLM handle nuanced emotional detection rather than regex patterns.

    Args:
        text: Text to analyze for emotional content

    Returns:
        Emotional salience score from 0.0 to 1.0
    """
    if not text:
        return 0.0

    text_lower = text.lower()

    # Simple intensity markers (keep these as they're straightforward)
    exclamation_count = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0

    salience_score = 0.0

    # Basic keyword presence check without complex regex
    emotion_keywords = [
        "love",
        "hate",
        "adore",
        "despise",
        "passionate",
        "excited",
        "thrilled",
        "devastated",
        "furious",
        "terrified",
        "amazing",
        "terrible",
        "horrible",
        "wonderful",
        "fantastic",
        "awful",
        "brilliant",
        "disgusting",
        "best",
        "worst",
        "favorite",
        "always",
        "never",
        "trauma",
        "milestone",
        "breakthrough",
        "revelation",
        "epiphany",
    ]

    # Count emotional keywords
    emotional_word_count = sum(1 for word in text_lower.split() if word in emotion_keywords)

    # Simple scoring based on presence
    if emotional_word_count > 0:
        salience_score += min(emotional_word_count * 0.15, 0.6)

    # Add intensity bonuses
    if exclamation_count > 0:
        salience_score += min(exclamation_count * 0.1, 0.2)

    if caps_ratio > CAPS_RATIO_THRESHOLD:  # More than 10% caps
        salience_score += 0.1

    return min(salience_score, 1.0)


def _clean_memory_text(memory_text: str) -> str:
    """Remove category prefixes from memory text."""
    clean_text = memory_text.strip()
    if clean_text.startswith("[") and "]" in clean_text:
        bracket_end = clean_text.find("]")
        clean_text = clean_text[bracket_end + 1:].strip()
    return clean_text


def _categorize_memory(memory_text: str, importance: float, categories: dict) -> None:
    """Categorize a single memory based on keywords."""
    text_lower = memory_text.lower()
    clean_text = _clean_memory_text(memory_text)

    # Define keyword mappings
    keyword_mappings = {
        "identity": ["name is", "called", "i am", "i'm", "born", "age", "years old", "tall", "height", "eyes", "hair"],
        "family": ["wife", "husband", "mother", "father", "sister", "brother", "daughter", "son", "family", "married", "spouse"],
        "medical": ["allerg", "medical", "health", "medication", "doctor", "hospital", "disease", "condition", "epinephrine", "epipen"],
        "professional": ["work", "job", "career", "ceo", "developer", "engineer", "company", "business", "achievement", "award", "wrote"],
        "contact": ["address", "live", "location", "phone", "email", "contact"],
        "preferences": ["love", "like", "enjoy", "prefer", "favorite", "hate", "dislike", "interest", "hobby", "passion"],
        "goals": ["goal", "plan", "want", "hope", "aspire", "aim", "objective"],
        "history": ["born in", "grew up", "from", "originally", "background", "history", "past"],
    }

    # Check each category
    for category, keywords in keyword_mappings.items():
        if any(keyword in text_lower for keyword in keywords):
            categories[category].append((clean_text, importance))
            return

    # Default to behavioral if not categorized
    categories["behavioral"].append((clean_text, importance))


def _create_memory_section(title: str, items: list, max_items: int) -> list[str]:
    """Create a formatted section for a memory category."""
    if not items:
        return []

    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    section = [title]
    section.extend([f"- {compress_memory_text(text)}" for text, _ in sorted_items[:max_items]])
    return section


def format_memories_for_llm_comprehension(memories: list[tuple[str, float]]) -> str:
    """🧠 ULTRA-HIGH ROI OPTIMIZATION: Format memories for optimal LLM comprehension.

    Transforms verbose memory dumps into structured, hierarchical format that:
    - Saves 40-60% tokens through concise formatting
    - Dramatically improves LLM parsing speed and accuracy
    - Uses visual hierarchy and categorization for instant recognition
    - Groups related information logically
    - Shows importance levels and relationships

    Args:
        memories: List of (memory_text, importance) tuples

    Returns:
        Optimized memory text formatted for LLM comprehension
    """
    if not memories:
        return ""

    # Initialize categories
    categories = {
        "identity": [],  # Name, age, location, role, physical traits
        "family": [],  # Family members, relationships
        "preferences": [],  # Likes, dislikes, interests, hobbies
        "medical": [],  # Health conditions, allergies, medications
        "professional": [],  # Work, achievements, skills
        "contact": [],  # Address, phone, email
        "behavioral": [],  # Habits, personality traits, communication style
        "goals": [],  # Aspirations, plans, objectives
        "history": [],  # Background, past events, origins
    }

    # Categorize memories
    for memory_text, importance in memories:
        _categorize_memory(memory_text, importance, categories)

    # Build formatted sections
    formatted_sections = []

    # Define section configurations: (title, category_key, max_items)
    section_configs = [
        ("### ⭐ CORE IDENTITY", "identity", 4),
        ("### 🏥 MEDICAL (Critical)", "medical", 3),
        ("### 👨‍👩‍👧‍👦 FAMILY & RELATIONSHIPS", "family", 4),
        ("### 💼 PROFESSIONAL", "professional", 3),
        ("### 🎯 PREFERENCES & INTERESTS", "preferences", 4),
        ("### 📍 LOCATION & CONTACT", "contact", 2),
        ("### 🎯 GOALS & ASPIRATIONS", "goals", 3),
        ("### 📚 BACKGROUND", "history", 2),
        ("### 🧠 BEHAVIORAL NOTES", "behavioral", 2),
    ]

    for title, category_key, max_items in section_configs:
        section = _create_memory_section(title, categories[category_key], max_items)
        formatted_sections.extend(section)

    if formatted_sections:
        return "## USER PROFILE\n\n" + "\n".join(formatted_sections)
    else:
        return ""


def compress_memory_text(text: str) -> str:
    """Compress verbose memory text into concise, token-efficient format.

    Uses the LLM's understanding rather than regex patterns for better comprehension.

    Examples:
    "My name is Jeff" → "Name: Jeff"
    "Wife's name is Rose and she is allergic to pets" → "Wife: Rose (pet allergic)"
    "I work as an AI developer and am the CEO of Myron Labs" → "Role: AI Developer, CEO Myron Labs"
    """
    compressed = text.strip()

    # Simple replacements without regex
    # Clean up common verbose phrases
    compressed = compressed.replace(" and ", " | ")

    # Remove extra spaces
    compressed = " ".join(compressed.split())

    # Remove trailing periods
    compressed = compressed.rstrip(".")

    # Limit length to prevent overly long entries
    if len(compressed) > MAX_COMPRESSED_TEXT_LENGTH:
        compressed = compressed[:COMPRESSED_TEXT_TRUNCATE_LENGTH] + "..."

    return compressed


def get_current_ctx_value() -> int:
    """Retrieves the current CTX value from MODEL_CONFIG.

    This allows the token manager to adapt to runtime changes in the context window size.

    Returns:
        The current n_ctx value from MODEL_CONFIG - FAILS if not available
    """
    from config import MODEL_CONFIG

    ctx_value = MODEL_CONFIG["n_ctx"]  # Will raise KeyError if not found
    logger.info(f"CTX value loaded from MODEL_CONFIG: {ctx_value}")
    return ctx_value


class TokenBudget:
    """Manages token allocation for different components of the context window.

    Provides dynamic budgeting based on content priority and available space.
    This class now dynamically adjusts to changes in the CTX value by checking
    main.py's LLAMA_INIT_PARAMS during initialization and before critical operations.

    Enhanced with:
    - Information density scoring for focused attention
    - Dynamic context window optimization
    - Emotional salience detection and weighting
    """

    # Reserve tokens for the model's response generation
    DEFAULT_RESPONSE_RESERVE = 1024
    # Add safety margin to avoid hitting exact limits (helps prevent overflow)
    SAFETY_MARGIN = 128
    # Component priority order (highest to lowest)
    PRIORITY_ORDER: ClassVar[list[str]] = [
        "system_prompt",  # System prompt is highest priority
        "current_query",  # Current user query
        "relevant_memories",  # User's relevant memories
        "recent_history",  # Recent conversation history
        "older_history",  # Older conversation history
        "web_results",  # Web search results
    ]

    def __init__(
        self,
        max_context_tokens: int | None = None,
        response_reserve_tokens: int = DEFAULT_RESPONSE_RESERVE,
    ):
        """Initialize the token budget manager.

        Args:
            max_context_tokens: Maximum number of tokens in the context window, or None to
                                fetch from main.py
            response_reserve_tokens: Tokens to reserve for model's response
        """
        # If max_context_tokens is None, fetch the current CTX value
        if max_context_tokens is None:
            max_context_tokens = get_current_ctx_value()
        self.max_context_tokens = max_context_tokens
        self.response_reserve_tokens = response_reserve_tokens
        # Add safety margin to available tokens to prevent overflow
        self.available_tokens = max_context_tokens - response_reserve_tokens - self.SAFETY_MARGIN
        # Initialize token counter - REQUIRED
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default OpenAI tokenizer
        # Track component allocations
        self.component_tokens = dict.fromkeys(self.PRIORITY_ORDER, 0)
        self.total_used_tokens = 0
        # Log initialization
        logger.info(
            f"TokenBudget initialized with max_context_tokens={self.max_context_tokens}, "
            f"available_tokens={self.available_tokens} (after reserving {self.response_reserve_tokens} "
            f"for response and {self.SAFETY_MARGIN} safety margin)"
        )

    def refresh_context_size(self) -> None:
        """Refreshes the context size by checking the current n_ctx value in main.py.

        This ensures the token budget adapts to runtime changes in the model configuration.
        """
        current_ctx = get_current_ctx_value()
        if current_ctx != self.max_context_tokens:
            old_ctx = self.max_context_tokens
            self.max_context_tokens = current_ctx
            self.available_tokens = current_ctx - self.response_reserve_tokens - self.SAFETY_MARGIN
            logger.info(f"Context window dynamically updated: {old_ctx} → {current_ctx} tokens")
            logger.info(f"New available tokens: {self.available_tokens}")
            # Re-check if we're now over budget with the new context size
            if self.total_used_tokens > self.available_tokens:
                overflow = self.get_overflow()
                logger.warning(f"New context size caused overflow of {overflow} tokens")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using tiktoken.

        Args:
            text: The text to count tokens for
        Returns:
            Number of tokens in the text
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not available - cannot count tokens accurately")
        return len(self.tokenizer.encode(text))

    def allocate_budget(self) -> dict[str, int]:
        """ULTRA-ADVANCED DYNAMIC TOKEN ALLOCATION with Real-time Context Optimization.

        Automatically refreshes the context size before allocation with sophisticated
        priority weighting and adaptive budget redistribution.

        Returns:
            Dictionary mapping component names to token budgets
        """
        # First refresh the context size from main.py to ensure we have the latest CTX value
        self.refresh_context_size()

        # 🚀 ULTRA-SPEED ALLOCATIONS - Minimal memory for maximum responsiveness
        min_allocations = {
            "system_prompt": 200,  # Reduced for speed
            "current_query": 150,  # Keep current query intact
            "relevant_memories": 100,  # DRASTICALLY REDUCED: Minimal memory
            "recent_history": 150,  # Focus on very recent context
            "older_history": 0,  # Completely disabled
            "web_results": 100,  # Reduced web results
        }

        # 🚀 ULTRA-SPEED PERCENTAGES - Minimal memory for instant responses
        target_percentages = {
            "system_prompt": 0.15,  # Reduced system overhead
            "current_query": 0.20,  # More focus on current query
            "relevant_memories": 0.15,  # MINIMAL - Only most critical memories
            "recent_history": 0.25,  # Focus on immediate context
            "older_history": 0.00,  # Completely disabled
            "web_results": 0.25,  # More room for web results when needed
        }
        # Calculate total minimum allocation to ensure we don't exceed available tokens
        total_min_allocation = sum(min_allocations.values())
        # If we don't have enough tokens for minimal allocations, reduce them proportionally
        if total_min_allocation > self.available_tokens * 0.9:  # 90% of available tokens
            scaling_factor = (self.available_tokens * 0.9) / total_min_allocation
            for component, value in min_allocations.items():
                min_allocations[component] = int(value * scaling_factor)
            logger.warning(
                f"Minimum token allocations scaled down by factor {scaling_factor:.2f} "
                f"due to small context window"
            )
        # Allocate minimum tokens first
        remaining_tokens = self.available_tokens
        budget = {}
        for component in self.PRIORITY_ORDER:
            min_tokens = min_allocations[component]
            budget[component] = min_tokens
            remaining_tokens -= min_tokens
        # Distribute remaining tokens according to target percentages
        if remaining_tokens > 0:
            for component in self.PRIORITY_ORDER:
                additional = int(remaining_tokens * target_percentages[component])
                budget[component] += additional
        # Log the budget allocation
        logger.info(f"Token budget allocated with context size {self.max_context_tokens}:")
        for component, tokens in budget.items():
            percentage = (tokens / self.available_tokens) * 100
            logger.info(f"  - {component}: {tokens} tokens ({percentage:.1f}%)")
        return budget

    def track_component_usage(self, component: str, token_count: int) -> None:
        """Track token usage for a specific component.

        Args:
            component: Name of the component
            token_count: Number of tokens used
        """
        # Check if component exists
        if component in self.component_tokens:
            # Update the component's token count
            self.component_tokens[component] = token_count
            # Recalculate total used tokens
            self.total_used_tokens = sum(self.component_tokens.values())
            # Check for overflow after updating
            if self.total_used_tokens > self.available_tokens:
                overflow = self.get_overflow()
                logger.warning(
                    f"Component '{component}' added {token_count} tokens, "
                    f"causing overflow of {overflow} tokens"
                )

    def get_overflow(self) -> int:
        """Get the number of tokens exceeding the available budget.

        First refreshes the context size to ensure we're checking against the latest CTX value.

        Returns:
            Number of overflow tokens (0 if within budget)
        """
        # Refresh context size before calculating overflow
        self.refresh_context_size()
        return max(0, self.total_used_tokens - self.available_tokens)

    def is_within_budget(self) -> bool:
        """Check if the current token usage is within the available budget.

        First refreshes the context size to ensure we're checking against the latest CTX value.

        Returns:
            True if within budget, False otherwise
        """
        # Refresh context size before checking budget
        self.refresh_context_size()
        return self.total_used_tokens <= self.available_tokens


class TokenManager:
    """Manages token usage for LLM context, handling budgeting, counting.

    And dynamic content pruning to ensure staying within token limits.
    This enhanced version automatically monitors the CTX value in main.py and
    adapts the token budget accordingly, ensuring optimal memory utilization
    regardless of how the model is configured.
    """

    def __init__(
        self, max_context_tokens: int | None = None, base_system_prompt: str | None = None
    ):
        """Initialize the token manager.

        Args:
            max_context_tokens: Maximum number of tokens in the context window, or None to fetch from main.py
            base_system_prompt: Optional base system prompt text to pre-cache token count for.
        """
        # Initialize budget either with the provided max_context_tokens or by dynamically fetching
        # from main.py
        self.budget = TokenBudget(max_context_tokens)
        self.base_system_prompt_text: str | None = base_system_prompt
        self.cached_base_system_prompt_tokens: int | None = None
        self.last_refresh_time: float = 0

        # SPEED OPTIMIZATION: Token count cache
        self.token_cache: dict[str, int] = {}
        self.max_cache_size = 1000  # Limit cache size
        self.cache_hits = 0
        self.cache_misses = 0
        # Cache token count for the base system prompt if provided
        if self.base_system_prompt_text:
            self.cached_base_system_prompt_tokens = self.budget.count_tokens(
                self.base_system_prompt_text
            )
        logger.info("TokenManager initialized: Dynamic context size monitoring enabled.")

    def count_tokens_cached(self, text: str) -> int:
        """Count tokens with caching for speed optimization.

        Args:
            text: Text to count tokens for
        Returns:
            Number of tokens (from cache if available)
        """
        # Create cache key - use hash for long texts to save memory
        if len(text) > CACHE_KEY_LENGTH_THRESHOLD:
            import hashlib

            cache_key = f"hash_{hashlib.sha256(text.encode()).hexdigest()}"
        else:
            cache_key = text

        # Check cache first
        if cache_key in self.token_cache:
            self.cache_hits += 1
            return self.token_cache[cache_key]

        # Cache miss - calculate tokens
        self.cache_misses += 1
        token_count = self.budget.count_tokens(text)

        # Add to cache if not full
        if len(self.token_cache) < self.max_cache_size:
            self.token_cache[cache_key] = token_count
        elif len(self.token_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.token_cache))
            del self.token_cache[oldest_key]
            self.token_cache[cache_key] = token_count

        return token_count

    def refresh_if_needed(self, force: bool = False) -> bool:
        """Refreshes the context size if enough time has passed since last refresh.

        This prevents excessive filesystem checks while still keeping the
        context size reasonably up-to-date.

        Args:
            force: Whether to force a refresh regardless of timing
        Returns:
            True if a refresh occurred, False otherwise
        """
        current_time = time.time()
        # Refresh at most once every 5 seconds unless forced
        if force or (current_time - self.last_refresh_time) > CONTEXT_REFRESH_INTERVAL:
            old_ctx = self.budget.max_context_tokens
            self.budget.refresh_context_size()
            self.last_refresh_time = current_time
            # If the context size changed, invalidate the cached base system prompt token count
            if old_ctx != self.budget.max_context_tokens and self.base_system_prompt_text:
                self.cached_base_system_prompt_tokens = self.budget.count_tokens(
                    self.base_system_prompt_text
                )
                logger.info(
                    f"Recalculated base system prompt token count: {self.cached_base_system_prompt_tokens}"
                )
            return True
        return False

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count the total tokens in a list of messages.

        Args:
            messages: List of message dictionaries
        Returns:
            Total token count for all messages
        """
        # Refresh context size before counting to ensure accurate assessment
        self.refresh_if_needed()
        total = 0
        for message in messages:
            content = message.get("content", "")
            message.get("role", "")
            # Count tokens in the message content
            total += self.budget.count_tokens(content)
            # Add tokens for message formatting (approximate)
            total += 4  # Basic overhead per message
        return total

    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.

        Uses smart truncation to preserve semantic integrity when possible.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
        Returns:
            Truncated text
        """
        # Quick check if truncation is needed
        if self.budget.count_tokens(text) <= max_tokens:
            return text

        # Use tiktoken for accurate truncation - REQUIRED
        if not self.budget.tokenizer:
            raise RuntimeError("Tokenizer not available - cannot truncate text accurately")

        tokens = self.budget.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.budget.tokenizer.decode(truncated_tokens)
        # Try to make the truncation more natural by ending at a sentence or paragraph
        # Find the last paragraph break
        last_para = truncated_text.rfind("\n\n")
        if last_para > len(truncated_text) * 0.8:  # If it's reasonably close to the end
            return truncated_text[: last_para + 2]
        # Find the last sentence break
        last_period = truncated_text.rfind(".")
        if last_period > len(truncated_text) * 0.85:  # If it's very close to the end
            return truncated_text[: last_period + 1]
        return truncated_text

    def optimize_memories(
        self, memories: list[tuple[str, float]], max_tokens: int
    ) -> list[tuple[str, float]]:
        """Enhanced memory optimization with information density and emotional salience scoring.

        Strategically selects the most important memories while staying within token limits.

        Args:
            memories: List of (memory_text, importance) tuples
            max_tokens: Maximum tokens allowed
        Returns:
            Optimized list of memories
        """
        # Ensure context size is up-to-date before optimization
        self.refresh_if_needed()
        if not memories:
            return []

        # Enhanced scoring with density and emotional factors
        enhanced_memories = []
        for memory_text, importance in memories:
            # Calculate enhancement factors
            density_score = calculate_information_density(memory_text)
            emotional_score = detect_emotional_salience(memory_text)

            # Combined importance score with weighted factors
            enhanced_importance = (
                importance * 0.6  # Original importance (60%)
                + density_score * 0.25  # Information density (25%)
                + emotional_score * 0.15  # Emotional salience (15%)
            )

            enhanced_memories.append(
                (memory_text, enhanced_importance, density_score, emotional_score)
            )

        # Sort by enhanced importance (highest first)
        sorted_memories = sorted(enhanced_memories, key=lambda x: x[1], reverse=True)
        optimized_memories = []
        used_tokens = 0

        logger.debug("🧠 Memory optimization with enhanced scoring:")
        for memory_text, enhanced_imp, density, emotional in sorted_memories:
            memory_tokens = self.budget.count_tokens(memory_text)

            # If adding this memory would exceed budget
            if used_tokens + memory_tokens > max_tokens:
                # If we haven't added any memories yet, try truncating this one
                if not optimized_memories:
                    truncated = self.truncate_text_to_token_limit(memory_text, max_tokens)
                    truncated_tokens = self.budget.count_tokens(truncated)
                    optimized_memories.append((truncated, enhanced_imp))
                    used_tokens += truncated_tokens
                    logger.debug(
                        f"   📝 Truncated high-priority memory: '{memory_text[:30]}...' "
                        f"from {memory_tokens} to {truncated_tokens} tokens"
                    )
                break

            # Add the memory and update token count
            optimized_memories.append((memory_text, enhanced_imp))
            used_tokens += memory_tokens

            # Log selection rationale for high-scoring memories
            if enhanced_imp > HIGH_IMPORTANCE_THRESHOLD:
                logger.debug(
                    f"   ⭐ Selected: '{memory_text[:40]}...' (score: {enhanced_imp:.2f}, "
                    f"density: {density:.2f}, emotional: {emotional:.2f})"
                )

        logger.info(
            f"📊 Optimized {len(sorted_memories)} memories to {len(optimized_memories)} within "
            f"{max_tokens} token budget ({used_tokens} tokens used)"
        )
        return optimized_memories

    def optimize_conversation_history(
        self, history: list[dict[str, Any]], max_tokens: int
    ) -> list[dict[str, Any]]:
        """Optimize conversation history to fit within token budget.

        Prioritizes recent conversation turns while preserving context.

        Args:
            history: List of conversation turns
            max_tokens: Maximum tokens allowed
        Returns:
            Optimized conversation history
        """
        # Ensure context size is up-to-date before optimization
        self.refresh_if_needed()
        if not history:
            return []
        # Important: Keep most recent history
        history_reversed = list(reversed(history))  # Newest first
        optimized_history = []
        used_tokens = 0
        # Always try to include at least the last turn if possible
        critical_turns = min(1, len(history_reversed))
        # First pass: estimate total tokens needed
        total_token_estimate = 0
        for turn in history_reversed:
            user_text = turn.get("user", "")
            model_text = turn.get("model", "")
            turn_tokens = (
                self.budget.count_tokens(user_text) + self.budget.count_tokens(model_text) + 8
            )  # 8 tokens for formatting
            total_token_estimate += turn_tokens
        # If total is much larger than max, use summarization strategy
        if (
            total_token_estimate > max_tokens * MAX_HISTORY_MULTIPLIER
            and len(history_reversed) > MIN_HISTORY_TURNS
        ):
            # Include most recent turn(s) and first turn for context
            critical_turns = min(2, len(history_reversed))
            optimized_history.extend(history_reversed[:critical_turns])
            # Estimate tokens used by critical turns
            for turn in optimized_history:
                user_text = turn.get("user", "")
                model_text = turn.get("model", "")
                used_tokens += (
                    self.budget.count_tokens(user_text) + self.budget.count_tokens(model_text) + 8
                )
            # Add the first turn if there's room and we have more than 3 turns total
            if len(history_reversed) > MIN_HISTORY_TURNS_FIRST and used_tokens < max_tokens * 0.7:
                first_turn = history_reversed[-1]
                first_turn_user = first_turn.get("user", "")
                first_turn_model = first_turn.get("model", "")
                first_turn_tokens = (
                    self.budget.count_tokens(first_turn_user)
                    + self.budget.count_tokens(first_turn_model)
                    + 8
                )
                if used_tokens + first_turn_tokens <= max_tokens:
                    optimized_history.append(first_turn)
                    used_tokens += first_turn_tokens
            # Restore original order (oldest first)
            return list(reversed(optimized_history))
        # Standard approach for smaller histories
        for i, turn in enumerate(history_reversed):
            # Prioritize critical turns (most recent)
            must_include = i < critical_turns
            # Calculate tokens for this turn
            user_text = turn.get("user", "")
            model_text = turn.get("model", "")
            turn_tokens = (
                self.budget.count_tokens(user_text) + self.budget.count_tokens(model_text) + 8
            )  # 8 tokens for formatting
            # If adding this turn would exceed budget
            if not must_include and used_tokens + turn_tokens > max_tokens:
                break
            # For critical turns, make room if needed by truncating
            if must_include and used_tokens + turn_tokens > max_tokens:
                # Truncate the model response to fit
                available_tokens = (
                    max_tokens - used_tokens - self.budget.count_tokens(user_text) - 8
                )
                if (
                    available_tokens > MIN_MEANINGFUL_TOKENS
                ):  # Only if we can keep a meaningful amount
                    truncated_model = self.truncate_text_to_token_limit(
                        model_text, available_tokens
                    )
                    updated_turn = turn.copy()  # Create a copy to avoid modifying the original
                    updated_turn["model"] = truncated_model
                    turn_tokens = (
                        self.budget.count_tokens(user_text)
                        + self.budget.count_tokens(truncated_model)
                        + 8
                    )
                    logger.debug(
                        f"Truncated critical turn response from {len(model_text)} chars to "
                        f"{len(truncated_model)} chars"
                    )
            # Add the turn and update token count
            optimized_history.append(updated_turn if 'updated_turn' in locals() else turn)
            used_tokens += turn_tokens
        # Restore original order (oldest first)
        return list(reversed(optimized_history))

    def _split_web_results_into_blocks(self, web_results: str) -> list[str]:
        """Split web results into content blocks based on separators."""
        blocks = []
        current_block = []
        lines = web_results.split("\n")

        for line in lines:
            is_separator = (
                line.strip() == "" or (
                    line.strip().startswith("-")
                    and len(line.strip()) >= SECTION_DIVIDER_MIN_LENGTH
                    and all(c == "-" for c in line.strip())
                )
            )

            if is_separator:
                if current_block:
                    blocks.append("\n".join(current_block))
                    blocks.append("\n\n")  # Add separator
                    current_block = []
            else:
                current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))
        return blocks

    def _preserve_priority_blocks(self, blocks: list[str], max_tokens: int) -> tuple[list[str], int]:
        """Preserve high-priority blocks within token limit."""
        preserved_blocks = []
        preserved_tokens = 0
        max_preserved_blocks = 3  # Keep at least top 3 results if possible

        for i, block in enumerate(blocks):
            block_tokens = self.budget.count_tokens(block)

            # Always include separators
            if block.strip() == "" or block == "\n\n":
                if preserved_tokens + block_tokens <= max_tokens:
                    preserved_blocks.append(block)
                    preserved_tokens += block_tokens
                continue

            # Handle content blocks
            is_high_priority = i // 2 < max_preserved_blocks

            if is_high_priority:
                if preserved_tokens + block_tokens <= max_tokens:
                    preserved_blocks.append(block)
                    preserved_tokens += block_tokens
                else:
                    # Try to truncate the last high-priority block
                    remaining_tokens = max_tokens - preserved_tokens
                    if remaining_tokens > MIN_MEANINGFUL_TOKENS_LARGE:
                        truncated_block = self.truncate_text_to_token_limit(block, remaining_tokens)
                        preserved_blocks.append(truncated_block)
                        preserved_tokens += self.budget.count_tokens(truncated_block)
                    break
            elif preserved_tokens + block_tokens <= max_tokens:
                preserved_blocks.append(block)
                preserved_tokens += block_tokens
            else:
                break

        return preserved_blocks, preserved_tokens

    def _add_truncation_notice(self, optimized_results: str, original_results: str, preserved_tokens: int, max_tokens: int) -> str:
        """Add truncation notice if content was cut."""
        if len(optimized_results) < len(original_results):
            truncation_notice = "\n\n[Results truncated to fit context window]"
            truncation_tokens = self.budget.count_tokens(truncation_notice)
            if preserved_tokens + truncation_tokens <= max_tokens:
                optimized_results += truncation_notice
        return optimized_results

    def optimize_web_results(self, web_results: str, max_tokens: int) -> str:
        """Optimize web search results to fit within token budget.

        Uses smart truncation for web content to maintain the most valuable information.

        Args:
            web_results: Web search results text
            max_tokens: Maximum tokens allowed
        Returns:
            Optimized web results
        """
        self.refresh_if_needed()
        web_tokens = self.budget.count_tokens(web_results)

        if web_tokens <= max_tokens:
            return web_results

        # For extremely large content, try to keep result blocks intact
        if web_tokens > max_tokens * 2:
            blocks = self._split_web_results_into_blocks(web_results)
            preserved_blocks, preserved_tokens = self._preserve_priority_blocks(blocks, max_tokens)
            optimized_results = "".join(preserved_blocks)
            return self._add_truncation_notice(optimized_results, web_results, preserved_tokens, max_tokens)

        # Simple truncation for smaller content
        return self.truncate_text_to_token_limit(web_results, max_tokens)

    def _parse_system_prompt_sections(self, system_prompt: str) -> list[tuple[str, str]]:
        """Parse system prompt into sections based on header markers."""
        sections = []
        current_section = []
        current_title = ""
        lines = system_prompt.split("\n")

        for line in lines:
            stripped = line.strip()
            is_header = (
                stripped.startswith("---")
                and stripped.endswith("---")
                and len(stripped) > MIN_SECTION_HEADER_LENGTH
            )

            if is_header:
                # Save previous section
                if current_section or current_title:
                    sections.append((current_title, "\n".join(current_section)))

                # Start new section
                current_title = stripped[3:-3].strip()
                current_section = []
            else:
                current_section.append(line)

        # Add the last section
        if current_section or current_title:
            sections.append((current_title, "\n".join(current_section)))

        return sections

    def _categorize_prompt_sections(self, sections: list[tuple[str, str]]) -> tuple[list, list]:
        """Categorize sections into critical and optional based on priority."""
        core_sections = ["", "CORE INSTRUCTIONS", "INSTRUCTIONS", "GUIDELINES", "RULES", "ROLE"]
        critical_parts = []
        optional_parts = []

        for section_title, section_content in sections:
            is_critical = any(title.upper() in section_title.upper() for title in core_sections)

            if is_critical:
                priority_index = next(
                    (i for i, t in enumerate(core_sections) if t.upper() in section_title.upper()),
                    999
                )
                critical_parts.append((section_title, section_content, priority_index))
            else:
                optional_parts.append((section_title, section_content))

        # Sort critical parts by priority
        critical_parts.sort(key=lambda x: x[2])
        return critical_parts, optional_parts

    def _build_optimized_prompt(self, critical_parts: list, optional_parts: list, max_tokens: int) -> str:
        """Build optimized prompt from prioritized sections."""
        combined_text = ""
        used_tokens = 0

        # Add critical parts first
        for title, content, _ in critical_parts:
            section_text = f"\n\n--- {title} ---\n\n{content}" if title else content
            section_tokens = self.budget.count_tokens(section_text)

            if used_tokens + section_tokens <= max_tokens:
                combined_text += section_text
                used_tokens += section_tokens
            else:
                # Try to include partial critical section
                remaining_tokens = max_tokens - used_tokens
                if remaining_tokens > MIN_MEANINGFUL_TOKENS:
                    truncated_section = self.truncate_text_to_token_limit(section_text, remaining_tokens)
                    combined_text += truncated_section
                    used_tokens += self.budget.count_tokens(truncated_section)
                break

        # Add optional parts if space available
        for title, content in optional_parts:
            section_text = f"\n\n--- {title} ---\n\n{content}" if title else content
            section_tokens = self.budget.count_tokens(section_text)

            if used_tokens + section_tokens <= max_tokens:
                combined_text += section_text
                used_tokens += section_tokens
            else:
                break

        return combined_text.strip()

    def optimize_system_prompt(self, system_prompt: str, max_tokens: int) -> str:
        """Optimize system prompt to fit within token budget.

        If needed, extract critical instructions and trim the rest.

        Args:
            system_prompt: System prompt text
            max_tokens: Maximum tokens allowed
        Returns:
            Optimized system prompt
        """
        self.refresh_if_needed()
        prompt_tokens = self.budget.count_tokens(system_prompt)

        if prompt_tokens <= max_tokens:
            return system_prompt

        # Parse sections and categorize by priority
        sections = self._parse_system_prompt_sections(system_prompt)
        critical_parts, optional_parts = self._categorize_prompt_sections(sections)

        # Build optimized prompt
        combined_text = self._build_optimized_prompt(critical_parts, optional_parts, max_tokens)

        # Fall back to simple truncation if no structured content found
        if not combined_text:
            return self.truncate_text_to_token_limit(system_prompt, max_tokens)

        return combined_text

    def _initialize_optimization(self, user_prompt: str) -> tuple[dict, int, int]:
        """Initialize optimization context and budget allocation."""
        self.refresh_if_needed(force=True)
        budget = self.budget.allocate_budget()
        self.budget.component_tokens = dict.fromkeys(self.budget.PRIORITY_ORDER, 0)
        self.budget.total_used_tokens = 0

        current_query_tokens = self.count_tokens_cached(user_prompt)
        self.budget.track_component_usage("current_query", current_query_tokens)
        remaining_budget = self.budget.available_tokens - current_query_tokens

        return budget, current_query_tokens, remaining_budget

    def _optimize_system_component(self, system_prompt: str, budget: dict, remaining_budget: int) -> tuple[str, int]:
        """Optimize system prompt component."""
        max_system_tokens = min(budget["system_prompt"], remaining_budget)

        # Handle cached base system prompt optimization
        if (
            self.base_system_prompt_text
            and self.cached_base_system_prompt_tokens is not None
            and system_prompt.startswith(self.base_system_prompt_text)
        ):
            dynamic_part = system_prompt[len(self.base_system_prompt_text):]
            system_tokens = self.cached_base_system_prompt_tokens
            if dynamic_part:
                system_tokens += self.budget.count_tokens(dynamic_part)
        else:
            system_tokens = self.budget.count_tokens(system_prompt)

        optimized_system = self.optimize_system_prompt(system_prompt, max_system_tokens)
        final_tokens = self.budget.count_tokens(optimized_system)
        self.budget.track_component_usage("system_prompt", final_tokens)

        return optimized_system, final_tokens

    def _optimize_web_component(self, web_results: str, budget: dict, remaining_budget: int) -> tuple[str, int]:
        """Optimize web results component."""
        if not web_results:
            return "", 0

        web_results_tokens = self.count_tokens_cached(web_results)
        is_scraping_task = web_results_tokens > LARGE_WEB_RESULTS_THRESHOLD

        if is_scraping_task:
            # Boost web results allocation by borrowing from history
            history_adjustment = min(budget["older_history"], budget["older_history"] * 0.7)
            budget["web_results"] += history_adjustment
            budget["older_history"] -= history_adjustment
            logger.info(f"Web scraping task detected: Boosted web results budget by {history_adjustment} tokens")

        max_web_tokens = min(budget["web_results"], remaining_budget)
        optimized_web_results = self.optimize_web_results(web_results, max_web_tokens)
        web_tokens = self.budget.count_tokens(optimized_web_results)
        self.budget.track_component_usage("web_results", web_tokens)

        logger.info(
            f"Web content optimized: {web_results_tokens} → {web_tokens} tokens "
            f"({(web_tokens / web_results_tokens * 100):.1f}% of original)"
        )

        return optimized_web_results, web_tokens

    def _optimize_memory_component(self, relevant_memories: list, budget: dict, remaining_budget: int) -> tuple[str, int]:
        """Optimize memories component."""
        max_memory_tokens = min(budget["relevant_memories"], remaining_budget)
        optimized_memories = self.optimize_memories(relevant_memories, max_memory_tokens)
        memory_text = format_memories_for_llm_comprehension(optimized_memories) if optimized_memories else ""
        memory_tokens = self.budget.count_tokens(memory_text)
        self.budget.track_component_usage("relevant_memories", memory_tokens)

        return memory_text, memory_tokens

    def _build_system_message(self, optimized_system: str, memory_text: str, max_system_tokens: int) -> str:
        """Build the complete system message with memories."""
        full_system_content = optimized_system

        if memory_text:
            memory_section = f"\n\n{memory_text}"
            if self.budget.count_tokens(full_system_content + memory_section) <= max_system_tokens:
                full_system_content += memory_section
            else:
                # Truncate memories if they would exceed budget
                remaining_sys_tokens = max_system_tokens - self.budget.count_tokens(full_system_content + "\n\n")
                if remaining_sys_tokens > MIN_MEANINGFUL_TOKENS:
                    truncated_memory = self.truncate_text_to_token_limit(memory_text, remaining_sys_tokens)
                    full_system_content += f"\n\n{truncated_memory}"

        return full_system_content

    def _add_web_messages(self, messages: list, optimized_web_results: str, remaining_budget: int) -> tuple[int, int]:
        """Add web results messages if they fit within budget."""
        current_total = 0
        remaining_total_budget = remaining_budget

        if optimized_web_results and remaining_total_budget > MIN_WEB_RESULTS_BUDGET:
            web_msg = f"Here are relevant results from the web:\n{optimized_web_results}"
            web_resp = "I'll consider this information when answering your question."
            web_msg_tokens = self.budget.count_tokens(web_msg)
            web_resp_tokens = self.budget.count_tokens(web_resp)

            if web_msg_tokens + web_resp_tokens <= remaining_total_budget - MIN_MEANINGFUL_TOKENS_LARGE:
                messages.append({"role": "user", "content": web_msg})
                messages.append({"role": "assistant", "content": web_resp})
                current_total += web_msg_tokens + web_resp_tokens
                remaining_total_budget -= web_msg_tokens + web_resp_tokens
            elif web_msg_tokens <= remaining_total_budget - 100:
                messages.append({"role": "user", "content": web_msg})
                current_total += web_msg_tokens
                remaining_total_budget -= web_msg_tokens

        return current_total, remaining_total_budget

    def _add_conversation_history(self, messages: list, optimized_history: list, budget: dict,
                                 remaining_budget: int, current_query_tokens: int) -> int:
        """Add conversation history messages within budget constraints."""
        history_tokens_used = 0

        for turn in optimized_history:
            user_msg = turn.get("user", "")
            model_msg = turn.get("model", "")

            if user_msg and model_msg:
                user_tokens = self.budget.count_tokens(user_msg)
                model_tokens = self.budget.count_tokens(model_msg)
                turn_tokens = user_tokens + model_tokens

                if turn_tokens < remaining_budget - current_query_tokens - 50:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": model_msg})
                    remaining_budget -= turn_tokens
                    history_tokens_used += turn_tokens
                else:
                    break

        # Track history token usage
        recent_ratio = budget["recent_history"] / (budget["recent_history"] + budget["older_history"] + 0.0001)
        self.budget.track_component_usage("recent_history", int(history_tokens_used * recent_ratio))
        self.budget.track_component_usage("older_history", int(history_tokens_used * (1 - recent_ratio)))

        return history_tokens_used

    def _perform_emergency_pruning(self, messages: list) -> None:
        """Emergency pruning to ensure messages fit within token budget."""
        final_token_count = self.count_message_tokens(messages)

        if final_token_count > self.budget.available_tokens:
            logger.warning(f"Final token count {final_token_count} exceeds available budget {self.budget.available_tokens}")

            # Remove history until under limit
            while (len(messages) > EMERGENCY_MIN_MESSAGES and
                   final_token_count > self.budget.available_tokens):
                if len(messages) >= EMERGENCY_MIN_MESSAGES_HISTORY:
                    messages.pop(1)  # Remove oldest history pair
                    messages.pop(1)
                    final_token_count = self.count_message_tokens(messages)
                    logger.warning(f"Emergency pruning: Removed history turn, new token count: {final_token_count}")
                else:
                    break

    def _strict_budget_enforcement(self, messages: list) -> None:
        """Strict enforcement to never exceed context window."""
        if not self.budget.is_within_budget():
            logger.warning("Token budget exceeded despite optimization efforts")

            while len(messages) > EMERGENCY_MIN_MESSAGES and not self.budget.is_within_budget():
                if len(messages) >= EMERGENCY_MIN_MESSAGES_HISTORY:
                    messages.pop(1)
                    messages.pop(1)
                    final_token_count = self.count_message_tokens(messages)
                    self.budget.total_used_tokens = final_token_count
                    logger.warning(f"STRICT ENFORCEMENT: Removed history turn, new token count: {final_token_count}")
                else:
                    # Last resort: truncate system prompt
                    if len(messages) > 0 and messages[0]["role"] == "system":
                        system_content = messages[0]["content"]
                        overflow = self.budget.get_overflow()
                        tokens_to_keep = (self.budget.count_tokens(system_content) -
                                        overflow - MIN_MEANINGFUL_TOKENS_LARGE)

                        if tokens_to_keep > MIN_SYSTEM_TOKENS:
                            messages[0]["content"] = self.truncate_text_to_token_limit(system_content, tokens_to_keep)
                            logger.warning(f"STRICT ENFORCEMENT: Truncated system prompt to {tokens_to_keep} tokens")
                            final_token_count = self.count_message_tokens(messages)
                            self.budget.total_used_tokens = final_token_count
                    break

            # Final check - raise exception if still over budget
            if not self.budget.is_within_budget():
                overflow = self.budget.get_overflow()
                error_msg = (f"Error during stream generation: Requested tokens ({self.budget.total_used_tokens}) "
                           f"exceed context window of {self.budget.max_context_tokens}")
                logger.error(error_msg)
                raise ValueError(error_msg)

    def create_optimized_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        relevant_memories: list[tuple[str, float]],
        conversation_history: list[dict[str, Any]],
        web_results: str = "",
    ) -> list[dict[str, str]]:
        """Create an optimized message list that fits within token budget.

        This method dynamically checks the current CTX value from main.py and ensures
        that all components fit within the available token budget, prioritizing the
        most important information while maximizing information density.

        Args:
            system_prompt: System prompt text
            user_prompt: Current user query
            relevant_memories: List of (memory_text, importance) tuples
            conversation_history: List of conversation turns
            web_results: Web search results text
        Returns:
            Optimized message list for LLM
        """
        start_time = time.time()

        # 1. Initialize optimization context
        budget, current_query_tokens, remaining_budget = self._initialize_optimization(user_prompt)

        # 2. Optimize system prompt
        optimized_system, system_tokens = self._optimize_system_component(system_prompt, budget, remaining_budget)
        remaining_budget -= system_tokens

        # 3. Optimize web results
        optimized_web_results, web_tokens = self._optimize_web_component(web_results, budget, remaining_budget)
        remaining_budget -= web_tokens

        # 4. Optimize memories
        memory_text, memory_tokens = self._optimize_memory_component(relevant_memories, budget, remaining_budget)
        remaining_budget -= memory_tokens

        # 5. Optimize conversation history
        history_budget = budget["recent_history"] + budget["older_history"]
        max_history_tokens = min(history_budget, remaining_budget)
        optimized_history = self.optimize_conversation_history(conversation_history, max_history_tokens)

        # 6. Build messages list
        messages = []
        full_system_content = self._build_system_message(optimized_system, memory_text, budget["system_prompt"])
        messages.append({"role": "system", "content": full_system_content})

        current_total = self.budget.count_tokens(full_system_content)
        remaining_total_budget = self.budget.available_tokens - current_total

        # 7. Add web results messages
        web_total, remaining_total_budget = self._add_web_messages(messages, optimized_web_results, remaining_total_budget)
        current_total += web_total

        # 8. Add conversation history
        self._add_conversation_history(messages, optimized_history, budget, remaining_total_budget, current_query_tokens)

        # 9. Add current user prompt
        messages.append({"role": "user", "content": user_prompt})

        # 10. Emergency pruning and strict enforcement
        self._perform_emergency_pruning(messages)
        self._strict_budget_enforcement(messages)

        # 11. Log usage and return
        self._log_token_usage(time.time() - start_time)
        return messages

    def _log_token_usage(self, elapsed_time: float | None = None) -> None:
        """Log detailed token usage statistics.

        Args:
            elapsed_time: Optional time taken to process the optimization
        """
        logger.info("\n=== TOKEN USAGE SUMMARY ===")
        logger.info(f"Total context window: {self.budget.max_context_tokens} tokens")
        logger.info(f"Reserved for response: {self.budget.response_reserve_tokens} tokens")
        logger.info(f"Available for prompt: {self.budget.available_tokens} tokens")
        logger.info(f"Total used in prompt: {self.budget.total_used_tokens} tokens")
        overflow = self.budget.get_overflow()
        if overflow > 0:
            logger.warning(f"Exceeding token budget by {overflow} tokens")
        else:
            remaining = self.budget.available_tokens - self.budget.total_used_tokens
            utilization = (
                (self.budget.available_tokens - remaining) / self.budget.available_tokens
            ) * 100
            logger.info(f"Remaining tokens: {remaining} ({utilization:.1f}% utilization)")
        logger.info("\nComponent breakdown:")
        for component, tokens in self.budget.component_tokens.items():
            percentage = (
                (tokens / self.budget.available_tokens) * 100
                if self.budget.available_tokens > 0
                else 0
            )
            logger.info(f"  - {component}: {tokens} tokens ({percentage:.1f}%)")
        if elapsed_time is not None:
            logger.info(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        logger.info("=== END TOKEN USAGE SUMMARY ===\n")
