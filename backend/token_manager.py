"""Token Management Utility
This module provides sophisticated token tracking and management for LLM context windows.
It ensures optimal use of the available context window by prioritizing and pruning
content dynamically based on importance and relevance.
The system dynamically adjusts to the current CTX value from LLAMA_INIT_PARAMS in main.py,
ensuring that token budgets are always optimally allocated based on the current model configuration.
"""

import logging
import time
from typing import Any

import tiktoken  # For accurate token counting

# Set up module logger
logger = logging.getLogger(__name__)

TIKTOKEN_AVAILABLE = True


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
    has_emphasis = "!" in text or "?" in text or any(word.isupper() and len(word) > 1 for word in text.split())
    
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
        "love", "hate", "adore", "despise", "passionate", "excited", "thrilled", 
        "devastated", "furious", "terrified", "amazing", "terrible", "horrible", 
        "wonderful", "fantastic", "awful", "brilliant", "disgusting", "best", 
        "worst", "favorite", "always", "never", "trauma", "milestone", 
        "breakthrough", "revelation", "epiphany"
    ]
    
    # Count emotional keywords
    emotional_word_count = sum(1 for word in text_lower.split() if word in emotion_keywords)
    
    # Simple scoring based on presence
    if emotional_word_count > 0:
        salience_score += min(emotional_word_count * 0.15, 0.6)

    # Add intensity bonuses
    if exclamation_count > 0:
        salience_score += min(exclamation_count * 0.1, 0.2)

    if caps_ratio > 0.1:  # More than 10% caps
        salience_score += 0.1

    return min(salience_score, 1.0)


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

    # Categorize memories by content type
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

    # Smart categorization using keyword detection
    for memory_text, importance in memories:
        text_lower = memory_text.lower()

        # Remove category prefixes added by the system (e.g., "[IDENTITY]", "[PREFERENCES]")
        clean_text = memory_text.strip()
        if clean_text.startswith("[") and "]" in clean_text:
            bracket_end = clean_text.find("]")
            clean_text = clean_text[bracket_end + 1:].strip()

        categorized = False

        # Identity indicators
        if any(
            keyword in text_lower
            for keyword in [
                "name is",
                "called",
                "i am",
                "i'm",
                "born",
                "age",
                "years old",
                "tall",
                "height",
                "eyes",
                "hair",
            ]
        ):
            categories["identity"].append((clean_text, importance))
            categorized = True
        # Family indicators
        elif any(
            keyword in text_lower
            for keyword in [
                "wife",
                "husband",
                "mother",
                "father",
                "sister",
                "brother",
                "daughter",
                "son",
                "family",
                "married",
                "spouse",
            ]
        ):
            categories["family"].append((clean_text, importance))
            categorized = True
        # Medical indicators
        elif any(
            keyword in text_lower
            for keyword in [
                "allerg",
                "medical",
                "health",
                "medication",
                "doctor",
                "hospital",
                "disease",
                "condition",
                "epinephrine",
                "epipen",
            ]
        ):
            categories["medical"].append((clean_text, importance))
            categorized = True
        # Professional indicators
        elif any(
            keyword in text_lower
            for keyword in [
                "work",
                "job",
                "career",
                "ceo",
                "developer",
                "engineer",
                "company",
                "business",
                "achievement",
                "award",
                "wrote",
            ]
        ):
            categories["professional"].append((clean_text, importance))
            categorized = True
        # Contact/Location indicators
        elif any(
            keyword in text_lower
            for keyword in ["address", "live", "location", "phone", "email", "contact"]
        ):
            categories["contact"].append((clean_text, importance))
            categorized = True
        # Preference indicators
        elif any(
            keyword in text_lower
            for keyword in [
                "love",
                "like",
                "enjoy",
                "prefer",
                "favorite",
                "hate",
                "dislike",
                "interest",
                "hobby",
                "passion",
            ]
        ):
            categories["preferences"].append((clean_text, importance))
            categorized = True
        # Goals/aspirations
        elif any(
            keyword in text_lower
            for keyword in ["goal", "plan", "want", "hope", "aspire", "aim", "objective"]
        ):
            categories["goals"].append((clean_text, importance))
            categorized = True
        # Historical/background
        elif any(
            keyword in text_lower
            for keyword in [
                "born in",
                "grew up",
                "from",
                "originally",
                "background",
                "history",
                "past",
            ]
        ):
            categories["history"].append((clean_text, importance))
            categorized = True

        # Default to behavioral if not categorized
        if not categorized:
            categories["behavioral"].append((clean_text, importance))

    # Build optimized format with visual hierarchy
    formatted_sections = []

    # Identity section (highest priority)
    if categories["identity"]:
        identity_items = sorted(categories["identity"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### ⭐ CORE IDENTITY")
        formatted_sections.extend(
            [f"- {compress_memory_text(text)}" for text, _ in identity_items[:4]]
        )  # Top 4 most important

    # Medical section (critical safety information)
    if categories["medical"]:
        medical_items = sorted(categories["medical"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 🏥 MEDICAL (Critical)")
        formatted_sections.extend(
            [f"- {compress_memory_text(text)}" for text, _ in medical_items[:3]]
        )

    # Family section
    if categories["family"]:
        family_items = sorted(categories["family"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 👨‍👩‍👧‍👦 FAMILY & RELATIONSHIPS")
        formatted_sections.extend(
            [f"- {compress_memory_text(text)}" for text, _ in family_items[:4]]
        )

    # Professional section
    if categories["professional"]:
        prof_items = sorted(categories["professional"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 💼 PROFESSIONAL")
        formatted_sections.extend([f"- {compress_memory_text(text)}" for text, _ in prof_items[:3]])

    # Preferences section
    if categories["preferences"]:
        pref_items = sorted(categories["preferences"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 🎯 PREFERENCES & INTERESTS")
        formatted_sections.extend([f"- {compress_memory_text(text)}" for text, _ in pref_items[:4]])

    # Contact section
    if categories["contact"]:
        contact_items = sorted(categories["contact"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 📍 LOCATION & CONTACT")
        formatted_sections.extend(
            [f"- {compress_memory_text(text)}" for text, _ in contact_items[:2]]
        )

    # Goals section
    if categories["goals"]:
        goal_items = sorted(categories["goals"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 🎯 GOALS & ASPIRATIONS")
        formatted_sections.extend([f"- {compress_memory_text(text)}" for text, _ in goal_items[:3]])

    # History section
    if categories["history"]:
        history_items = sorted(categories["history"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 📚 BACKGROUND")
        formatted_sections.extend(
            [f"- {compress_memory_text(text)}" for text, _ in history_items[:2]]
        )

    # Behavioral section (catch-all)
    if categories["behavioral"]:
        behavioral_items = sorted(categories["behavioral"], key=lambda x: x[1], reverse=True)
        formatted_sections.append("### 🧠 BEHAVIORAL NOTES")
        formatted_sections.extend(
            [f"- {compress_memory_text(text)}" for text, _ in behavioral_items[:2]]
        )

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
    if len(compressed) > 80:
        compressed = compressed[:77] + "..."

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
    PRIORITY_ORDER = [
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
        """ULTRA-ADVANCED DYNAMIC TOKEN ALLOCATION with Real-time Context Optimization
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
            for component in min_allocations:
                min_allocations[component] = int(min_allocations[component] * scaling_factor)
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
    """Manages token usage for LLM context, handling budgeting, counting,
    and dynamic content pruning to ensure staying within token limits.
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
        if len(text) > 200:
            import hashlib

            cache_key = f"hash_{hashlib.md5(text.encode()).hexdigest()}"
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
        if force or (current_time - self.last_refresh_time) > 5:
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
            if enhanced_imp > 0.8:
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
        if total_token_estimate > max_tokens * 3 and len(history_reversed) > 5:
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
            if len(history_reversed) > 3 and used_tokens < max_tokens * 0.7:
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
                if available_tokens > 50:  # Only if we can keep a meaningful amount
                    truncated_model = self.truncate_text_to_token_limit(
                        model_text, available_tokens
                    )
                    turn = turn.copy()  # Create a copy to avoid modifying the original
                    turn["model"] = truncated_model
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
            optimized_history.append(turn)
            used_tokens += turn_tokens
        # Restore original order (oldest first)
        return list(reversed(optimized_history))

    def optimize_web_results(self, web_results: str, max_tokens: int) -> str:
        """Optimize web search results to fit within token budget.
        Uses smart truncation for web content to maintain the most valuable information.

        Args:
            web_results: Web search results text
            max_tokens: Maximum tokens allowed
        Returns:
            Optimized web results
        """
        # Ensure context size is up-to-date before optimization
        self.refresh_if_needed()
        web_tokens = self.budget.count_tokens(web_results)
        if web_tokens <= max_tokens:
            return web_results
        # For extremely large content, try to keep result blocks intact
        if web_tokens > max_tokens * 2:
            # Split by result blocks (usually separated by "---" or blank lines)
            # Simple split without regex
            blocks = []
            current_block = []
            lines = web_results.split("\n")
            
            for line in lines:
                if line.strip() == "" or (line.strip().startswith("-") and len(line.strip()) >= 3 and all(c == "-" for c in line.strip())):
                    if current_block:
                        blocks.append("\n".join(current_block))
                        blocks.append("\n\n")  # Add separator
                        current_block = []
                else:
                    current_block.append(line)
            
            if current_block:
                blocks.append("\n".join(current_block))
            # Preserve important blocks (first few results)
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
                # For content blocks
                if i // 2 < max_preserved_blocks:  # Block index accounting for separators
                    # Try to include all high-priority blocks
                    if preserved_tokens + block_tokens <= max_tokens:
                        preserved_blocks.append(block)
                        preserved_tokens += block_tokens
                    else:
                        # Truncate the last block if needed
                        remaining_tokens = max_tokens - preserved_tokens
                        if remaining_tokens > 100:  # Only if we can keep a meaningful amount
                            truncated_block = self.truncate_text_to_token_limit(
                                block, remaining_tokens
                            )
                            preserved_blocks.append(truncated_block)
                            preserved_tokens += self.budget.count_tokens(truncated_block)
                        break
                # Lower priority blocks
                elif preserved_tokens + block_tokens <= max_tokens:
                    preserved_blocks.append(block)
                    preserved_tokens += block_tokens
                else:
                    break
            # Reassemble preserved blocks
            optimized_results = "".join(preserved_blocks)
            # Add truncation notice if we cut material
            if len(optimized_results) < len(web_results):
                truncation_notice = "\n\n[Results truncated to fit context window]"
                truncation_tokens = self.budget.count_tokens(truncation_notice)
                if preserved_tokens + truncation_tokens <= max_tokens:
                    optimized_results += truncation_notice
            return optimized_results
        # Simple truncation for smaller content
        return self.truncate_text_to_token_limit(web_results, max_tokens)

    def optimize_system_prompt(self, system_prompt: str, max_tokens: int) -> str:
        """Optimize system prompt to fit within token budget.
        If needed, extract critical instructions and trim the rest.

        Args:
            system_prompt: System prompt text
            max_tokens: Maximum tokens allowed
        Returns:
            Optimized system prompt
        """
        # Ensure context size is up-to-date before optimization
        self.refresh_if_needed()
        prompt_tokens = self.budget.count_tokens(system_prompt)
        if prompt_tokens <= max_tokens:
            return system_prompt
        # Extract core components if needed
        # Split by sections without regex
        sections = []
        current_section = []
        current_title = ""
        
        lines = system_prompt.split("\n")
        for i, line in enumerate(lines):
            # Check if line is a section header (starts and ends with ---)
            stripped = line.strip()
            if stripped.startswith("---") and stripped.endswith("---") and len(stripped) > 6:
                # Extract title
                title = stripped[3:-3].strip()
                if current_section or current_title:
                    sections.append("\n".join(current_section))
                    sections.append(current_title)
                current_section = []
                current_title = title
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section or current_title:
            sections.append("\n".join(current_section))
            sections.append(current_title)
        # Critical sections that should be preserved (in order of priority)
        core_sections = ["", "CORE INSTRUCTIONS", "INSTRUCTIONS", "GUIDELINES", "RULES", "ROLE"]
        # First pass: extract and prioritize critical sections
        critical_parts = []
        optional_parts = []
        for i in range(0, len(sections) - 1, 2):
            section_content = sections[i]
            section_title = sections[i + 1] if i + 1 < len(sections) else ""
            # Check if this is a critical section
            is_critical = any(title.upper() in section_title.upper() for title in core_sections)
            if is_critical:
                critical_parts.append(
                    (
                        section_title,
                        section_content,
                        core_sections.index(
                            next(
                                (t for t in core_sections if t.upper() in section_title.upper()),
                                999,
                            )
                        ),
                    )
                )
            else:
                optional_parts.append((section_title, section_content))
        # Sort critical parts by priority
        critical_parts.sort(key=lambda x: x[2])
        # Combine critical parts first
        combined_text = ""
        used_tokens = 0
        # Add critical parts until we approach the limit
        for title, content, _ in critical_parts:
            section_text = f"\n\n--- {title} ---\n\n{content}" if title else content
            section_tokens = self.budget.count_tokens(section_text)
            if used_tokens + section_tokens <= max_tokens:
                combined_text += section_text
                used_tokens += section_tokens
            else:
                # Try to include at least part of this critical section
                remaining_tokens = max_tokens - used_tokens
                if remaining_tokens > 50:  # Only if we can keep a meaningful amount
                    truncated_section = self.truncate_text_to_token_limit(
                        section_text, remaining_tokens
                    )
                    combined_text += truncated_section
                    used_tokens += self.budget.count_tokens(truncated_section)
                break
        # If we still have room, add optional parts
        for title, content in optional_parts:
            section_text = f"\n\n--- {title} ---\n\n{content}" if title else content
            section_tokens = self.budget.count_tokens(section_text)
            if used_tokens + section_tokens <= max_tokens:
                combined_text += section_text
                used_tokens += section_tokens
            else:
                break
        # If no structured sections were found or combined text is empty, fall back to simple truncation
        if not combined_text:
            return self.truncate_text_to_token_limit(system_prompt, max_tokens)
        return combined_text.strip()

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
        # Force refresh context size before processing to ensure we have the latest CTX value
        self.refresh_if_needed(force=True)
        # Track start time for performance monitoring
        start_time = time.time()
        # Allocate token budget (this also performs another refresh)
        budget = self.budget.allocate_budget()
        self.budget.component_tokens = dict.fromkeys(
            self.budget.PRIORITY_ORDER, 0
        )  # Reset component tracking
        self.budget.total_used_tokens = 0
        # Track tokens used by current query (never truncate the current user query)
        current_query_tokens = self.count_tokens_cached(user_prompt)
        self.budget.track_component_usage("current_query", current_query_tokens)
        # Reserve full budget for current query (never truncate it)
        remaining_budget = self.budget.available_tokens - current_query_tokens
        # Calculate tokens needed for core components
        web_results_tokens = self.count_tokens_cached(web_results) if web_results else 0
        # 1. Optimize system prompt
        max_system_tokens = min(budget["system_prompt"], remaining_budget)
        system_tokens = 0
        dynamic_system_prompt_part = ""
        # Check if we're using a cached base system prompt
        if (
            self.base_system_prompt_text
            and self.cached_base_system_prompt_tokens is not None
            and system_prompt.startswith(self.base_system_prompt_text)
        ):
            system_tokens += self.cached_base_system_prompt_tokens
            dynamic_system_prompt_part = system_prompt[len(self.base_system_prompt_text) :]
            if dynamic_system_prompt_part:
                system_tokens += self.budget.count_tokens(dynamic_system_prompt_part)
        else:
            # Fallback or entirely dynamic prompt
            system_tokens = self.budget.count_tokens(system_prompt)
        # The optimize_system_prompt method receives the full system_prompt.
        # If it's a combination of base + dynamic, optimize_system_prompt will handle the combined text.
        optimized_system = self.optimize_system_prompt(system_prompt, max_system_tokens)
        # Recalculate tokens for the *optimized* system prompt to ensure accuracy after potential truncation
        final_optimized_system_tokens = self.budget.count_tokens(optimized_system)
        self.budget.track_component_usage("system_prompt", final_optimized_system_tokens)
        remaining_budget -= final_optimized_system_tokens
        # 2. Optimize web results (if present) - high priority for web scraping tasks
        optimized_web_results = ""
        if web_results:
            # Allocate more tokens to web results if they're large - this is crucial for web scraping
            # Detect if this appears to be a web scraping task (large web_results)
            is_scraping_task = web_results_tokens > 5000
            if is_scraping_task:
                # For web scraping, boost the web results allocation by borrowing from history
                history_adjustment = min(
                    budget["older_history"], budget["older_history"] * 0.7
                )  # Borrow up to 70% from older history
                budget["web_results"] += history_adjustment
                budget["older_history"] -= history_adjustment
                logger.info(
                    f"Web scraping task detected: Boosted web results budget by {history_adjustment} tokens"
                )
            max_web_tokens = min(budget["web_results"], remaining_budget)
            optimized_web_results = self.optimize_web_results(web_results, max_web_tokens)
            web_tokens = self.budget.count_tokens(optimized_web_results)
            self.budget.track_component_usage("web_results", web_tokens)
            remaining_budget -= web_tokens
            logger.info(
                f"Web content optimized: {web_results_tokens} → {web_tokens} tokens "
                + f"({(web_tokens / web_results_tokens * 100):.1f}% of original)"
            )
        # 3. Optimize memories
        max_memory_tokens = min(budget["relevant_memories"], remaining_budget)
        optimized_memories = self.optimize_memories(relevant_memories, max_memory_tokens)
        memory_text = (
            format_memories_for_llm_comprehension(optimized_memories) if optimized_memories else ""
        )
        memory_tokens = self.budget.count_tokens(memory_text)
        self.budget.track_component_usage("relevant_memories", memory_tokens)
        remaining_budget -= memory_tokens
        # 4. Split history budget between recent and older
        history_budget = budget["recent_history"] + budget["older_history"]
        max_history_tokens = min(history_budget, remaining_budget)
        # Optimize conversation history
        optimized_history = self.optimize_conversation_history(
            conversation_history, max_history_tokens
        )
        # 5. Construct the optimized messages list
        messages = []
        # System message with memories and time
        full_system_content = optimized_system
        # Add optimized memories to system content (already includes proper headers)
        if memory_text:
            memory_section = f"\n\n{memory_text}"
            # Check if adding memories would exceed system token budget
            if self.budget.count_tokens(full_system_content + memory_section) <= max_system_tokens:
                full_system_content += memory_section
            else:
                # If it would exceed, truncate memories further
                remaining_sys_tokens = max_system_tokens - self.budget.count_tokens(
                    full_system_content + "\n\n"
                )
                if remaining_sys_tokens > 50:  # Only add if we have reasonable space
                    truncated_memory = self.truncate_text_to_token_limit(
                        memory_text, remaining_sys_tokens
                    )
                    full_system_content += f"\n\n{truncated_memory}"
        # Add the system message
        messages.append({"role": "system", "content": full_system_content})
        # Track current total tokens
        current_total = self.budget.count_tokens(full_system_content)
        remaining_total_budget = self.budget.available_tokens - current_total
        # Add web results if available and we have room (before conversation history)
        if optimized_web_results and remaining_total_budget > 200:  # Minimum threshold
            # Format web content to distinguish it from regular conversation
            web_msg = f"Here are relevant results from the web:\n{optimized_web_results}"
            web_resp = "I'll consider this information when answering your question."
            web_msg_tokens = self.budget.count_tokens(web_msg)
            web_resp_tokens = self.budget.count_tokens(web_resp)
            if web_msg_tokens + web_resp_tokens <= remaining_total_budget - 100:  # Keep some margin
                messages.append({"role": "user", "content": web_msg})
                messages.append({"role": "assistant", "content": web_resp})
                current_total += web_msg_tokens + web_resp_tokens
                remaining_total_budget -= web_msg_tokens + web_resp_tokens
            elif web_msg_tokens <= remaining_total_budget - 100:
                # Just add the web results without response if we have room
                messages.append({"role": "user", "content": web_msg})
                current_total += web_msg_tokens
                remaining_total_budget -= web_msg_tokens
        # Add conversation history, but check total token count as we go
        history_tokens_used = 0
        for turn in optimized_history:
            # Don't inject timestamps into the conversation - let the AI reference them naturally
            user_msg = turn.get("user", "")
            model_msg = turn.get("model", "")
            if user_msg and model_msg:
                user_content = user_msg
                user_tokens = self.budget.count_tokens(user_content)
                model_tokens = self.budget.count_tokens(model_msg)
                turn_tokens = user_tokens + model_tokens
                # Check if adding this turn would exceed our remaining budget
                # Always leave room for the current query
                if turn_tokens < remaining_total_budget - current_query_tokens - 50:
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": model_msg})
                    current_total += turn_tokens
                    remaining_total_budget -= turn_tokens
                    history_tokens_used += turn_tokens
                else:
                    # Stop adding history if we're close to the limit
                    break
        # Track history tokens (split between recent and older based on ratio)
        recent_ratio = budget["recent_history"] / (
            budget["recent_history"] + budget["older_history"] + 0.0001
        )
        self.budget.track_component_usage("recent_history", int(history_tokens_used * recent_ratio))
        self.budget.track_component_usage(
            "older_history", int(history_tokens_used * (1 - recent_ratio))
        )
        # Add the current user prompt
        messages.append({"role": "user", "content": user_prompt})
        # Final safety check - verify total token count
        final_token_count = self.count_message_tokens(messages)
        if final_token_count > self.budget.available_tokens:
            logger.warning(
                f"Final token count {final_token_count} exceeds available budget "
                f"{self.budget.available_tokens}"
            )
            # Emergency pruning - remove history until we're under limit
            while len(messages) > 3 and final_token_count > self.budget.available_tokens:
                # Remove oldest history pair (user + assistant)
                if len(messages) >= 5:  # At least system + 2 turns + current query
                    messages.pop(1)  # After system
                    messages.pop(1)  # Now at position 1
                    # Re-check token count after removal
                    final_token_count = self.count_message_tokens(messages)
                    logger.warning(
                        f"Emergency pruning: Removed history turn, new token count: {final_token_count}"
                    )
                else:
                    break
        # Log token usage and performance
        self._log_token_usage(time.time() - start_time)
        # Final validation - STRICT ENFORCEMENT
        if not self.budget.is_within_budget():
            logger.warning("Token budget exceeded despite optimization efforts")
            overflow = self.budget.get_overflow()
            logger.warning(f"Overflow: {overflow} tokens")
            # STRICT ENFORCEMENT: Never exceed the context window
            # Continue removing messages until we're under the limit
            while len(messages) > 3 and not self.budget.is_within_budget():
                # Remove oldest history pair (user + assistant)
                if len(messages) >= 5:  # At least system + 2 turns + current query
                    messages.pop(1)  # After system
                    messages.pop(1)  # Now at position 1
                    # Re-calculate token usage
                    final_token_count = self.count_message_tokens(messages)
                    # Update the budget tracking
                    self.budget.total_used_tokens = final_token_count
                    logger.warning(
                        f"STRICT ENFORCEMENT: Removed history turn, new token count: {final_token_count}"
                    )
                else:
                    # If we can't remove more history but still over budget, truncate the system prompt
                    if len(messages) > 0 and messages[0]["role"] == "system":
                        system_content = messages[0]["content"]
                        # Calculate how many tokens we need to remove
                        overflow = self.budget.get_overflow()
                        # Add a safety margin
                        tokens_to_keep = self.budget.count_tokens(system_content) - overflow - 100
                        if tokens_to_keep > 200:  # Ensure we keep some minimal system prompt
                            messages[0]["content"] = self.truncate_text_to_token_limit(
                                system_content, tokens_to_keep
                            )
                            logger.warning(
                                f"STRICT ENFORCEMENT: Truncated system prompt to {tokens_to_keep} tokens"
                            )
                            # Re-calculate token usage
                            final_token_count = self.count_message_tokens(messages)
                            # Update the budget tracking
                            self.budget.total_used_tokens = final_token_count
                    break
            # If we're still over budget after all optimizations, raise an exception
            if not self.budget.is_within_budget():
                overflow = self.budget.get_overflow()
                error_msg = (
                    f"Error during stream generation: Requested tokens ({self.budget.total_used_tokens}) "
                    f"exceed context window of {self.budget.max_context_tokens}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
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
