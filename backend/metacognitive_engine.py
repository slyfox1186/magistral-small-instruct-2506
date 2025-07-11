#!/usr/bin/env python3
"""Metacognitive Engine v1.

Implements true "neural consciousness" through self-assessment and iterative improvement.
Based on Gemini's strategic analysis for hybrid heuristics + LLM criticism.
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum

import utils
from gpu_lock import GPULock, Priority

logger = logging.getLogger(__name__)

# Response evaluation constants
MIN_HELPFUL_RESPONSE_LENGTH = 30
LONG_RESPONSE_THRESHOLD = 100
MIN_CONFIDENT_RESPONSE_LENGTH = 20
MEDIUM_CONFIDENT_RESPONSE_LENGTH = 100

# Query classification constants
SHORT_QUERY_THRESHOLD = 20
LONG_QUERY_THRESHOLD = 200

# Quality thresholds
IMPROVEMENT_THRESHOLD = 0.7
WEAK_AREA_THRESHOLD = 0.6
MINIMUM_RESPONSE_LENGTH = 10
SHORT_RESPONSE_LENGTH = 50
EXCESSIVE_RESPONSE_LENGTH = 2000


class QualityTier(Enum):
    """Quality of Service tiers for metacognitive processing."""

    REAL_TIME = "real_time"  # Heuristics only, minimal latency
    BALANCED = "balanced"  # Heuristics + one LLM critic pass
    ANALYTICAL = "analytical"  # Full metacognitive loop with multiple passes


@dataclass
class ResponseAssessment:
    """Assessment scores for a generated response."""

    factual_accuracy: float = 0.0  # Is information correct?
    relevance: float = 0.0  # Does it answer the question?
    completeness: float = 0.0  # Is anything missing?
    clarity: float = 0.0  # Is it understandable?
    coherence: float = 0.0  # Does it flow logically?
    helpfulness: float = 0.0  # Does it solve the user's problem?
    confidence: float = 0.0  # How confident is the response?

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "factual_accuracy": 0.25,
            "relevance": 0.20,
            "completeness": 0.15,
            "clarity": 0.15,
            "coherence": 0.10,
            "helpfulness": 0.10,
            "confidence": 0.05,
        }

        total_score = 0.0
        for dimension, weight in weights.items():
            total_score += getattr(self, dimension) * weight

        return min(1.0, max(0.0, total_score))

    @property
    def needs_improvement(self) -> bool:
        """Check if response needs improvement based on threshold."""
        return self.overall_score < IMPROVEMENT_THRESHOLD

    def get_weak_areas(self, threshold: float = WEAK_AREA_THRESHOLD) -> list[str]:
        """Get list of areas that need improvement."""
        weak_areas = []
        for field_name, value in asdict(self).items():
            if field_name not in ["overall_score", "needs_improvement"] and value < threshold:
                weak_areas.append(field_name.replace("_", " "))
        return weak_areas


@dataclass
class ImprovementSuggestion:
    """Specific suggestion for response improvement."""

    area: str  # Which assessment area to improve
    issue: str  # What the specific issue is
    suggestion: str  # How to improve it
    priority: int = 1  # Priority level (1=high, 2=medium, 3=low)


class HeuristicEvaluator:
    """Fast, rule-based response evaluation.

    Implements the "System 1" thinking for immediate quality checks.
    """

    def __init__(self):
        """Initialize the heuristic evaluator."""
        # No longer rely on regex patterns - trust the LLM's capabilities
        pass

    def evaluate_response(
        self, response: str, user_query: str, context: str = ""
    ) -> ResponseAssessment:
        """Fast heuristic evaluation of response quality.

        Args:
            response: The generated response to evaluate
            user_query: The user's original query
            context: Additional context for evaluation

        Returns:
            ResponseAssessment with heuristic scores
        """
        assessment = ResponseAssessment()

        if not response or not response.strip():
            # Empty response
            return assessment

        # Basic checks
        assessment.clarity = self._evaluate_clarity(response)
        assessment.completeness = self._evaluate_completeness(response, user_query)
        assessment.coherence = self._evaluate_coherence(response)
        assessment.relevance = self._evaluate_relevance(response, user_query)
        assessment.helpfulness = self._evaluate_helpfulness(response, user_query)
        assessment.confidence = self._evaluate_confidence(response)

        # Factual accuracy is harder for heuristics - use conservative estimate
        assessment.factual_accuracy = 0.7  # Neutral assumption

        return assessment

    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity using simple length-based heuristics."""
        score = 1.0

        # Check for empty or very short responses
        response_length = len(response.strip())
        if response_length < MINIMUM_RESPONSE_LENGTH:
            score -= 0.5
        elif response_length < SHORT_RESPONSE_LENGTH:
            score -= 0.2

        # Check for excessive length (might be unclear rambling)
        if response_length > EXCESSIVE_RESPONSE_LENGTH:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _evaluate_completeness(self, response: str, user_query: str) -> float:
        """Evaluate response completeness based on length and query type."""
        score = 0.5  # Base score

        # Check for question words in query
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        query_lower = user_query.lower()

        # If query asks specific questions, give bonus for longer responses
        query_has_question = any(word in query_lower for word in question_words)
        if query_has_question and len(response) > LONG_RESPONSE_THRESHOLD:
            score += 0.3

        # Length consideration - very short responses likely incomplete
        response_length = len(response.strip())
        if response_length < SHORT_RESPONSE_LENGTH:
            score -= 0.3
        elif response_length > LONG_QUERY_THRESHOLD:
            score += 0.2

        return max(0.0, min(1.0, score))

    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate logical flow and coherence."""
        # Trust the LLM to produce coherent responses
        # Use a high base score with minor adjustments for length
        score = 0.8  # Higher base score - trust the model

        response_length = len(response.strip())
        if response_length < MIN_CONFIDENT_RESPONSE_LENGTH:
            # Very short responses might lack coherence
            score -= 0.2
        elif response_length > MEDIUM_CONFIDENT_RESPONSE_LENGTH:
            # Longer responses likely have better structure
            score += 0.1

        return max(0.0, min(1.0, score))

    def _evaluate_relevance(self, response: str, user_query: str) -> float:
        """Evaluate relevance to user query."""
        if not user_query:
            return 0.5

        # Simple word overlap check without regex
        query_words = set(user_query.lower().split())
        response_words = set(response.lower().split())

        # Calculate word overlap
        if query_words:
            overlap = len(query_words.intersection(response_words))
            relevance_score = min(1.0, 0.5 + (overlap / len(query_words)) * 0.5)
        else:
            relevance_score = 0.7  # Trust the model by default

        # Boost for question queries
        question_boost_threshold = 50
        if "?" in user_query and len(response) > question_boost_threshold:
            relevance_score += 0.1

        return max(0.0, min(1.0, relevance_score))

    def _evaluate_helpfulness(self, response: str, user_query: str) -> float:
        """Evaluate how helpful the response is."""
        # Trust the model to be helpful by default
        score = 0.7  # Higher base score

        # Simple length-based heuristic
        response_length = len(response.strip())
        if response_length < MIN_HELPFUL_RESPONSE_LENGTH:
            score -= 0.3  # Very short responses are less helpful
        elif response_length > LONG_RESPONSE_THRESHOLD:
            score += 0.2  # Longer responses tend to be more helpful

        return max(0.0, min(1.0, score))

    def _evaluate_confidence(self, response: str) -> float:
        """Evaluate confidence level of the response."""
        # Trust the model's natural confidence level
        # Use response length as a simple proxy
        response_length = len(response.strip())

        if response_length < MIN_CONFIDENT_RESPONSE_LENGTH:
            return 0.5  # Very short responses show less confidence
        elif response_length < MEDIUM_CONFIDENT_RESPONSE_LENGTH:
            return 0.7  # Medium confidence
        else:
            return 0.8  # Longer, detailed responses show confidence


class LLMCritic:
    """LLM-based response critic for deeper quality assessment.

    Implements the "System 2" thinking for nuanced evaluation.
    """

    def __init__(self, llm, model_lock: GPULock):
        """Initialize LLM critic with model and GPU lock."""
        self.llm = llm
        self.model_lock = model_lock

    async def evaluate_response(
        self, response: str, user_query: str, context: str = ""
    ) -> ResponseAssessment:
        """LLM-based evaluation of response quality.

        Args:
            response: The generated response to evaluate
            user_query: The user's original query
            context: Additional context for evaluation

        Returns:
            ResponseAssessment with LLM-derived scores
        """
        try:
            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(response, user_query, context)

            # Use model with LOWEST priority to avoid blocking main responses
            async with self.model_lock.acquire_context(Priority.LOW, timeout=20.0):
                # Generate evaluation using the main LLM
                formatted_prompt = utils.format_prompt(
                    "You are a precise AI response evaluator. Analyze responses objectively.",
                    evaluation_prompt,
                )

                evaluation_text = ""
                for chunk in self.llm(
                    formatted_prompt,
                    max_tokens=500,
                    temperature=0.1,  # Low temperature for consistent evaluation
                    stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                    stream=True,
                ):
                    if chunk["choices"][0]["finish_reason"] is None:
                        evaluation_text += chunk["choices"][0]["text"]

            # Parse the evaluation result
            assessment = self._parse_evaluation_result(evaluation_text)

        except Exception:
            logger.exception("LLM evaluation failed")
            # Return neutral assessment on failure
            assessment = ResponseAssessment(
                factual_accuracy=0.5,
                relevance=0.5,
                completeness=0.5,
                clarity=0.5,
                coherence=0.5,
                helpfulness=0.5,
                confidence=0.5,
            )

        return assessment

    def _create_evaluation_prompt(self, response: str, user_query: str, context: str) -> str:
        """Create structured evaluation prompt for the LLM."""
        return f"""Please evaluate this AI response based on the user's query and context provided.

**User Query:** {user_query}

**Context:** {context if context else "No additional context provided"}

**AI Response to Evaluate:**
{response}

Please evaluate the response on these criteria and provide a score from 0.0 to 1.0 for each:

1. **Factual Accuracy**: Is the information correct and truthful?
2. **Relevance**: Does it directly address the user's question or need?
3. **Completeness**: Does it provide a thorough answer without missing key points?
4. **Clarity**: Is it easy to understand and well-structured?
5. **Coherence**: Does it flow logically without contradictions?
6. **Helpfulness**: Does it actually help solve the user's problem?
7. **Confidence**: How confident and authoritative is the response?

For each criterion, also provide a brief justification (1-2 sentences).

Format your response as JSON:
{{
    "factual_accuracy": 0.8,
    "factual_accuracy_reason": "Information appears accurate based on general knowledge",
    "relevance": 0.9,
    "relevance_reason": "Directly addresses the user's question",
    "completeness": 0.7,
    "completeness_reason": "Covers main points but could include more detail",
    "clarity": 0.9,
    "clarity_reason": "Well-structured and easy to understand",
    "coherence": 0.8,
    "coherence_reason": "Logical flow with no contradictions",
    "helpfulness": 0.8,
    "helpfulness_reason": "Provides actionable information",
    "confidence": 0.7,
    "confidence_reason": "Appropriately confident without being overconfident"
}}"""

    def _parse_evaluation_result(self, evaluation_text: str) -> ResponseAssessment:
        """Parse LLM evaluation result into ResponseAssessment."""
        try:
            # Try to find JSON-like content by looking for curly braces
            start_idx = evaluation_text.find("{")
            end_idx = evaluation_text.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = evaluation_text[start_idx : end_idx + 1]
                eval_data = json.loads(json_str)

                return ResponseAssessment(
                    factual_accuracy=float(eval_data.get("factual_accuracy", 0.5)),
                    relevance=float(eval_data.get("relevance", 0.5)),
                    completeness=float(eval_data.get("completeness", 0.5)),
                    clarity=float(eval_data.get("clarity", 0.5)),
                    coherence=float(eval_data.get("coherence", 0.5)),
                    helpfulness=float(eval_data.get("helpfulness", 0.5)),
                    confidence=float(eval_data.get("confidence", 0.5)),
                )
            else:
                logger.warning("No JSON found in LLM evaluation response")
                # Return neutral assessment if no JSON found
                return ResponseAssessment(
                    factual_accuracy=0.7,
                    relevance=0.7,
                    completeness=0.7,
                    clarity=0.7,
                    coherence=0.7,
                    helpfulness=0.7,
                    confidence=0.7,
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM evaluation JSON: {e}")
            # Return neutral assessment on parse error
            return ResponseAssessment(
                factual_accuracy=0.7,
                relevance=0.7,
                completeness=0.7,
                clarity=0.7,
                coherence=0.7,
                helpfulness=0.7,
                confidence=0.7,
            )
        except Exception:
            logger.exception("Error parsing LLM evaluation")
            return ResponseAssessment()  # Default neutral scores


class MetacognitiveEngine:
    """Main metacognitive engine that orchestrates response evaluation and improvement.

    Implements true "neural consciousness" through self-reflection.
    """

    def __init__(self, llm, model_lock: GPULock):
        """Initialize metacognitive engine with model and GPU lock."""
        self.llm = llm
        self.model_lock = model_lock
        self.heuristic_evaluator = HeuristicEvaluator()
        self.llm_critic = LLMCritic(llm, model_lock)

        # Quality thresholds for different tiers
        self.quality_thresholds = {
            QualityTier.REAL_TIME: 0.5,  # Very lenient
            QualityTier.BALANCED: 0.7,  # Standard threshold
            QualityTier.ANALYTICAL: 0.8,  # High standard
        }

    async def evaluate_and_improve_response(
        self,
        response: str,
        user_query: str,
        context: str = "",
        quality_tier: QualityTier = QualityTier.BALANCED,
        max_iterations: int = 3,
    ) -> tuple[str, ResponseAssessment]:
        """Evaluate response and iteratively improve if needed.

        Args:
            response: Initial response to evaluate
            user_query: User's original query
            context: Additional context
            quality_tier: Quality of service tier
            max_iterations: Maximum improvement iterations

        Returns:
            Tuple of (final_response, final_assessment)
        """
        current_response = response
        iteration = 0

        while iteration < max_iterations:
            # Perform evaluation based on quality tier
            assessment = await self._evaluate_response(
                current_response, user_query, context, quality_tier
            )

            logger.info(
                f"Metacognitive evaluation (iter {iteration}): overall={assessment.overall_score:.3f}"
            )

            # Check if improvement is needed
            threshold = self.quality_thresholds[quality_tier]
            if assessment.overall_score >= threshold:
                logger.info(f"✅ Response meets quality threshold ({threshold:.1f})")
                break

            # Skip improvement for real-time tier
            if quality_tier == QualityTier.REAL_TIME:
                logger.info("🚀 Real-time tier - skipping improvement loop")
                break

            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                assessment, current_response, user_query
            )

            if not suggestions:
                logger.info("No specific improvements identified")
                break

            # Attempt to improve the response
            logger.info(f"🔄 Attempting response improvement (iteration {iteration + 1})")
            improved_response = await self._improve_response(
                current_response, user_query, context, suggestions
            )

            if improved_response and improved_response != current_response:
                current_response = improved_response
                iteration += 1
            else:
                logger.info("No improvement generated, stopping iteration")
                break

        # Final evaluation
        final_assessment = await self._evaluate_response(
            current_response, user_query, context, quality_tier
        )

        logger.info(f"Final metacognitive score: {final_assessment.overall_score:.3f}")
        return current_response, final_assessment

    async def _evaluate_response(
        self, response: str, user_query: str, context: str, quality_tier: QualityTier
    ) -> ResponseAssessment:
        """Evaluate response based on quality tier."""
        # Always start with fast heuristic evaluation
        heuristic_assessment = self.heuristic_evaluator.evaluate_response(
            response, user_query, context
        )

        # For real-time tier, only use heuristics
        if quality_tier == QualityTier.REAL_TIME:
            return heuristic_assessment

        # For balanced and analytical tiers, also use LLM critic
        try:
            llm_assessment = await self.llm_critic.evaluate_response(response, user_query, context)

            # Blend heuristic and LLM assessments (70% LLM, 30% heuristic)
            blended_assessment = ResponseAssessment(
                factual_accuracy=0.7 * llm_assessment.factual_accuracy
                + 0.3 * heuristic_assessment.factual_accuracy,
                relevance=0.7 * llm_assessment.relevance + 0.3 * heuristic_assessment.relevance,
                completeness=0.7 * llm_assessment.completeness
                + 0.3 * heuristic_assessment.completeness,
                clarity=0.7 * llm_assessment.clarity + 0.3 * heuristic_assessment.clarity,
                coherence=0.7 * llm_assessment.coherence + 0.3 * heuristic_assessment.coherence,
                helpfulness=0.7 * llm_assessment.helpfulness
                + 0.3 * heuristic_assessment.helpfulness,
                confidence=0.7 * llm_assessment.confidence + 0.3 * heuristic_assessment.confidence,
            )

        except Exception as e:
            logger.warning(f"LLM evaluation failed, using heuristics only: {e}")
            blended_assessment = heuristic_assessment

        return blended_assessment

    def _generate_improvement_suggestions(
        self, assessment: ResponseAssessment, response: str, user_query: str
    ) -> list[ImprovementSuggestion]:
        """Generate specific improvement suggestions based on assessment."""
        suggestions = []

        weak_areas = assessment.get_weak_areas(threshold=0.6)

        for area in weak_areas:
            if area == "factual accuracy":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Information may be inaccurate or unverified",
                        suggestion="Verify facts and provide more reliable information",
                        priority=1,
                    )
                )
            elif area == "relevance":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Response doesn't directly address the user's query",
                        suggestion="Focus more directly on answering the specific question asked",
                        priority=1,
                    )
                )
            elif area == "completeness":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Response is missing important information",
                        suggestion="Provide more comprehensive coverage of the topic",
                        priority=2,
                    )
                )
            elif area == "clarity":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Response is unclear or confusing",
                        suggestion="Restructure for better clarity and readability",
                        priority=1,
                    )
                )
            elif area == "coherence":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Response lacks logical flow or contains contradictions",
                        suggestion="Improve logical structure and remove contradictions",
                        priority=2,
                    )
                )
            elif area == "helpfulness":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Response doesn't help solve the user's problem",
                        suggestion="Provide more actionable and practical information",
                        priority=1,
                    )
                )
            elif area == "confidence":
                suggestions.append(
                    ImprovementSuggestion(
                        area=area,
                        issue="Response shows uncertainty or lack of confidence",
                        suggestion="Provide more authoritative and confident information",
                        priority=3,
                    )
                )

        # Sort by priority
        suggestions.sort(key=lambda s: s.priority)
        return suggestions

    async def _improve_response(
        self,
        original_response: str,
        user_query: str,
        context: str,
        suggestions: list[ImprovementSuggestion],
    ) -> str | None:
        """Generate an improved version of the response."""
        try:
            # Create improvement prompt
            improvement_prompt = self._create_improvement_prompt(
                original_response, user_query, context, suggestions
            )

            # Use model to generate improved response with LOWEST priority
            async with self.model_lock.acquire_context(Priority.LOW, timeout=20.0):
                formatted_prompt = utils.format_prompt(
                    "You are a helpful AI assistant that improves responses based on feedback.",
                    improvement_prompt,
                )

                improved_text = ""
                for chunk in self.llm(
                    formatted_prompt,
                    max_tokens=1000,
                    temperature=0.3,  # Slightly creative for improvements
                    stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                    stream=True,
                ):
                    if chunk["choices"][0]["finish_reason"] is None:
                        improved_text += chunk["choices"][0]["text"]

            return improved_text.strip() if improved_text.strip() else None

        except Exception:
            logger.exception("Response improvement failed")
            return None

    def _create_improvement_prompt(
        self,
        original_response: str,
        user_query: str,
        context: str,
        suggestions: list[ImprovementSuggestion],
    ) -> str:
        """Create prompt for response improvement."""
        suggestions_text = "\n".join(
            [f"- {s.area.title()}: {s.suggestion}" for s in suggestions[:3]]  # Top 3 suggestions
        )

        return f"""Please improve the following AI response based on the specific feedback provided.

**User's Original Query:** {user_query}

**Context:** {context if context else "No additional context"}

**Current Response:**
{original_response}

**Improvement Areas:**
{suggestions_text}

Please provide an improved version that addresses these specific issues while maintaining
the helpful and informative tone. Keep the response concise but comprehensive."""


# ===================== QUALITY TIER SELECTOR =====================


def select_quality_tier(user_query: str, context: str = "") -> QualityTier:
    """Automatically select appropriate quality tier based on query analysis.

    Args:
        user_query: User's query to analyze
        context: Additional context

    Returns:
        Appropriate QualityTier for the query
    """
    query_lower = user_query.lower()
    query_length = len(user_query.strip())

    # Check for real-time keywords (quick, casual queries)
    real_time_keywords = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "joke",
        "funny",
        "quick",
        "briefly",
    ]
    for keyword in real_time_keywords:
        if keyword in query_lower:
            return QualityTier.REAL_TIME

    # Check for analytical keywords (complex, detailed requests)
    analytical_keywords = [
        "detailed",
        "comprehensive",
        "thorough",
        "complete",
        "in-depth",
        "analyze",
        "analysis",
        "report",
        "research",
        "study",
        "explain everything",
        "tell me all",
        "full explanation",
        "business plan",
        "strategy",
        "architecture",
        "design",
    ]
    for keyword in analytical_keywords:
        if keyword in query_lower:
            return QualityTier.ANALYTICAL

    # Check query length
    if query_length < SHORT_QUERY_THRESHOLD:
        return QualityTier.REAL_TIME
    elif query_length > LONG_QUERY_THRESHOLD:
        return QualityTier.ANALYTICAL

    # Default to balanced
    return QualityTier.BALANCED


# ===================== GLOBAL INSTANCE =====================

_global_metacognitive_engine: MetacognitiveEngine | None = None


def initialize_metacognitive_engine(llm, model_lock: GPULock) -> MetacognitiveEngine:
    """Initialize global metacognitive engine."""
    globals()["_global_metacognitive_engine"] = MetacognitiveEngine(llm, model_lock)
    logger.info("🧠 Metacognitive Engine v1 initialized successfully")
    return _global_metacognitive_engine


def get_metacognitive_engine() -> MetacognitiveEngine | None:
    """Get global metacognitive engine instance."""
    return _global_metacognitive_engine
