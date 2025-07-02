#!/usr/bin/env python3
"""Metacognitive Engine v1
Implements true "neural consciousness" through self-assessment and iterative improvement.
Based on Gemini's strategic analysis for hybrid heuristics + LLM criticism.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum

from llama_cpp import Llama

import utils
from gpu_lock import GPULock, Priority

logger = logging.getLogger(__name__)


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
        return self.overall_score < 0.7

    def get_weak_areas(self, threshold: float = 0.6) -> list[str]:
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
        # Common patterns for poor responses
        self.empty_response_patterns = [
            r"^\s*$",
            r"^I don\'t know\.?\s*$",
            r"^I\'m not sure\.?\s*$",
            r"^I cannot answer\.?\s*$",
        ]

        self.repetition_patterns = [
            r"(.{10,}?)\1{2,}",  # Same phrase repeated 3+ times
            r"(\b\w+\b)(\s+\1){3,}",  # Same word repeated 4+ times
        ]

        self.placeholder_patterns = [
            r"\[.*?\]",  # [placeholder text]
            r"{{.*?}}",  # {{placeholder}}
            r"TODO:",  # TODO markers
            r"FIXME:",  # FIXME markers
            r"XXX",  # XXX markers
        ]

        # Quality indicators
        self.positive_indicators = [
            r"\b(?:specific|detailed|comprehensive|thorough|clear|precise)\b",
            r"\b(?:example|instance|illustration|demonstration)\b",
            r"\b(?:because|therefore|consequently|as a result)\b",
            r"\b(?:first|second|third|finally|in conclusion)\b",
        ]

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
        """Evaluate response clarity using heuristics."""
        score = 1.0

        # Check for empty or very short responses
        if len(response.strip()) < 10:
            score -= 0.5

        # Check for repetition
        for pattern in self.repetition_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score -= 0.3
                break

        # Check for placeholder text
        for pattern in self.placeholder_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score -= 0.4
                break

        # Check for excessive length (might be unclear rambling)
        if len(response) > 2000:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _evaluate_completeness(self, response: str, user_query: str) -> float:
        """Evaluate response completeness."""
        score = 0.5  # Base score

        # Check for question words in query and corresponding answers
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        query_lower = user_query.lower()
        response_lower = response.lower()

        # If query asks specific questions, check if response addresses them
        query_has_question = any(word in query_lower for word in question_words)
        if query_has_question:
            # Look for answering patterns
            answer_patterns = [
                r"\b(?:the answer is|this is|here\'s how|because|due to)\b",
                r"\b(?:you can|to do this|follow these|step by step)\b",
            ]

            if any(re.search(pattern, response_lower) for pattern in answer_patterns):
                score += 0.3

        # Check for structured content (lists, steps, examples)
        if re.search(r"(?:\d+\.|•|\*|\-)\s+", response):
            score += 0.2

        # Length consideration - very short responses likely incomplete
        if len(response) < 50:
            score -= 0.3
        elif len(response) > 200:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate logical flow and coherence."""
        score = 0.7  # Base score

        # Check for connecting words
        connecting_words = [
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
            "meanwhile",
            "similarly",
            "in contrast",
            "for example",
        ]

        response_lower = response.lower()
        connections_found = sum(1 for word in connecting_words if word in response_lower)

        if connections_found > 0:
            score += min(0.2, connections_found * 0.05)

        # Check for contradictions (simple patterns)
        contradiction_patterns = [
            r"\b(?:yes|true|correct)\b.*?\b(?:no|false|incorrect|wrong)\b",
            r"\b(?:always|never)\b.*?\b(?:sometimes|often|rarely)\b",
        ]

        for pattern in contradiction_patterns:
            if re.search(pattern, response_lower, re.DOTALL):
                score -= 0.3
                break

        return max(0.0, min(1.0, score))

    def _evaluate_relevance(self, response: str, user_query: str) -> float:
        """Evaluate relevance to user query."""
        if not user_query:
            return 0.5

        # Extract key terms from query
        query_words = set(re.findall(r"\b\w{3,}\b", user_query.lower()))
        response_words = set(re.findall(r"\b\w{3,}\b", response.lower()))

        # Calculate word overlap
        if query_words:
            overlap = len(query_words.intersection(response_words))
            relevance_score = min(1.0, overlap / len(query_words))
        else:
            relevance_score = 0.5

        # Boost for direct question answering
        if "?" in user_query:
            answer_patterns = [
                r"\byes\b",
                r"\bno\b",
                r"\bthe answer is\b",
                r"\bthis is\b",
                r"\byou can\b",
            ]
            if any(re.search(pattern, response.lower()) for pattern in answer_patterns):
                relevance_score += 0.2

        return max(0.0, min(1.0, relevance_score))

    def _evaluate_helpfulness(self, response: str, user_query: str) -> float:
        """Evaluate how helpful the response is."""
        score = 0.5  # Base score

        response_lower = response.lower()

        # Check for helpful patterns
        helpful_patterns = [
            r"\b(?:here\'s how|you can|try this|follow these steps)\b",
            r"\b(?:example|for instance|such as)\b",
            r"\b(?:tip|advice|suggestion|recommendation)\b",
        ]

        for pattern in helpful_patterns:
            if re.search(pattern, response_lower):
                score += 0.2
                break

        # Check for unhelpful patterns
        unhelpful_patterns = [
            r"\bi don\'t know\b",
            r"\bi can\'t help\b",
            r"\bi\'m not sure\b",
            r"\bthat\'s not possible\b",
        ]

        for pattern in unhelpful_patterns:
            if re.search(pattern, response_lower):
                score -= 0.4
                break

        return max(0.0, min(1.0, score))

    def _evaluate_confidence(self, response: str) -> float:
        """Evaluate confidence level of the response."""
        response_lower = response.lower()

        # Confident indicators
        confident_patterns = [
            r"\b(?:definitely|certainly|absolutely|clearly|obviously)\b",
            r"\b(?:the answer is|this is|you should|you can)\b",
        ]

        uncertain_patterns = [
            r"\b(?:maybe|perhaps|possibly|might|could|probably)\b",
            r"\b(?:i think|i believe|it seems|appears to)\b",
            r"\b(?:not sure|don\'t know|unclear)\b",
        ]

        confident_count = sum(
            1 for pattern in confident_patterns if re.search(pattern, response_lower)
        )
        uncertain_count = sum(
            1 for pattern in uncertain_patterns if re.search(pattern, response_lower)
        )

        # Calculate confidence score
        if confident_count > uncertain_count:
            return min(1.0, 0.7 + (confident_count * 0.1))
        elif uncertain_count > confident_count:
            return max(0.2, 0.7 - (uncertain_count * 0.15))
        else:
            return 0.7  # Neutral


class LLMCritic:
    """LLM-based response critic for deeper quality assessment.
    Implements the "System 2" thinking for nuanced evaluation.
    """

    def __init__(self, llm: Llama, model_lock: GPULock):
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
            return assessment

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Return neutral assessment on failure
            return ResponseAssessment(
                factual_accuracy=0.5,
                relevance=0.5,
                completeness=0.5,
                clarity=0.5,
                coherence=0.5,
                helpfulness=0.5,
                confidence=0.5,
            )

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
            # Extract JSON from the response
            json_match = re.search(r"\{.*?\}", evaluation_text, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())

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
                return self._fallback_parse(evaluation_text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM evaluation JSON: {e}")
            return self._fallback_parse(evaluation_text)
        except Exception as e:
            logger.error(f"Error parsing LLM evaluation: {e}")
            return ResponseAssessment()  # Default neutral scores

    def _fallback_parse(self, evaluation_text: str) -> ResponseAssessment:
        """Fallback parsing using regex for scores."""
        assessment = ResponseAssessment()

        # Try to extract scores using regex
        score_patterns = {
            "factual_accuracy": r"factual[_\s]*accuracy[:\s]*([0-9]\.[0-9])",
            "relevance": r"relevance[:\s]*([0-9]\.[0-9])",
            "completeness": r"completeness[:\s]*([0-9]\.[0-9])",
            "clarity": r"clarity[:\s]*([0-9]\.[0-9])",
            "coherence": r"coherence[:\s]*([0-9]\.[0-9])",
            "helpfulness": r"helpfulness[:\s]*([0-9]\.[0-9])",
            "confidence": r"confidence[:\s]*([0-9]\.[0-9])",
        }

        text_lower = evaluation_text.lower()
        for dimension, pattern in score_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    score = float(match.group(1))
                    setattr(assessment, dimension, score)
                except ValueError:
                    pass

        return assessment


class MetacognitiveEngine:
    """Main metacognitive engine that orchestrates response evaluation and improvement.
    Implements true "neural consciousness" through self-reflection.
    """

    def __init__(self, llm: Llama, model_lock: GPULock):
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

            return blended_assessment

        except Exception as e:
            logger.warning(f"LLM evaluation failed, using heuristics only: {e}")
            return heuristic_assessment

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

        except Exception as e:
            logger.error(f"Response improvement failed: {e}")
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

    # Real-time tier patterns (quick, casual queries)
    real_time_patterns = [
        r"\b(?:hi|hello|hey|thanks|thank you)\b",
        r"\b(?:joke|funny|laugh)\b",
        r"\b(?:quick|fast|briefly)\b",
        r"^\w{1,3}\?$",  # Very short questions like "Why?"
    ]

    # Analytical tier patterns (complex, detailed requests)
    analytical_patterns = [
        r"\b(?:detailed|comprehensive|thorough|complete|in-depth)\b",
        r"\b(?:analyze|analysis|report|research|study)\b",
        r"\b(?:explain everything|tell me all|full explanation)\b",
        r"\b(?:business plan|strategy|architecture|design)\b",
        r"\b(?:take your time|be thorough|don\'t rush)\b",
        r"write.{0,20}(?:detailed|comprehensive|thorough)",
    ]

    # Check for real-time patterns
    for pattern in real_time_patterns:
        if re.search(pattern, query_lower):
            return QualityTier.REAL_TIME

    # Check for analytical patterns
    for pattern in analytical_patterns:
        if re.search(pattern, query_lower):
            return QualityTier.ANALYTICAL

    # Check query length and complexity
    if len(user_query) < 20:
        return QualityTier.REAL_TIME
    elif len(user_query) > 200:
        return QualityTier.ANALYTICAL

    # Default to balanced
    return QualityTier.BALANCED


# ===================== GLOBAL INSTANCE =====================

_global_metacognitive_engine: MetacognitiveEngine | None = None


def initialize_metacognitive_engine(llm: Llama, model_lock: GPULock) -> MetacognitiveEngine:
    """Initialize global metacognitive engine."""
    global _global_metacognitive_engine
    _global_metacognitive_engine = MetacognitiveEngine(llm, model_lock)
    logger.info("🧠 Metacognitive Engine v1 initialized successfully")
    return _global_metacognitive_engine


def get_metacognitive_engine() -> MetacognitiveEngine | None:
    """Get global metacognitive engine instance."""
    return _global_metacognitive_engine
