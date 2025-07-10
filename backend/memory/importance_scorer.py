"""Sophisticated memory importance scoring system using NLP and heuristics.
Calculates dynamic importance scores based on content analysis, entities, sentiment, and conversation context.
"""

import logging
import re

try:
    import spacy
    from spacytextblob.spacytextblob import SpacyTextBlob
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("SpaCy not available. Falling back to heuristic-only scoring.")

logger = logging.getLogger(__name__)


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(value, max_val))


class MemoryImportanceScorer:
    """Calculates an importance score using a hybrid heuristic and NLP model.
    
    Score ranges:
    - 0.0-0.3: Low importance (greetings, fillers)
    - 0.3-0.6: Medium importance (general conversation)
    - 0.6-0.8: High importance (questions, answers, facts)
    - 0.8-1.0: Critical importance (instructions, personal info, key decisions)
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}

        # Load NLP model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_md")
                self.nlp.add_pipe("spacytextblob")
                logger.info("Loaded spaCy model for advanced importance scoring")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}. Using heuristic-only scoring.")
                self.nlp = None

        # Tunable weights for each feature
        self.weights = {
            # Basic heuristic weights
            "length_weight": config.get("length_weight", 0.05),
            "question_weight": config.get("question_weight", 0.15),
            "keyword_weight": config.get("keyword_weight", 0.25),
            "instruction_weight": config.get("instruction_weight", 0.20),
            "answer_boost": config.get("answer_boost", 0.1),
            # Advanced NLP-based weights
            "entity_weight": config.get("entity_weight", 0.3),
            "sentiment_weight": config.get("sentiment_weight", 0.15),
            "pii_weight": config.get("pii_weight", 0.4),  # High weight for PII
            "topic_shift_weight": config.get("topic_shift_weight", 0.2),
        }

        # Base scores by role
        self.base_scores = {
            "user": config.get("base_score_user", 0.2),
            "assistant": config.get("base_score_assistant", 0.25),
        }

        # Keywords and patterns
        self.high_importance_keywords = [
            "remember", "important", "critical", "key point", "don't forget",
            "fact is", "rule is", "my name is", "my birthday is", "my email is",
            "my phone is", "deadline", "urgent", "password", "credential"
        ]

        self.low_importance_keywords = [
            "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "cool",
            "sounds good", "got it", "understood", "bye", "goodbye", "see you"
        ]

        self.instructional_verbs = [
            "create", "generate", "summarize", "analyze", "list", "find", "explain",
            "code", "write", "build", "run", "implement", "design", "calculate",
            "develop", "configure", "setup", "install", "debug", "fix"
        ]

        # Regex patterns for PII detection
        self.pii_patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "DATE": re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
        }

        # Important entity types for spaCy
        self.important_entities = ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "TIME", "PERCENT"]

    def _score_length(self, text: str) -> float:
        """Scores based on length with diminishing returns."""
        # Short messages (<20 chars) get 0.0
        # Medium messages (20-200 chars) scale linearly
        # Long messages (>200 chars) cap at 1.0
        if len(text) < 20:
            return 0.0
        elif len(text) > 200:
            return 1.0
        else:
            return (len(text) - 20) / 180

    def _is_question(self, text: str) -> bool:
        """Checks if the text is a question."""
        text = text.strip().lower()
        # Check for question mark or question words at start
        question_starters = ["who", "what", "where", "when", "why", "how", "do", "does", "is", "are", "can", "could", "will", "would", "should"]
        return text.endswith("?") or any(text.startswith(q + " ") for q in question_starters)

    def _score_keywords(self, text: str) -> float:
        """Scores based on matching high/low importance keywords."""
        text_lower = text.lower()

        # Check high importance keywords
        high_count = sum(1 for keyword in self.high_importance_keywords if keyword in text_lower)
        if high_count > 0:
            return min(1.0, high_count * 0.5)  # Cap at 1.0, but allow multiple keywords to increase score

        # Check low importance keywords
        low_count = sum(1 for keyword in self.low_importance_keywords if keyword in text_lower)
        if low_count > 0:
            return max(-1.0, -low_count * 0.3)  # Negative score, but don't go below -1.0

        return 0.0

    def _score_instructions(self, text: str) -> float:
        """Scores based on presence of instructional verbs."""
        text_lower = text.lower()
        verb_count = sum(1 for verb in self.instructional_verbs if verb in text_lower)
        return min(1.0, verb_count * 0.5)  # Multiple verbs increase score

    def _score_pii(self, text: str) -> float:
        """Scores based on presence of PII patterns."""
        pii_found = []
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(text):
                pii_found.append(pii_type)

        if pii_found:
            logger.debug(f"Found PII types: {pii_found}")
            return 1.0
        return 0.0

    def _score_nlp_features(self, doc) -> dict[str, float]:
        """Extracts scores for entities and sentiment from a spaCy doc."""
        if not self.nlp:
            return {"sentiment": 0.0, "entities": 0.0}

        # Sentiment: Polarity is [-1, 1]. We care about magnitude, not direction.
        try:
            sentiment_score = abs(doc._.blob.polarity)  # Score is [0, 1]
        except:
            sentiment_score = 0.0

        # Entities: Score based on number and type of entities
        entity_score = 0.0
        important_entity_count = sum(1 for ent in doc.ents if ent.label_ in self.important_entities)
        if important_entity_count > 0:
            entity_score = min(1.0, important_entity_count * 0.3)

        return {"sentiment": sentiment_score, "entities": entity_score}

    def _score_topic_shift(self, doc, prev_doc) -> float:
        """Scores based on semantic similarity to the previous message."""
        if not self.nlp or prev_doc is None:
            return 0.0

        try:
            if not prev_doc.has_vector or not doc.has_vector:
                return 0.0

            similarity = doc.similarity(prev_doc)
            # Low similarity (<0.6) suggests a topic shift, which is important
            if similarity < 0.6:
                return 1.0
            elif similarity < 0.8:
                return 0.5
            return 0.0
        except:
            return 0.0

    def calculate_importance(
        self,
        text: str,
        role: str,
        conversation_history: list[dict[str, str]] | None = None
    ) -> float:
        """Calculates the final importance score for a given message.
        
        Args:
            text: The message content
            role: "user" or "assistant"
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Float between 0.0 and 1.0 representing importance
        """
        if not text or not text.strip():
            return 0.0

        # Start with base score
        base_score = self.base_scores.get(role, 0.2)

        # Process text with spaCy if available
        doc = None
        if self.nlp:
            try:
                doc = self.nlp(text)
            except Exception as e:
                logger.warning(f"spaCy processing failed: {e}")

        # Basic heuristic scores
        length_score = self._score_length(text)
        keyword_score = self._score_keywords(text)
        instruction_score = self._score_instructions(text)
        question_score = 1.0 if self._is_question(text) else 0.0
        pii_score = self._score_pii(text)

        # Advanced NLP scores (if available)
        entity_score = 0.0
        sentiment_score = 0.0
        topic_shift_score = 0.0

        if doc:
            nlp_scores = self._score_nlp_features(doc)
            entity_score = nlp_scores["entities"]
            sentiment_score = nlp_scores["sentiment"]

            # Check for topic shift
            if conversation_history and len(conversation_history) > 0:
                last_msg = conversation_history[-1]
                if self.nlp:
                    try:
                        last_doc = self.nlp(last_msg["content"])
                        topic_shift_score = self._score_topic_shift(doc, last_doc)
                    except:
                        pass

        # Contextual answer boost
        if role == "assistant" and conversation_history:
            # Look for the last user message
            for msg in reversed(conversation_history):
                if msg["role"] == "user":
                    if self._is_question(msg["content"]):
                        base_score += self.weights["answer_boost"]
                    break

        # Combine all scores using weights
        final_score = (
            base_score
            + length_score * self.weights["length_weight"]
            + question_score * self.weights["question_weight"]
            + keyword_score * self.weights["keyword_weight"]
            + instruction_score * self.weights["instruction_weight"]
            + entity_score * self.weights["entity_weight"]
            + sentiment_score * self.weights["sentiment_weight"]
            + pii_score * self.weights["pii_weight"]
            + topic_shift_score * self.weights["topic_shift_weight"]
        )

        # Log detailed scoring for high-importance messages
        clamped_score = _clamp(final_score)
        if clamped_score >= 0.7:
            logger.info(f"High importance score {clamped_score:.3f} for {role} message: '{text[:100]}...'")
            logger.debug(f"Score breakdown: base={base_score}, length={length_score*self.weights['length_weight']:.3f}, "
                        f"question={question_score*self.weights['question_weight']:.3f}, "
                        f"keyword={keyword_score*self.weights['keyword_weight']:.3f}, "
                        f"pii={pii_score*self.weights['pii_weight']:.3f}")

        return clamped_score


# Singleton instance for reuse
_scorer_instance = None

def get_importance_scorer(config: dict | None = None) -> MemoryImportanceScorer:
    """Get or create the singleton importance scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = MemoryImportanceScorer(config)
    return _scorer_instance
