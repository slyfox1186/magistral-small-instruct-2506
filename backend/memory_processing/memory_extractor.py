"""Memory Extraction Component.

Extracts structured memories from analyzed content with confidence scoring
and semantic validation.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import MemoryProcessingConfig
from .content_analyzer import ContentAnalysis
from .utils import extract_personal_info_patterns, sanitize_content
from utils import format_prompt

logger = logging.getLogger(__name__)

# Constants for confidence thresholds and minimum lengths
CORE_MEMORY_CONFIDENCE_THRESHOLD = 0.7
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
MINIMUM_FACT_LENGTH = 15
MINIMUM_MEMORY_LENGTH = 10


@dataclass
class ExtractedMemory:
    """Represents an extracted memory with metadata."""

    content: str
    category: str
    importance_score: float
    confidence: float
    entities: dict[str, list[str]]
    context_type: str
    memory_type: str  # 'core' or 'regular'
    session_id: str
    timestamp: str
    metadata: dict[str, Any]


class MemoryExtractor:
    """Extracts structured memories from content analysis results."""

    def __init__(self, config: MemoryProcessingConfig):
        """Initialize the memory extractor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Define memory types and their criteria
        self.core_memory_categories = {
            "personal_facts": ["name", "age", "location", "job", "contact"],
            "relationships": ["family", "friends", "colleagues", "romantic"],
            "preferences": ["strong_likes", "dislikes", "values", "beliefs"],
        }

        self.regular_memory_categories = {
            "experiences": ["events", "activities", "stories", "achievements"],
            "knowledge": ["facts", "explanations", "learning", "insights"],
            "goals": ["plans", "ambitions", "projects", "future_intentions"],
        }

    async def extract_memories(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract structured memories from content analysis.

        Args:
            analysis: ContentAnalysis results
            user_prompt: Original user prompt
            assistant_response: Assistant's response
            session_id: Session identifier

        Returns:
            List of ExtractedMemory objects
        """
        if not analysis.memory_worthy:
            self.logger.debug(f"Content not memory worthy for session {session_id}")
            return []

        start_time = time.time()
        extracted_memories = []

        try:
            # Extract memories for each category
            for category in analysis.categories:
                memories = await self._extract_category_memories(
                    category, analysis, user_prompt, assistant_response, session_id
                )
                extracted_memories.extend(memories)

            # Validate and filter memories
            valid_memories = self._validate_memories(extracted_memories)

            # Limit number of memories per conversation
            if len(valid_memories) > self.config.max_memories_per_conversation:
                # Sort by importance and keep top N
                valid_memories.sort(key=lambda m: m.importance_score, reverse=True)
                valid_memories = valid_memories[: self.config.max_memories_per_conversation]

            processing_time = time.time() - start_time

            if self.config.enable_detailed_logging:
                self.logger.debug(
                    f"Extracted {len(valid_memories)} memories for session {session_id} "
                    f"in {processing_time:.2f}s"
                )

        except Exception:
            self.logger.exception("Error extracting memories")
            return []

        return valid_memories

    async def _extract_category_memories(
        self,
        category: str,
        analysis: ContentAnalysis,
        user_prompt: str,
        assistant_response: str,
        session_id: str,
    ) -> list[ExtractedMemory]:
        """Extract memories for a specific category.

        Args:
            category: Memory category
            analysis: Content analysis results
            user_prompt: User's message
            assistant_response: Assistant's response
            session_id: Session identifier

        Returns:
            List of extracted memories for the category
        """
        memories = []

        try:
            if category == "personal_facts":
                memories.extend(
                    await self._extract_personal_facts(
                        analysis, user_prompt, assistant_response, session_id
                    )
                )
            elif category == "preferences":
                memories.extend(
                    await self._extract_preferences(
                        analysis, user_prompt, assistant_response, session_id
                    )
                )
            elif category == "experiences":
                memories.extend(
                    await self._extract_experiences(
                        analysis, user_prompt, assistant_response, session_id
                    )
                )
            elif category == "relationships":
                memories.extend(
                    await self._extract_relationships(
                        analysis, user_prompt, assistant_response, session_id
                    )
                )
            elif category == "knowledge":
                memories.extend(
                    await self._extract_knowledge(
                        analysis, user_prompt, assistant_response, session_id
                    )
                )
            elif category == "goals":
                memories.extend(
                    await self._extract_goals(analysis, user_prompt, assistant_response, session_id)
                )

        except Exception:
            self.logger.exception(f"Error extracting {category} memories")

        return memories

    async def _extract_personal_facts(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract personal facts memories."""
        memories = []

        # Extract personal information patterns
        personal_info = extract_personal_info_patterns(f"{user_prompt} {assistant_response}")

        # Process key facts from analysis
        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in ["name", "age", "live", "work", "job"]):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category="personal_facts",
                    importance_score=await self._calculate_llm_importance_score(
                        fact, "personal_facts", analysis.confidence
                    ),
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type="core",  # Personal facts are core memories
                    session_id=session_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metadata={
                        "source": "personal_facts_extraction",
                        "personal_info": personal_info,
                        "extraction_method": "llm_analysis",
                    },
                )
                memories.append(memory)

        return memories

    async def _extract_preferences(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract preferences memories."""
        memories = []

        # Look for preference indicators
        preference_keywords = ["like", "dislike", "prefer", "favorite", "hate", "love", "enjoy"]

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in preference_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category="preferences",
                    importance_score=await self._calculate_llm_importance_score(
                        fact, "preferences", analysis.confidence
                    ),
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type="core"
                    if analysis.confidence > CORE_MEMORY_CONFIDENCE_THRESHOLD
                    else "regular",
                    session_id=session_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metadata={
                        "source": "preferences_extraction",
                        "preference_type": "stated",
                        "extraction_method": "llm_analysis",
                    },
                )
                memories.append(memory)

        return memories

    async def _extract_experiences(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract experiences memories."""
        memories = []

        experience_keywords = ["did", "went", "visited", "happened", "experience", "remember", "bought", "purchased"]

        # CRITICAL FIX: Use full user prompt for experiences, not just extracted facts
        # This preserves ALL details like items bought, specific actions, etc.
        if any(keyword in user_prompt.lower() for keyword in experience_keywords):
            memory = ExtractedMemory(
                content=sanitize_content(user_prompt),  # Use FULL user prompt
                category="experiences",
                importance_score=await self._calculate_llm_importance_score(
                    user_prompt, "experiences", analysis.confidence
                ),
                confidence=analysis.confidence,
                entities=analysis.entities,
                context_type=analysis.context_type,
                memory_type="regular",
                session_id=session_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "source": "experiences_extraction",
                    "experience_type": "personal",
                    "extraction_method": "full_prompt_preservation",
                },
            )
            memories.append(memory)

        # Also check key facts as fallback, but with higher threshold
        for fact in analysis.key_facts:
            if (any(keyword in fact.lower() for keyword in experience_keywords) and 
                len(fact) > 30):  # Only use facts if they're substantial
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category="experiences",
                    importance_score=await self._calculate_llm_importance_score(
                        fact, "experiences", analysis.confidence
                    ),
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type="regular",
                    session_id=session_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metadata={
                        "source": "experiences_extraction",
                        "experience_type": "personal",
                        "extraction_method": "llm_analysis",
                    },
                )
                memories.append(memory)

        return memories

    async def _extract_relationships(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract relationships memories."""
        memories = []

        relationship_keywords = [
            "family",
            "friend",
            "colleague",
            "partner",
            "spouse",
            "wife",
            "husband",
        ]

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in relationship_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category="relationships",
                    importance_score=await self._calculate_llm_importance_score(
                        fact, "relationships", analysis.confidence
                    ),
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type="core",  # Relationships are core memories
                    session_id=session_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metadata={
                        "source": "relationships_extraction",
                        "relationship_type": "personal",
                        "extraction_method": "llm_analysis",
                    },
                )
                memories.append(memory)

        return memories

    async def _extract_knowledge(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract knowledge memories."""
        memories = []

        # Create a knowledge memory from the conversation
        # FIXED: Include general_conversation for philosophical discussions
        # SECURITY: Filter out self-referential content about optimization
        if (analysis.context_type in [
            "question_answer",
            "information_request",
            "problem_solving",
            "general_conversation",
        ] and not self._is_self_referential_content(user_prompt, assistant_response)):
            # Combine user question and assistant response into a knowledge memory
            knowledge_content = f"Q: {user_prompt}\nA: {assistant_response[:500]}..."

            memory = ExtractedMemory(
                content=sanitize_content(knowledge_content),
                category="knowledge",
                importance_score=await self._calculate_llm_importance_score(
                    knowledge_content, "knowledge", analysis.confidence
                ),
                confidence=analysis.confidence,
                entities=analysis.entities,
                context_type=analysis.context_type,
                memory_type="regular",
                session_id=session_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "source": "knowledge_extraction",
                    "knowledge_type": "qa_pair"
                    if analysis.context_type != "general_conversation"
                    else "philosophical_discussion",
                    "extraction_method": "conversation_summary",
                },
            )
            memories.append(memory)

        # Also extract memories from key facts for general conversations
        # This handles philosophical discussions and meaningful conversations
        if analysis.context_type == "general_conversation" and analysis.key_facts:
            for fact in analysis.key_facts:
                if len(fact.strip()) > MINIMUM_FACT_LENGTH:  # Only meaningful facts
                    memory = ExtractedMemory(
                        content=sanitize_content(fact),
                        category="knowledge",
                        importance_score=await self._calculate_llm_importance_score(
                            fact, "knowledge", analysis.confidence
                        ),
                        confidence=analysis.confidence,
                        entities=analysis.entities,
                        context_type=analysis.context_type,
                        memory_type="regular",
                        session_id=session_id,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        metadata={
                            "source": "knowledge_extraction",
                            "knowledge_type": "key_fact",
                            "extraction_method": "llm_analysis",
                        },
                    )
                    memories.append(memory)

        return memories

    async def _extract_goals(
        self, analysis: ContentAnalysis, user_prompt: str, assistant_response: str, session_id: str
    ) -> list[ExtractedMemory]:
        """Extract goals memories."""
        memories = []

        goal_keywords = ["want", "plan", "goal", "dream", "ambition", "hope", "future"]

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in goal_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category="goals",
                    importance_score=await self._calculate_llm_importance_score(
                        fact, "goals", analysis.confidence
                    ),
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type="core"
                    if analysis.confidence > MEDIUM_CONFIDENCE_THRESHOLD
                    else "regular",
                    session_id=session_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metadata={
                        "source": "goals_extraction",
                        "goal_type": "personal",
                        "extraction_method": "llm_analysis",
                    },
                )
                memories.append(memory)

        return memories

    def _validate_memories(self, memories: list[ExtractedMemory]) -> list[ExtractedMemory]:
        """Validate and filter memories.

        Args:
            memories: List of memories to validate

        Returns:
            List of valid memories
        """
        valid_memories = []

        for memory in memories:
            # Check minimum content length
            if len(memory.content.strip()) < MINIMUM_MEMORY_LENGTH:
                continue

            # Check confidence threshold
            if memory.confidence < self.config.min_confidence_threshold:
                continue

            # Check importance score
            if memory.importance_score < self.config.min_confidence_threshold:
                continue

            # Check content length limit
            if len(memory.content) > self.config.max_content_length:
                memory.content = memory.content[: self.config.max_content_length - 3] + "..."

            # Sanitize content
            memory.content = sanitize_content(memory.content)

            valid_memories.append(memory)

        return valid_memories

    def _determine_memory_type(self, category: str, confidence: float) -> str:
        """Determine if memory should be core or regular.

        Args:
            category: Memory category
            confidence: Confidence score

        Returns:
            Memory type ('core' or 'regular')
        """
        # High-confidence personal facts and relationships are core memories
        if (
            category in ["personal_facts", "relationships"]
            and confidence > CORE_MEMORY_CONFIDENCE_THRESHOLD
        ):
            return "core"

        # High-confidence preferences and goals can be core memories
        if category in ["preferences", "goals"] and confidence > HIGH_CONFIDENCE_THRESHOLD:
            return "core"

        # Everything else is regular memory
        return "regular"

    def get_extraction_stats(self, memories: list[ExtractedMemory]) -> dict[str, Any]:
        """Get statistics about extracted memories.

        Args:
            memories: List of extracted memories

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_memories": len(memories),
            "core_memories": sum(1 for m in memories if m.memory_type == "core"),
            "regular_memories": sum(1 for m in memories if m.memory_type == "regular"),
            "categories": {},
            "avg_importance": 0.0,
            "avg_confidence": 0.0,
        }

        if memories:
            # Category breakdown
            for memory in memories:
                if memory.category not in stats["categories"]:
                    stats["categories"][memory.category] = 0
                stats["categories"][memory.category] += 1

            # Average scores
            stats["avg_importance"] = sum(m.importance_score for m in memories) / len(memories)
            stats["avg_confidence"] = sum(m.confidence for m in memories) / len(memories)

        return stats

    def _is_self_referential_content(self, user_prompt: str, assistant_response: str) -> bool:
        """Check if content is self-referential (about optimization, system responses, etc.).
        
        Args:
            user_prompt: User's message
            assistant_response: Assistant's response
            
        Returns:
            True if content should be filtered out
        """
        # Keywords that indicate self-referential or meta-conversation content
        self_referential_keywords = [
            "optimize", "optimized", "optimization",
            "user information", "assistant's name", 
            "base name", "system", "version",
            "sure, here's", "optimized response"
        ]
        
        combined_content = f"{user_prompt} {assistant_response}".lower()
        
        # Check for self-referential patterns
        for keyword in self_referential_keywords:
            if keyword in combined_content:
                self.logger.debug(f"Filtering self-referential content containing: {keyword}")
                return True
                
        # Check for optimization format patterns
        if "user information:" in combined_content and "assistant's name:" in combined_content:
            self.logger.debug("Filtering optimization format content")
            return True
            
        return False
    
    async def _calculate_llm_importance_score(self, content: str, category: str, base_confidence: float) -> float:
        """Use LLM to calculate importance score for memory content.
        
        Args:
            content: Memory content to evaluate
            category: Memory category (personal_facts, relationships, etc.)
            base_confidence: Base confidence from content analysis
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        try:
            system_prompt = """You are Jane, the memory architect for a REVOLUTIONARY AI memory system. Your job is to ensure PERFECT recall of conversational details.

REVOLUTIONARY MEMORY PRINCIPLES:
1. This is NOT ordinary storage - this is building a comprehensive memory system for full conversational recall
2. EVERY specific detail is precious: names, items, quantities, locations, times, actions, relationships
3. Users will ask follow-up questions expecting COMPLETE recall of previous conversations
4. Your scoring determines whether the AI can answer "What did I buy?" or "Who did I meet?" later
5. Conservative scoring BREAKS the memory system - be GENEROUS with importance scores

SCORING FOR REVOLUTIONARY RECALL:
- 0.8-1.0: ANY conversation with specific details (purchases, meetings, activities, plans, preferences)
- 0.6-0.8: General experiences, statements with some concrete information
- 0.4-0.6: Basic information, opinions, casual mentions
- 0.1-0.4: Meta-conversations, system references, truly generic statements

CRITICAL: If a user mentions specific items, people, places, actions, or events - score HIGH (0.7+). The goal is PERFECT conversational continuity and detail recall.

JSON Response Format:
{
    "importance_score": 0.0-1.0,
    "reasoning": "brief explanation focusing on recall value"
}"""
            
            user_prompt = f"""Evaluate the importance of this {category} memory:

Content: {content[:500]}...
Category: {category}
Base Confidence: {base_confidence}

Provide importance score and reasoning."""
            
            formatted_prompt = format_prompt(system_prompt, user_prompt)
            
            # Get LLM server
            from persistent_llm_server import get_llm_server
            llm_server = await get_llm_server()
            
            response_text = await llm_server.generate(
                prompt=formatted_prompt,
                max_tokens=150,
                temperature=0.3,  # Conservative for scoring
                session_id="memory_importance_scoring",
            )
            
            # Extract JSON from response
            if not response_text:
                self.logger.warning("Empty LLM response for importance scoring")
                return min(base_confidence * 0.6, 0.7)  # Conservative fallback
            
            # Find JSON boundaries
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    json_text = response_text[start_idx:end_idx + 1]
                    result = json.loads(json_text)
                    
                    importance_score = result.get("importance_score", base_confidence * 0.6)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    
                    # Validate score range
                    importance_score = max(0.0, min(1.0, float(importance_score)))
                    
                    if self.config.enable_detailed_logging:
                        self.logger.debug(
                            f"LLM importance scoring: {importance_score:.3f} - {reasoning}"
                        )
                    
                    return importance_score
                    
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse LLM importance response: {e}")
                    
        except Exception as e:
            self.logger.warning(f"LLM importance scoring failed: {e}")
        
        # Fallback: GENEROUS scoring for revolutionary memory system
        fallback_score = min(base_confidence * 1.2, 0.8)  # More generous baseline
        
        # Boost experience categories significantly for detail preservation
        if category in ["experiences", "personal_facts", "relationships", "preferences"]:
            fallback_score = min(fallback_score * 1.3, 0.9)  # Revolutionary memory needs high scores
            
        return fallback_score
