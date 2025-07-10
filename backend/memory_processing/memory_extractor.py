"""Memory Extraction Component

Extracts structured memories from analyzed content with confidence scoring
and semantic validation.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import MemoryProcessingConfig
from .content_analyzer import ContentAnalysis
from .utils import extract_personal_info_patterns, sanitize_content

logger = logging.getLogger(__name__)

@dataclass
class ExtractedMemory:
    """Represents an extracted memory with metadata"""
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
    """Extracts structured memories from content analysis results
    """

    def __init__(self, config: MemoryProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Define memory types and their criteria
        self.core_memory_categories = {
            'personal_facts': ['name', 'age', 'location', 'job', 'contact'],
            'relationships': ['family', 'friends', 'colleagues', 'romantic'],
            'preferences': ['strong_likes', 'dislikes', 'values', 'beliefs']
        }

        self.regular_memory_categories = {
            'experiences': ['events', 'activities', 'stories', 'achievements'],
            'knowledge': ['facts', 'explanations', 'learning', 'insights'],
            'goals': ['plans', 'ambitions', 'projects', 'future_intentions']
        }

    async def extract_memories(self, analysis: ContentAnalysis, user_prompt: str,
                              assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract structured memories from content analysis
        
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
                valid_memories = valid_memories[:self.config.max_memories_per_conversation]

            processing_time = time.time() - start_time

            if self.config.enable_detailed_logging:
                self.logger.debug(f"Extracted {len(valid_memories)} memories for session {session_id} "
                                f"in {processing_time:.2f}s")

            return valid_memories

        except Exception as e:
            self.logger.error(f"Error extracting memories: {e!s}")
            return []

    async def _extract_category_memories(self, category: str, analysis: ContentAnalysis,
                                        user_prompt: str, assistant_response: str,
                                        session_id: str) -> list[ExtractedMemory]:
        """Extract memories for a specific category
        
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
            if category == 'personal_facts':
                memories.extend(await self._extract_personal_facts(
                    analysis, user_prompt, assistant_response, session_id
                ))
            elif category == 'preferences':
                memories.extend(await self._extract_preferences(
                    analysis, user_prompt, assistant_response, session_id
                ))
            elif category == 'experiences':
                memories.extend(await self._extract_experiences(
                    analysis, user_prompt, assistant_response, session_id
                ))
            elif category == 'relationships':
                memories.extend(await self._extract_relationships(
                    analysis, user_prompt, assistant_response, session_id
                ))
            elif category == 'knowledge':
                memories.extend(await self._extract_knowledge(
                    analysis, user_prompt, assistant_response, session_id
                ))
            elif category == 'goals':
                memories.extend(await self._extract_goals(
                    analysis, user_prompt, assistant_response, session_id
                ))

        except Exception as e:
            self.logger.error(f"Error extracting {category} memories: {e!s}")

        return memories

    async def _extract_personal_facts(self, analysis: ContentAnalysis, user_prompt: str,
                                     assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract personal facts memories"""
        memories = []

        # Extract personal information patterns
        personal_info = extract_personal_info_patterns(f"{user_prompt} {assistant_response}")

        # Process key facts from analysis
        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in ['name', 'age', 'live', 'work', 'job']):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category='personal_facts',
                    importance_score=analysis.importance_score + 0.1,  # Boost for personal facts
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type='core',  # Personal facts are core memories
                    session_id=session_id,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    metadata={
                        'source': 'personal_facts_extraction',
                        'personal_info': personal_info,
                        'extraction_method': 'llm_analysis'
                    }
                )
                memories.append(memory)

        return memories

    async def _extract_preferences(self, analysis: ContentAnalysis, user_prompt: str,
                                  assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract preferences memories"""
        memories = []

        # Look for preference indicators
        preference_keywords = ['like', 'dislike', 'prefer', 'favorite', 'hate', 'love', 'enjoy']

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in preference_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category='preferences',
                    importance_score=analysis.importance_score,
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type='core' if analysis.confidence > 0.7 else 'regular',
                    session_id=session_id,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    metadata={
                        'source': 'preferences_extraction',
                        'preference_type': 'stated',
                        'extraction_method': 'llm_analysis'
                    }
                )
                memories.append(memory)

        return memories

    async def _extract_experiences(self, analysis: ContentAnalysis, user_prompt: str,
                                  assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract experiences memories"""
        memories = []

        experience_keywords = ['did', 'went', 'visited', 'happened', 'experience', 'remember']

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in experience_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category='experiences',
                    importance_score=analysis.importance_score,
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type='regular',
                    session_id=session_id,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    metadata={
                        'source': 'experiences_extraction',
                        'experience_type': 'personal',
                        'extraction_method': 'llm_analysis'
                    }
                )
                memories.append(memory)

        return memories

    async def _extract_relationships(self, analysis: ContentAnalysis, user_prompt: str,
                                    assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract relationships memories"""
        memories = []

        relationship_keywords = ['family', 'friend', 'colleague', 'partner', 'spouse', 'wife', 'husband']

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in relationship_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category='relationships',
                    importance_score=analysis.importance_score + 0.1,  # Boost for relationships
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type='core',  # Relationships are core memories
                    session_id=session_id,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    metadata={
                        'source': 'relationships_extraction',
                        'relationship_type': 'personal',
                        'extraction_method': 'llm_analysis'
                    }
                )
                memories.append(memory)

        return memories

    async def _extract_knowledge(self, analysis: ContentAnalysis, user_prompt: str,
                                assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract knowledge memories"""
        memories = []

        # Create a knowledge memory from the conversation
        # FIXED: Include general_conversation for philosophical discussions
        if analysis.context_type in ['question_answer', 'information_request', 'problem_solving', 'general_conversation']:
            # Combine user question and assistant response into a knowledge memory
            knowledge_content = f"Q: {user_prompt}\nA: {assistant_response[:500]}..."

            memory = ExtractedMemory(
                content=sanitize_content(knowledge_content),
                category='knowledge',
                importance_score=analysis.importance_score,
                confidence=analysis.confidence,
                entities=analysis.entities,
                context_type=analysis.context_type,
                memory_type='regular',
                session_id=session_id,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={
                    'source': 'knowledge_extraction',
                    'knowledge_type': 'qa_pair' if analysis.context_type != 'general_conversation' else 'philosophical_discussion',
                    'extraction_method': 'conversation_summary'
                }
            )
            memories.append(memory)

        # Also extract memories from key facts for general conversations
        # This handles philosophical discussions and meaningful conversations
        if analysis.context_type == 'general_conversation' and analysis.key_facts:
            for fact in analysis.key_facts:
                if len(fact.strip()) > 15:  # Only meaningful facts
                    memory = ExtractedMemory(
                        content=sanitize_content(fact),
                        category='knowledge',
                        importance_score=analysis.importance_score,
                        confidence=analysis.confidence,
                        entities=analysis.entities,
                        context_type=analysis.context_type,
                        memory_type='regular',
                        session_id=session_id,
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                        metadata={
                            'source': 'knowledge_extraction',
                            'knowledge_type': 'key_fact',
                            'extraction_method': 'llm_analysis'
                        }
                    )
                    memories.append(memory)

        return memories

    async def _extract_goals(self, analysis: ContentAnalysis, user_prompt: str,
                            assistant_response: str, session_id: str) -> list[ExtractedMemory]:
        """Extract goals memories"""
        memories = []

        goal_keywords = ['want', 'plan', 'goal', 'dream', 'ambition', 'hope', 'future']

        for fact in analysis.key_facts:
            if any(keyword in fact.lower() for keyword in goal_keywords):
                memory = ExtractedMemory(
                    content=sanitize_content(fact),
                    category='goals',
                    importance_score=analysis.importance_score,
                    confidence=analysis.confidence,
                    entities=analysis.entities,
                    context_type=analysis.context_type,
                    memory_type='core' if analysis.confidence > 0.6 else 'regular',
                    session_id=session_id,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    metadata={
                        'source': 'goals_extraction',
                        'goal_type': 'personal',
                        'extraction_method': 'llm_analysis'
                    }
                )
                memories.append(memory)

        return memories

    def _validate_memories(self, memories: list[ExtractedMemory]) -> list[ExtractedMemory]:
        """Validate and filter memories
        
        Args:
            memories: List of memories to validate
            
        Returns:
            List of valid memories
        """
        valid_memories = []

        for memory in memories:
            # Check minimum content length
            if len(memory.content.strip()) < 10:
                continue

            # Check confidence threshold
            if memory.confidence < self.config.min_confidence_threshold:
                continue

            # Check importance score
            if memory.importance_score < self.config.min_confidence_threshold:
                continue

            # Check content length limit
            if len(memory.content) > self.config.max_content_length:
                memory.content = memory.content[:self.config.max_content_length-3] + "..."

            # Sanitize content
            memory.content = sanitize_content(memory.content)

            valid_memories.append(memory)

        return valid_memories

    def _determine_memory_type(self, category: str, confidence: float) -> str:
        """Determine if memory should be core or regular
        
        Args:
            category: Memory category
            confidence: Confidence score
            
        Returns:
            Memory type ('core' or 'regular')
        """
        # High-confidence personal facts and relationships are core memories
        if category in ['personal_facts', 'relationships'] and confidence > 0.7:
            return 'core'

        # High-confidence preferences and goals can be core memories
        if category in ['preferences', 'goals'] and confidence > 0.8:
            return 'core'

        # Everything else is regular memory
        return 'regular'

    def get_extraction_stats(self, memories: list[ExtractedMemory]) -> dict[str, Any]:
        """Get statistics about extracted memories
        
        Args:
            memories: List of extracted memories
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_memories': len(memories),
            'core_memories': sum(1 for m in memories if m.memory_type == 'core'),
            'regular_memories': sum(1 for m in memories if m.memory_type == 'regular'),
            'categories': {},
            'avg_importance': 0.0,
            'avg_confidence': 0.0
        }

        if memories:
            # Category breakdown
            for memory in memories:
                if memory.category not in stats['categories']:
                    stats['categories'][memory.category] = 0
                stats['categories'][memory.category] += 1

            # Average scores
            stats['avg_importance'] = sum(m.importance_score for m in memories) / len(memories)
            stats['avg_confidence'] = sum(m.confidence for m in memories) / len(memories)

        return stats
