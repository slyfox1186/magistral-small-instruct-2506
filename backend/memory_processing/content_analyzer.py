"""
Content Analysis Component for Memory Processing

Uses LLM-powered analysis to understand conversation content,
categorize memories, and determine what information is worth storing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time

from .utils import extract_entities, detect_emotional_content, validate_memory_content
from .config import MemoryProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class ContentAnalysis:
    """Results of content analysis"""
    categories: List[str]
    importance_score: float
    confidence: float
    entities: Dict[str, List[str]]
    emotional_content: Dict[str, Any]
    memory_worthy: bool
    key_facts: List[str]
    context_type: str
    processing_time: float

class ContentAnalyzer:
    """
    Advanced content analyzer using LLM for intelligent content understanding
    """
    
    def __init__(self, llm_server, config: MemoryProcessingConfig):
        self.llm_server = llm_server
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory categories and their patterns
        self.memory_categories = {
            'personal_facts': [
                'name', 'age', 'location', 'job', 'profession', 'education', 'background'
            ],
            'preferences': [
                'like', 'dislike', 'prefer', 'favorite', 'hate', 'love', 'enjoy', 'opinion'
            ],
            'experiences': [
                'did', 'went', 'visited', 'happened', 'experience', 'event', 'story', 'memory'
            ],
            'relationships': [
                'family', 'friend', 'colleague', 'partner', 'spouse', 'parent', 'child', 'sibling'
            ],
            'knowledge': [
                'learned', 'discovered', 'understand', 'know', 'teach', 'explain', 'fact'
            ],
            'goals': [
                'want', 'plan', 'goal', 'dream', 'ambition', 'hope', 'wish', 'future'
            ]
        }
    
    async def analyze_content(self, user_prompt: str, assistant_response: str, session_id: str) -> ContentAnalysis:
        """
        Analyze conversation content for memory extraction
        
        Args:
            user_prompt: User's message
            assistant_response: Assistant's response
            session_id: Conversation session ID
            
        Returns:
            ContentAnalysis object with analysis results
        """
        start_time = time.time()
        
        try:
            # Combine user and assistant content
            combined_content = f"User: {user_prompt}\nAssistant: {assistant_response}"
            
            # Validate content
            is_valid, error_msg = validate_memory_content(combined_content)
            if not is_valid:
                self.logger.warning(f"Invalid content for analysis: {error_msg}")
                return self._create_empty_analysis(time.time() - start_time)
            
            # Extract basic entities and emotional content
            entities = extract_entities(combined_content)
            emotional_content = detect_emotional_content(combined_content)
            
            # Perform LLM-based analysis
            llm_analysis = await self._perform_llm_analysis(combined_content)
            
            # Calculate importance score
            importance_score = self._calculate_importance_score(
                combined_content, entities, emotional_content, llm_analysis
            )
            
            # Determine memory categories
            categories = self._determine_categories(combined_content, llm_analysis)
            
            # Extract key facts
            key_facts = self._extract_key_facts(combined_content, llm_analysis)
            
            # Determine context type
            context_type = self._determine_context_type(combined_content, llm_analysis)
            
            # Check if content is memory worthy
            memory_worthy = self._is_memory_worthy(importance_score, categories, key_facts)
            
            processing_time = time.time() - start_time
            
            analysis = ContentAnalysis(
                categories=categories,
                importance_score=importance_score,
                confidence=llm_analysis.get('confidence', 0.5),
                entities=entities,
                emotional_content=emotional_content,
                memory_worthy=memory_worthy,
                key_facts=key_facts,
                context_type=context_type,
                processing_time=processing_time
            )
            
            if self.config.enable_detailed_logging:
                self.logger.debug(f"Content analysis complete for session {session_id}: "
                                f"importance={importance_score:.2f}, "
                                f"categories={categories}, "
                                f"memory_worthy={memory_worthy}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            return self._create_empty_analysis(time.time() - start_time)
    
    async def _perform_llm_analysis(self, content: str) -> Dict[str, Any]:
        """
        Use LLM to analyze content for memory extraction
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with LLM analysis results
        """
        try:
            # Import format_prompt function
            from utils import format_prompt
            
            system_prompt = """You are an advanced memory analysis system. Analyze conversations to extract meaningful information for long-term storage.

Your task is to identify what information is worth remembering and categorize it appropriately.

Categories to consider:
- personal_facts: Names, age, location, job, personal identifiers
- preferences: Likes, dislikes, opinions, tastes
- experiences: Events, activities, stories, memories
- relationships: Family, friends, colleagues, connections
- knowledge: Facts learned, information shared, explanations
- goals: Plans, ambitions, future intentions

Respond with ONLY a valid JSON object containing:
- "key_facts": List of important facts to remember
- "categories": List of applicable categories
- "importance_reasons": List of reasons why this is important
- "confidence": Float between 0-1 indicating analysis confidence
- "memory_worthy": Boolean indicating if this should be stored
- "context_type": String describing the type of conversation"""

            user_prompt = f"""Analyze the following conversation:

{content}

Provide your analysis as a JSON object:"""
            
            # Format prompt properly
            formatted_prompt = format_prompt(system_prompt, user_prompt)
            
            # Call LLM with timeout
            try:
                response = await asyncio.wait_for(
                    self.llm_server.generate(formatted_prompt, max_tokens=512, temperature=0.7),
                    timeout=self.config.llm_timeout
                )
                
                # Parse JSON response
                try:
                    analysis = json.loads(response)
                    return analysis
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        analysis = json.loads(json_str)
                        return analysis
                    else:
                        self.logger.warning("Failed to parse LLM response as JSON")
                        return self._create_fallback_analysis(content)
                
            except asyncio.TimeoutError:
                self.logger.warning(f"LLM analysis timed out after {self.config.llm_timeout}s")
                return self._create_fallback_analysis(content)
                
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return self._create_fallback_analysis(content)
    
    def _create_fallback_analysis(self, content: str) -> Dict[str, Any]:
        """
        Create fallback analysis when LLM analysis fails
        
        Args:
            content: Content to analyze
            
        Returns:
            Basic analysis dictionary
        """
        # Simple keyword-based analysis
        categories = []
        key_facts = []
        
        content_lower = content.lower()
        
        # Check for personal information patterns
        if any(keyword in content_lower for keyword in ['name', 'age', 'live', 'work', 'job']):
            categories.append('personal_facts')
            key_facts.append("Contains personal information")
        
        # Check for preferences
        if any(keyword in content_lower for keyword in ['like', 'dislike', 'prefer', 'favorite']):
            categories.append('preferences')
            key_facts.append("Contains preferences")
        
        # Check for experiences
        if any(keyword in content_lower for keyword in ['did', 'went', 'visited', 'happened']):
            categories.append('experiences')
            key_facts.append("Contains experiences")
        
        return {
            'key_facts': key_facts,
            'categories': categories,
            'importance_reasons': ['Fallback analysis'],
            'confidence': 0.3,
            'memory_worthy': len(key_facts) > 0,
            'context_type': 'general_conversation'
        }
    
    def _calculate_importance_score(self, content: str, entities: Dict[str, List[str]], 
                                  emotional_content: Dict[str, Any], 
                                  llm_analysis: Dict[str, Any]) -> float:
        """
        Calculate importance score for memory storage
        
        Args:
            content: Content to analyze
            entities: Extracted entities
            emotional_content: Emotional analysis
            llm_analysis: LLM analysis results
            
        Returns:
            Importance score between 0 and 1
        """
        base_score = 0.3  # Base score for all content
        
        # LLM confidence boost
        llm_confidence = llm_analysis.get('confidence', 0.5)
        base_score += llm_confidence * 0.3
        
        # Entity boost
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        entity_boost = min(entity_count * 0.1, 0.3)
        base_score += entity_boost
        
        # Emotional content boost
        if emotional_content.get('has_emotional_content', False):
            emotional_boost = min(emotional_content.get('intensity', 0) * 0.1, 0.2)
            base_score += emotional_boost
        
        # Personal information boost
        if any(category in llm_analysis.get('categories', []) for category in ['personal_facts', 'relationships']):
            base_score += self.config.personal_info_boost
        
        # Key facts boost
        key_facts_count = len(llm_analysis.get('key_facts', []))
        key_facts_boost = min(key_facts_count * 0.05, 0.2)
        base_score += key_facts_boost
        
        # Apply multiplier
        final_score = base_score * self.config.importance_multiplier
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, final_score))
    
    def _determine_categories(self, content: str, llm_analysis: Dict[str, Any]) -> List[str]:
        """
        Determine memory categories for content
        
        Args:
            content: Content to analyze
            llm_analysis: LLM analysis results
            
        Returns:
            List of applicable categories
        """
        categories = llm_analysis.get('categories', [])
        
        # Validate categories
        valid_categories = []
        for category in categories:
            if category in self.memory_categories:
                valid_categories.append(category)
        
        # Add fallback category if none found
        if not valid_categories:
            valid_categories.append('knowledge')
        
        return valid_categories
    
    def _extract_key_facts(self, content: str, llm_analysis: Dict[str, Any]) -> List[str]:
        """
        Extract key facts from content
        
        Args:
            content: Content to analyze
            llm_analysis: LLM analysis results
            
        Returns:
            List of key facts
        """
        key_facts = llm_analysis.get('key_facts', [])
        
        # Validate and filter key facts
        valid_facts = []
        for fact in key_facts:
            if isinstance(fact, str) and len(fact.strip()) > 10:
                valid_facts.append(fact.strip())
        
        return valid_facts[:5]  # Limit to top 5 facts
    
    def _determine_context_type(self, content: str, llm_analysis: Dict[str, Any]) -> str:
        """
        Determine the context type of the conversation
        
        Args:
            content: Content to analyze
            llm_analysis: LLM analysis results
            
        Returns:
            Context type string
        """
        context_type = llm_analysis.get('context_type', 'general_conversation')
        
        # Validate context type
        valid_context_types = [
            'personal_introduction', 'casual_conversation', 'question_answer',
            'problem_solving', 'information_request', 'creative_collaboration',
            'general_conversation'
        ]
        
        if context_type not in valid_context_types:
            context_type = 'general_conversation'
        
        return context_type
    
    def _is_memory_worthy(self, importance_score: float, categories: List[str], 
                         key_facts: List[str]) -> bool:
        """
        Determine if content is worth storing in memory
        
        Args:
            importance_score: Calculated importance score
            categories: Memory categories
            key_facts: Extracted key facts
            
        Returns:
            True if content should be stored
        """
        # Check importance threshold
        if importance_score < self.config.min_confidence_threshold:
            return False
        
        # Check if we have meaningful facts to store
        if not key_facts:
            return False
        
        # High-value categories are always worth storing
        high_value_categories = ['personal_facts', 'relationships', 'goals']
        if any(category in high_value_categories for category in categories):
            return True
        
        # Check if importance score is high enough
        return importance_score >= self.config.high_confidence_threshold
    
    def _create_empty_analysis(self, processing_time: float) -> ContentAnalysis:
        """
        Create empty analysis for failed processing
        
        Args:
            processing_time: Time taken for processing
            
        Returns:
            Empty ContentAnalysis object
        """
        return ContentAnalysis(
            categories=[],
            importance_score=0.0,
            confidence=0.0,
            entities={},
            emotional_content={},
            memory_worthy=False,
            key_facts=[],
            context_type='unknown',
            processing_time=processing_time
        )