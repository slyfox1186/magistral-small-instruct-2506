"""LLM-based memory importance scoring system.

Uses only the local LLM model to determine memory importance scores.
No hardcoded patterns, keywords, or regex - pure AI-driven scoring.
"""

import json
import logging

logger = logging.getLogger(__name__)


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(value, max_val))


class MemoryImportanceScorer:
    """LLM-powered memory importance scorer.
    
    Uses the local language model to intelligently analyze message content
    and determine importance scores without any hardcoded patterns.
    
    Score ranges:
    - 0.0-0.3: Low importance (greetings, fillers, casual chat)
    - 0.3-0.6: Medium importance (general conversation, questions)
    - 0.6-0.8: High importance (answers, facts, instructions)
    - 0.8-1.0: Critical importance (personal info, key decisions, commands)
    """

    def __init__(self, config: dict | None = None):
        """Initialize the LLM-based importance scorer."""
        if config is None:
            config = {}
        
        # Base scores by role (fallback values if LLM fails)
        self.base_scores = {
            "user": config.get("base_score_user", 0.4),
            "assistant": config.get("base_score_assistant", 0.3),
        }

    async def _get_llm_importance_score(self, text: str, role: str, conversation_history: list[dict[str, str]] | None = None) -> float:
        """Get importance score from LLM using format_prompt pattern."""
        try:
            from persistent_llm_server import get_llm_server
            import utils
            
            # Build conversation context if available
            context_text = ""
            if conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-3:]  # Last 3 messages for context
                context_parts = []
                for msg in recent_messages:
                    msg_role = msg.get("role", "unknown")
                    msg_content = msg.get("content", "")[:200]  # Limit length
                    context_parts.append(f"{msg_role}: {msg_content}")
                context_text = f"\n\nRecent conversation context:\n" + "\n".join(context_parts)
            
            system_prompt = f"""You are an expert memory importance analyzer. Your task is to analyze a {role} message and determine its importance for long-term memory storage.

Analyze the message content and assign an importance score between 0.0 and 1.0:

**IMPORTANCE SCALE:**
0.0-0.3: Low importance
- Simple greetings, casual chat, filler words
- Basic acknowledgments like "ok", "thanks", "got it"
- Small talk without meaningful information

0.3-0.6: Medium importance  
- General conversation with some context
- Questions that might be referenced later
- Discussions about topics or preferences
- Routine instructions or explanations

0.6-0.8: High importance
- Specific facts, decisions, or answers
- Important questions and their responses
- Technical information or detailed explanations
- Instructions that might be referenced frequently
- User preferences and settings

0.8-1.0: Critical importance
- Personal information (names, contact details, addresses)
- Passwords, credentials, or security information
- Critical instructions or rules
- Important deadlines or commitments
- Key facts that define the user or relationship

**ROLE CONSIDERATIONS:**
- User messages often contain questions, personal info, or instructions (slightly higher importance)
- Assistant messages contain answers and explanations (evaluate based on usefulness)

**CONTEXT AWARENESS:**
- Consider if this message provides new information or clarifies something important
- Messages that answer previous questions are often more important
- Follow-up questions or clarifications may have elevated importance

Return ONLY a JSON object with the importance score:
{{"importance": 0.75}}

The score should be a float between 0.0 and 1.0."""

            user_prompt = f"""Analyze this {role} message for memory importance:

Message: "{text}"{context_text}

Provide the importance score as JSON."""

            # Use the standardized format_prompt function
            formatted_prompt = utils.format_prompt(system_prompt, user_prompt)
            
            # Get LLM server
            llm_server = await get_llm_server()
            
            # Generate response
            response = await llm_server.generate(
                prompt=formatted_prompt,
                max_tokens=50,  # Short response needed
                temperature=0.2,  # Low temperature for consistent scoring
                session_id="importance_scorer",
            )
            
            # Extract JSON from response
            response_text = response.strip()
            logger.debug(f"Raw LLM response for importance scoring: {response_text}")
            
            # Remove any markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    response_text = response_text[start:end].strip()
            
            # Find JSON object boundaries
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx + 1]
                
                try:
                    result = json.loads(json_text)
                    importance = result.get("importance", 0.5)
                    
                    # Validate and clamp the score
                    if isinstance(importance, (int, float)):
                        clamped_score = _clamp(float(importance))
                        logger.debug(f"LLM importance score for {role} message: {clamped_score}")
                        return clamped_score
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM importance JSON: {e}")
                    
            logger.warning(f"Could not extract valid importance score from LLM response: {response_text[:100]}")
            return self.base_scores.get(role, 0.4)
            
        except Exception as e:
            logger.error(f"LLM importance scoring failed: {e}", exc_info=True)
            return self.base_scores.get(role, 0.4)

    async def calculate_importance_async(
        self, text: str, role: str, conversation_history: list[dict[str, str]] | None = None
    ) -> float:
        """Calculate importance score using LLM analysis (async version).
        
        Args:
            text: The message content
            role: "user" or "assistant"
            conversation_history: List of previous messages for context
            
        Returns:
            Float between 0.0 and 1.0 representing importance
        """
        if not text or not text.strip():
            return 0.0
        
        try:
            return await self._get_llm_importance_score(text, role, conversation_history)
        except Exception as e:
            logger.error(f"Error in calculate_importance_async: {e}", exc_info=True)
            # Fallback to base score
            return self.base_scores.get(role, 0.4)

    def calculate_importance(
        self, text: str, role: str, conversation_history: list[dict[str, str]] | None = None
    ) -> float:
        """Calculate importance score using LLM analysis.
        
        This is a synchronous wrapper - use calculate_importance_async when possible.
        
        Args:
            text: The message content
            role: "user" or "assistant"
            conversation_history: List of previous messages for context
            
        Returns:
            Float between 0.0 and 1.0 representing importance
        """
        if not text or not text.strip():
            return 0.0
        
        try:
            import asyncio
            
            # Handle both sync and async contexts more safely
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # We're already in an async context - this shouldn't be called
                logger.warning("calculate_importance called from async context - use calculate_importance_async instead")
                # Fallback to base score to avoid deadlock
                return self.base_scores.get(role, 0.4)
            except RuntimeError:
                # No event loop running, safe to create one
                return asyncio.run(self._get_llm_importance_score(text, role, conversation_history))
                
        except Exception as e:
            logger.error(f"Error in calculate_importance: {e}", exc_info=True)
            # Fallback to base score
            return self.base_scores.get(role, 0.4)


# Singleton instance for reuse
_scorer_instance = None


def get_importance_scorer(config: dict | None = None) -> MemoryImportanceScorer:
    """Get or create the singleton importance scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = MemoryImportanceScorer(config)
    return _scorer_instance