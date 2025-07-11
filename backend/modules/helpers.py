#!/usr/bin/env python3
"""Helper functions for memory management and conversation processing."""

import asyncio
import logging
import time
from datetime import UTC, datetime

from constants import (
    DEFAULT_IMPORTANCE_SCORE,
    ECHO_SIMILARITY_THRESHOLD,
    MEMORY_CONSOLIDATION_INTERVAL,
    MIN_EMBEDDINGS_FOR_COMPARISON,
)

logger = logging.getLogger(__name__)


async def periodic_memory_consolidation():
    """Periodically consolidate old memories to save space and improve retrieval."""
    while True:
        await asyncio.sleep(MEMORY_CONSOLIDATION_INTERVAL)  # Run every hour
        try:
            # Import here to avoid circular imports
            from .globals import app_state

            if app_state.personal_memory:
                await app_state.personal_memory.consolidate_old_memories()
                logger.debug("Memory consolidation completed")
        except asyncio.CancelledError:
            logger.debug("Memory consolidation task cancelled")
            raise
        except (AttributeError, TypeError):
            logger.exception("Memory consolidation error - invalid memory system")
        except Exception as e:
            logger.error(f"Unexpected memory consolidation error: {e}", exc_info=True)


async def detect_echo(
    user_message: str, assistant_response: str, threshold: float = ECHO_SIMILARITY_THRESHOLD
) -> bool:
    """Detect if assistant response is just echoing user data using semantic similarity.

    Returns True if response is likely an echo, False otherwise.
    """
    try:
        # Run imports in thread to avoid blocking event loop
        def _import_and_get_embeddings():
            from sklearn.metrics.pairwise import cosine_similarity

            from resource_manager import get_sentence_transformer_embeddings

            # Generate embeddings for both messages
            embeddings = get_sentence_transformer_embeddings([user_message, assistant_response])
            return embeddings, cosine_similarity

        embeddings, cosine_similarity = await asyncio.to_thread(_import_and_get_embeddings)

        if embeddings is None or len(embeddings) < MIN_EMBEDDINGS_FOR_COMPARISON:
            logger.warning("[ECHO_DETECTION] Could not generate embeddings for comparison")
            return False

        # Calculate cosine similarity
        user_embedding = embeddings[0].reshape(1, -1)
        assistant_embedding = embeddings[1].reshape(1, -1)

        similarity = cosine_similarity(user_embedding, assistant_embedding)[0][0]

        is_echo = similarity > threshold
        logger.info(
            f"[ECHO_DETECTION] Similarity: {similarity:.4f} | "
            f"Threshold: {threshold} | Echo: {is_echo}"
        )

    except ImportError:
        logger.exception("[ECHO_DETECTION] Missing dependencies")
        return False
    except ValueError:
        logger.exception("[ECHO_DETECTION] Invalid input data")
        return False
    except Exception as e:
        logger.error(f"[ECHO_DETECTION] Unexpected error: {e}", exc_info=True)
        return False
    else:
        return is_echo


async def store_conversation_memory(user_prompt: str, assistant_response: str, session_id: str):
    """Store conversation turn in personal memory system with echo detection."""
    try:
        from .globals import app_state

        if not app_state.personal_memory:
            return

        start_time = time.time()
        logger.debug(f"[MEMORY_STORE] Storing memory for session {session_id}")

        is_echo = await detect_echo(user_prompt, assistant_response)

        user_importance = await _store_user_memory(user_prompt, session_id, app_state)

        if not is_echo:
            assistant_importance = await _store_assistant_memory(
                assistant_response, user_prompt, session_id, app_state
            )
        else:
            assistant_importance = None
            logger.debug("[MEMORY_STORE_ASSISTANT] Echo detected - skipping storage")

        await _handle_conversation_summarization(session_id, app_state)
        _log_memory_storage_complete(start_time, user_importance=user_importance,
                                   assistant_importance=assistant_importance, is_echo=is_echo)

    except Exception as e:
        logger.error(f"[MEMORY_STORE_ERROR] Failed to store conversation memory: {e}", exc_info=True)


async def _store_user_memory(user_prompt: str, session_id: str, app_state) -> float:
    """Store user message and return its importance score."""
    user_importance, user_analysis = await _calculate_message_importance(
        user_prompt, session_id, is_user_message=True, app_state=app_state
    )

    await app_state.personal_memory.add_memory(
        content=f"User: {user_prompt}",
        conversation_id=session_id,
        importance=user_importance,
        metadata={"analysis": user_analysis} if user_analysis else None,
    )

    return user_importance


async def _store_assistant_memory(assistant_response: str, user_prompt: str,
                                session_id: str, app_state) -> float:
    """Store assistant response and return its importance score."""
    assistant_importance, assistant_analysis = await _calculate_message_importance(
        assistant_response, session_id, is_user_message=False,
        app_state=app_state, responding_to=user_prompt[:100]
    )

    await app_state.personal_memory.add_memory(
        content=f"Assistant: {assistant_response}",
        conversation_id=session_id,
        importance=assistant_importance,
        metadata={"analysis": assistant_analysis} if assistant_analysis else None,
    )

    return assistant_importance


async def _calculate_message_importance(message: str, session_id: str, is_user_message: bool,
                                      app_state, responding_to: str | None = None) -> tuple[float, dict]:
    """Calculate importance score for a message."""
    importance = DEFAULT_IMPORTANCE_SCORE
    analysis = None

    if not app_state.importance_calculator:
        return importance, analysis

    context = {
        "session_id": session_id,
        "timestamp": datetime.now(UTC).timestamp(),
    }

    if is_user_message:
        context["is_user_message"] = True
        message_type = "USER"
    else:
        context["is_assistant_response"] = True
        if responding_to:
            context["responding_to"] = responding_to
        message_type = "ASSISTANT"

    try:
        importance, analysis = app_state.importance_calculator.calculate_importance(message, context=context)
        logger.debug(f"[MEMORY_STORE_{message_type}] {message_type.title()} importance: {importance}")
    except (AttributeError, TypeError):
        logger.exception(f"[MEMORY_STORE_{message_type}] Invalid importance calculator")
    except ValueError:
        logger.exception(f"[MEMORY_STORE_{message_type}] Invalid message for importance calculation")
    except Exception as e:
        logger.error(f"[MEMORY_STORE_{message_type}] Unexpected error calculating importance: {e}", exc_info=True)

    return importance, analysis


async def _handle_conversation_summarization(session_id: str, app_state):
    """Handle conversation summarization for the session."""
    try:
        await app_state.personal_memory._summarize_conversation(session_id)
        logger.debug(f"[CONVERSATION_SUMMARY] Created summary for session {session_id}")
    except Exception:
        logger.exception("[CONVERSATION_SUMMARY] Failed to create summary")


def _log_memory_storage_complete(start_time: float, user_importance: float,
                                assistant_importance: float, is_echo: bool):
    """Log completion of memory storage with timing."""
    end_time = time.time()
    duration = end_time - start_time
    assistant_imp_text = assistant_importance if not is_echo else 'skipped'
    logger.debug(
        f"[MEMORY_STORE_COMPLETE] Stored in {duration:.2f}s, "
        f"importance: user={user_importance}, assistant={assistant_imp_text}"
    )
