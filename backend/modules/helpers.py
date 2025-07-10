#!/usr/bin/env python3
"""Helper functions for memory management and conversation processing."""

import asyncio
import logging
import time
from datetime import UTC, datetime

UTC = UTC

from constants import (
    DEFAULT_IMPORTANCE_SCORE,
    ECHO_SIMILARITY_THRESHOLD,
    MEMORY_CONSOLIDATION_INTERVAL,
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
        except (AttributeError, TypeError) as e:
            logger.error(f"Memory consolidation error - invalid memory system: {e}")
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

        if embeddings is None or len(embeddings) < 2:
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

        return is_echo

    except ImportError as e:
        logger.error(f"[ECHO_DETECTION] Missing dependencies: {e}")
        return False
    except ValueError as e:
        logger.error(f"[ECHO_DETECTION] Invalid input data: {e}")
        return False
    except Exception as e:
        logger.error(f"[ECHO_DETECTION] Unexpected error: {e}", exc_info=True)
        return False


async def store_conversation_memory(user_prompt: str, assistant_response: str, session_id: str):
    """Store conversation turn in personal memory system with echo detection."""
    try:
        # Import here to avoid circular imports
        from .globals import app_state

        if app_state.personal_memory:
            start_time = time.time()
            logger.debug(f"[MEMORY_STORE] Storing memory for session {session_id}")

            # Echo Detection: Check if assistant response is just echoing user data
            is_echo = await detect_echo(user_prompt, assistant_response)

            # Calculate importance for user message
            user_importance = DEFAULT_IMPORTANCE_SCORE  # Default
            user_analysis = None
            if app_state.importance_calculator:
                try:
                    user_importance, user_analysis = app_state.importance_calculator.calculate_importance(
                        user_prompt,
                        context={
                            "is_user_message": True,
                            "session_id": session_id,
                            "timestamp": datetime.now(UTC).timestamp(),
                        },
                    )
                    logger.debug(f"[MEMORY_STORE_USER] User importance: {user_importance}")
                except (AttributeError, TypeError) as e:
                    logger.error(f"[MEMORY_STORE_USER] Invalid importance calculator: {e}")
                except ValueError as e:
                    logger.error(
                        f"[MEMORY_STORE_USER] Invalid user message for importance calculation: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"[MEMORY_STORE_USER] Unexpected error calculating user importance: {e}",
                        exc_info=True,
                    )

            # Store user message with calculated importance
            await app_state.personal_memory.add_memory(
                content=f"User: {user_prompt}",
                conversation_id=session_id,
                importance=user_importance,
                metadata={"analysis": user_analysis} if user_analysis else None,
            )

            # Only store assistant response if it's NOT an echo
            if not is_echo:
                # Calculate importance for assistant response
                assistant_importance = DEFAULT_IMPORTANCE_SCORE  # Default
                assistant_analysis = None
                if app_state.importance_calculator:
                    try:
                        assistant_importance, assistant_analysis = (
                            app_state.importance_calculator.calculate_importance(
                                assistant_response,
                                context={
                                    "is_assistant_response": True,
                                    "session_id": session_id,
                                    "timestamp": datetime.now(UTC).timestamp(),
                                    "responding_to": user_prompt[
                                        :100
                                    ],  # Context of what we're responding to
                                },
                            )
                        )
                        logger.debug(
                            f"[MEMORY_STORE_ASSISTANT] Assistant importance: {assistant_importance}"
                        )
                    except (AttributeError, TypeError) as e:
                        logger.error(f"[MEMORY_STORE_ASSISTANT] Invalid importance calculator: {e}")
                    except ValueError as e:
                        logger.error(
                            f"[MEMORY_STORE_ASSISTANT] Invalid assistant response for importance calculation: {e}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[MEMORY_STORE_ASSISTANT] Unexpected error calculating assistant importance: {e}",
                            exc_info=True,
                        )

                # Store assistant response with calculated importance
                await app_state.personal_memory.add_memory(
                    content=f"Assistant: {assistant_response}",
                    conversation_id=session_id,
                    importance=assistant_importance,
                    metadata={"analysis": assistant_analysis} if assistant_analysis else None,
                )
            else:
                logger.debug("[MEMORY_STORE_ASSISTANT] Echo detected - skipping storage")

            # Trigger conversation summarization for this session
            try:
                await app_state.personal_memory._summarize_conversation(session_id)
                logger.debug(f"[CONVERSATION_SUMMARY] Created summary for session {session_id}")
            except Exception as e:
                logger.error(f"[CONVERSATION_SUMMARY] Failed to create summary: {e}")

            # Log summary of what was stored with timing
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(
                f"[MEMORY_STORE_COMPLETE] Stored in {duration:.2f}s, "
                f"importance: user={user_importance}, assistant="
                f"{assistant_importance if not is_echo else 'skipped'}"
            )

    except Exception as e:
        logger.error(
            f"[MEMORY_STORE_ERROR] Failed to store conversation memory: {e}", exc_info=True
        )
