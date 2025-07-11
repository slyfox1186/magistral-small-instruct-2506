#!/usr/bin/env python3
"""Global variables, configuration, and background processing system."""

import asyncio
import logging
import os
import time
from collections import deque, namedtuple

from config import GENERATION_CONFIG, MODEL_CONFIG, MODEL_PATH
from constants import (
    BATCH_PROCESSING_SIZE,
    DEFAULT_BATCH_SIZE,
    MAX_EMBEDDING_QUEUE_SIZE,
    MAX_MEMORY_EXTRACTION_QUEUE_SIZE,
)
from gpu_lock import gpu_for_inference
from token_manager import TokenManager

# Import prometheus metrics if available
try:
    import prometheus_client  # noqa: F401

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===================== Configuration =====================

# Error codes and constants
ENOSPC_ERROR_CODE = 28  # No space left on device
MAX_RESPONSE_LENGTH = 8000  # Maximum response length for memory processing

# Llama initialization parameters
LLAMA_INIT_PARAMS = {
    "model_path": MODEL_PATH,
    "n_ctx": MODEL_CONFIG["n_ctx"],
    "n_batch": DEFAULT_BATCH_SIZE,
    "n_threads": os.cpu_count() or 8,
    "main_gpu": 0,
    "n_gpu_layers": MODEL_CONFIG["n_gpu_layers"],
    "flash_attn": True,
    "use_mmap": True,
    "use_mlock": False,
    "offload_kqv": True,
    "verbose": MODEL_CONFIG["verbose"],
}

# Llama generation parameters
LLAMA_GENERATE_PARAMS = {
    "temperature": GENERATION_CONFIG["temperature"],
    "top_p": GENERATION_CONFIG["top_p"],
    "top_k": GENERATION_CONFIG["top_k"],
    "min_p": GENERATION_CONFIG["min_p"],
    "stream": GENERATION_CONFIG["stream"],
    "stop": GENERATION_CONFIG["stop"],
}

# Only add max_tokens if it's not None
if GENERATION_CONFIG["max_tokens"] is not None:
    LLAMA_GENERATE_PARAMS["max_tokens"] = GENERATION_CONFIG["max_tokens"]


# ===================== Redis-based State Management =====================


class RedisStateManager:
    """Manages global state in Redis for multi-worker safety."""

    def __init__(self, redis_client):
        """Initialize the Redis state manager with a Redis client."""
        self.redis_client = redis_client
        self.MEMORY_DISABLED_KEY = "system:memory_extraction_disabled"
        self.MEMORY_DISABLE_UNTIL_KEY = "system:memory_disable_until"

    async def set_memory_extraction_disabled(self, disabled: bool, duration_seconds: float = 0):
        """Set memory extraction disabled state."""
        if not self.redis_client:
            return

        if disabled:
            await self.redis_client.set(self.MEMORY_DISABLED_KEY, "1")
            if duration_seconds > 0:
                disable_until = time.time() + duration_seconds
                await self.redis_client.set(self.MEMORY_DISABLE_UNTIL_KEY, str(disable_until))
        else:
            await self.redis_client.delete(self.MEMORY_DISABLED_KEY)
            await self.redis_client.delete(self.MEMORY_DISABLE_UNTIL_KEY)

    async def is_memory_extraction_disabled(self) -> bool:
        """Check if memory extraction is disabled."""
        if not self.redis_client:
            return False

        current_time = time.time()

        # Check if explicitly disabled
        if await self.redis_client.get(self.MEMORY_DISABLED_KEY):
            # Check if time-based disable has expired
            disable_until_str = await self.redis_client.get(self.MEMORY_DISABLE_UNTIL_KEY)
            if disable_until_str:
                disable_until = float(disable_until_str)
                if current_time > disable_until:
                    # Time expired, re-enable
                    await self.set_memory_extraction_disabled(False)
                    return False
            return True
        return False


# ===================== Application State =====================


class ApplicationState:
    """Centralized application state management.

    This class encapsulates all global state to avoid scattered globals
    and provide better control over initialization and access.
    """

    def __init__(self):
        """Initialize the application state."""
        # Core services
        self.llm = None
        self.llm_error: str | None = None
        self.token_manager: TokenManager | None = None
        self.redis_client = None
        self.state_manager: RedisStateManager | None = None

        # Memory services
        self.personal_memory = None
        self.importance_calculator = None

        # Processing engines
        self.ultra_engine = None
        self.metacognitive_engine = None

        # Monitoring
        self.health_checker = None
        self.memory_analytics = None

        # External services
        self.crypto_trader = None
        self.stock_searcher = None

        # Background processing
        self.background_tasks: list = []

        # Prometheus metrics (initialized if available)
        self.prometheus_available = PROMETHEUS_AVAILABLE
        self.gpu_queue_depth = None
        self.model_lock_held = None
        self.request_duration = None
        self.request_total = None
        self.request_errors = None
        self.metacognitive_evaluations = None
        self.metacognitive_duration = None
        self.conversation_memories_total = None
        self.memory_consolidation_total = None
        self.memory_consolidation_duration = None
        self.conversation_turns_total = None

        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

    def is_initialized(self) -> bool:
        """Check if core services are initialized."""
        return self.llm is not None and self.redis_client is not None

    def reset(self):
        """Reset all services to initial state."""
        self.__init__()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        from prometheus_client import Counter, Gauge, Histogram

        # 1. LATENCY: Request processing time
        self.request_duration = Histogram(
            "neural_chat_request_duration_seconds",
            "Time spent processing chat requests",
            ["endpoint", "status"],
        )

        # 2. TRAFFIC: Request rate
        self.request_total = Counter(
            "neural_chat_requests_total", "Total number of chat requests", ["endpoint", "status"]
        )

        # 3. ERRORS: Error rate
        self.request_errors = Counter(
            "neural_chat_errors_total", "Total number of request errors", ["endpoint", "error_type"]
        )

        # 4. SATURATION: GPU and system resource utilization
        self.gpu_queue_depth = Gauge(
            "neural_chat_gpu_queue_depth", "Current depth of GPU processing queue", ["priority"]
        )

        self.model_lock_held = Gauge(
            "neural_chat_model_lock_held",
            "Whether the model lock is currently held (1=held, 0=free)",
        )

        # Additional neural consciousness specific metrics
        self.metacognitive_evaluations = Counter(
            "neural_chat_metacognitive_evaluations_total",
            "Number of metacognitive evaluations performed",
            ["quality_tier", "improved"],
        )

        self.metacognitive_duration = Histogram(
            "neural_chat_metacognitive_duration_seconds",
            "Time spent in metacognitive evaluation",
        )

        self.conversation_memories_total = Counter(
            "neural_chat_memories_total",
            "Total number of memories stored",
            ["memory_type", "session_id"],
        )

        self.memory_consolidation_total = Counter(
            "neural_chat_memory_consolidation_total",
            "Number of memory consolidation runs",
            ["trigger_type"],
        )

        self.memory_consolidation_duration = Histogram(
            "neural_chat_memory_consolidation_duration_seconds",
            "Time spent consolidating memories",
        )

        self.conversation_turns_total = Counter(
            "neural_chat_conversation_turns_total",
            "Total conversation turns processed",
            ["session_id"],
        )


# Single instance of application state
app_state = ApplicationState()


# ===================== Background Processing State =====================


class BackgroundProcessingState:
    """Centralized background processing state management.

    This class encapsulates all background processing state to avoid
    scattered globals and provide better control.
    """

    def __init__(self):
        """Initialize the background processing state."""
        # Task queues
        self.memory_extraction_queue = deque(maxlen=MAX_MEMORY_EXTRACTION_QUEUE_SIZE)
        self.embedding_queue = deque(maxlen=MAX_EMBEDDING_QUEUE_SIZE)

        # Processing state
        self.background_processor_running = False
        self.background_processor_task = None

        # Activity tracking
        self.last_chat_activity = 0.0

        # Processing parameters
        self.gpu_idle_threshold = 2.0  # Seconds of inactivity before considering GPU idle
        self.batch_size = BATCH_PROCESSING_SIZE  # Process this many memory extractions in one batch
        self.batch_timeout = 10.0  # Max seconds to wait for batch to fill up

    def reset(self):
        """Reset background processing state."""
        self.__init__()


# Single instance of background processing state
bg_state = BackgroundProcessingState()


# ===================== Background Processing Classes =====================


class MemoryExtractionTask:
    """Represents a queued memory extraction task."""

    def __init__(
        self,
        user_id: str,
        session_id: str,
        user_prompt: str,
        assistant_response: str,
        timestamp: float,
    ):
        """Initialize a memory extraction task."""
        self.user_id = user_id
        self.session_id = session_id
        self.user_prompt = user_prompt
        self.assistant_response = assistant_response
        self.timestamp = timestamp
        self.priority = 1.0  # Could be based on importance

    def __repr__(self):
        """Return string representation of the task."""
        return f"MemoryTask({self.user_id}, {self.timestamp})"


# Task structure for embedding generation
EmbeddingTask = namedtuple("EmbeddingTask", ["user_id", "user_prompt", "model_response"])


# ===================== Background Processing Functions =====================


async def queue_memory_extraction_task(
    user_id: str, session_id: str, user_prompt: str, assistant_response: str, timestamp: float
):
    """Queue a memory extraction task for intelligent background processing."""
    task = MemoryExtractionTask(user_id, session_id, user_prompt, assistant_response, timestamp)

    # Check if queue is at capacity before adding (for eviction logging)
    queue_was_full = len(bg_state.memory_extraction_queue) >= MAX_MEMORY_EXTRACTION_QUEUE_SIZE

    bg_state.memory_extraction_queue.append(task)

    if queue_was_full:
        logger.warning(
            f"🔥 Memory extraction queue full ({MAX_MEMORY_EXTRACTION_QUEUE_SIZE} items) - oldest task evicted"
        )

    logger.debug(
        f"Queued memory task, queue size: {len(bg_state.memory_extraction_queue)}/{MAX_MEMORY_EXTRACTION_QUEUE_SIZE}"
    )

    # Start background processor if not running
    await ensure_background_processor()


async def ensure_background_processor():
    """Ensure the background processor is running."""
    if not bg_state.background_processor_running:
        bg_state.background_processor_running = True
        bg_state.background_processor_task = asyncio.create_task(intelligent_background_processor())
        logger.debug("Started background processor")


async def intelligent_background_processor():
    """INTELLIGENT GPU OPTIMIZATION ENGINE.

    Monitors GPU usage and chat activity to optimally batch process
    memory extraction and embedding tasks when GPU resources are available.

    Key Features:
    - Waits for GPU idle periods
    - Batches multiple tasks for efficiency
    - Uses 100% GPU when processing
    - Prioritizes real-time chat responses
    - Prevents concurrent model access (SEGFAULT protection)
    """
    logger.debug("Background processor started")

    try:
        while True:
            if await _should_wait_for_tasks():
                continue

            if await _should_wait_for_gpu_idle():
                continue

            await _process_gpu_idle_batches()

    except asyncio.CancelledError:
        logger.info("Background processor cancelled")
        raise
    except Exception as e:
        await _handle_processor_error(e)
    finally:
        bg_state.background_processor_running = False
        logger.info("Background processor stopped")


async def _should_wait_for_tasks() -> bool:
    """Check if we need to wait for tasks to be available."""
    if not bg_state.memory_extraction_queue and not bg_state.embedding_queue:
        await asyncio.sleep(1.0)
        return True
    return False


async def _should_wait_for_gpu_idle() -> bool:
    """Check if we need to wait for GPU to be idle."""
    current_time = time.time()
    time_since_last_chat = current_time - bg_state.last_chat_activity

    if time_since_last_chat < bg_state.gpu_idle_threshold:
        wait_time = bg_state.gpu_idle_threshold - time_since_last_chat
        logger.debug(f"Waiting {wait_time:.1f}s for GPU idle period")
        await asyncio.sleep(min(wait_time, 1.0))
        return True
    return False


async def _process_gpu_idle_batches():
    """Process batches when GPU is idle."""
    memory_batch, embedding_batch = _collect_task_batches()

    if memory_batch:
        logger.debug(f"Processing batch of {len(memory_batch)} memory tasks")
        await process_memory_batch(memory_batch)
        logger.debug(f"Memory batch complete - {len(bg_state.memory_extraction_queue)} remaining")

    if embedding_batch:
        logger.info(f"SEGFAULT PROTECTION - Processing batch of {len(embedding_batch)} embedding tasks")
        await process_embedding_batch(embedding_batch)
        logger.info(f"Embedding batch complete - {len(bg_state.embedding_queue)} tasks remaining")


def _collect_task_batches() -> tuple[list, list]:
    """Collect batches of memory and embedding tasks."""
    memory_batch = []
    embedding_batch = []

    while bg_state.memory_extraction_queue and len(memory_batch) < bg_state.batch_size:
        memory_batch.append(bg_state.memory_extraction_queue.popleft())

    while bg_state.embedding_queue and len(embedding_batch) < bg_state.batch_size:
        embedding_batch.append(bg_state.embedding_queue.popleft())

    return memory_batch, embedding_batch


async def _handle_processor_error(e: Exception):
    """Handle different types of processor errors."""
    if isinstance(e, OSError):
        await _handle_os_error(e)
    elif isinstance(e, RuntimeError):
        await _handle_runtime_error(e)
    elif isinstance(e, MemoryError):
        await _handle_memory_error(e)
    else:
        logger.error(f"Background processor unexpected error: {type(e).__name__}: {e}", exc_info=True)
        await asyncio.sleep(5)


async def _handle_os_error(e: OSError):
    """Handle OS errors in background processor."""
    logger.exception("Background processor I/O error")
    if e.errno == ENOSPC_ERROR_CODE:
        logger.critical("Disk full - pausing background processing")
        await asyncio.sleep(300)


async def _handle_runtime_error(e: RuntimeError):
    """Handle runtime errors in background processor."""
    logger.exception("Background processor runtime error")
    if "cannot schedule new futures" in str(e):
        logger.critical("Event loop closed - stopping processor")
        return


async def _handle_memory_error(e: MemoryError):
    """Handle memory errors in background processor."""
    logger.critical(f"Out of memory in background processor: {e}")
    bg_state.memory_extraction_queue.clear()
    bg_state.embedding_queue.clear()


async def process_memory_batch(batch: list[MemoryExtractionTask]):
    """Process a batch of memory extraction tasks efficiently.

    Uses 100% GPU resources for optimal throughput.
    """
    if not batch:
        return

    logger.debug(f"Processing memory batch: {len(batch)} tasks")

    for task in batch:
        try:
            # CRASH PROTECTION: Skip processing very long responses
            if len(task.assistant_response) > MAX_RESPONSE_LENGTH:
                logger.warning(
                    f"CRASH PROTECTION: Skipping memory processing for very long response "
                    f"({len(task.assistant_response)} chars)"
                )
                continue

            # Use LOW priority to not interfere with real-time chat with SHORTER timeout
            async with gpu_for_inference("LLM initialization"):
                if app_state.personal_memory:
                    try:
                        # Store conversation in personal memory
                        await app_state.personal_memory.add_memory(
                            content=f"User: {task.user_prompt}",
                            conversation_id=task.session_id,
                            importance=0.5,
                        )

                        await app_state.personal_memory.add_memory(
                            content=f"Assistant: {task.assistant_response}",
                            conversation_id=task.session_id,
                            importance=0.6,
                        )

                        logger.debug(f"Memories stored for session {task.session_id}")
                    except TimeoutError:
                        logger.exception(
                            f"DEADLOCK PROTECTION: Memory extraction timed out for {task.user_id}"
                        )
                        # Continue processing other tasks even if one times out
                        continue
                    except MemoryError as e:
                        logger.critical(f"Out of memory during extraction for {task.user_id}: {e}")
                        # Skip remaining tasks to prevent further memory issues
                        break
                    except ValueError:
                        logger.exception(f"Invalid data for {task.user_id}")
                        continue
                    except Exception as memory_error:
                        logger.exception(
                            f"CRASH PROTECTION: Memory extraction failed for {task.user_id}: "
                            f"{type(memory_error).__name__}"
                        )
                        # Continue processing other tasks even if one fails
                        continue
        except Exception:
            logger.exception("CRASH PROTECTION: Task processing failed")
            # Continue with next task


async def process_embedding_batch(batch):
    """Process a batch of embedding tasks efficiently."""
    if not batch:
        return

    logger.debug(f"Processing embedding batch: {len(batch)} tasks")

    for task in batch:
        try:
            # Process embedding task here
            # Implementation depends on specific embedding requirements
            logger.debug(f"Processing embedding for user {task.user_id}")
        except (ValueError, TypeError):
            logger.exception(f"Invalid embedding data for {task.user_id}")
        except MemoryError as e:
            logger.critical(f"Out of memory processing embedding for {task.user_id}: {e}")
            break  # Stop processing to prevent further issues
        except Exception as e:
            logger.exception(f"Embedding processing failed for {task.user_id}: {type(e).__name__}")
