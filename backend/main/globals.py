#!/usr/bin/env python3
"""Global variables, configuration, and background processing system."""

import asyncio
import logging
import os
import time
from collections import deque, namedtuple

from config import API_CONFIG, GENERATION_CONFIG, MODEL_CONFIG, MODEL_PATH
from constants import (
    BATCH_PROCESSING_SIZE,
    DEFAULT_BATCH_SIZE,
    MAX_EMBEDDING_QUEUE_SIZE,
    MAX_MEMORY_EXTRACTION_QUEUE_SIZE,
)
from gpu_lock import gpu_for_inference
from llama_cpp import Llama
from token_manager import TokenManager

# Import prometheus metrics if available
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ===================== Configuration =====================

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


# ===================== Global Variables =====================
llm: Llama | None = None
llm_error: str | None = None
personal_memory = None  # Will be initialized with memory provider
importance_calculator = None  # Will be initialized on startup
token_manager: TokenManager | None = None
ultra_engine = None  # UltraAdvancedEngine
redis_client = None
state_manager: RedisStateManager | None = None
metacognitive_engine = None
background_tasks = []

# Monitoring instances
import monitoring
health_checker: monitoring.HealthChecker | None = None
memory_analytics: monitoring.MemoryAnalytics | None = None

# External service instances  
crypto_trader = None
stock_searcher = None

# Background task tracking
memory_extraction_queue = deque(maxlen=MAX_MEMORY_EXTRACTION_QUEUE_SIZE)
embedding_queue = deque(maxlen=MAX_EMBEDDING_QUEUE_SIZE)
background_processor_running = False
background_processor_task = None

# Chat activity tracking
last_chat_activity = 0.0

# Background processing parameters
GPU_IDLE_THRESHOLD = 2.0  # Seconds of inactivity before considering GPU idle
BATCH_SIZE = BATCH_PROCESSING_SIZE  # Process this many memory extractions in one batch
BATCH_TIMEOUT = 10.0  # Max seconds to wait for batch to fill up


# ===================== Prometheus Metrics =====================
if PROMETHEUS_AVAILABLE:
    # 1. LATENCY: Request processing time
    request_duration = Histogram(
        "neural_chat_request_duration_seconds",
        "Time spent processing chat requests",
        ["endpoint", "status"],
    )

    # 2. TRAFFIC: Request rate
    request_total = Counter(
        "neural_chat_requests_total", "Total number of chat requests", ["endpoint", "status"]
    )

    # 3. ERRORS: Error rate
    request_errors = Counter(
        "neural_chat_errors_total", "Total number of request errors", ["endpoint", "error_type"]
    )

    # 4. SATURATION: GPU and system resource utilization
    gpu_queue_depth = Gauge(
        "neural_chat_gpu_queue_depth", "Current depth of GPU processing queue", ["priority"]
    )

    model_lock_held = Gauge(
        "neural_chat_model_lock_held", "Whether the model lock is currently held (1=held, 0=free)"
    )

    # Additional neural consciousness specific metrics
    metacognitive_evaluations = Counter(
        "neural_chat_metacognitive_evaluations_total",
        "Number of metacognitive evaluations performed",
        ["quality_tier", "improved"],
    )

    metacognitive_duration = Histogram(
        "neural_chat_metacognitive_duration_seconds",
        "Time spent on metacognitive evaluation",
        ["quality_tier"],
    )

    response_quality_score = Histogram(
        "neural_chat_response_quality_score",
        "Quality scores from metacognitive evaluation",
        ["dimension"],
    )
else:
    # No-op metrics if Prometheus not available
    request_duration = request_total = request_errors = None
    gpu_queue_depth = model_lock_held = metacognitive_evaluations = None
    metacognitive_duration = response_quality_score = None


# ===================== Background Processing Classes =====================

class MemoryExtractionTask:
    """Represents a queued memory extraction task"""

    def __init__(
        self,
        user_id: str,
        session_id: str,
        user_prompt: str,
        assistant_response: str,
        timestamp: float,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.user_prompt = user_prompt
        self.assistant_response = assistant_response
        self.timestamp = timestamp
        self.priority = 1.0  # Could be based on importance

    def __repr__(self):
        return f"MemoryTask({self.user_id}, {self.timestamp})"


# Task structure for embedding generation
EmbeddingTask = namedtuple("EmbeddingTask", ["user_id", "user_prompt", "model_response"])


# ===================== Background Processing Functions =====================

async def queue_memory_extraction_task(
    user_id: str, session_id: str, user_prompt: str, assistant_response: str, timestamp: float
):
    """Queue a memory extraction task for intelligent background processing"""
    task = MemoryExtractionTask(user_id, session_id, user_prompt, assistant_response, timestamp)

    # Check if queue is at capacity before adding (for eviction logging)
    queue_was_full = len(memory_extraction_queue) >= MAX_MEMORY_EXTRACTION_QUEUE_SIZE

    memory_extraction_queue.append(task)

    if queue_was_full:
        logger.warning(
            f"🔥 Memory extraction queue full ({MAX_MEMORY_EXTRACTION_QUEUE_SIZE} items) - oldest task evicted"
        )

    logger.debug(
        f"Queued memory task, queue size: {len(memory_extraction_queue)}/{MAX_MEMORY_EXTRACTION_QUEUE_SIZE}"
    )

    # Start background processor if not running
    await ensure_background_processor()


async def ensure_background_processor():
    """Ensure the background processor is running"""
    global background_processor_running, background_processor_task

    if not background_processor_running:
        background_processor_running = True
        background_processor_task = asyncio.create_task(intelligent_background_processor())
        logger.debug("Started background processor")


async def intelligent_background_processor():
    """INTELLIGENT GPU OPTIMIZATION ENGINE

    Monitors GPU usage and chat activity to optimally batch process
    memory extraction and embedding tasks when GPU resources are available.

    Key Features:
    - Waits for GPU idle periods
    - Batches multiple tasks for efficiency
    - Uses 100% GPU when processing
    - Prioritizes real-time chat responses
    - Prevents concurrent model access (SEGFAULT protection)
    """
    global background_processor_running

    logger.debug("Background processor started")

    try:
        while True:
            # Check if we have any tasks to process
            if not memory_extraction_queue and not embedding_queue:
                await asyncio.sleep(1.0)  # Check every second
                continue

            # Check if GPU is idle (no recent chat activity)
            current_time = time.time()
            time_since_last_chat = current_time - last_chat_activity

            if time_since_last_chat >= GPU_IDLE_THRESHOLD:
                # GPU is idle - start batch processing
                memory_batch = []
                embedding_batch = []

                # Collect a batch of memory tasks (up to BATCH_SIZE)
                while memory_extraction_queue and len(memory_batch) < BATCH_SIZE:
                    memory_batch.append(memory_extraction_queue.popleft())

                # Collect a batch of embedding tasks
                while embedding_queue and len(embedding_batch) < BATCH_SIZE:
                    embedding_batch.append(embedding_queue.popleft())

                # Process memory tasks first (higher priority)
                if memory_batch:
                    logger.debug(f"Processing batch of {len(memory_batch)} memory tasks")
                    await process_memory_batch(memory_batch)
                    logger.debug(f"Memory batch complete - {len(memory_extraction_queue)} remaining")

                # Process embedding tasks (lower priority, SEGFAULT protection)
                if embedding_batch:
                    logger.info(f"SEGFAULT PROTECTION - Processing batch of {len(embedding_batch)} embedding tasks")
                    await process_embedding_batch(embedding_batch)
                    logger.info(f"Embedding batch complete - {len(embedding_queue)} tasks remaining")
            else:
                # GPU busy with chat - wait for idle period
                wait_time = GPU_IDLE_THRESHOLD - time_since_last_chat
                logger.debug(f"Waiting {wait_time:.1f}s for GPU idle period")
                await asyncio.sleep(min(wait_time, 1.0))

    except asyncio.CancelledError:
        logger.info("Background processor cancelled")
        raise
    except OSError as e:
        logger.error(f"Background processor I/O error: {e}")
    except RuntimeError as e:
        logger.error(f"Background processor runtime error: {e}")
    except Exception as e:
        logger.error(f"Background processor unexpected error: {e}", exc_info=True)
    finally:
        background_processor_running = False
        logger.info("Background processor stopped")


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
            if len(task.assistant_response) > 8000:
                logger.warning(
                    f"CRASH PROTECTION: Skipping memory processing for very long response "
                    f"({len(task.assistant_response)} chars)"
                )
                continue

            # Use LOW priority to not interfere with real-time chat with SHORTER timeout
            async with gpu_for_inference("LLM initialization"):
                if personal_memory:
                    try:
                        # Store conversation in personal memory
                        await personal_memory.add_memory(
                            content=f"User: {task.user_prompt}",
                            conversation_id=task.session_id,
                            importance=0.5,
                        )

                        await personal_memory.add_memory(
                            content=f"Assistant: {task.assistant_response}",
                            conversation_id=task.session_id,
                            importance=0.6,
                        )

                        logger.debug(f"Memories stored for session {task.session_id}")
                    except TimeoutError:
                        logger.error(f"DEADLOCK PROTECTION: Memory extraction timed out for {task.user_id}")
                        # Continue processing other tasks even if one times out
                    except Exception as memory_error:
                        logger.error(f"CRASH PROTECTION: Memory extraction failed for {task.user_id}: {memory_error}")
                        # Continue processing other tasks even if one fails
        except Exception as task_error:
            logger.error(f"CRASH PROTECTION: Task processing failed: {task_error}")
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
        except Exception as e:
            logger.error(f"Embedding processing failed for {task.user_id}: {e}")