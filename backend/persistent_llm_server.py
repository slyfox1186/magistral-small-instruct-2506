#!/usr/bin/env python3
"""State-of-the-art persistent LLM server optimized for serial processing.

Implements aggressive caching and minimal GPU lock time.
"""

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import redis

from gpu_lock import gpu_for_inference

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Request object for LLM processing."""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    session_id: str = ""
    priority: int = 1  # Lower = higher priority
    future: asyncio.Future | None = None
    request_id: str = ""  # For tracing

    def __post_init__(self):
        """Initialize the future if not provided."""
        if self.future is None:
            self.future = asyncio.Future()
        if not self.request_id:
            import uuid

            self.request_id = str(uuid.uuid4())[:8]


class PersistentLLMServer:
    """State-of-the-art LLM server optimized for serial processing.

    Key optimizations:
    1. Persistent model loading (no reload overhead)
    2. Multi-layer caching (exact + semantic)
    3. Minimal GPU lock time
    4. Efficient request queuing
    """

    def __init__(self, model_path: str, redis_url: str = "redis://localhost:6379"):
        """Initialize the LLM server."""
        self.model_path = model_path
        self.model = None
        self.request_queue = asyncio.PriorityQueue()
        self.gpu_lock = asyncio.Lock()
        self.is_running = False

        # Cache configuration
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour cache

        # Performance metrics
        self.stats = {
            "requests_processed": 0,
            "cache_hits": 0,
            "gpu_lock_time_total": 0.0,
            "queue_wait_time_total": 0.0,
            "startup_time": None,
        }

    async def start(self):
        """Initialize the persistent model server."""
        if self.is_running:
            logger.warning("Server already running")
            return

        logger.info("🚀 Starting Persistent LLM Server...")
        startup_start = time.time()

        try:
            # Load model once and keep in memory
            from config import MODEL_CONFIG

            logger.info(f"📥 Loading model: {self.model_path}")
            logger.info(f"📊 Using context length: {MODEL_CONFIG['n_ctx']}")

            # Model loading happens ONCE at startup, no need for lock
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python is not installed. The LLM server cannot start without it."
                ) from e

            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=MODEL_CONFIG.get("n_gpu_layers", -1),
                n_ctx=MODEL_CONFIG["n_ctx"],  # Use config value
                n_batch=MODEL_CONFIG.get("n_batch", 512),
                verbose=MODEL_CONFIG.get("verbose", False),
            )

            self.stats["startup_time"] = time.time() - startup_start
            logger.info(f"✅ Model loaded in {self.stats['startup_time']:.2f}s")

            # Start the worker task
            self.is_running = True
            self._worker_task = asyncio.create_task(self._worker_loop())

            logger.info("🎯 Persistent LLM Server ready for requests")

        except Exception:
            logger.exception("❌ Failed to start LLM server")
            raise

    async def stop(self):
        """Gracefully stop the server."""
        logger.info("🛑 Stopping Persistent LLM Server...")
        self.is_running = False

        # Clear any pending requests
        while not self.request_queue.empty():
            try:
                _, request = self.request_queue.get_nowait()
                if not request.future.done():
                    request.future.set_exception(RuntimeError("Server shutting down"))
            except asyncio.QueueEmpty:
                break

        # Unload model to free GPU memory
        if self.model:
            del self.model
            self.model = None

        logger.info("✅ Server stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        session_id: str = "",
        priority: int = 1,
        request_id: str = "",
    ) -> str:
        """Generate response for a prompt with aggressive caching.

        Returns response immediately if cached, otherwise queues for processing.
        """
        if not self.is_running or not self.model:
            raise RuntimeError("LLM server not running. Call start() first.")

        # Step 1: Check exact match cache
        cache_key = self._generate_cache_key(prompt, max_tokens, temperature, top_p)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            self.stats["cache_hits"] += 1
            logger.info(f"💰 Cache hit for prompt: {prompt[:50]}...")
            return cached_response

        # Step 2: Queue for GPU processing
        request = LLMRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            session_id=session_id,
            priority=priority,
            request_id=request_id,
        )

        logger.info(
            f"[🆔 {request.request_id}] generate() called for session {session_id}. "
            f"Adding to queue. Queue size: {self.request_queue.qsize()}"
        )

        queue_start = time.time()
        await self.request_queue.put((priority, request))

        # Step 3: Wait for processing
        try:
            logger.info(f"[🆔 {request.request_id}] Awaiting future...")
            response = await request.future
            logger.info(f"[🆔 {request.request_id}] Future resolved! Returning from generate()")
            queue_wait_time = time.time() - queue_start
            self.stats["queue_wait_time_total"] += queue_wait_time

            # Cache the response
            await self._cache_response(cache_key, response)
        except Exception:
            logger.exception("❌ Generation failed")
            raise
        else:
            return response

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        session_id: str = "",
        priority: int = 1,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as they are generated.

        This method bypasses caching to provide real-time streaming.
        """
        if not self.is_running or not self.model:
            raise RuntimeError("LLM server not running. Call start() first.")

        # For streaming, we need to bypass the queue and use the model directly
        # This is a limitation - streaming can't use the queue-based approach
        async with gpu_for_inference(f"stream_{session_id}"):
            try:
                # Create a queue to pass tokens from the executor thread to the async generator
                token_queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def stream_worker():
                    """Worker function to run in thread executor."""
                    try:
                        # Stream tokens directly from the model
                        stream = self.model(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=["[/INST]", "[/SYSTEM_PROMPT]"],
                            echo=False,
                            stream=True,  # Enable streaming
                        )

                        for output in stream:
                            if isinstance(output, dict) and "choices" in output:
                                token = output["choices"][0].get("text", "")
                                if token:
                                    # Debug: Log each token as it's generated
                                    print(f"🔄 STREAM TOKEN: '{token}' (len={len(token)})")
                                    # Put token in queue (thread-safe)
                                    loop.call_soon_threadsafe(token_queue.put_nowait, token)

                        # Signal end of stream
                        loop.call_soon_threadsafe(token_queue.put_nowait, None)
                    except Exception as e:
                        # Put exception in queue
                        loop.call_soon_threadsafe(token_queue.put_nowait, e)

                # Start streaming in thread executor
                loop.run_in_executor(None, stream_worker)

                # Yield tokens as they arrive
                while True:
                    token = await token_queue.get()
                    if token is None:
                        # End of stream
                        break
                    elif isinstance(token, Exception):
                        # Error occurred
                        self._raise_streaming_error(token)
                    else:
                        yield token

            except Exception:
                logger.exception("❌ Streaming generation failed")
                raise

    def _raise_streaming_error(self, error: Exception) -> None:
        """Helper method to raise streaming errors."""
        raise error

    async def _worker_loop(self):
        """Main worker loop that processes requests serially."""
        logger.info("🔄 Worker loop started")

        while self.is_running:
            try:
                # Get next request (blocks until available)
                logger.debug(
                    f"🔄 Worker waiting for next request. Queue size: {self.request_queue.qsize()}"
                )
                priority, request = await self.request_queue.get()
                logger.info(
                    f"[🆔 {request.request_id}] Worker dequeued request from session {request.session_id}"
                )

                if not self.is_running:
                    break

                # Process the request
                try:
                    logger.info(f"[🆔 {request.request_id}] Starting _process_request")
                    response = await self._process_request(request)
                    logger.info(
                        f"[🆔 {request.request_id}] Finished _process_request. Setting future..."
                    )
                    if not request.future.done():
                        request.future.set_result(response)
                        logger.info(f"[🆔 {request.request_id}] Future set successfully")
                except Exception as e:
                    logger.exception("❌ Request processing failed")
                    if not request.future.done():
                        request.future.set_exception(e)
                finally:
                    self.request_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception:
                logger.exception("❌ Worker loop error")
                await asyncio.sleep(0.1)  # Brief pause on error

        logger.info("🛑 Worker loop stopped")

    async def _process_request(self, request: LLMRequest) -> str:
        """Process a single request with minimal GPU lock time."""
        logger.info(f"[🆔 {request.request_id}] Entering _process_request")

        # PHASE 1: Preprocessing (outside GPU lock)
        start_time = time.time()
        prompt = request.prompt.strip()

        # PHASE 2: GPU generation (inside lock - minimize this time!)
        async with gpu_for_inference("llm_generation"):
            gpu_start = time.time()

            try:
                # This is the ONLY place where GPU is used
                # Run the synchronous model call in a thread to avoid blocking the event loop
                import asyncio

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model(
                        prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=["[/INST]", "[/SYSTEM_PROMPT]"],  # Magistral stop tokens
                        echo=False,
                    ),
                )

                gpu_time = time.time() - gpu_start
                self.stats["gpu_lock_time_total"] += gpu_time

                # Extract text from response
                if isinstance(response, dict) and "choices" in response:
                    text = response["choices"][0]["text"].strip()
                else:
                    text = str(response).strip()

                logger.info(f"🎯 GPU processing: {gpu_time:.3f}s | Tokens: ~{len(text.split())}")

            except Exception:
                logger.exception("❌ GPU generation failed")
                raise

        # PHASE 3: Postprocessing (outside GPU lock)
        total_time = time.time() - start_time
        self.stats["requests_processed"] += 1

        logger.info(
            f"✅ Request completed | Total: {total_time:.3f}s | GPU: {gpu_time:.3f}s | "
            f"Session: {request.session_id}"
        )
        logger.info(f"[🆔 {request.request_id}] Exiting _process_request")

        return text

    def _generate_cache_key(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> str:
        """Generate cache key for request parameters."""
        key_data = f"{prompt}|{max_tokens}|{temperature}|{top_p}"
        return f"llm_cache:{hashlib.sha256(key_data.encode()).hexdigest()}"

    async def _get_cached_response(self, cache_key: str) -> str | None:
        """Get cached response if exists."""
        try:
            cached = self.redis_client.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
        else:
            return cached if cached else None

    async def _cache_response(self, cache_key: str, response: str):
        """Cache response with TTL."""
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, response)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()

        if stats["requests_processed"] > 0:
            stats["avg_gpu_time"] = stats["gpu_lock_time_total"] / stats["requests_processed"]
            stats["avg_queue_wait"] = stats["queue_wait_time_total"] / stats["requests_processed"]
            stats["cache_hit_rate"] = stats["cache_hits"] / (
                stats["requests_processed"] + stats["cache_hits"]
            )

        stats["queue_size"] = self.request_queue.qsize()
        stats["is_running"] = self.is_running

        return stats


# Global instance
llm_server: PersistentLLMServer | None = None
_llm_server_lock = asyncio.Lock()  # Lock for initialization


async def get_llm_server() -> PersistentLLMServer:
    """Get or create the global LLM server instance with proper locking."""
    global llm_server  # noqa: PLW0603

    # Fast path: if already initialized, return immediately
    if llm_server is not None and llm_server.is_running:
        return llm_server

    # Slow path: need to initialize with lock
    async with _llm_server_lock:
        # Double-check pattern: another task might have initialized while we waited
        if llm_server is not None and llm_server.is_running:
            return llm_server

        # Initialize the server
        logger.info("🔒 Initializing LLM server (locked)...")
        from config import MODEL_PATH

        llm_server = PersistentLLMServer(MODEL_PATH)
        await llm_server.start()
        logger.info("✅ LLM server initialized successfully")

    return llm_server
