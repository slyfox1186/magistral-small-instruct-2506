"""Embedding Service.

=================

GPU-accelerated embedding generation with Redis integration.
Handles batch processing, caching, and async operations.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import redis.asyncio as redis
import yaml
from cachetools import TTLCache
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..utils import get_metrics, get_model_manager

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""

    def __init__(
        self,
        redis_client: redis.Redis,
        executor: ThreadPoolExecutor | None = None,
        config_path: str = "config/models.yaml",
    ):
        """Initialize embedding service.

        Args:
            redis_client: Async Redis client
            executor: Thread pool for CPU-bound operations
            config_path: Path to model configuration
        """
        self.redis = redis_client
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.config = self._load_config(config_path)
        self.metrics = get_metrics()

        # Initialize model
        self.model_manager = get_model_manager()
        self.model = None  # Lazy loading
        self._model_lock = asyncio.Lock()

        # Embedding cache
        cache_config = self.config["embedding"]["cache"]
        self.cache = (
            TTLCache(maxsize=cache_config["max_size"], ttl=cache_config["ttl_seconds"])
            if cache_config["enabled"]
            else None
        )

        # Redis stream for batch processing
        self.stream_key = "embedding_queue"
        self.consumer_group = "embedding_workers"
        self.consumer_name = f"worker_{int(time.time())}"

        # Vector index names
        self.stm_index = "idx:stm_vectors"
        self.ltm_index = "idx:ltm_vectors"

    def _load_config(self, config_path: str) -> dict:
        """Load model configuration."""
        config_file = Path(__file__).parent.parent / config_path
        return yaml.safe_load(config_file.read_text())

    async def initialize(self):
        """Initialize the embedding service."""
        # Load model
        await self._ensure_model_loaded()

        # Create Redis indices
        await self._create_vector_indices()

        # Create consumer group for stream processing
        try:
            await self.redis.xgroup_create(self.stream_key, self.consumer_group, id="0")
        except Exception:
            # Group already exists - this is expected and safe to ignore
            logger.debug("Consumer group already exists")

        logger.info("Embedding service initialized")

    async def _ensure_model_loaded(self):
        """Ensure embedding model is loaded."""
        async with self._model_lock:
            if self.model is None:
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    self.executor, self.model_manager.get_current_model
                )

                # Get actual dimension from model
                test_embedding = await self._generate_single_embedding("test")
                self.vector_dimension = len(test_embedding)
                logger.info(f"Loaded embedding model with dimension {self.vector_dimension}")

    async def _create_vector_indices(self):
        """Create vector search indices in Redis."""
        await self._ensure_model_loaded()

        # STM Index
        await self._create_index(
            self.stm_index,
            "stm:",
            [
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": self.vector_dimension, "DISTANCE_METRIC": "COSINE"},
                    as_name="embedding",
                ),
                TextField("$.content", as_name="content"),
                TagField("$.tags", as_name="tags"),
                TextField("$.circle", as_name="circle"),
                NumericField("$.timestamp", as_name="timestamp"),
                TextField("$.source", as_name="source"),
            ],
        )

        # LTM Index
        await self._create_index(
            self.ltm_index,
            "ltm:",
            [
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": self.vector_dimension, "DISTANCE_METRIC": "COSINE"},
                    as_name="embedding",
                ),
                TextField("$.content", as_name="content"),
                TagField("$.tags", as_name="tags"),
                TextField("$.circle", as_name="circle"),
                NumericField("$.retrieval_score", as_name="retrieval_score"),
                NumericField("$.last_accessed", as_name="last_accessed"),
            ],
        )

    async def _create_index(self, index_name: str, prefix: str, schema: list):
        """Create a single index with error handling."""
        try:
            # Check if index exists
            await self.redis.ft(index_name).info()
            logger.info(f"Index {index_name} already exists")
        except Exception:
            # Create index
            try:
                await self.redis.ft(index_name).create_index(
                    schema, definition=IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
                )
                logger.info(f"Created vector index: {index_name}")
            except Exception:
                logger.exception(f"Failed to create index {index_name}")

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Check cache first
        if self.cache and text in self.cache:
            self.metrics.increment_counter("cache_hits")
            return self.cache[text]

        self.metrics.increment_counter("cache_misses")

        # Generate embedding
        async with self.metrics.track_operation("embedding_generation"):
            embedding = await self._generate_single_embedding(text)

            # Cache result
            if self.cache:
                self.cache[text] = embedding

            return embedding

    async def _generate_single_embedding(self, text: str) -> list[float]:
        """Generate embedding using the model."""
        await self._ensure_model_loaded()

        loop = asyncio.get_event_loop()

        # Run in executor to avoid blocking
        embedding = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                text,
                convert_to_tensor=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config["embedding"]["primary"]["normalize"],
            ),
        )

        # Convert to list of floats
        return embedding.astype(np.float32).tolist()

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []

        async with self.metrics.track_operation("batch_embedding_generation"):
            # Check cache for each text
            results = [None] * len(texts)
            uncached_indices = []
            uncached_texts = []

            for i, text in enumerate(texts):
                if self.cache and text in self.cache:
                    results[i] = self.cache[text]
                    self.metrics.increment_counter("cache_hits")
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
                    self.metrics.increment_counter("cache_misses")

            # Generate embeddings for uncached texts
            if uncached_texts:
                await self._ensure_model_loaded()

                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    self.executor,
                    lambda: self.model.encode(
                        uncached_texts,
                        batch_size=self.config["embedding"]["primary"]["batch_size"],
                        convert_to_tensor=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self.config["embedding"]["primary"]["normalize"],
                    ),
                )

                # Convert to lists and update results
                for i, idx in enumerate(uncached_indices):
                    embedding_list = embeddings[i].astype(np.float32).tolist()
                    results[idx] = embedding_list

                    # Cache result
                    if self.cache:
                        self.cache[uncached_texts[i]] = embedding_list

            self.metrics.increment_counter("total_embeddings", len(texts))
            return results

    async def queue_for_embedding(self, memory_id: str, content: str):
        """Queue a memory for batch embedding processing."""
        await self.redis.xadd(self.stream_key, {"memory_id": memory_id, "content": content})

    async def process_embedding_queue(self):
        """Process queued embeddings in batches."""
        batch_size = self.config["embedding"]["primary"]["batch_size"]

        while True:
            try:
                # Read batch from stream
                messages = await self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_key: ">"},
                    count=batch_size,
                    block=1000,  # 1 second timeout
                )

                if not messages:
                    continue

                # Extract texts and IDs
                batch_items = []
                for _stream_name, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        batch_items.append(
                            {
                                "msg_id": msg_id,
                                "memory_id": data[b"memory_id"].decode(),
                                "content": data[b"content"].decode(),
                            }
                        )

                if not batch_items:
                    continue

                # Generate embeddings
                texts = [item["content"] for item in batch_items]
                embeddings = await self.generate_embeddings_batch(texts)

                # Update memories with embeddings
                pipeline = self.redis.pipeline()

                for item, embedding in zip(batch_items, embeddings, strict=False):
                    # Determine if STM or LTM
                    key_prefix = "stm:" if item["memory_id"].startswith("stm:") else "ltm:"
                    key = f"{key_prefix}{item['memory_id']}"

                    # Update embedding field
                    pipeline.json().set(key, "$.embedding", embedding)

                    # Acknowledge message
                    pipeline.xack(self.stream_key, self.consumer_group, item["msg_id"])

                await pipeline.execute()

                logger.info(f"Processed batch of {len(batch_items)} embeddings")

            except Exception:
                logger.exception("Error processing embedding queue")
                await asyncio.sleep(5)  # Back off on error

    async def vector_search(
        self,
        query_embedding: list[float],
        index_name: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query vector
            index_name: Index to search (stm or ltm)
            limit: Number of results
            filters: Additional filters

        Returns:
            List of search results with scores
        """
        # Convert embedding to bytes
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        # Build query
        base_query = f"*=>[KNN {limit} @embedding $query_vector AS score]"

        # Add filters if provided
        if filters:
            filter_parts = []
            if "tags" in filters:
                filter_parts.append(f"@tags:{{{' '.join(filters['tags'])}}}")
            if "circle" in filters:
                filter_parts.append(f"@circle:{filters['circle']}")
            if "min_score" in filters:
                filter_parts.append(f"@retrieval_score:[{filters['min_score']} inf]")

            if filter_parts:
                base_query = f"({' '.join(filter_parts)}) {base_query}"

        # Create query
        query = (
            Query(base_query)
            .return_fields("content", "tags", "circle", "score", "id")
            .sort_by("score")
            .dialect(2)
        )

        # Execute search
        results = await self.redis.ft(index_name).search(
            query, query_params={"query_vector": query_vector}
        )

        # Format results
        formatted_results = []
        for doc in results.docs:
            result = {
                "id": doc.id,
                "score": float(doc.score),
                "content": doc.content,
                "tags": doc.tags.split(",") if hasattr(doc, "tags") else [],
                "circle": doc.circle if hasattr(doc, "circle") else None,
            }
            formatted_results.append(result)

        return formatted_results

    async def reindex_memories(self, memory_type: str = "all"):
        """Reindex memories with new embeddings (for model updates)."""
        logger.info(f"Starting reindexing for {memory_type} memories")

        patterns = []
        if memory_type in ["all", "stm"]:
            patterns.append("stm:*")
        if memory_type in ["all", "ltm"]:
            patterns.append("ltm:*")

        total_reindexed = 0

        for pattern in patterns:
            async for key in self.redis.scan_iter(match=pattern):
                try:
                    # Get memory content
                    content = await self.redis.json().get(key, "$.content")
                    if content:
                        # Generate new embedding
                        embedding = await self.generate_embedding(content[0])

                        # Update embedding
                        await self.redis.json().set(key, "$.embedding", embedding)

                        total_reindexed += 1

                        if total_reindexed % 100 == 0:
                            logger.info(f"Reindexed {total_reindexed} memories")

                except Exception:
                    logger.exception(f"Failed to reindex {key}")

        logger.info(f"Reindexing complete. Total: {total_reindexed}")

        return total_reindexed

    async def shutdown(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Embedding service shutdown complete")


# Factory function
async def create_embedding_service(redis_url: str = "redis://localhost:6379") -> EmbeddingService:
    """Create and initialize embedding service."""
    redis_client = await redis.from_url(redis_url)
    service = EmbeddingService(redis_client)
    await service.initialize()
    return service
