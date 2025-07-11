#!/usr/bin/env python3
"""Centralized Resource Manager for ML Models.

==========================================

This module provides a thread-safe, singleton resource manager that handles:
- Model loading and caching (preventing duplicate loads)
- GPU/CPU resource locking (preventing concurrent usage)
- Device management and allocation

Key Features:
- Singleton pattern ensures only one instance
- Fine-grained loading locks prevent race conditions
- GPU execution lock prevents VRAM contention
- Supports both SentenceTransformers and LLM models
"""

import collections
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResourceManager:
    """Thread-safe singleton resource manager for ML models."""

    _instance: Optional["ResourceManager"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "ResourceManager":
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the resource manager (runs only once)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        with self._instance_lock:
            if hasattr(self, "_initialized") and self._initialized:
                return

            logger.info("🚀 Initializing ResourceManager singleton...")

            # Determine optimal device
            self.device = self._determine_device()
            logger.info(f"📱 ResourceManager using device: {self.device}")

            # Model cache: {model_id: (model_instance, execution_lock)}
            self._model_cache: dict[str, tuple[Any, threading.Lock | None]] = {}

            # Fine-grained loading locks to prevent race conditions during model loading
            self._model_loading_locks = collections.defaultdict(threading.Lock)

            # GPU execution lock to prevent VRAM contention
            # Only create if we have CUDA available
            self._gpu_execution_lock = threading.Lock() if self.device == "cuda" else None

            # Statistics tracking
            self._stats = {
                "models_loaded": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_inference_calls": 0,
                "gpu_lock_acquisitions": 0,
            }

            self._initialized = True
            logger.info("✅ ResourceManager initialized successfully")

    def _determine_device(self) -> str:
        """Determine the best device to use for models - FORCE CUDA for sentence-transformers."""
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"🔥 CUDA available with {device_count} GPU(s) - FORCING CUDA usage")
                return "cuda"
            else:
                logger.warning("💻 CUDA not available according to PyTorch - FORCING CUDA anyway")
                return "cuda"  # Force CUDA even if PyTorch says it's not available
        except ImportError:
            logger.warning("📦 PyTorch not available - FORCING CUDA for sentence-transformers")
            return "cuda"  # Force CUDA even without PyTorch

    def get_model(
        self, model_identifier: str, force_device: str | None = None
    ) -> tuple[Any, Any]:
        """Get a model instance with thread-safe loading and caching.

        Args:
            model_identifier: Unique identifier for the model
            force_device: Override default device selection

        Returns:
            Tuple of (model_instance, execution_lock)
        """
        # Check cache first (fast path, no locks)
        if model_identifier in self._model_cache:
            self._stats["cache_hits"] += 1
            model, execution_lock = self._model_cache[model_identifier]
            logger.debug(f"📋 Cache HIT for model: {model_identifier}")
            return model, execution_lock

        # Model not in cache, need to load with proper locking
        self._stats["cache_misses"] += 1
        logger.debug(f"📋 Cache MISS for model: {model_identifier}")

        with self._model_loading_locks[model_identifier]:
            # Double-check pattern: another thread might have loaded it while we waited
            if model_identifier in self._model_cache:
                self._stats["cache_hits"] += 1
                model, execution_lock = self._model_cache[model_identifier]
                logger.debug(f"📋 Cache HIT after lock wait for model: {model_identifier}")
                return model, execution_lock

            # Critical section: Load the model
            logger.info(f"🔄 Loading model: {model_identifier}")
            start_time = time.time()

            model = self._load_model_from_source(model_identifier)

            # Move to appropriate device - FORCE CUDA
            target_device = force_device or "cuda"  # Always use CUDA, ignore self.device
            if hasattr(model, "to"):
                model = model.to(target_device)
                logger.info(f"📱 Moved model {model_identifier} to {target_device} (FORCED)")
            else:
                logger.info(f"📱 Model {model_identifier} already on {target_device} (FORCED)")

            # Set to evaluation mode if applicable
            if hasattr(model, "eval"):
                model.eval()

            # Determine execution lock based on actual device
            model_device = getattr(model, "device", target_device)
            device_type = model_device.type if hasattr(model_device, "type") else str(model_device)

            execution_lock = self._gpu_execution_lock if "cuda" in device_type else None

            # Cache the loaded model
            self._model_cache[model_identifier] = (model, execution_lock)

            load_time = time.time() - start_time
            self._stats["models_loaded"] += 1

            logger.info(
                f"✅ Successfully loaded model '{model_identifier}' onto {device_type} in {load_time:.2f}s"
            )

            return model, execution_lock

    def _load_model_from_source(self, model_identifier: str) -> Any:
        """Load a model from its source based on the identifier."""
        if "bge" in model_identifier.lower() or "sentence-transformer" in model_identifier.lower():
            return self._load_sentence_transformer(model_identifier)
        elif "llama" in model_identifier.lower() or "mistral" in model_identifier.lower():
            return self._load_llm_model(model_identifier)
        else:
            raise ValueError(f"❌ Unknown model type for identifier: {model_identifier}")

    def _load_sentence_transformer(self, model_identifier: str) -> Any:
        """Load a SentenceTransformer model - FORCE CUDA usage with FP16."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # FORCE CUDA for sentence-transformers - no fallback to CPU
            initial_device = "cuda"

            logger.info(
                f"🚀 Loading SentenceTransformer {model_identifier} on {initial_device} with FP32 (FORCED)"
            )

            # Load with full precision for better accuracy
            model = SentenceTransformer(
                model_identifier, device=initial_device, model_kwargs={"torch_dtype": torch.float32}
            )

            # Get embedding dimension for logging
            embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"📐 SentenceTransformer embedding dimension: {embedding_dim}")
            logger.info("💾 Model loaded in FP32 for full precision")

        except ImportError:
            logger.exception("❌ Failed to import sentence_transformers")
            raise
        except Exception:
            logger.exception(f"❌ Failed to load SentenceTransformer {model_identifier}")
            raise
        else:
            return model

    def _load_llm_model(self, model_identifier: str) -> Any:
        """Load an LLM model (placeholder for future implementation)."""
        logger.warning(f"⚠️ LLM loading not yet implemented for: {model_identifier}")
        raise NotImplementedError("LLM loading will be implemented in a future version")

    def run_inference(
        self,
        model_identifier: str,
        task_function: Callable[[Any], Any],
        force_device: str | None = None,
    ) -> Any:
        """Run inference with proper resource locking.

        Args:
            model_identifier: Unique identifier for the model
            task_function: Function that takes the model and performs the work
            force_device: Override default device selection

        Returns:
            Result of the task_function
        """
        self._stats["total_inference_calls"] += 1

        model, execution_lock = self.get_model(model_identifier, force_device)

        if execution_lock:
            # GPU model - need to acquire execution lock
            with execution_lock:
                self._stats["gpu_lock_acquisitions"] += 1
                logger.debug(f"🔒 Acquired GPU lock for {model_identifier}")
                start_time = time.time()

                try:
                    result = task_function(model)
                    execution_time = time.time() - start_time
                    logger.debug(
                        f"🔓 Released GPU lock for {model_identifier} (execution: {execution_time:.3f}s)"
                    )
                except Exception:
                    logger.exception(f"❌ Error during GPU inference for {model_identifier}")
                    raise
                else:
                    return result
        else:
            # CPU model - no lock needed
            logger.debug(f"💻 Running CPU inference for {model_identifier}")
            start_time = time.time()

            try:
                result = task_function(model)
                execution_time = time.time() - start_time
                logger.debug(
                    f"✅ CPU inference completed for {model_identifier} (execution: {execution_time:.3f}s)"
                )
            except Exception:
                logger.exception(f"❌ Error during CPU inference for {model_identifier}")
                raise
            else:
                return result

    def get_stats(self) -> dict[str, Any]:
        """Get resource manager statistics."""
        return {
            **self._stats,
            "cached_models": list(self._model_cache.keys()),
            "device": self.device,
            "gpu_lock_enabled": self._gpu_execution_lock is not None,
        }

    def clear_cache(self, model_identifier: str | None = None) -> None:
        """Clear model cache.

        Args:
            model_identifier: Specific model to clear, or None to clear all
        """
        with self._instance_lock:
            if model_identifier:
                if model_identifier in self._model_cache:
                    del self._model_cache[model_identifier]
                    logger.info(f"🗑️ Cleared cache for model: {model_identifier}")
                else:
                    logger.warning(f"⚠️ Model not in cache: {model_identifier}")
            else:
                cleared_count = len(self._model_cache)
                self._model_cache.clear()
                logger.info(f"🗑️ Cleared cache for {cleared_count} models")

    def is_model_loaded(self, model_identifier: str) -> bool:
        """Check if a model is currently loaded in cache."""
        return model_identifier in self._model_cache


# Global instance getter function
def get_resource_manager() -> ResourceManager:
    """Get the singleton ResourceManager instance."""
    return ResourceManager()


# Convenience functions for common model operations
def get_sentence_transformer_embeddings(
    texts: list, model_id: str = "BAAI/bge-large-en-v1.5"
) -> list:
    """Get embeddings using the ResourceManager."""
    resource_manager = get_resource_manager()

    def embedding_task(model):
        # Use convert_to_tensor=False to get numpy arrays regardless of device
        # This ensures backward compatibility with existing code
        return model.encode(
            texts, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True
        )

    return resource_manager.run_inference(model_id, embedding_task)


def ensure_sentence_transformer_loaded(model_id: str = "BAAI/bge-large-en-v1.5") -> None:
    """Ensure a SentenceTransformer model is loaded (for backwards compatibility)."""
    resource_manager = get_resource_manager()
    resource_manager.get_model(model_id)
    logger.info(f"✅ SentenceTransformer model {model_id} ensured loaded")
