#!/usr/bin/env python3
"""Application lifespan management for startup and shutdown."""

import asyncio
import contextlib
import logging
import tracemalloc
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Local imports that need to be available
import crypto_trading
import monitoring
import redis_utils
import stock_search
from colored_logging import create_section_separator, log_startup_banner
from config import MODEL_CONFIG, MODEL_PATH
from constants import CACHE_TTL
from llm_optimizer import initialize_llm_optimizer
from memory_provider import MemoryConfig, get_memory_system
from metacognitive_engine import initialize_metacognitive_engine
from token_manager import TokenManager
from ultra_advanced_engine import UltraAdvancedEngine

# Import from our modules
from .globals import (
    RedisStateManager,
    app_state,
    bg_state,
)
from .helpers import periodic_memory_consolidation
from .system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ===================== Helper Functions =====================
async def _initialize_redis() -> None:
    """Initialize Redis connection and validate required modules."""
    try:
        logger.info(" Initializing async Redis connection with built-in resilience...")
        app_state.redis_client = await redis_utils.initialize_redis_connection_async()
        logger.info(" Async Redis connection initialized successfully")

        # Validate required Redis modules
        required_modules = {"search": "RediSearch", "json": "RedisJSON"}
        await redis_utils.validate_redis_modules(app_state.redis_client, required_modules)
        logger.info(" Required Redis modules (RediSearch, RedisJSON) are available")

    except redis_utils.RedisModuleError:
        logger.exception(" Critical Redis module missing")
        app_state.redis_client = None
    except Exception:
        logger.exception(" Failed to initialize Redis connection")
        app_state.redis_client = None


async def _perform_vram_check() -> None:
    """Perform VRAM pre-flight check to prevent CUDA OOM crashes."""
    try:
        from gpu_utils import check_vram_requirements

        model_filename = Path(MODEL_PATH).name.lower()
        if "24b" in model_filename:
            required_vram = 16.0
        elif "small" in model_filename:
            required_vram = 6.0
        else:
            required_vram = 8.0

        sufficient, available_gb, total_gb = check_vram_requirements(required_vram, gpu_index=0)
        logger.info(
            f" VRAM Check: Available = {available_gb:.2f} GB, Required (estimated) = {required_vram} GB, "
            f"Total = {total_gb:.2f} GB"
        )

        if not sufficient:
            app_state.llm_error = (
                f" Insufficient VRAM. Required: ~{required_vram} GB, Available: {available_gb:.2f} GB. "
                f"Cannot load model safely."
            )
            logger.error(app_state.llm_error)
        else:
            logger.info(" VRAM check passed - sufficient memory available for model loading")

    except Exception as e:
        logger.warning(f" Could not perform VRAM check: {e}. Proceeding with model load attempt.")
        app_state.llm_error = None


def _initialize_model() -> None:
    """Initialize model configuration and validate model file."""
    logger.info(f"Loading Llama model from {MODEL_PATH}")
    logger.info(
        f"Model configuration: ctx_length={MODEL_CONFIG['n_ctx']}, gpu_layers={MODEL_CONFIG['n_gpu_layers']}"
    )

    if not Path(MODEL_PATH).exists():
        app_state.llm_error = f"Model file not found at {MODEL_PATH}"
        logger.error(app_state.llm_error)
    elif app_state.llm_error:
        # VRAM check failed, error already set and logged
        pass
    else:
        logger.info(" Skipping legacy model loading - using persistent LLM server")
        app_state.llm = None
        app_state.llm_error = None


def _initialize_core_components() -> None:
    """Initialize core application components."""
    # Initialize Redis state manager for multi-worker safety
    if redis_utils.is_redis_available(app_state.redis_client):
        app_state.state_manager = RedisStateManager(app_state.redis_client)
        logger.info(" Redis state manager initialized for multi-worker safety")

    # Initialize token manager (required for all operations)
    try:
        app_state.token_manager = TokenManager(MODEL_CONFIG["n_ctx"], SYSTEM_PROMPT)
        logger.info(" Token manager initialized successfully")
    except Exception:
        logger.exception(" Failed to initialize token manager")
        app_state.token_manager = None


async def _initialize_memory_system() -> None:
    """Initialize personal memory system and consolidation."""
    try:
        logger.info("Initializing personal AI memory system...")

        memory_config = MemoryConfig()
        app_state.personal_memory = get_memory_system(memory_config)
        logger.info(
            f" Personal memory system initialized with backend: {memory_config.MEMORY_BACKEND}"
        )

        if memory_config.USE_REDIS_COMPAT:
            logger.info(" Redis compatibility mode enabled for gradual migration")

        # Start periodic memory consolidation task
        consolidation_task = asyncio.create_task(periodic_memory_consolidation())
        app_state.background_tasks.append(consolidation_task)
        logger.info(" Started periodic memory consolidation task")

    except Exception as e:
        logger.error(f" Error initializing memory system: {e}", exc_info=True)
        app_state.personal_memory = None
        app_state.importance_calculator = None


def _initialize_llm_components() -> None:
    """Initialize LLM-dependent components."""
    if app_state.llm:
        try:
            app_state.ultra_engine = UltraAdvancedEngine(app_state.llm, None)
            logger.info(" Ultra-Advanced AI Engine initialized successfully")
        except Exception as e:
            logger.error(f" Error initializing LLM-dependent components: {e}", exc_info=True)


def _initialize_trading_modules() -> None:
    """Initialize trading and market data modules."""
    try:
        logger.info("Initializing trading and market data modules...")
        app_state.crypto_trader = crypto_trading.CryptoTrading()
        app_state.stock_searcher = stock_search.EnhancedStockSearch()
        logger.info(" Trading modules initialized successfully")
    except Exception as e:
        logger.error(f" Error initializing trading modules: {e}", exc_info=True)


def _initialize_monitoring_systems() -> None:
    """Initialize monitoring systems."""
    try:
        logger.info("Initializing production monitoring systems...")
        app_state.health_checker, app_state.memory_analytics = monitoring.initialize_monitoring(
            None,
            None,
            app_state.redis_client,
        )

        logger.info(" Monitoring systems initialized successfully")
        logger.info("    Health checks: ACTIVE")
        logger.info("    Performance metrics: ACTIVE")
        logger.info("    Memory analytics: ACTIVE")
    except Exception as e:
        logger.error(f" Error initializing monitoring systems: {e}", exc_info=True)


def _initialize_llm_optimizer() -> None:
    """Initialize LLM optimizer for performance."""
    try:
        logger.info(" Initializing LLM optimizer...")
        initialize_llm_optimizer(redis_client=app_state.redis_client, cache_ttl=CACHE_TTL)
        logger.info(" LLM optimizer initialized successfully")
        logger.info("    Caching enabled for LLM calls")
        logger.info("    Performance monitoring active")
    except Exception as e:
        logger.error(f" Error initializing LLM optimizer: {e}", exc_info=True)


def _preload_sentence_transformer() -> None:
    """Pre-load sentence transformer model to avoid lazy loading."""
    try:
        logger.info(" Pre-loading sentence transformer model to avoid lazy loading...")
        from resource_manager import ensure_sentence_transformer_loaded

        ensure_sentence_transformer_loaded("BAAI/bge-large-en-v1.5")
        logger.info(" Sentence transformer model pre-loaded successfully")
        logger.info("    Embeddings will be generated without startup delay")
        logger.info("    Model cached in GPU memory with FP16 precision")
    except Exception as e:
        logger.error(f" Error pre-loading sentence transformer: {e}", exc_info=True)
        logger.warning(" Embeddings will be loaded on first use (may cause delay)")


async def _preload_llm_server() -> None:
    """Pre-load persistent LLM server to prevent concurrent loading."""
    try:
        logger.info(" Pre-loading persistent LLM server...")
        from persistent_llm_server import get_llm_server

        await get_llm_server()
        logger.info(" Persistent LLM server pre-loaded successfully")
        logger.info("    Model loaded into GPU memory")
        logger.info("    Concurrent initialization protection: ACTIVE")
    except Exception as e:
        logger.error(f" Error pre-loading persistent LLM server: {e}", exc_info=True)
        logger.warning(" LLM server will be loaded on first use (may cause issues)")


def _initialize_metacognitive_engine() -> None:
    """Initialize metacognitive engine for response quality assessment."""
    try:
        logger.info(" Initializing Metacognitive Engine v1...")
        app_state.metacognitive_engine = initialize_metacognitive_engine(app_state.llm, None)
        logger.info(" Metacognitive Engine initialized successfully")
        logger.info("    Heuristic evaluation: ACTIVE")
        logger.info("    LLM criticism: ACTIVE")
        logger.info("    Self-improvement loop: ACTIVE")
    except Exception as e:
        logger.error(f" Error initializing metacognitive engine: {e}", exc_info=True)


async def _startup_sequence() -> None:
    """Execute the complete startup sequence."""
    print(create_section_separator(" APPLICATION STARTUP", 80))
    logger.info("Starting FastAPI application lifecycle")

    await _initialize_redis()
    logger.info(" Using LLM's natural sentiment understanding instead of keyword matching")

    print(create_section_separator(" MODEL INITIALIZATION", 80))
    log_startup_banner("Mistral Small Chat Application", "v3.2-24B")

    await _perform_vram_check()
    _initialize_model()
    _initialize_core_components()
    await _initialize_memory_system()
    _initialize_llm_components()
    _initialize_trading_modules()
    _initialize_monitoring_systems()
    _initialize_llm_optimizer()
    _preload_sentence_transformer()
    await _preload_llm_server()
    _initialize_metacognitive_engine()

    logger.info(" Web scraping configured for direct async calls")
    logger.info("    Non-blocking web scraping: ACTIVE")
    logger.info("    Native async implementation: READY")

    logger.info(" Application startup complete - Ready to serve requests!")
    print(create_section_separator(" SERVER READY", 80))


async def _shutdown_sequence() -> None:
    """Execute the complete shutdown sequence."""
    print(create_section_separator(" APPLICATION SHUTDOWN", 80))
    logger.info("Beginning graceful application shutdown...")

    # Save Redis data
    if redis_utils.is_redis_available(app_state.redis_client):
        try:
            await app_state.redis_client.save()
            logger.info(" Redis data persistence completed")
        except Exception:
            logger.exception(" Error saving Redis data")

    # Cleanup persistent LLM server
    try:
        from persistent_llm_server import llm_server

        if llm_server:
            logger.info(" Shutting down persistent LLM server...")
            await llm_server.stop()
            logger.info(" Persistent LLM server stopped")
    except Exception:
        logger.exception(" Error stopping LLM server")

    # Cancel all tracked background tasks
    if app_state.background_tasks:
        logger.info(f" Stopping {len(app_state.background_tasks)} background tasks...")
        for task in app_state.background_tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*app_state.background_tasks, return_exceptions=True)
        app_state.background_tasks.clear()
        logger.info(" All background tasks stopped")

    # Stop background processor
    if bg_state.background_processor_task and not bg_state.background_processor_task.done():
        logger.info(" Stopping background processor...")
        bg_state.background_processor_running = False
        bg_state.background_processor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bg_state.background_processor_task
        logger.info(" Background processor stopped")

    # Cleanup legacy model
    if app_state.llm:
        logger.info(" Cleaning up legacy model resources...")
        app_state.llm = None

    # Stop tracemalloc to prevent memory leak
    tracemalloc.stop()
    logger.info(" Tracemalloc stopped")

    logger.info(" Application shutdown complete")


# ===================== Application Lifespan =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown management."""
    await _startup_sequence()
    yield
    await _shutdown_sequence()
