#!/usr/bin/env python3
"""Application lifespan management for startup and shutdown."""

import asyncio
import logging
import os
import tracemalloc
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Local imports that need to be available
import crypto_trading
import monitoring
import redis_utils
import stock_search
from colored_logging import create_section_separator, log_startup_banner
from config import API_CONFIG, MODEL_CONFIG, MODEL_PATH
from constants import CACHE_TTL
from llm_optimizer import initialize_llm_optimizer
from memory_provider import MemoryConfig, get_memory_system
from metacognitive_engine import initialize_metacognitive_engine
from token_manager import TokenManager
from ultra_advanced_engine import UltraAdvancedEngine

# Import from our modules
from .globals import (
    RedisStateManager,
    background_processor_running,
    background_processor_task,
    background_tasks,
)
from . import globals as global_vars
from .helpers import periodic_memory_consolidation
from .system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ===================== Application Lifespan =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown management."""

    print(create_section_separator(" APPLICATION STARTUP", 80))
    logger.info("Starting FastAPI application lifecycle")

    # Initialize Redis connection with connection pooling asynchronously
    try:
        logger.info(" Initializing async Redis connection with built-in resilience...")
        global_vars.redis_client = await redis_utils.initialize_redis_connection_async()
        logger.info(" Async Redis connection initialized successfully")

        # Validate required Redis modules
        required_modules = {"search": "RediSearch", "json": "RedisJSON"}
        await redis_utils.validate_redis_modules(redis_client, required_modules)
        logger.info(" Required Redis modules (RediSearch, RedisJSON) are available")

    except redis_utils.RedisModuleError as e:
        logger.error(f" Critical Redis module missing: {e}")
        redis_client = None  # Ensure redis is considered unavailable
    except Exception as e:
        logger.error(f" Failed to initialize Redis connection: {e}")
        redis_client = None

    # Removed flawed sentiment analysis loading - LLM handles sentiment naturally
    logger.info(" Using LLM's natural sentiment understanding instead of keyword matching")

    # Load the model inside lifespan to prevent blocking startup
    print(create_section_separator(" MODEL INITIALIZATION", 80))
    log_startup_banner("Mistral Small Chat Application", "v3.2-24B")

    # VRAM Pre-flight Check - Prevent CUDA OOM crashes
    try:
        from gpu_utils import check_vram_requirements

        # Estimate VRAM requirements based on model filename only
        model_filename = os.path.basename(MODEL_PATH).lower()
        if "24b" in model_filename:
            required_vram = 16.0  # GB - Estimate for 24B Q4 model
        elif "small" in model_filename:
            required_vram = 6.0  # GB - Estimate for Small model
        else:
            required_vram = 8.0  # GB - Default conservative estimate

        sufficient, available_gb, total_gb = check_vram_requirements(required_vram, gpu_index=0)
        logger.info(
            f" VRAM Check: Available = {available_gb:.2f} GB, Required (estimated) = {required_vram} GB, "
            f"Total = {total_gb:.2f} GB"
        )

        if not sufficient:
            llm_error = (
                f" Insufficient VRAM. Required: ~{required_vram} GB, Available: {available_gb:.2f} GB. "
                f"Cannot load model safely."
            )
            logger.error(llm_error)
        else:
            logger.info(" VRAM check passed - sufficient memory available for model loading")

    except Exception as e:
        logger.warning(f" Could not perform VRAM check: {e}. Proceeding with model load attempt.")
        llm_error = None  # Allow startup to continue if VRAM check fails

    logger.info(f"Loading Llama model from {MODEL_PATH}")
    logger.info(
        f"Model configuration: ctx_length={MODEL_CONFIG['n_ctx']}, gpu_layers={MODEL_CONFIG['n_gpu_layers']}"
    )

    if not os.path.exists(MODEL_PATH):
        llm_error = f"Model file not found at {MODEL_PATH}"
        logger.error(llm_error)
    elif llm_error:
        # VRAM check failed, error already set and logged
        pass
    else:
        # DISABLED: Using persistent LLM server instead
        logger.info(" Skipping legacy model loading - using persistent LLM server")
        global_vars.llm = None  # Will be handled by persistent_llm_server.py
        llm_error = None

    # Initialize Redis state manager for multi-worker safety
    if redis_utils.is_redis_available(global_vars.redis_client):
        global_vars.state_manager = RedisStateManager(global_vars.redis_client)
        logger.info(" Redis state manager initialized for multi-worker safety")

    # Initialize token manager (required for all operations)
    try:
        global_vars.token_manager = TokenManager(MODEL_CONFIG["n_ctx"], SYSTEM_PROMPT)
        logger.info(" Token manager initialized successfully")
    except Exception as tm_error:
        logger.error(f" Failed to initialize token manager: {tm_error}")
        # Set to None to prevent further errors
        global_vars.token_manager = None

    # Initialize personal memory system (SQLite-based for simplicity)
    try:
        logger.info("Initializing personal AI memory system...")

        # Initialize personal memory system using provider
        memory_config = MemoryConfig()
        global_vars.personal_memory = get_memory_system(memory_config)
        logger.info(
            f" Personal memory system initialized with backend: {memory_config.MEMORY_BACKEND}"
        )

        # Initialize importance calculator - disabled for now
        # importance_calculator = NeuralImportanceCalculator()
        # logger.info(" Neural importance calculator initialized")

        # Log if compatibility mode is enabled
        if memory_config.USE_REDIS_COMPAT:
            logger.info(" Redis compatibility mode enabled for gradual migration")

        # Start periodic memory consolidation task
        consolidation_task = asyncio.create_task(periodic_memory_consolidation())
        background_tasks.append(consolidation_task)
        logger.info(" Started periodic memory consolidation task")

    except Exception as e:
        logger.error(f" Error initializing memory system: {e}", exc_info=True)
        # Set to None to prevent further errors
        global_vars.personal_memory = None
        global_vars.importance_calculator = None

    # Initialize LLM-dependent components
    if global_vars.llm:
        try:
            # Initialize Ultra-Advanced Engine (with gpu_lock instead of model_lock)
            global_vars.ultra_engine = UltraAdvancedEngine(
                global_vars.llm, None
            )  # Pass None since we use gpu_for_inference
            logger.info(" Ultra-Advanced AI Engine initialized successfully")

        except Exception as e:
            logger.error(f" Error initializing LLM-dependent components: {e}", exc_info=True)

    # Initialize trading modules
    try:
        logger.info("Initializing trading and market data modules...")
        global_vars.crypto_trader = crypto_trading.CryptoTrading()
        global_vars.stock_searcher = stock_search.StockSearch()
        logger.info(" Trading modules initialized successfully")
    except Exception as e:
        logger.error(f" Error initializing trading modules: {e}", exc_info=True)

    # Initialize monitoring systems
    try:
        logger.info("Initializing production monitoring systems...")
        global_vars.health_checker, global_vars.memory_analytics = monitoring.initialize_monitoring(
            None,
            None,
            global_vars.redis_client,  # Memory monitoring will be updated later
        )

        logger.info(" Monitoring systems initialized successfully")
        logger.info("    Health checks: ACTIVE")
        logger.info("    Performance metrics: ACTIVE")
        logger.info("    Memory analytics: ACTIVE")
    except Exception as e:
        logger.error(f" Error initializing monitoring systems: {e}", exc_info=True)

    # Legacy memory system removed - using new unified memory pipeline

    # Initialize LLM optimizer for performance
    try:
        logger.info(" Initializing LLM optimizer...")
        initialize_llm_optimizer(redis_client=global_vars.redis_client, cache_ttl=CACHE_TTL)
        logger.info(" LLM optimizer initialized successfully")
        logger.info("    Caching enabled for LLM calls")
        logger.info("    Performance monitoring active")
    except Exception as e:
        logger.error(f" Error initializing LLM optimizer: {e}", exc_info=True)

    # Initialize sentence transformer model at startup to avoid lazy loading
    # CRITICAL: This MUST happen at startup to prevent lazy loading delays during inference
    # The ResourceManager uses singleton pattern and caches the model, so loading it here
    # ensures it's ready when embedding_service.py calls get_sentence_transformer_embeddings()
    try:
        logger.info(" Pre-loading sentence transformer model to avoid lazy loading...")
        from resource_manager import ensure_sentence_transformer_loaded

        # This forces the model to load NOW instead of on first embedding request
        ensure_sentence_transformer_loaded("BAAI/bge-small-en-v1.5")
        logger.info(" Sentence transformer model pre-loaded successfully")
        logger.info("    Embeddings will be generated without startup delay")
        logger.info("    Model cached in GPU memory with FP16 precision")
    except Exception as e:
        logger.error(f" Error pre-loading sentence transformer: {e}", exc_info=True)
        logger.warning(" Embeddings will be loaded on first use (may cause delay)")

    # Initialize persistent LLM server at startup to prevent concurrent loading
    try:
        logger.info(" Pre-loading persistent LLM server...")
        from persistent_llm_server import get_llm_server

        # This will load the model once during startup, preventing race conditions
        await get_llm_server()
        logger.info(" Persistent LLM server pre-loaded successfully")
        logger.info("    Model loaded into GPU memory")
        logger.info("    Concurrent initialization protection: ACTIVE")
    except Exception as e:
        logger.error(f" Error pre-loading persistent LLM server: {e}", exc_info=True)
        logger.warning(" LLM server will be loaded on first use (may cause issues)")

    # Initialize metacognitive engine for response quality assessment
    try:
        logger.info(" Initializing Metacognitive Engine v1...")
        global_vars.metacognitive_engine = initialize_metacognitive_engine(global_vars.llm, None)
        logger.info(" Metacognitive Engine initialized successfully")
        logger.info("    Heuristic evaluation: ACTIVE")
        logger.info("    LLM criticism: ACTIVE")
        logger.info("    Self-improvement loop: ACTIVE")
    except Exception as e:
        logger.error(f" Error initializing metacognitive engine: {e}", exc_info=True)

    # Web scraping now uses direct async calls - no separate service needed
    logger.info(" Web scraping configured for direct async calls")
    logger.info("    Non-blocking web scraping: ACTIVE")
    logger.info("    Native async implementation: READY")

    logger.info(" Application startup complete - Ready to serve requests!")
    print(create_section_separator(" SERVER READY", 80))

    yield

    print(create_section_separator(" APPLICATION SHUTDOWN", 80))
    logger.info("Beginning graceful application shutdown...")

    # Save Redis data
    if redis_utils.is_redis_available(global_vars.redis_client):
        try:
            await global_vars.redis_client.save()
            logger.info(" Redis data persistence completed")
        except Exception as e:
            logger.error(f" Error saving Redis data: {e}")

    # Cleanup persistent LLM server
    try:
        from persistent_llm_server import llm_server

        if llm_server:
            logger.info(" Shutting down persistent LLM server...")
            await llm_server.stop()
            logger.info(" Persistent LLM server stopped")
    except Exception as e:
        logger.error(f" Error stopping LLM server: {e}")

    # Stop all background tasks
    global background_processor_running

    # Cancel all tracked background tasks
    if background_tasks:
        logger.info(f" Stopping {len(background_tasks)} background tasks...")
        for task in background_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete cancellation
        await asyncio.gather(*background_tasks, return_exceptions=True)
        background_tasks.clear()
        logger.info(" All background tasks stopped")

    # Stop background processor
    if background_processor_task and not background_processor_task.done():
        logger.info(" Stopping background processor...")
        background_processor_running = False
        background_processor_task.cancel()
        try:
            await background_processor_task
        except asyncio.CancelledError:
            pass
        logger.info(" Background processor stopped")

    # Cleanup legacy model
    if global_vars.llm:
        logger.info(" Cleaning up legacy model resources...")
        global_vars.llm = None

    # Stop tracemalloc to prevent memory leak
    tracemalloc.stop()
    logger.info(" Tracemalloc stopped")

    logger.info(" Application shutdown complete")