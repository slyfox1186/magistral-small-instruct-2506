#!/usr/bin/env python3
"""Configuration module for FastAPI backend.
Contains all imports, environment setup, and logging configuration.
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import logging
import os
import textwrap
import time
import tracemalloc
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
UTC = timezone.utc
from typing import Any

# Third-party imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

# Try to import prometheus_client
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Local module imports
import crypto_trading
import monitoring
import redis_utils
import stock_search
import utils
import web_scraper
from circuit_breaker import circuit_breaker_manager
from colored_logging import create_section_separator, log_startup_banner, setup_colored_logging
from config import API_CONFIG, GENERATION_CONFIG, MODEL_CONFIG, MODEL_PATH
from constants import (
    BATCH_PROCESSING_SIZE,
    CACHE_TTL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMPORTANCE_SCORE,
    ECHO_SIMILARITY_THRESHOLD,
    MAX_EMBEDDING_QUEUE_SIZE,
    MAX_MEMORY_EXTRACTION_QUEUE_SIZE,
    MEMORY_CONSOLIDATION_INTERVAL,
)
from gpu_lock import gpu_for_inference, gpu_lock
from llm_optimizer import get_llm_optimizer, initialize_llm_optimizer
from memory_provider import MemoryConfig, get_memory_stats, get_memory_system
from metacognitive_engine import initialize_metacognitive_engine
from token_manager import TokenManager
from ultra_advanced_engine import UltraAdvancedEngine

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable tracemalloc for better coroutine debugging
tracemalloc.start()

# Configure logging
logger = logging.getLogger(__name__)
if not PROMETHEUS_AVAILABLE:
    logger.warning("prometheus_client not available. Install with: pip install prometheus_client")

# Configure beautiful colored logging
setup_colored_logging(level=logging.DEBUG, enable_stream_formatting=True)
logger = logging.getLogger(__name__)