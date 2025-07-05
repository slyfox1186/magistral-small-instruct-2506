#!/usr/bin/env python3
"""Configuration module for FastAPI backend.
Contains all imports, environment setup, and logging configuration.
"""

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv

load_dotenv()

import logging
import os
import tracemalloc
from datetime import UTC

UTC = UTC

# Third-party imports

# Try to import prometheus_client
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Local module imports
from colored_logging import setup_colored_logging

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
