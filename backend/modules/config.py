#!/usr/bin/env python3
"""Configuration module for FastAPI backend.

Contains all imports, environment setup, and logging configuration.
"""

# Standard library imports
import logging
import os
import tracemalloc

# Third-party imports
from dotenv import load_dotenv

# Check for prometheus_client availability
try:
    # Import is only used to check availability, not actually used in this module
    import prometheus_client as _prometheus_client  # noqa: F401

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Local module imports
from colored_logging import setup_colored_logging

# Load environment variables FIRST before any other setup
load_dotenv()

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

# API Configuration
API_CONFIG = {
    "cors_origins": ["http://localhost:3000", "http://localhost:4000"],
    "enable_metrics": PROMETHEUS_AVAILABLE,
    "debug": os.getenv("DEBUG", "false").lower() == "true"
}
