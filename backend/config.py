"""Configuration settings for the backend application."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Model configuration - Magistral-Small-2506
MAGISTRAL_MODEL_PATH = str(MODELS_DIR / "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf")

# Default model selection (Magistral by default)
MODEL_PATH = os.environ.get("MODEL_PATH", MAGISTRAL_MODEL_PATH)

# Model initialization parameters
MODEL_CONFIG = {
    "n_ctx": 8192,  # Reduced from 32768 to help with CUDA OOM
    "n_batch": 2048,
    "n_threads": os.cpu_count(),
    "main_gpu": 0,
    "n_gpu_layers": -1,
    "flash_attn": True,
    "use_mmap": True,
    "use_mlock": False,
    "offload_kqv": True,
    "verbose": False,
}

# Model completion/generation parameters - Magistral-Small-2506
GENERATION_CONFIG = {
    "temperature": 0.7,  # Recommended for Magistral
    "max_tokens": 2048,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.0,
    "stream": True,
    "echo": False,
    "stop": ["[/INST]", "[/SYSTEM_PROMPT]"],  # Magistral stop tokens
}

# Redis configuration
REDIS_CONFIG = {
    "host": os.environ.get("REDIS_HOST", "localhost"),
    "port": int(os.environ.get("REDIS_PORT", 6379)),
    "decode_responses": True,
    "max_history": int(os.environ.get("REDIS_MAX_HISTORY", 10)),
}

# API configuration
API_CONFIG = {
    "host": os.environ.get("API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("API_PORT", 8000)),
    "cors_origins": os.environ.get("CORS_ORIGINS", "http://localhost:4000").split(","),
    "timeout": float(os.environ.get("API_TIMEOUT", 120.0)),
}


# Cache configuration
CACHE_CONFIG = {
    "scrape_cache_ttl": int(os.environ.get("SCRAPE_CACHE_TTL", 900)),  # 15 minutes
    "memory_cache_size": int(os.environ.get("MEMORY_CACHE_SIZE", 10000)),
}

# External services
EXTERNAL_SERVICES = {
    "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
    "google_cse_id": os.environ.get("GOOGLE_CSE_ID", ""),
    "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
}

# Logging configuration
LOGGING_CONFIG = {
    "level": os.environ.get("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
