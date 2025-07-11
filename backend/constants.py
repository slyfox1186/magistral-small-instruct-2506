"""Backend Constants for Neural Consciousness Chat System.

Centralized configuration values to prevent magic numbers throughout the codebase.
"""

# Memory and Queue Management
MAX_EMBEDDING_QUEUE_SIZE = 1000  # Maximum items in embedding queue
MAX_MEMORY_EXTRACTION_QUEUE_SIZE = 1000  # Maximum items in memory extraction queue
MEMORY_CONSOLIDATION_INTERVAL = 3600  # Memory consolidation interval in seconds (1 hour)

# Text processing constants
MINIMAL_TEXT_LENGTH = 20  # Minimum meaningful text length
MAX_REASONABLE_TEXT_LENGTH = 200  # Maximum reasonable text length
CONTENT_PREVIEW_LENGTH = 100  # Length for content previews
EXTENDED_CONTENT_PREVIEW_LENGTH = 200  # Extended content preview length

# Echo detection constants
MIN_EMBEDDINGS_FOR_COMPARISON = 2  # Minimum embeddings needed for similarity comparison

# Redis module constants
MIN_MODULE_LIST_LENGTH = 2  # Minimum module list length for Redis module parsing

# Crypto trading constants
TRILLION_THRESHOLD = 1_000_000_000_000
BILLION_THRESHOLD = 1_000_000_000
MILLION_THRESHOLD = 1_000_000
THOUSAND_THRESHOLD = 1_000

# Sentiment thresholds
POSITIVE_SENTIMENT_THRESHOLD = 0.1
NEGATIVE_SENTIMENT_THRESHOLD = -0.1

# Model similarity thresholds
LOW_SIMILARITY_THRESHOLD = 0.6
MEDIUM_SIMILARITY_THRESHOLD = 0.8
HIGH_SIMILARITY_THRESHOLD = 0.9

# Performance constants
SLOW_OPERATION_THRESHOLD_MS = 100  # Milliseconds
EMBEDDING_DIMENSION_COUNT = 10  # Number of dimensions for version tracking

# Model and Processing
DEFAULT_BATCH_SIZE = 2048  # Default n_batch for Llama model
DEFAULT_GPU_LAYERS = -1  # Use all GPU layers by default
DEFAULT_CONTEXT_SIZE = 32768  # Default context window size

# Timeouts and Intervals
GPU_LOCK_MAX_HOLD_TIME = 90.0  # Maximum GPU lock hold time in seconds
DEFAULT_REQUEST_TIMEOUT = 60.0  # Default timeout for lock acquisition
GRACEFUL_SHUTDOWN_TIMEOUT = 5.0  # Maximum time for graceful shutdown

# Echo Detection
ECHO_SIMILARITY_THRESHOLD = 0.98  # Threshold for detecting echoed responses

# Performance and Monitoring
BATCH_PROCESSING_SIZE = 50  # Size for batch operations
LOG_PERFORMANCE_INTERVAL = 300  # Performance logging interval in seconds (5 minutes)

# Memory Management
DEFAULT_IMPORTANCE_SCORE = 0.5  # Default importance for memory items
HIGH_IMPORTANCE_THRESHOLD = 0.7  # Threshold for high importance items

# Network and API
DEFAULT_API_TIMEOUT = 30.0  # Default API request timeout
MAX_RETRIES = 3  # Maximum retry attempts for failed operations
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)
