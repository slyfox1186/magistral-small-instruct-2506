import asyncio
import hashlib
import json
import logging
import os  # For os.cpu_count()
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import redis
from redis.commands.search.field import (
    NumericField,  # Although not used yet, good to have
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# Local imports
from async_redis_client import initialize_async_redis_connection
from resource_manager import get_resource_manager

# Try to import optional dependencies
try:
    from memory.local_cache import ResilientRedisClient

    RESILIENT_CLIENT_AVAILABLE = True
except ImportError:
    RESILIENT_CLIENT_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)
if not RESILIENT_CLIENT_AVAILABLE:
    logger.warning("Resilient Redis client not available - fallback disabled")
if not NUMPY_AVAILABLE:
    logger.warning(
        "numpy library not found. Install it to enable vector similarity search features."
    )

# Basic config if no handlers are present, adjust as needed
if not logger.handlers:
    # Use simple basic config if colored logging isn't set up yet
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
# Global ThreadPoolExecutor for CPU-bound tasks like embedding generation
# Adjust max_workers based on your application's typical load and core count.
# Using os.cpu_count() can be a sensible default.
_THREAD_POOL_EXECUTOR = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)

# --- Resource Manager Integration ---
# Constants for embedding model
EMBEDDING_MODEL_ID = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = None  # Will be set when model is first loaded


def _ensure_embedding_model():
    """Get embedding model via ResourceManager (backwards compatibility)."""
    global EMBEDDING_DIM, embedding_model

    resource_manager = get_resource_manager()
    model, _ = resource_manager.get_model(EMBEDDING_MODEL_ID)

    # Set embedding dimension if not set
    if EMBEDDING_DIM is None and hasattr(model, "get_sentence_embedding_dimension"):
        EMBEDDING_DIM = model.get_sentence_embedding_dimension()
        logger.debug(f"📐 Set EMBEDDING_DIM to {EMBEDDING_DIM}")

    # Set global for backwards compatibility
    embedding_model = model

    return model


def is_embedding_model_available() -> bool:
    """Check if embedding model is available via ResourceManager."""
    try:
        resource_manager = get_resource_manager()
        # Try to get the model - if it's available, this will succeed
        model, _ = resource_manager.get_model(EMBEDDING_MODEL_ID)
        return model is not None
    except Exception as e:
        logger.debug(f"Embedding model not available: {e}")
        return False


# Backwards compatibility: create a simple embedding_model attribute
embedding_model = None  # Will be populated by _ensure_embedding_model when needed
# ----------------------------------
# --- Configuration ---
# Redis configuration with environment variable support
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)  # Redis authentication
REDIS_SENTINEL_ENABLED = os.getenv("REDIS_SENTINEL_ENABLED", "false").lower() == "true"
# --- Memory Keys Structure ---
# Core Memory Keys
VITAL_MEMORY_KEY = "vital_memories"  # SET (Could be deprecated if B handles all)
CONVERSATION_HISTORY_KEY_PREFIX = "conversation:"
MAX_NON_VITAL_HISTORY = 30  # Increased from 20 to 30 to ensure we can store more history
# JSON Keys for Memory B (Replacing Hash/Sorted Set)
MEMORY_B_PREFIX = "memory_b:"  # Prefix for JSON documents
# Vector Index Names
VECTOR_INDEX_NAME = "idx:memory_vectors_json"  # New index name for JSON
VECTOR_FIELD_NAME = "embedding"
MEMORY_TEXT_FIELD = "text"
IMPORTANCE_FIELD = "importance"  # Field name for importance score in JSON
TIMESTAMP_FIELD = "timestamp"  # Field name for creation timestamp in JSON
USER_ID_FIELD = "user_id"  # Field name for user identifier in JSON
# Memory optimization settings
VECTOR_DISTANCE_METRIC = "COSINE"
VECTOR_TYPE = "FLOAT32"


# --- Helper Function to Ensure Redis Persistence ---
def _ensure_redis_persistence(client):
    """Configures Redis for data persistence to prevent data loss on restart."""
    if not client:
        logger.error("Cannot configure Redis persistence: Client is None")
        return False
    try:
        # Get current persistence configuration
        current_save = client.config_get("save")
        logger.info(f"Current Redis save configuration: {current_save}")

        # Enable AOF (Append Only File) for better durability
        try:
            current_aof = client.config_get("appendonly")
            if current_aof.get("appendonly", "no") == "no":
                client.config_set("appendonly", "yes")
                logger.info("✅ Enabled Redis AOF persistence for exceptional memory durability")
            else:
                logger.info("✅ Redis AOF persistence already enabled")
        except redis.exceptions.ResponseError as e:
            logger.warning(f"Could not enable AOF persistence: {e}")

        # Only try to configure RDB snapshots if the current save setting is empty
        # This avoids errors on Redis instances with protected configs
        if current_save.get("save", "") == "":
            try:
                # Configure RDB snapshots (save after 60 seconds if at least 1 key changed)
                client.config_set("save", "60 1")
                logger.info("Successfully configured Redis save settings")
            except redis.exceptions.ResponseError as e:
                logger.warning(f"Could not set Redis save configuration: {e}")
                logger.warning("Redis may be in protected mode. Will use existing configuration.")
        # Don't try to change the directory, as this often fails in protected Redis instances
        # Instead, just trigger a save with the current configuration
        # Trigger an immediate save
        try:
            save_result = client.save()
            if save_result:
                logger.info("Redis data saved successfully.")
                return True
            else:
                logger.warning(
                    "Redis save command returned False. Data may not persist between restarts."
                )
                return False
        except redis.exceptions.ResponseError as e:
            logger.error(f"Redis error during save operation: {e}")
            logger.warning(
                "Redis persistence may not be enabled. Data might not be saved between restarts."
            )
            return False
    except redis.exceptions.ResponseError as e:
        logger.error(f"Redis error configuring persistence: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error configuring Redis persistence: {e}")
        return False


# --- Helper Function for JSON Index Creation ---
async def _create_json_vector_index(client):
    """Attempts to create the Redis vector search index ON JSON if it doesn't exist."""
    if not client:
        logger.error("Cannot create Redis index: Client is None")
        return False
    if not NUMPY_AVAILABLE:
        logger.error("Cannot create Redis index: NumPy is not available")
        return False
    # Ensure embedding model is loaded to get EMBEDDING_DIM
    model = _ensure_embedding_model()
    if not model or EMBEDDING_DIM is None:
        logger.error(
            "Cannot create Redis index: EMBEDDING_DIM is None (embedding model not loaded)"
        )
        return False
    try:
        # Define schema using JSONPath
        schema = (
            # TextField for better exact phrase matching on memory content
            TextField(
                f"$.{MEMORY_TEXT_FIELD}", as_name=MEMORY_TEXT_FIELD, no_stem=True, sortable=False
            ),
            VectorField(
                f"$.{VECTOR_FIELD_NAME}",
                "FLAT",
                {
                    "TYPE": VECTOR_TYPE,
                    "DIM": EMBEDDING_DIM,
                    "DISTANCE_METRIC": VECTOR_DISTANCE_METRIC,
                },
                as_name=VECTOR_FIELD_NAME,
            ),
            NumericField(
                f"$.{IMPORTANCE_FIELD}", as_name=IMPORTANCE_FIELD
            ),  # Index importance score
        )
        # Define index ON JSON for keys with the new prefix
        definition = IndexDefinition(prefix=[MEMORY_B_PREFIX], index_type=IndexType.JSON)
        logger.info(f"Attempting to create/verify Redis JSON index: '{VECTOR_INDEX_NAME}'...")
        # Use the programmatic approach
        await client.ft(VECTOR_INDEX_NAME).create_index(fields=schema, definition=definition)
        logger.info(f"Successfully created or verified JSON vector index '{VECTOR_INDEX_NAME}'.")
        return True
    except redis.exceptions.ResponseError as idx_err:
        if "Index already exists" in str(idx_err):
            logger.info(f"JSON Vector index '{VECTOR_INDEX_NAME}' already exists.")
            return True
        else:
            logger.error(f"ERROR creating JSON vector index '{VECTOR_INDEX_NAME}': {idx_err}")
            return False
    except Exception as e:
        logger.error(
            f"Unexpected error during JSON index creation/verification for "
            f"'{VECTOR_INDEX_NAME}': {e}"
        )
        return False


# --- Redis Connection Pool Configuration ---
def initialize_redis_connection():
    """Initialize Redis connection with high availability support."""
    if REDIS_SENTINEL_ENABLED:
        # Use Sentinel for high availability
        try:
            import asyncio

            from redis_sentinel import initialize_sentinel

            logger.info("🔄 Initializing Redis Sentinel for high availability...")

            # Initialize Sentinel in a new event loop if needed
            try:
                sentinel_manager = asyncio.get_event_loop().run_until_complete(
                    initialize_sentinel()
                )
                client = sentinel_manager.get_master_client()
                logger.info("✅ Redis Sentinel connection established with automatic failover")
                return client
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                sentinel_manager = loop.run_until_complete(initialize_sentinel())
                client = sentinel_manager.get_master_client()
                logger.info("✅ Redis Sentinel connection established with automatic failover")
                return client

        except ImportError:
            logger.error("redis_sentinel module not available, falling back to direct connection")
        except Exception as e:
            logger.error(
                f"Failed to initialize Redis Sentinel: {e}, falling back to direct connection"
            )

    # Direct Redis connection with connection pooling
    connection_pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,  # Redis authentication
        decode_responses=True,
        socket_timeout=5,
        socket_keepalive=True,
        retry_on_timeout=True,
        max_connections=20,  # Connection pool size
        socket_connect_timeout=5,
        health_check_interval=30,  # Health check every 30 seconds
    )

    # Configure base Redis connection with connection pool
    base_redis_client = redis.Redis(connection_pool=connection_pool)

    # Verify connection
    try:
        base_redis_client.ping()
    except Exception as e:
        logger.error(f"Failed to ping Redis: {e}")
        raise

    # Create resilient client with fallback if available
    if RESILIENT_CLIENT_AVAILABLE:
        client = ResilientRedisClient(base_redis_client)
        logger.info(
            f"✅ Resilient Redis connection pool established at {REDIS_HOST}:{REDIS_PORT} with local fallback"
        )
    else:
        client = base_redis_client
        logger.info(
            f"✅ Redis connection pool established at {REDIS_HOST}:{REDIS_PORT} (pool size: 20)"
        )
        logger.warning(
            "🔥 Local cache fallback not available - Redis failures will cause service disruption"
        )

    # Configure Redis persistence
    config_client = base_redis_client if RESILIENT_CLIENT_AVAILABLE else client
    configure_redis_persistence(config_client)

    return client


async def initialize_redis_connection_async():
    """Initialize async Redis connection with resilience features.
    This is the preferred method for new code.
    """
    logger.info("🚀 Initializing async Redis connection...")

    # TODO: Add Sentinel support for async client if needed

    try:
        client = await initialize_async_redis_connection(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            max_retries=3,
            decode_responses=True,
        )

        # Configure persistence (async version)
        await configure_redis_persistence_async(client)

        logger.info("✅ Async Redis client initialized with built-in resilience")
        return client

    except Exception as e:
        logger.error(f"❌ Failed to initialize async Redis connection: {e}")
        raise


class RedisModuleError(Exception):
    """Exception raised when required Redis modules are not available."""

    pass


async def validate_redis_modules(redis_client, required_modules: dict):
    """Checks if required Redis modules are loaded.

    Args:
        redis_client: Async Redis client instance
        required_modules: Dict mapping module keys to human-readable names
                         e.g., {"search": "RediSearch", "json": "RedisJSON"}

    Raises:
        RedisModuleError: If any required module is missing
    """
    if not redis_client:
        raise RedisModuleError("Redis client is not connected.")

    try:
        # Get loaded modules using MODULE LIST command
        loaded_modules = await redis_client.module_list()

        # Extract module names - handle both dict and list formats
        loaded_module_names = set()
        for module in loaded_modules:
            if isinstance(module, dict):
                # Format: {'name': 'search', 'ver': 20807, ...}
                name = module.get("name", "")
            elif isinstance(module, list) and len(module) >= 2:
                # Format: [b'name', b'search', b'ver', 20807, ...]
                name = module[1].decode("utf-8") if isinstance(module[1], bytes) else str(module[1])
            else:
                continue
            loaded_module_names.add(name.lower())

        logger.info(f"🔍 Redis modules detected: {sorted(loaded_module_names)}")

        # Check each required module
        missing_modules = []
        for module_key, module_name in required_modules.items():
            # Check both the key and alternative names for the module
            module_aliases = {
                "json": ["json", "rejson"],
                "search": ["search", "redisearch"],
                "timeseries": ["timeseries", "redis-timeseries"],
                "bf": ["bf", "redisbloom"],
            }

            aliases_to_check = module_aliases.get(module_key.lower(), [module_key.lower()])
            found = any(alias in loaded_module_names for alias in aliases_to_check)

            if not found:
                missing_modules.append(module_name)

        if missing_modules:
            raise RedisModuleError(
                f"Missing required Redis modules: {', '.join(missing_modules)}. "
                f"Install with: redis-server --loadmodule RedisJSON.so --loadmodule redisearch.so"
            )

    except RedisModuleError:
        raise  # Re-raise our custom exception
    except Exception as e:
        # Handle potential command errors if 'MODULE LIST' isn't supported
        raise RedisModuleError(f"Failed to validate Redis modules: {e}")


async def configure_redis_persistence_async(client):
    """Configure Redis persistence settings (async version)."""
    try:
        # Check persistence config
        save_config = await client.config_get("save")
        if save_config.get("save", "") == "":
            logger.warning("Redis appears to be running without persistence enabled!")
        else:
            logger.info(f"Redis persistence enabled: {save_config['save']}")

        # Trigger background save
        await client.bgsave()
        logger.info("Triggered background save operation")

    except redis.exceptions.ResponseError as e:
        logger.warning(f"Could not configure Redis persistence: {e}")
    except Exception as e:
        logger.error(f"Error checking Redis persistence: {e}")


def configure_redis_persistence(config_client):
    """Configure Redis persistence settings."""
    try:
        # Check if Redis is configured for RDB persistence
        save_config = config_client.config_get("save")
        if save_config.get("save", "") == "":
            logger.warning("Redis appears to be running without persistence enabled!")
            logger.warning("Chat history may be lost when Redis restarts.")
        else:
            logger.info(
                f"Redis persistence appears to be enabled with config: {save_config['save']}"
            )
    except redis.exceptions.ResponseError:
        logger.warning(
            "Could not check Redis persistence configuration - Redis may be in protected mode."
        )
        logger.warning(
            "If Redis is not configured for persistence, chat history may be lost on restart."
        )

    # Try to configure Redis for persistence
    persistence_configured = _ensure_redis_persistence(config_client)
    if persistence_configured:
        logger.info("Redis persistence has been configured successfully.")
    else:
        logger.warning(
            "Redis persistence configuration failed. Chat history may not be saved between restarts."
        )
        logger.info(
            "To ensure chat history persistence, edit your redis.conf file to enable RDB snapshots."
        )
        logger.info("Example config: save 60 1 (save after 60 seconds if at least 1 key changed)")
        logger.info("Then restart your Redis server with that configuration.")

    # Try to trigger an immediate save regardless of configuration
    try:
        config_client.bgsave()
        logger.info("Triggered a background save operation.")
    except redis.exceptions.ResponseError as e:
        if "Background save already in progress" in str(e):
            logger.info("Background save already in progress.")
        else:
            logger.warning(f"Could not trigger background save: {e}")
            logger.warning("This may indicate Redis is not configured for persistence.")
    except Exception as e:
        logger.error(f"Error triggering background save: {e}")
        logger.warning("This may indicate Redis is not configured for persistence.")

    # Try to detect if Redis Stack is available (but don't create index yet)
    try:
        # This is a synchronous redis client, not async
        config_client.execute_command("COMMAND INFO", "FT.CREATE")
        logger.info("Redis Stack FT.CREATE command detected.")
        logger.info(
            "Vector index creation will be deferred until first use to avoid blocking startup."
        )
    except redis.exceptions.ResponseError:
        logger.warning(
            "Redis Stack vector search capabilities (FT.CREATE command) not detected. "
            "Vector search will use fallback."
        )


# Redis client initialized lazily to prevent blocking imports
redis_client = None


# --- Helper Functions ---
async def _perform_history_save_async(
    user_id: str,
    user_prompt: str,
    model_response: str,
    redis_client_instance,
    history_key_prefix: str,
    history_limit: int,
):
    """Asynchronously generates embeddings and saves conversation history to Redis."""
    if not redis_client_instance:
        logger.error(
            "_perform_history_save_async: Redis client instance is not available. Cannot save history."
        )
        return
    try:
        loop = asyncio.get_running_loop()
        logger.debug(f"Async history: Generating user prompt embedding for user {user_id}")
        # Use ResourceManager-powered generate_embedding (no model instance needed)
        user_embedding = await loop.run_in_executor(
            _THREAD_POOL_EXECUTOR, generate_embedding, user_prompt
        )
        logger.debug(f"Async history: Generating model response embedding for user {user_id}")
        model_embedding = await loop.run_in_executor(
            _THREAD_POOL_EXECUTOR, generate_embedding, model_response
        )
        # Get current timestamp in local time zone for consistency, as in original function
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            "user": user_prompt,
            "model": model_response,
            "timestamp": timestamp_str,
            "user_embedding": user_embedding,  # Ensure key matches original if critical
            "model_embedding": model_embedding,  # Ensure key matches original
        }
        redis_key = f"{history_key_prefix}{user_id}"
        serialized_entry = json.dumps(history_entry)
        logger.debug(f"Async history: Pushing to Redis for user {user_id} at key {redis_key}")
        await redis_client_instance.lpush(redis_key, serialized_entry)
        logger.debug(f"Async history: Trimming Redis list for user {user_id} at key {redis_key}")
        await redis_client_instance.ltrim(redis_key, 0, history_limit - 1)
        logger.info(
            f"Asynchronously added conversation turn to history for user_id: {user_id}. "
            f"Embeddings {'generated' if user_embedding else 'not generated'}."
        )
    except Exception as e:
        logger.error(f"Error in _perform_history_save_async for user {user_id}: {e}", exc_info=True)


def add_to_conversation_history(
    user_id: str,
    user_prompt: str,
    model_response: str,
    redis_client_instance,
    history_key_prefix: str,
    history_limit: int,
):
    """Non-blockingly adds a new turn (text and embedding) to the conversation history for the given user_id
    by scheduling an asynchronous task. Uses ResourceManager for embedding generation.

    Args:
        user_id: The ID of the user.
        user_prompt: The user's message.
        model_response: The model's response.
        redis_client_instance: Instance of the Redis client.
        history_key_prefix: Prefix for the Redis history key (e.g., CONVERSATION_HISTORY_KEY_PREFIX).
        history_limit: Max number of history entries (e.g., MAX_NON_VITAL_HISTORY).
    """
    if not redis_client_instance:
        logger.warning(
            "add_to_conversation_history: Redis client not provided. History will not be saved."
        )
        return

    # Note: Embedding model is now handled internally by ResourceManager
    try:
        # Check if we're in an async context and can create tasks
        try:
            asyncio.get_running_loop()
            # We have a running loop, can create task
            asyncio.create_task(
                _perform_history_save_async(
                    user_id,
                    user_prompt,
                    model_response,
                    redis_client_instance,
                    history_key_prefix,
                    history_limit,
                )
            )
            logger.info(f"Scheduled background task for saving history for user_id: {user_id}")
        except RuntimeError:
            # No running loop - this should not happen in normal operation
            logger.warning(f"No event loop running, cannot save history for user_id: {user_id}")
            logger.debug("This indicates the function was called outside the main async context")
    except Exception as e:
        logger.error(
            f"Error in add_to_conversation_history for user {user_id}: {e}. History save failed.",
            exc_info=True,
        )
        # OPTIONAL: Synchronous fallback if critical and no loop is available, though this defeats
        # the purpose.


def generate_embedding(text: str, model_instance=None) -> list[float] | None:
    """Generates an embedding vector for the given text using ResourceManager."""
    if not text or not isinstance(text, str):
        logger.warning("Invalid input for embedding generation.")
        return None

    try:
        # Use ResourceManager for thread-safe, cached model access
        resource_manager = get_resource_manager()

        def embedding_task(model):
            embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return embedding.tolist()  # Convert numpy array to list for JSON serialization

        return resource_manager.run_inference(EMBEDDING_MODEL_ID, embedding_task)

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


async def get_conversation_history(
    user_id: str, redis_client_instance, history_key_prefix: str, history_limit: int
) -> list[dict[str, str]]:
    """Retrieves the non-vital conversation history (text only) for a user."""
    if not redis_client_instance:
        logger.warning("Redis client not available in get_conversation_history")
        return []
    key = f"{history_key_prefix}{user_id}"
    try:
        # Retrieve only the last MAX_NON_VITAL_HISTORY items (index 0 is newest)
        history_json = await redis_client_instance.lrange(key, 0, history_limit - 1)
        # Ensure backwards compatibility: only return items that are valid JSON dicts
        history = []
        for item in history_json:
            try:
                data = json.loads(item)
                if isinstance(data, dict):
                    # We only need text fields here, remove embedding if present for this func
                    data.pop("user_embedding", None)
                    data.pop("model_embedding", None)
                    history.append(data)
            except json.JSONDecodeError:
                logger.warning(f"Skipping corrupted history item: {item[:50]}...")
        return history
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error in get_conversation_history: {e}")
        return []


# --- Similarity Search (Now on JSON Index) ---
async def find_similar_vital_memories(
    query_text: str, top_n: int = 5, min_similarity: float = 0.5, redis_client_instance=None
) -> list[tuple[str, float, float | None]]:
    """Finds vital memories (Type B - JSON) semantically similar to the query text.
    Uses efficient Redis Stack JSON vector search (FT.SEARCH KNN) if available.
    Returns a list of tuples: (memory_text, similarity_score, importance_score)
    """
    # Use the provided redis_client_instance parameter
    client = redis_client_instance if redis_client_instance is not None else redis_client
    # Log which client we're using
    if client is redis_client:
        logger.debug("Using global redis_client for similarity search")
    else:
        logger.debug("Using provided redis_client_instance for similarity search")
    # Check each component separately for better error messages
    if not client:
        logger.warning("Redis client is missing for similarity search.")
        return []

    # Get embedding model via ResourceManager
    try:
        resource_manager = get_resource_manager()
        model, _ = resource_manager.get_model(EMBEDDING_MODEL_ID)
    except Exception as e:
        logger.warning(f"Failed to get embedding model for similarity search: {e}")
        return []
    if not NUMPY_AVAILABLE:
        logger.warning("NumPy is missing for similarity search.")
        return []
    # Generate the embedding for the input query text
    query_embedding = generate_embedding(query_text, model)
    if query_embedding is None:
        return []  # Added return if embedding fails
    query_vector_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    # Check if JSON index exists
    try:
        client.ft(VECTOR_INDEX_NAME).info()  # Use client for info
        logger.info(f"Redis Stack JSON vector search index '{VECTOR_INDEX_NAME}' found.")
    except redis.exceptions.ResponseError as e:
        if "Unknown index name" in str(e):
            logger.warning(f"Index '{VECTOR_INDEX_NAME}' not found. Attempting to create it now...")
            index_created = await _create_json_vector_index(client)
            if not index_created:
                logger.error(
                    f"Failed to create index '{VECTOR_INDEX_NAME}' on the fly. Cannot perform vector search."
                )
                return []
            else:
                logger.info(
                    f"Successfully created index '{VECTOR_INDEX_NAME}' on the fly. Proceeding with search."
                )
                # Index created, proceed with the search logic below
        else:
            # Different ResponseError occurred
            logger.error(
                f"Redis ResponseError checking index '{VECTOR_INDEX_NAME}': {e}. "
                f"Cannot perform vector search."
            )
            return []
    except Exception as e:
        logger.error(
            f"Error checking JSON index '{VECTOR_INDEX_NAME}': {e}. Cannot perform vector search."
        )
        return []
    # --- Execute FT.SEARCH ON JSON ---
    try:
        knn_query = (
            Query(f"(*)=>[KNN {top_n} @{VECTOR_FIELD_NAME} $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields(
                "id", "vector_score", f"$.{MEMORY_TEXT_FIELD}", f"$.{IMPORTANCE_FIELD}"
            )  # Return ID, score, text, importance
            .dialect(2)
        )
        query_params = {"query_vector": query_vector_bytes}
        logger.debug(f"Executing Redis FT.SEARCH JSON KNN query: {knn_query.query_string()}")
        # Use the passed client for search command
        results = client.ft(VECTOR_INDEX_NAME).search(knn_query, query_params)
        logger.debug(f"FT.SEARCH JSON returned {results.total} potential matches.")
        similarities = []
        for doc in results.docs:
            memory_key = doc.id  # Define key early in the loop
            distance = float(doc.vector_score)
            similarity = 1.0 - distance
            if similarity >= min_similarity:
                # Access other fields
                memory_text = getattr(doc, f"$.{MEMORY_TEXT_FIELD}", None)
                importance = getattr(doc, f"$.{IMPORTANCE_FIELD}", None)
                if memory_text and memory_key:  # Check key existence here
                    # Return key, text, similarity, and importance (or None)
                    importance_float = float(importance) if importance is not None else None
                    similarities.append((memory_key, memory_text, similarity, importance_float))
                    logger.debug(
                        f"  - Match: Key: {memory_key}, Text: '{memory_text[:50]}...' "
                        f"(Similarity: {similarity:.4f}, Importance: {importance_float})"
                    )
                else:
                    logger.warning(
                        f"Could not retrieve key or memory text for matched document ID {doc.id}"
                    )
        return similarities
    except Exception as e:
        logger.error(f"Error during FT.SEARCH JSON execution: {e}")
        return []


# --- Vital Memory B Functions (Now using JSON) ---
async def add_vital_memory_b(memory_text: str, importance: float, redis_client_instance=None):
    """Adds or updates a vital memory B as a JSON document."""
    # Use the provided redis_client_instance parameter
    client = redis_client_instance if redis_client_instance is not None else redis_client
    # Log which client we're using
    if client is redis_client:
        logger.debug("Using global redis_client for adding vital memory")
    else:
        logger.debug("Using provided redis_client_instance for adding vital memory")
    if not client:
        logger.error("No Redis client available for adding vital memory")
        return
    # Clean the input text
    cleaned_memory = memory_text.strip()
    if not cleaned_memory:
        logger.warning("Skipping empty vital memory B add/update.")
        return
    # Clamp importance score
    if not (0.0 <= importance <= 1.0):
        logger.warning(
            f"Importance score {importance} is outside the valid range [0.0, 1.0]. Clamping."
        )
        importance = max(0.0, min(1.0, importance))
    embedding = generate_embedding(cleaned_memory)
    if not embedding:
        logger.warning(
            f"Skipping memory save for '{cleaned_memory[:50]}...' due to embedding failure."
        )
        return
    # Create a clean key based on category prefix if available
    # Handle format "Category: Value" specially
    if ":" in cleaned_memory:
        # Extract category and create a concise key
        category, value = cleaned_memory.split(":", 1)
        category = category.strip().lower().replace(" ", "_")
        # Add hash for uniqueness (8 chars)
        value_hash = hashlib.md5(value.strip().encode()).hexdigest()[:8]
        memory_key = f"{MEMORY_B_PREFIX}{category}_{value_hash}"
    else:
        # Fall back to old method for non-categorized memories
        memory_key = f"{MEMORY_B_PREFIX}{cleaned_memory.replace(' ', '_').lower()[:50]}"
    memory_doc = {
        MEMORY_TEXT_FIELD: cleaned_memory,
        VECTOR_FIELD_NAME: embedding,
        IMPORTANCE_FIELD: importance,
    }
    try:
        # Store using the client we determined earlier
        await client.json().set(memory_key, ".", memory_doc)
        logger.info(
            f"Successfully added/updated vital memory B JSON (Key: {memory_key}): '{cleaned_memory[:50]}...'"
        )
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error in add_vital_memory_b (JSON): {e}")
    except TypeError as e:
        logger.error(f"Type error (likely JSON serialization) in add_vital_memory_b (JSON): {e}")


async def get_vital_memories_b(threshold: float, redis_client_instance=None) -> list[str]:
    """Retrieves vital memories TEXT from JSON docs with importance >= threshold."""
    # This now requires querying the JSON index, not ZRANGEBYSCORE
    if not redis_client_instance:
        return []
    try:
        # Query the index for documents meeting the importance threshold
        # Note: Requires the index 'idx:memory_vectors_json' to exist and include importance
        query_str = f"@{IMPORTANCE_FIELD}:[{threshold} +inf]"
        query = Query(query_str).return_fields(f"$.{MEMORY_TEXT_FIELD}")  # Only return text
        results = await redis_client_instance.ft(VECTOR_INDEX_NAME).search(query)
        # Extract text from results
        members = [
            getattr(doc, MEMORY_TEXT_FIELD)
            for doc in results.docs
            if hasattr(doc, MEMORY_TEXT_FIELD)
        ]
        return members
    except redis.exceptions.ResponseError as e:
        # Handle case where index might not exist yet
        if "Unknown Index name" in str(e):
            logger.warning(
                f"Index {VECTOR_INDEX_NAME} not found in get_vital_memories_b. Returning empty list."
            )
            return []
        logger.error(f"Redis error in get_vital_memories_b (JSON Query): {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_vital_memories_b (JSON Query): {e}")
        return []


# --- Simplified Deletion Function (Uses DEL command by Key) ---
async def delete_vital_memory_b(memory_key: str, redis_client_instance=None):
    """Deletes a vital memory B JSON document using its exact key."""
    if not redis_client_instance:
        logger.warning("Redis client not available for deletion.")
        return 0  # Indicate no deletion occurred
    if (
        not memory_key
        or not isinstance(memory_key, str)
        or not memory_key.startswith(MEMORY_B_PREFIX)
    ):
        logger.warning(f"Invalid or potentially unsafe key provided for deletion: '{memory_key}'")
        return 0  # Indicate no deletion occurred
    logger.debug(f"Attempting to delete memory with key: '{memory_key}'")
    deleted_count = 0
    try:
        # Use the direct DEL command
        deleted_count = await redis_client_instance.delete(memory_key)
        if deleted_count > 0:
            logger.info(f"Successfully deleted memory with key '{memory_key}'.")
        else:
            logger.warning(f"Key '{memory_key}' not found or already deleted.")
        return deleted_count
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error during DEL operation for key '{memory_key}': {e}")
        return 0  # Indicate deletion failed
    except Exception as e:
        logger.error(f"Unexpected error in delete_vital_memory_b (using DEL): {e}")
        return 0  # Indicate deletion failed


# --- End Simplified Deletion ---
def is_redis_available(redis_client_instance) -> bool:
    """Checks if the Redis client connected successfully."""
    # Check only main client
    return redis_client_instance is not None


# --- Function to escape query characters ---
def escape_query_chars(text: str) -> str:
    """Escape special characters in Redis FT.SEARCH queries."""
    # Characters that need escaping in Redis FT.SEARCH queries
    special_chars = ["@", "(", ")", "[", "]", "{", "}", "|", "&", "*", "?", "~", '"', "'", "-", "+"]
    escaped = text
    for char in special_chars:
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


# --- Function to find memories containing specific terms ---
async def find_memories_by_term(term: str, redis_client_instance=None) -> list[tuple[str, str]]:
    """Finds all memories (key, text) where the text field contains a specific term."""
    if not redis_client_instance or not term:
        return []
    memories = []
    try:
        # Basic escaping for the search term itself
        # Escape punctuation that might interfere with query structure
        escaped_term = term.replace("-", "\\-").replace("(", "\\(").replace(")", "\\)")
        # Ensure common separators aren't escaped if they are part of the term search itself
        # This simple search relies on TextField tokenization
        # Query for the term within the text field
        term_query_str = f"@{MEMORY_TEXT_FIELD}:({escaped_term})"
        term_query = Query(term_query_str).return_fields("id", f"$.{MEMORY_TEXT_FIELD}")
        logger.debug(f"Executing broad term search query: {term_query_str}")
        results = await redis_client_instance.ft(VECTOR_INDEX_NAME).search(term_query)
        logger.debug(f"Broad term search for '{term}' found {results.total} potential matches.")
        for doc in results.docs:
            key = doc.id
            text = getattr(doc, f"$.{MEMORY_TEXT_FIELD}", None)
            if key and text:
                memories.append((key, text))
    except redis.exceptions.ResponseError as e:
        logger.error(f"Redis error during broad term search for '{term}': {e}")
        # Optionally fallback to SCAN here if FT.SEARCH fails unexpectedly
    except Exception as e:
        logger.error(f"Unexpected error during broad term search for '{term}': {e}")
    return memories


async def clear_all_redis_memory_data(redis_client_instance=None) -> int:
    """Clear ALL memory-related data from Redis.
    Returns the total number of keys deleted.
    """
    if not redis_client_instance:
        return 0

    total_deleted = 0

    try:
        # Helper function to clear keys by pattern
        async def clear_by_pattern(pattern_name, pattern):
            cursor = "0"  # Redis SCAN uses string cursors
            deleted_count = 0
            while True:
                cursor, keys = await redis_client_instance.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted = await redis_client_instance.delete(*keys)
                    deleted_count += deleted
                    logger.debug(f"Deleted {deleted} {pattern_name} keys (pattern: {pattern})")
                if cursor == "0":  # cursor returns to "0" when scan is complete
                    break
            return deleted_count

        # 1. Clear all memory_b keys
        deleted = await clear_by_pattern("memory_b", f"{MEMORY_B_PREFIX}*")
        total_deleted += deleted

        # 2. Clear all conversation history keys
        deleted = await clear_by_pattern(
            "conversation history", f"{CONVERSATION_HISTORY_KEY_PREFIX}*"
        )
        total_deleted += deleted

        # 3. Clear legacy vital memories key if it exists
        if await redis_client_instance.exists(VITAL_MEMORY_KEY):
            await redis_client_instance.delete(VITAL_MEMORY_KEY)
            total_deleted += 1
            logger.debug("Deleted legacy vital_memories key")

        # 4. Clear message-related keys (msgs:*, conv:*, msg:*, user_convs:*)
        message_patterns = ["msgs:*", "conv:*", "msg:*", "user_convs:*"]
        for pattern in message_patterns:
            deleted = await clear_by_pattern(f"message ({pattern})", pattern)
            total_deleted += deleted

        # 5. Clear neural memory keys
        deleted = await clear_by_pattern("neural memory", "memory_b:neural_mem_*")
        total_deleted += deleted

        # 6. Clear any other memory-related keys (be thorough)
        other_patterns = ["memory:*", "user_memory:*", "session_memory:*", "essential_memory:*"]
        for pattern in other_patterns:
            deleted = await clear_by_pattern(f"other memory ({pattern})", pattern)
            total_deleted += deleted

        # 7. Clear Redis indexes and any index-related keys
        index_patterns = ["idx:*", "index:*", "*_idx", "*_index"]
        for pattern in index_patterns:
            deleted = await clear_by_pattern(f"index ({pattern})", pattern)
            total_deleted += deleted

        # 8. Clear any embedding or vector keys
        vector_patterns = ["embedding:*", "vector:*", "emb:*", "*_emb", "*_vector", "*_embedding"]
        for pattern in vector_patterns:
            deleted = await clear_by_pattern(f"vector ({pattern})", pattern)
            total_deleted += deleted

        # 9. NUCLEAR OPTION: Clear ALL keys that could possibly be memory-related
        # This catches any patterns we might have missed
        all_possible_patterns = [
            "*conversation*",
            "*memory*",
            "*msg*",
            "*conv*",
            "*session*",
            "*user*",
            "*vital*",
            "*essential*",
            "*neural*",
            "*brain*",
            "*history*",
        ]
        for pattern in all_possible_patterns:
            deleted = await clear_by_pattern(f"comprehensive ({pattern})", pattern)
            total_deleted += deleted

        # 10. FINAL CHECK: Get all remaining keys and check for any memory-related ones
        try:
            all_keys = []
            cursor = 0
            while True:
                cursor, keys = await redis_client_instance.scan(cursor, count=1000)
                all_keys.extend(keys)
                if cursor == 0:
                    break

            # Filter for any remaining memory-related keys
            memory_keywords = [
                "memory",
                "neural",
                "conversation",
                "history",
                "session",
                "user",
                "vital",
                "msg",
                "conv",
            ]
            remaining_memory_keys = []
            for key in all_keys:
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in memory_keywords):
                    remaining_memory_keys.append(key)

            if remaining_memory_keys:
                logger.warning(
                    f"Found {len(remaining_memory_keys)} additional memory-related keys to delete"
                )
                if remaining_memory_keys:
                    deleted = await redis_client_instance.delete(*remaining_memory_keys)
                    total_deleted += deleted
                    logger.info(f"Deleted {deleted} additional memory keys")
        except Exception as e:
            logger.warning(f"Error in final memory key cleanup: {e}")

        logger.info(f"🧹 Total Redis memory keys deleted: {total_deleted}")
        return total_deleted

    except Exception as e:
        logger.error(f"Error clearing all Redis memory data: {e}")
        return total_deleted
