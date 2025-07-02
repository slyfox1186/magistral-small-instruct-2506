# Critical Performance Fixes - Implementation Guide

## 1. Fix Streaming Memory Accumulation

### Problem
The `accumulatedResponseRef.current` grows unbounded during streaming, causing memory to increase with each token.

### Solution: Implement Sliding Window

```python
# backend/main.py - Add these constants
STREAMING_BUFFER_SIZE = 4096  # Max chars to keep in buffer
STREAMING_CHUNK_SIZE = 512    # Chars to flush at once

# Modify the streaming handler
class StreamingBuffer:
    def __init__(self, max_size=STREAMING_BUFFER_SIZE):
        self.buffer = ""
        self.max_size = max_size
        self.total_sent = 0
    
    def add_token(self, token):
        self.buffer += token
        
        # Flush if buffer exceeds size
        if len(self.buffer) > self.max_size:
            chunk = self.buffer[:STREAMING_CHUNK_SIZE]
            self.buffer = self.buffer[STREAMING_CHUNK_SIZE:]
            self.total_sent += len(chunk)
            return chunk
        return None
```

### Frontend Fix

```typescript
// client/src/App.tsx - Replace accumulation with chunking
const MAX_DISPLAY_CHARS = 10000; // Only keep last 10k chars in DOM

// In streaming loop, replace:
// accumulatedResponseRef.current += tokenText

// With:
if (accumulatedResponseRef.current.length > MAX_DISPLAY_CHARS) {
  // Keep only the last MAX_DISPLAY_CHARS
  accumulatedResponseRef.current = 
    accumulatedResponseRef.current.slice(-MAX_DISPLAY_CHARS) + tokenText;
} else {
  accumulatedResponseRef.current += tokenText;
}
```

## 2. Fix GPU Memory Management

### Problem
Model stays loaded in GPU memory even when idle.

### Solution: Implement Auto-Unloading

```python
# backend/persistent_llm_server.py
import asyncio
from datetime import datetime, timedelta

class PersistentLLMServer:
    def __init__(self, model_path: str, redis_url: str = "redis://localhost:6379"):
        # ... existing init ...
        self.last_request_time = datetime.now()
        self.model_unload_timeout = 300  # 5 minutes
        self.unload_task = None
    
    async def _schedule_unload(self):
        """Schedule model unloading after inactivity"""
        await asyncio.sleep(self.model_unload_timeout)
        
        # Check if still inactive
        if (datetime.now() - self.last_request_time).seconds > self.model_unload_timeout:
            await self._unload_model()
    
    async def _unload_model(self):
        """Unload model from GPU memory"""
        if self.model:
            logger.info("🔻 Unloading model from GPU after inactivity")
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    async def generate(self, prompt: str, **kwargs) -> str:
        self.last_request_time = datetime.now()
        
        # Cancel any pending unload
        if self.unload_task:
            self.unload_task.cancel()
        
        # Reload model if unloaded
        if self.model is None:
            await self._load_model()
        
        # ... existing generate logic ...
        
        # Schedule unload after request
        self.unload_task = asyncio.create_task(self._schedule_unload())
```

## 3. Fix Queue Memory Growth

### Problem
Background queues can grow indefinitely.

### Solution: Implement Bounded Queues

```python
# backend/main.py
from collections import deque

# Replace unbounded deques with size limits
MAX_QUEUE_SIZE = 1000
QUEUE_OVERFLOW_POLICY = "drop_oldest"  # or "reject_new"

class BoundedQueue:
    def __init__(self, maxsize=MAX_QUEUE_SIZE):
        self.queue = deque(maxlen=maxsize)
        self.dropped_count = 0
    
    def put(self, item):
        if len(self.queue) >= self.queue.maxlen:
            self.dropped_count += 1
            logger.warning(f"Queue overflow! Dropped {self.dropped_count} items")
        self.queue.append(item)
    
    def get(self):
        return self.queue.popleft() if self.queue else None
    
    def qsize(self):
        return len(self.queue)

# Replace global queues
memory_extraction_queue = BoundedQueue()
embedding_queue = BoundedQueue()
```

## 4. Fix Redis Connection Leaks

### Problem
Redis connections not properly cleaned up.

### Solution: Use Connection Context Manager

```python
# backend/redis_utils.py
from contextlib import asynccontextmanager
import asyncio

class RedisConnectionManager:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.pool = None
        self._lock = asyncio.Lock()
    
    async def get_pool(self):
        if self.pool is None:
            async with self._lock:
                if self.pool is None:
                    self.pool = await aioredis.create_redis_pool(
                        self.redis_url,
                        minsize=5,
                        maxsize=20,
                        timeout=10
                    )
        return self.pool
    
    @asynccontextmanager
    async def get_connection(self):
        pool = await self.get_pool()
        conn = await pool.acquire()
        try:
            yield conn
        finally:
            pool.release(conn)
    
    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

# Usage
redis_manager = RedisConnectionManager("redis://localhost")

async def safe_redis_operation():
    async with redis_manager.get_connection() as conn:
        # Use connection
        await conn.set("key", "value")
```

## 5. Fix Token Processing Performance

### Problem
Each token causes full UI re-render.

### Solution: Batch Updates

```typescript
// client/src/App.tsx
import { useRef, useCallback } from 'react';

// Token batching implementation
const useTokenBatcher = (onBatch: (tokens: string[]) => void, delay = 50) => {
  const batchRef = useRef<string[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  const addToken = useCallback((token: string) => {
    batchRef.current.push(token);
    
    // Clear existing timer
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    
    // Set new timer
    timerRef.current = setTimeout(() => {
      if (batchRef.current.length > 0) {
        onBatch(batchRef.current);
        batchRef.current = [];
      }
    }, delay);
  }, [onBatch, delay]);
  
  return addToken;
};

// In component
const handleTokenBatch = useCallback((tokens: string[]) => {
  const combined = tokens.join('');
  setCurrentResponseRaw(prev => prev + combined);
}, []);

const addToken = useTokenBatcher(handleTokenBatch);

// In streaming loop, replace direct update with:
addToken(tokenText);
```

## 6. Extract Magic Numbers

### Problem
Magic numbers throughout codebase.

### Solution: Configuration Constants

```python
# backend/config.py - Add these constants
class StreamingConfig:
    BUFFER_SIZE = 4096
    CHUNK_SIZE = 512
    TOKEN_BATCH_DELAY_MS = 50
    MAX_RESPONSE_LENGTH = 100000

class MemoryConfig:
    CONSOLIDATION_INTERVAL_SECONDS = 3600  # 1 hour
    MAX_QUEUE_SIZE = 1000
    EMBEDDING_BATCH_SIZE = 5
    GPU_IDLE_THRESHOLD_SECONDS = 2.0

class ModelConfig:
    UNLOAD_TIMEOUT_SECONDS = 300  # 5 minutes
    LOAD_TIMEOUT_SECONDS = 60
    MAX_CONTEXT_LENGTH = 32768
    DEFAULT_BATCH_SIZE = 2048

class NetworkConfig:
    REQUEST_TIMEOUT_SECONDS = 60.0
    RETRY_COUNT = 3
    RETRY_DELAY_SECONDS = 0.5
    CONNECTION_POOL_SIZE = 20
```

## 7. Monitoring Implementation

Add performance tracking:

```python
# backend/monitoring.py
import time
import psutil
import GPUtil
from prometheus_client import Gauge, Histogram

# Memory metrics
memory_usage_bytes = Gauge('app_memory_usage_bytes', 'Memory usage in bytes')
gpu_memory_usage_mb = Gauge('app_gpu_memory_usage_mb', 'GPU memory usage in MB')
queue_depth = Gauge('app_queue_depth', 'Queue depth by name', ['queue_name'])

# Performance metrics
response_generation_time = Histogram('app_response_generation_seconds', 'Response generation time')
token_generation_rate = Gauge('app_token_generation_rate', 'Tokens per second')

class PerformanceMonitor:
    @staticmethod
    def update_metrics():
        # Memory
        process = psutil.Process()
        memory_usage_bytes.set(process.memory_info().rss)
        
        # GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory_usage_mb.set(gpus[0].memoryUsed)
        except:
            pass
        
        # Queues
        queue_depth.labels(queue_name='memory_extraction').set(
            memory_extraction_queue.qsize()
        )
        queue_depth.labels(queue_name='embedding').set(
            embedding_queue.qsize()
        )
```

## Implementation Priority

1. **Day 1**: Fix memory accumulation in streaming (both frontend and backend)
2. **Day 2**: Implement GPU memory management and bounded queues
3. **Day 3**: Fix Redis connections and add monitoring
4. **Day 4**: Extract magic numbers and improve error handling
5. **Day 5**: Testing and validation

These fixes address the most critical performance issues and should significantly improve system stability and resource usage.