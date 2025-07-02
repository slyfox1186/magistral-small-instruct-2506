# Context Findings

Based on deep analysis of the codebase, here are the specific issues identified:

## Performance Bottlenecks & Memory Leaks (Priority 1)

### 1. Frontend Memory Accumulation (client/src/App.tsx)
- **Issue**: `accumulatedResponseRef.current` grows unbounded during streaming
- **Location**: Lines 95, 907, 910 - accumulates all tokens without cleanup
- **Impact**: Memory usage scales with response length, can cause browser crash
- **Pattern**: Using useRef to accumulate streaming data without size limits

### 2. Backend Queue Growth (backend/main.py)
- **Issue**: `memory_extraction_queue` and `embedding_queue` can grow infinitely
- **Location**: Line 420 - `embedding_queue = deque()` with no max size
- **Impact**: Server memory exhaustion under heavy load
- **Pattern**: Using collections.deque without maxlen parameter

### 3. GPU Memory Lock (backend/gpu_lock.py)
- **Issue**: Model stays loaded in GPU memory even when idle
- **Location**: No GPU memory release mechanism in GPULock class
- **Impact**: ~16GB VRAM permanently occupied
- **Pattern**: Missing context manager cleanup for GPU resources

### 4. Process Cleanup (start.py)
- **Issue**: Zombie processes from improper signal handling
- **Location**: Lines 300-347 - cleanup function doesn't wait for process termination
- **Impact**: Port conflicts, resource leaks
- **Pattern**: Using process.terminate() without proper wait

## Code Quality Issues (Priority 2)

### 1. Magic Numbers
- **backend/main.py**:
  - Line 100: `await asyncio.sleep(3600)` - hardcoded hour
  - Line 269: `"n_batch": 2048` - hardcoded batch size
  - Line 54: `self._max_hold_time = 90.0` - hardcoded timeout
- **client/src/App.tsx**:
  - Line 156: `setTimeout(() => setCopiedMessageId(null), 2000)` - hardcoded 2 seconds

### 2. Poor Error Handling
- **backend/main.py**:
  - Lines 105-111: Broad exception catching without proper logging
  - No retry logic for failed operations
- **backend/gpu_lock.py**:
  - Missing error handling in acquire/release methods

### 3. Commented-Out Code
- **backend/main.py**:
  - Lines 66: `# from memory.importance_calculator import NeuralImportanceCalculator  # Module not found`
  - Multiple blocks of commented code throughout

## Configuration Issues (Priority 3)

### 1. Hardcoded Values
- **backend/config.py**: Ports, model paths, API keys scattered
- **client/vite.config.ts**: Line 37: `port: 4000` hardcoded
- **start.py**: Lines 16-17: Default ports hardcoded

### 2. Missing Environment Validation
- No validation for required environment variables
- No default fallbacks for critical settings

### 3. Logging Configuration
- **backend/colored_logging.py**: No log rotation
- **start.py**: Log file grows unbounded

## Specific Files That Need Modification

### High Priority (Performance/Memory):
1. `client/src/App.tsx` - Implement streaming buffer with size limit
2. `backend/main.py` - Add queue size limits and monitoring
3. `backend/gpu_lock.py` - Add GPU memory release mechanism
4. `start.py` - Fix process cleanup and signal handling

### Medium Priority (Code Quality):
1. `backend/config.py` - Extract all magic numbers to constants
2. `backend/main.py` - Improve error handling and retry logic
3. `backend/memory/services/embedding_service.py` - Add queue monitoring

### Low Priority (Configuration):
1. Create `.env.example` with all required variables
2. Add environment validation on startup
3. Implement log rotation in logging setup

## Similar Features Analyzed

### Streaming Implementations:
- Current pattern uses Server-Sent Events (SSE)
- No backpressure handling or flow control
- Missing stream cleanup on client disconnect

### Memory Management Patterns:
- Redis-based storage with vector search
- No memory pressure monitoring
- Missing cleanup for old embeddings

### Process Management:
- Custom Python script instead of systemd/supervisor
- No health checks or automatic restarts
- Missing graceful shutdown handling

## Technical Constraints

1. **GPU Memory**: Limited to available VRAM (typically 16-24GB)
2. **Redis Memory**: No eviction policy configured
3. **Node Process Memory**: Default heap size may be insufficient
4. **Python GIL**: Blocking operations affect async performance

## Integration Points Identified

1. **Frontend-Backend**: SSE streaming at `/api/chat-stream`
2. **Backend-Redis**: Memory storage and retrieval
3. **Backend-GPU**: Model inference through llama-cpp-python
4. **Process Management**: Signal handling between start.py and services