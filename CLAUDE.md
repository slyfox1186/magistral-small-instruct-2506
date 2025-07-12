# CLAUDE.md

**WARNING: This codebase has critical architectural flaws. System is over-engineered and unreliable.**

## 🚨 CRITICAL FIXES NEEDED

### Resource Management
- **GPU race conditions** (`gpu_lock.py:135`): Task cleanup missing
- **Memory leaks**: Background tasks never canceled
- **Connection leaks** (`memory_api.py:503`): Redis connections not managed

### Concurrency Issues  
- **LLM server races** (`persistent_llm_server.py:434-447`): Double-checked locking broken
- **Frontend state confusion** (`chat.ts:196`): Multiple completion signals
- **Redis deadlocks**: Lock handling improper

### Architecture Problems
- **Circular dependencies**: `globals.py`, `lifespan.py` import everything
- **No dependency injection**: Hardcoded service discovery
- **Untestable code**: Components can't be isolated

## 🔧 CRITICAL FIXES

```python
# gpu_lock.py:135 - Track tasks properly
task = asyncio.create_task(self._wake_next_waiter())
self._background_tasks.add(task)
task.add_done_callback(self._background_tasks.discard)
```

```typescript
// App.tsx - Use state machine
type StreamState = 'idle' | 'thinking' | 'streaming' | 'completed' | 'error';
const [streamState, setStreamState] = useState<StreamState>('idle');
```

```python
# consolidation_worker.py - Task lifecycle
class ConsolidationWorker:
    def __init__(self):
        self._background_tasks: set[asyncio.Task] = set()
    async def stop(self):
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
```

## 📋 COMMANDS

### Start
```bash
cd docker && docker-compose up -d  # Redis first
python start.py                    # Frontend + Backend
```

### Development
- **All**: `npm run lint`, `npm run format`, `npm run type-check` 
- **Backend**: `ruff check backend/`, `mypy backend/`, `pytest`
- **Frontend**: `cd client && npm run dev/build/lint/type-check`

### Access
- Frontend: http://localhost:4000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Redis Insight: http://localhost:8002

## 🏗️ ARCHITECTURE

Neural Consciousness Chat System: React TS → FastAPI → Magistral LLM + Redis

### Problems
- **Backend**: Circular imports, race conditions, connection leaks, no error boundaries
- **Frontend**: 1800+ line monolith, state management anti-patterns  
- **Overall**: Over-engineered, scope creep (crypto/stocks/weather APIs unnecessary)

## ⚠️ DEVELOPMENT WARNINGS

### Known Issues
1. Backend startup: 2-5 minutes (model loading)
2. Memory leaks during long sessions  
3. GPU deadlocks require restart
4. Hot reload fails (circular dependencies)

### Security Risks
- No input validation
- Resource exhaustion attacks possible
- Redis data loaded without validation
- Unbounded queues allow DoS

### Performance Problems  
- P95 latency: 3-8s (target <2s)
- Unbounded memory growth
- Poor GPU utilization

## 🎯 PRIORITY FIXES

### Critical Files
1. `/backend/gpu_lock.py` - Race conditions
2. `/backend/persistent_llm_server.py` - Initialization races  
3. `/client/src/App.tsx` - State machine needed
4. `/backend/memory/services/consolidation_worker.py` - Task leaks
5. `/backend/modules/chat_routes.py` - Input validation

### Recommendations
- **Remove scope creep**: crypto/stocks/weather APIs
- **Simplify architecture**: Focus on core chat functionality
- **Break circular dependencies**: Implement dependency injection
- **Add proper error handling**: Unified pattern across system