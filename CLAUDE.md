# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**⚠️ WARNING: Critical architectural flaws exist - race conditions, memory leaks, circular dependencies compromise system reliability.**

## 🚀 Quick Start & Commands

### Start System
```bash
cd docker && docker-compose up -d  # Redis first (required)
python start.py                    # Frontend + Backend
```

### Development Commands
| Command | Purpose | Details |
|---------|---------|---------|
| `npm run lint` | Lint all code | Python (Ruff + MyPy) + TypeScript (ESLint) |
| `npm run format` | Format all code | Python (Ruff) + TypeScript (Prettier) |
| `npm run type-check` | Type validation | MyPy + TypeScript compiler |
| `cd client && npm run dev` | Frontend dev server | Port 4000, hot reload |
| `cd backend && pytest` | Run backend tests | Limited coverage |

### Access Points
- **Frontend**: http://localhost:4000
- **Backend**: http://localhost:8000 + `/docs` (Swagger)
- **Redis Insight**: http://localhost:8002

## 🏗️ Architecture: Neural Consciousness Chat System

**Stack**: React TypeScript → FastAPI Python → Local Magistral LLM + Redis Stack

### Key Components
| Layer | Technology | Critical Files |
|-------|------------|----------------|
| **Frontend** | React 18 + TypeScript + Vite | `App.tsx` (1800+ lines monolith), `api/chat.ts` |
| **Backend** | FastAPI + Python 3.11 | `main.py`, `modules/lifespan.py`, `persistent_llm_server.py` |
| **AI/Memory** | Magistral LLM + Redis | `memory/`, `gpu_lock.py` ⚠️, `modules/chat_routes.py` |
| **Infrastructure** | Docker + Redis Stack | `docker/docker-compose.yml`, `monitoring/` |

## 🚨 Critical Issues & Warnings

### Critical Bugs (Immediate Attention Required)
| Issue | Location | Impact | Fix Priority |
|-------|----------|--------|--------------|
| **LLM Server Race Conditions** | `persistent_llm_server.py:467-479` | Double-checked locking issues | 🔴 Critical |
| **Memory Connection Leaks** | `memory_api.py:502-503` | Redis connections not pooled | 🟠 High |
| **Task Lifecycle Leaks** | Background tasks never canceled | Unbounded growth | 🔴 Critical |
| **Frontend State Chaos** | `App.tsx` multiple signals | Race conditions | 🟠 High |
| **Circular Dependencies** | `globals.py`, `lifespan.py` | Untestable code | 🟠 High |

### Development Hazards
- **Startup**: 2-5 minutes, may fail with GPU OOM
- **Stability**: GPU deadlocks require full restart
- **Security**: No input validation, resource exhaustion attacks possible
- **Performance**: P95 latency 3-8s (target <2s), poor GPU utilization
- **Scope Creep**: Unnecessary crypto/stock/weather APIs add complexity

## 🔧 Code Standards & Patterns

### Python (Backend) - Ruff + MyPy + Pytest
- **Lint/Format**: `ruff check . && ruff format .` (100 char limit, double quotes)
- **Types**: MyPy strict mode, Pydantic models for API validation
- **Testing**: Pytest (minimal coverage currently)

### TypeScript (Frontend) - ESLint + Prettier + Vite
- **Lint/Format**: ESLint (React + a11y rules) + Prettier
- **Build**: Vite for fast dev/build, strict TypeScript config
- **Patterns**: React context + hooks (poorly implemented), streaming chat via SSE

### Key System Patterns
- **Memory**: Complex embedding-based system with Redis storage
- **Streaming**: Server-sent events for real-time LLM responses  
- **Resource Mgmt**: Attempted GPU/memory management (buggy)
- **Error Handling**: Minimal and inconsistent (needs unified pattern)

## 💡 Essential Development Guidelines

### Prerequisites & Startup
1. **Redis First**: Always `cd docker && docker-compose up -d` before starting system
2. **Slow Startup**: Backend takes 2-5 minutes to load LLM model, may fail with GPU OOM
3. **Resource Monitoring**: Watch GPU memory, system prone to CUDA errors

### Working with the System
4. **Test Carefully**: System is fragile, small changes can cause cascading failures
5. **Isolation Issues**: Components tightly coupled, testing difficult
6. **Deadlock Recovery**: GPU locks may require full process restart
7. **Memory Vigilance**: Known leaks in long sessions

### Before Making Changes
🚨 **Backup Redis data first** - system state and conversation history
🚨 **Start small** - incremental changes only, monitor resource usage
🚨 **Consider refactoring** - system may need architectural fixes vs patches

### Quick Fixes Priority
1. `persistent_llm_server.py:467-479` - Fix double-checked locking race condition
2. `memory_api.py:502-503` - Implement Redis connection pooling 
3. `App.tsx` - Implement proper state machine
4. `consolidation_worker.py` - Fix task lifecycle management
5. `chat_routes.py` - Add input validation

### Note on GPU Lock
- `gpu_lock.py:136-138` background task tracking is actually correctly implemented
- Original issue identification was inaccurate