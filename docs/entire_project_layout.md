# Magistral-Small-2506-1.2 Project Layout Documentation

## 📋 Project Overview

This is a **Neural Consciousness Chat System** implementing advanced metacognitive AI capabilities with a local LLM (Magistral-Small-2506). The system consists of a FastAPI backend, React/TypeScript frontend, and supporting infrastructure.

## 🏗️ High-Level Architecture

```markdown
┌────────────────────────────────────────────────────────────────────┐
│                    MAGISTRAL CHAT SYSTEM                           │
├────────────────────────────────────────────────────────────────────┤
│  Frontend (React/TS)  │  Backend (FastAPI)    │  Infrastructure    │
│  ┌─────────────────┐  │  ┌────────────────┐   │  ┌──────────────┐  │
│  │   Chat UI       │◄─┤  │   API Server   │◄─┤  |   Redis      │  │
│  │   Components    │  │  │   Memory Sys   │   │  │   Docker     │  │
│  │   Utilities     │  │  │   LLM Engine   │   │  │ Monitoring   │  │
│  └─────────────────┘  │  └────────────────┘   │  └──────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## 🗂️ Project Structure

### 📁 Root Level Files

| File | Type | Purpose |
|------|------|---------|
| `start.py` | Python | **Main Application Launcher** - Orchestrates frontend and backend startup |
| `package.json` | JSON | Root-level Node.js dependencies |
| `CLAUDE.md` | Markdown | Development guidelines and build commands |

---

## 🔧 Backend Architecture (`/backend/`)

### 🚀 Core Application Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | **FastAPI Entry Point** - Main application setup | App initialization, middleware, route setup |
| `start.py` | **Process Manager** - Handles startup sequence | Frontend/backend orchestration, port management |
| `config.py` | **Configuration** - System settings | Environment variables, API configuration |
| `constants.py` | **System Constants** - Shared values | Fixed values used across the application |

### 🧠 AI & Memory System

#### Memory Management (`/backend/memory/`)
```
memory/
├── __init__.py              # Memory system initialization
├── importance_scorer.py     # Rates memory importance for retention
├── memory_provider.py       # Main memory interface
├── config/                  # Memory configuration files
│   ├── circles.yaml        # Memory organization circles
│   ├── models.yaml         # Embedding model settings
│   ├── redis.yaml          # Redis connection config
│   └── storage_rules.yaml  # Memory retention rules
├── schemas/
│   ├── __init__.py
│   └── memory_schema.py    # Memory data structures
├── services/
│   ├── __init__.py
│   ├── consolidation_worker.py  # Background memory optimization
│   ├── embedding_service.py     # Semantic embeddings
│   ├── memory_api.py           # Memory API endpoints
│   ├── retrieval_engine.py     # Memory search/retrieval
│   └── storage_rules_manager.py # Memory lifecycle management
└── utils/
    ├── __init__.py
    ├── metrics.py          # Memory system metrics
    └── model_manager.py    # Embedding model management
```

#### AI Engine Components
| File | Purpose | Key Features |
|------|---------|--------------|
| `persistent_llm_server.py` | **LLM Engine** - GPU-accelerated local model server | Model loading, inference, GPU management |
| `metacognitive_engine.py` | **Self-Evaluation** - AI self-improvement system | Response evaluation, learning adaptation |
| `llm_optimizer.py` | **Performance Optimization** - Model efficiency tuning | Memory optimization, inference speed |
| `ultra_advanced_engine.py` | **Advanced AI Features** - Enhanced capabilities | Complex reasoning, multi-turn context |

### 🔌 API & Routes (`/backend/modules/`)

| File | Purpose | Endpoints |
|------|---------|-----------|
| `chat_routes.py` | **Chat API** - Main chat functionality | `/api/chat-stream` |
| `health_routes.py` | **Health Monitoring** - System status | `/api/health` |
| `memory_routes.py` | **Memory Management** - Memory operations | `/clear-vital-memories` |
| `trading_routes.py` | **Trading Features** - Financial integrations | Trading-related endpoints |
| `app_factory.py` | **App Factory** - Application creation | FastAPI app setup |
| `chat_helpers.py` | **Chat Utilities** - Chat processing helpers | Message handling, formatting |
| `lifespan.py` | **Application Lifecycle** - Startup/shutdown | Resource management |
| `models.py` | **Data Models** - Pydantic schemas | Request/response models |
| `system_prompt.py` | **AI Prompts** - System instructions | AI behavior configuration |

### 🔐 Security & Infrastructure

| File | Purpose | Key Features |
|------|---------|--------------|
| `security.py` | **Security Layer** - Authentication, authorization | API security, input validation |
| `error_handlers.py` | **Error Management** - Exception handling | Global error handling, logging |
| `monitoring.py` | **System Monitoring** - Performance metrics | Prometheus metrics, health checks |
| `circuit_breaker.py` | **Resilience** - Failure protection | Circuit breaker pattern |
| `gpu_lock.py` | **Resource Management** - GPU coordination | GPU resource locking |
| `resource_manager.py` | **Resource Allocation** - System resources | Memory, CPU management |

### 📊 Data & External Services

| File | Purpose | Functionality |
|------|---------|---------------|
| `redis_utils.py` | **Redis Operations** - Cache management | Redis client, data operations |
| `async_redis_client.py` | **Async Redis** - Asynchronous operations | Async Redis operations |
| `redis_compat_memory.py` | **Redis Compatibility** - Memory interface | Redis-backed memory system |
| `web_scraper.py` | **Web Scraping** - External data retrieval | Web content extraction |
| `stock_search.py` | **Stock Data** - Financial data integration | Stock market information |
| `crypto_trading.py` | **Cryptocurrency** - Crypto trading features | Crypto market operations |
| `crypto_config.py` | **Crypto Configuration** - Crypto settings | Trading configuration |
| `token_manager.py` | **Token Management** - API token handling | Authentication tokens |

### 🛠️ Utilities & Helpers

| File | Purpose | Features |
|------|---------|----------|
| `utils.py` | **General Utilities** - Common functions | Shared utility functions |
| `colored_logging.py` | **Enhanced Logging** - Colored log output | Color-coded logging system |
| `async_personal_memory_system.py` | **Async Memory** - Async memory operations | Asynchronous memory handling |
| `personal_memory_system.py` | **Personal Memory** - User-specific memory | Personal context management |

### 🤖 AI Model Storage
```
models/
└── Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf
    # Local LLM model file (GGUF format)
    # 32K context window, GPU-accelerated
```

---

## 🎨 Frontend Architecture (`/client/`)

### ⚛️ React Application Core

| File | Purpose | Key Components |
|------|---------|----------------|
| `src/App.tsx` | **Main Application** - Root component | Chat interface, message handling, streaming |
| `src/main.tsx` | **Entry Point** - React app initialization | React DOM rendering, providers |
| `index.html` | **HTML Template** - Base HTML structure | App container, meta tags |

### 🧩 Components (`/client/src/components/`)

#### Core Components
| File | Purpose | Features |
|------|---------|----------|
| `ChatMessage.tsx` | **Message Display** - Individual chat messages | Message formatting, user/AI distinction |
| `MarkdownItRenderer.tsx` | **Markdown Rendering** - Rich text display | Syntax highlighting, code blocks |
| `ErrorBoundary.tsx` | **Error Handling** - React error boundaries | Error catching, fallback UI |
| `ThemeToggleButton.tsx` | **Theme Switching** - Light/dark mode | Theme management, user preferences |

#### Alert System (`/client/src/components/alerts/`)
| File | Purpose | Features |
|------|---------|----------|
| `AlertContainer.tsx` | **Alert Management** - Notification container | Alert display, positioning |
| `NotificationToast.tsx` | **Toast Notifications** - Popup messages | Success/error notifications |

#### Status System (`/client/src/components/status/`)
| File | Purpose | Features |
|------|---------|----------|
| `StatusIndicator.tsx` | **Status Display** - System status visualization | Loading, success, error states |

### 🏗️ Application Structure

#### Contexts (`/client/src/contexts/`)
| File | Purpose | Functionality |
|------|---------|---------------|
| `AlertContext.tsx` | **Alert State** - Global alert management | Alert providers, hooks |
| `ThemeContext.tsx` | **Theme State** - Theme management | Theme providers, switching |

#### API Layer (`/client/src/api/`)
| File | Purpose | Features |
|------|---------|----------|
| `chat.ts` | **Chat API Client** - Backend communication | Streaming chat, WebSocket handling |

#### Type Definitions (`/client/src/types/`)
| File | Purpose | Types |
|------|---------|-------|
| `status.ts` | **Status Types** - Status-related types | Status enums, interfaces |

### 🔧 Utilities (`/client/src/utils/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `config.ts` | **Configuration** - App settings | API URLs, timeouts, constants |
| `logger.ts` | **Logging** - Client-side logging | Debug logging, error tracking |
| `network.ts` | **Network Utilities** - HTTP helpers | Request handling, retries |
| `performance.ts` | **Performance Monitoring** - Performance tracking | Timing, metrics |
| `types.ts` | **Type Definitions** - TypeScript types | Interface definitions |
| `typeGuards.ts` | **Type Guards** - Runtime type checking | Type safety, validation |
| `messageUtils.ts` | **Message Utilities** - Message processing | Message formatting, parsing |
| `markdownUtils.ts` | **Markdown Utilities** - Markdown processing | Markdown parsing, rendering |
| `cacheUtils.ts` | **Cache Management** - Client-side caching | Cache operations, storage |
| `index.ts` | **Utility Exports** - Centralized exports | Utility re-exports |

### 🎨 Styling (`/client/src/styles/`)

| File | Purpose | Styling Features |
|------|---------|------------------|
| `App.css` | **Main Styles** - Primary application styles | Layout, chat interface, responsive design |
| `index.css` | **Global Styles** - Base styles, CSS variables | Theme variables, global resets |
| `MarkdownIt.css` | **Markdown Styles** - Code syntax highlighting | Code blocks, syntax coloring |
| `StatusAnimations.css` | **Status Animations** - Loading and status animations | Spinning indicators, transitions |

### ⚙️ Build Configuration

| File | Purpose | Configuration |
|------|---------|---------------|
| `package.json` | **Dependencies** - Node.js packages | React, TypeScript, build tools |
| `tsconfig.json` | **TypeScript Config** - TS compilation | Type checking, module resolution |
| `tsconfig.node.json` | **Node TypeScript** - Node.js specific TS config | Build tools configuration |
| `vite.config.ts` | **Vite Configuration** - Build system | Hot reload, bundling, proxying |
| `eslint.config.js` | **ESLint Configuration** - Code quality | Linting rules, code standards |

---

## 🐳 Infrastructure (`/docker/`, `/monitoring/`)

### 🗄️ Database & Caching (`/docker/`)
| File | Purpose | Configuration |
|------|---------|---------------|
| `docker-compose.yml` | **Redis Stack** - Database services | Redis with vector search |
| `redis.conf` | **Redis Configuration** - Redis settings | Memory, persistence, performance |

### 📊 Monitoring Stack (`/monitoring/`)
| File | Purpose | Features |
|------|---------|----------|
| `docker-compose.yml` | **Monitoring Services** - Prometheus, Grafana | Metrics collection, dashboards |
| `prometheus.yml` | **Prometheus Config** - Metrics scraping | Metric collection rules |
| `grafana-dashboard.json` | **Grafana Dashboard** - Visualization | System metrics, performance graphs |
| `README.md` | **Monitoring Documentation** - Setup guide | Monitoring system instructions |

---

## 🔄 Data Flow Architecture

### 1. **User Interaction Flow**
```
User Input → React Components → API Client → FastAPI Backend → LLM Engine → Memory System
    ↓                                                                            ↓
Response UI ← Message Processing ← Streaming Response ← AI Processing ← Memory Retrieval
```

### 2. **Memory System Flow**
```
New Information → Importance Scoring → Embedding Generation → Redis Storage
                                                                    ↓
Memory Retrieval ← Semantic Search ← Consolidation Worker ← Storage Rules
```

### 3. **AI Processing Flow**
```
User Message → System Prompt → Context Retrieval → LLM Inference → Response Generation
                                      ↓                               ↓
             Memory Storage ← Metacognitive Analysis ← Response Evaluation
```

## 🚀 Application Startup Sequence

1. **`start.py`** - Main orchestrator
2. **Redis Docker** - Database services
3. **Backend Services** - FastAPI server, LLM loading
4. **Frontend Development** - React dev server
5. **Monitoring** - Prometheus/Grafana (optional)

## 📊 Key Technologies

### Backend Stack
- **FastAPI** - Async web framework
- **LLaMA.cpp** - Local LLM inference
- **Redis Stack** - Vector database
- **Uvicorn** - ASGI server

### Frontend Stack
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **CSS3** - Styling with variables

### Infrastructure
- **Docker** - Containerization
- **Prometheus** - Metrics
- **Grafana** - Monitoring

## 🔗 Inter-Component Communication

### API Endpoints
- **`/api/chat-stream`** - Streaming chat interface
- **`/api/health`** - System health checks
- **`/clear-vital-memories`** - Memory management
- **`/docs`** - API documentation

### Real-time Features
- **Server-Sent Events** - Streaming responses
- **WebSocket** - Real-time updates
- **Redis Pub/Sub** - Internal messaging

---

## 🎯 Development Workflows

### Starting Development
```bash
# 1. Start Redis
cd docker && docker-compose up -d

# 2. Start both frontend and backend
python start.py

# 3. Access application
# Frontend: http://localhost:4000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Code Organization Principles
- **Separation of Concerns** - Clear boundaries between layers
- **Modular Architecture** - Reusable components and services
- **Type Safety** - Full TypeScript coverage
- **Async Operations** - Non-blocking I/O throughout
- **Error Handling** - Comprehensive error management
- **Performance** - Optimized for real-time streaming

This architecture enables a scalable, maintainable neural consciousness chat system with advanced AI capabilities and robust real-time communication.
