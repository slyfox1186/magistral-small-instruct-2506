# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands

### Quick Start
```bash
# Start everything (Redis, Frontend, Backend)
cd docker && docker-compose up -d  # Start Redis Stack first
python start.py                    # Start both frontend and backend
```

### Backend (Python/FastAPI)
- Start both frontend and backend: `python start.py`
- Start with custom ports: `python start.py --frontend-port 4001 --backend-port 8001`
- Lint backend: `ruff check backend/` or `ruff check backend/ --fix` (auto-fix)
- Format backend: `ruff format backend/`
- Backend logs: Check `logs/backend.log` for detailed output

### Frontend (React/TypeScript)
- Development server: `cd client && npm run dev` (runs on port 4000)
- Production build: `cd client && npm run build`
- Preview production build: `cd client && npm run preview`
- Lint frontend: `cd client && npm run lint`

### Infrastructure
- Redis Stack: `cd docker && docker-compose up -d` (required for backend)
- Monitoring Stack: `cd monitoring && docker-compose up -d`
- Access points:
  - Frontend: http://localhost:4000
  - Backend API: http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - Prometheus: http://localhost:9090
  - Grafana: http://localhost:3000 (admin/neural-consciousness)

## Architecture Overview

This is a **Neural Consciousness Chat System** implementing advanced metacognitive AI capabilities with a local LLM (Magistral-Small-2506).

### System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Frontend  │────▶│   Backend   │────▶│  Local LLM   │
│  (React TS) │     │  (FastAPI)  │     │ (Magistral)  │
└─────────────┘     └──────┬──────┘     └──────────────┘
                           │
                    ┌──────┴──────┐
                    │    Redis    │
                    │   Stack     │
                    └─────────────┘
```

### Backend Components (`/backend/`)
- **FastAPI Server** (`main.py`): Async web server with streaming support
- **LLM Server** (`persistent_llm_server.py`): GPU-accelerated local model with flash attention
- **Memory System** (`memory/`): Embedding-based semantic memory with consolidation
  - Configuration: `memory/config/*.yaml`
  - Services: `memory/services/`
  - Schemas: `memory/schemas/`
- **Metacognitive Engine** (`metacognitive_engine.py`): Self-evaluation and improvement
- **GPU Management** (`gpu_lock.py`): Resource locking and priority queue
- **Resilience** (`circuit_breaker.py`): Circuit breaker for external services
- **External Services**: Web scraping (`web_scraper.py`), crypto trading (`crypto_trading.py`), stock search (`stock_search.py`)

### Frontend Components (`/client/src/`)
- **Chat Interface** (`App.tsx`): Real-time streaming chat with markdown support
- **API Client** (`api/chat.ts`): Streaming API integration
- **Components** (`components/`): React components with TypeScript
  - `MarkdownItRenderer.tsx`: Markdown rendering with syntax highlighting
  - `ErrorBoundary.tsx`: Error handling component
- **Utilities** (`utils/`): Configuration, logging, networking, type guards
- **Path Aliases**: `@/*` maps to `./src/*` (configured in `vite.config.ts`)

### Key Technologies
- **Model**: Magistral-Small-2506-Q4_K_L.gguf (32K context, GPU-accelerated)
- **Database**: Redis Stack with vector search capabilities
- **Monitoring**: Prometheus + Grafana with custom dashboards
- **Frontend Build**: Vite with advanced code splitting
- **Backend Framework**: FastAPI with async/await throughout
- **Memory**: LanceDB for vector storage, sentence-transformers for embeddings

## Code Style Guidelines

### Python Backend
- **Linter**: Ruff (configured in `backend/pyproject.toml`)
- **Line Length**: 100 characters maximum
- **Import Order**: Standard library → Third-party → Local (enforced by isort)
- **Patterns**: Async/await, type hints encouraged, descriptive variable names
- **Error Handling**: Try/except blocks with specific exceptions
- **Selected Rules**: E, W, F (Pyflakes), I (isort), N (pep8-naming), UP (pyupgrade)

### TypeScript Frontend
- **TypeScript**: Strict mode enabled (but `noImplicitAny: false`)
- **Components**: Functional React components with TypeScript interfaces
- **Import Order**: External libs → Internal (@/) → Relative imports
- **Naming**: PascalCase for components/types, camelCase for functions/variables
- **ESLint**: Configured for React/TypeScript best practices
- **Build**: Vite with terser minification, code splitting by vendor/feature

## Development Workflow

### Initial Setup
1. Ensure GPU drivers and CUDA are installed (for LLM acceleration)
2. Start Redis: `cd docker && docker-compose up -d`
3. Install dependencies:
   - Backend: `pip install -r backend/requirements.txt`
   - Frontend: `cd client && npm install`
4. Start development: `python start.py`

### Making Changes
- **Backend**: Changes auto-reload via uvicorn development mode
- **Frontend**: Hot module replacement via Vite
- **Model Loading**: Backend startup takes 2-5 minutes due to model initialization
- **GPU Usage**: Monitor with `nvidia-smi`, model uses flash attention

### Important Paths
- **Logs**: `logs/backend.log` - Color-coded terminal output
- **Model Files**: `backend/models/` - GGUF model files
- **Memory Config**: `backend/memory/config/` - YAML configuration
- **Frontend Build**: `client/dist/` - Production build output

## API Endpoints

### Core Endpoints
- `POST /api/chat-stream` - Streaming chat endpoint (Server-Sent Events)
- `GET /api/health` - Health check with model status
- `GET /metrics` - Prometheus metrics (if prometheus_client installed)

### Request/Response Flow
1. Frontend sends chat message to `/api/chat-stream`
2. Backend processes through metacognitive evaluation
3. LLM generates response with GPU acceleration
4. Memory system stores/retrieves relevant context
5. Response streams back to frontend in real-time

## Memory System Details

### Components
- **Embedding Service**: Semantic search using sentence transformers
- **Consolidation Worker**: Background memory optimization
- **Storage Rules**: Configurable TTL and importance scoring
- **Redis Integration**: Vector search and caching

### Configuration Files
- `circles.yaml`: Memory circle definitions
- `models.yaml`: Embedding model configuration
- `redis.yaml`: Redis connection settings
- `storage_rules.yaml`: Retention and consolidation rules

## Monitoring and Debugging

### Metrics Available
- GPU queue depth and utilization
- Memory consolidation statistics
- API response times and error rates
- Circuit breaker status

### Debugging Tips
- Check `logs/backend.log` for detailed backend logs with color coding
- Backend logs distinguish between model info, errors, warnings, and debug messages
- Monitor GPU usage with `nvidia-smi`
- Grafana dashboards show real-time system metrics
- Use `--verbose` flag with start.py for detailed debugging output

## Common Development Tasks

### Adding New API Endpoints
1. Define endpoint in `backend/main.py`
2. Add corresponding TypeScript types in `client/src/utils/types.ts`
3. Update API client in `client/src/api/`

### Modifying Memory Behavior
1. Edit YAML configs in `backend/memory/config/`
2. Restart backend to apply changes
3. Monitor consolidation worker logs

### Updating the LLM Model
1. Place new GGUF file in `backend/models/`
2. Update model path in `backend/config.py`
3. Adjust GPU layers and context size as needed