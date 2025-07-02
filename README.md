# Magistral Small Instruct 2506 - Neural Consciousness Chat System

A sophisticated AI chat system implementing advanced metacognitive capabilities with a local LLM (Magistral-Small-2506).

## 🚀 Quick Start

```bash
# Clone and navigate to project root
git clone https://github.com/slyfox1186/magistral-small-instruct-2506.git
cd magistral-small-instruct-2506

# Start everything (Redis, Frontend, Backend)
cd docker && docker-compose up -d  # Start Redis Stack first
cd ../                             # Go back to project root
python start.py                    # Start both frontend and backend
```

## 🏗️ Architecture

This is a **Neural Consciousness Chat System** with a modular FastAPI backend and React TypeScript frontend.

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

### 🔧 Recent Improvements

**Modular Architecture Refactoring** (Latest):
- **Broke down monolithic 3,421-line main.py** into 12 focused modules
- **99% reduction** in main entry point (from 3,421 → 32 lines)
- **Clean separation of concerns**: routes, models, helpers, configuration
- **All import issues fixed** with comprehensive dependency management

### Backend Components (`/backend/`)
- **FastAPI Server** (`main_new.py`): Modular entry point with clean imports
- **Modular Architecture** (`main/`): 
  - `config.py` - Configuration and imports
  - `globals.py` - Global variables and background processing
  - `helpers.py` - Memory management functions
  - `models.py` - Pydantic request/response models
  - `chat_routes.py` - Main streaming chat endpoint
  - `trading_routes.py` - Crypto/stock trading APIs
  - `health_routes.py` - Health monitoring endpoints
  - `memory_routes.py` - Memory management APIs
- **LLM Server** (`persistent_llm_server.py`): GPU-accelerated local model with flash attention
- **Memory System** (`memory/`): Embedding-based semantic memory with consolidation
- **Metacognitive Engine** (`metacognitive_engine.py`): Self-evaluation and improvement
- **External Services**: Web scraping, crypto trading, stock search

### Frontend Components (`/client/src/`)
- **Chat Interface** (`App.tsx`): Real-time streaming chat with markdown support
- **API Client** (`api/chat.ts`): Streaming API integration
- **Components** (`components/`): React components with TypeScript
- **Path Aliases**: `@/*` maps to `./src/*`

### Key Technologies
- **Model**: Magistral-Small-2506-Q4_K_L.gguf (32K context, GPU-accelerated)
- **Database**: Redis Stack with vector search capabilities
- **Monitoring**: Prometheus + Grafana with custom dashboards
- **Frontend Build**: Vite with advanced code splitting
- **Backend Framework**: FastAPI with async/await throughout

## 🛠️ Development Commands

### Backend (Python/FastAPI)
- Start both frontend and backend: `python start.py` *(from project root)*
- Start with custom ports: `python start.py --frontend-port 4001 --backend-port 8001`
- Lint backend: `ruff check backend/` or `ruff check backend/ --fix`
- Format backend: `ruff format backend/`
- Backend logs: Check `logs/backend.log` for detailed output

### Frontend (React/TypeScript)
- Development server: `cd client && npm run dev` (runs on port 4000)
- Production build: `cd client && npm run build`
- Preview production build: `cd client && npm run preview`
- Lint frontend: `cd client && npm run lint`

### Infrastructure
- Redis Stack: `cd docker && docker-compose up -d && cd ../` (required for backend)
- Monitoring Stack: `cd monitoring && docker-compose up -d && cd ../`

## 🌐 Access Points
- **Frontend**: http://localhost:4000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/neural-consciousness)

## 📋 Requirements

### Initial Setup
1. Ensure GPU drivers and CUDA are installed (for LLM acceleration)
2. Start Redis: `cd docker && docker-compose up -d && cd ../`
3. Install dependencies:
   - Backend: `pip install -r backend/requirements.txt`
   - Frontend: `cd client && npm install && cd ../`
4. Start development: `python start.py` *(from project root)*

## 🧠 Memory System

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

## 📊 Monitoring and Debugging

### Metrics Available
- GPU queue depth and utilization
- Memory consolidation statistics
- API response times and error rates
- Circuit breaker status

### Debugging Tips
- Check `logs/backend.log` for detailed backend logs with color coding
- Monitor GPU usage with `nvidia-smi`
- Grafana dashboards show real-time system metrics
- Use `--verbose` flag with start.py for detailed debugging output

## 🎯 Code Quality

### Python Backend
- **Linter**: Ruff (configured in `backend/pyproject.toml`)
- **Line Length**: 100 characters maximum
- **Import Order**: Standard library → Third-party → Local
- **Patterns**: Async/await, type hints, descriptive variable names

### TypeScript Frontend
- **TypeScript**: Strict mode enabled
- **Components**: Functional React components with TypeScript interfaces
- **Import Order**: External libs → Internal (@/) → Relative imports
- **Naming**: PascalCase for components/types, camelCase for functions/variables

## 🔄 Development Workflow

### Making Changes
- **Backend**: Changes auto-reload via uvicorn development mode
- **Frontend**: Hot module replacement via Vite
- **Model Loading**: Backend startup takes 2-5 minutes due to model initialization
- **GPU Usage**: Model uses flash attention for optimal performance

## 📁 Important Paths
- **Logs**: `logs/backend.log` - Color-coded terminal output
- **Model Files**: `backend/models/` - GGUF model files
- **Memory Config**: `backend/memory/config/` - YAML configuration
- **Frontend Build**: `client/dist/` - Production build output

## 🤝 Contributing

This project follows clean architecture principles with:
- **Separation of concerns** across modules
- **Type safety** throughout the codebase  
- **Comprehensive error handling** and logging
- **Performance optimization** for real-time chat
- **Modular design** for easy maintenance and testing

## 📄 License

[Add your license information here]