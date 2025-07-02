# Magistral Small Instruct 2506 - Neural Consciousness Chat System

A **local AI chat system** that runs entirely on your machine - no internet required after setup! Features advanced memory, web scraping, crypto/stock data, and GPU acceleration.

## 🎯 What You Get

- **Private AI Assistant** that remembers your conversations
- **Real-time streaming chat** with markdown support  
- **Web search integration** for current information
- **Cryptocurrency & stock data** with live prices
- **Memory system** that learns about you over time
- **100% local** - your data never leaves your machine
- **GPU accelerated** for fast responses

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **RAM**: 16GB (32GB recommended for better performance)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 20GB free space
- **Internet**: Only for initial setup and web search features

### Software Prerequisites
- **Python**: 3.11 or 3.12
- **Node.js**: 18+ and npm
- **Docker**: Latest version
- **CUDA**: 11.8 or 12.x (for GPU acceleration)
- **Git**: For cloning the repository

## 🚀 Quick Start

### Step 1: Install Prerequisites

**Ubuntu/Debian:**
```bash
# Install Python 3.11+, Node.js, Docker
sudo apt update
sudo apt install python3.11 python3.11-pip nodejs npm docker.io git
sudo usermod -aG docker $USER  # Logout and login after this
```

**macOS:**
```bash
# Install with Homebrew
brew install python@3.11 node docker git
```

**Windows:**
- Install [Python 3.11+](https://www.python.org/downloads/)
- Install [Node.js 18+](https://nodejs.org/)
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Install [Git](https://git-scm.com/download/win)

### Step 2: Download the AI Model

The system uses **Magistral-Small-2506**, a high-quality local language model. Choose one of these official sources:

**Option A: Unsloth (Original)**
```bash
# Download from Unsloth (requires ~8GB download)
mkdir -p ~/ai-models
cd ~/ai-models
wget https://huggingface.co/unsloth/Magistral-Small-2506-GGUF/resolve/main/Magistral-Small-2506-Q4_K_M.gguf
```

**Option B: Bartowski (Alternative)**
```bash
# Download from Bartowski (requires ~8GB download)
mkdir -p ~/ai-models
cd ~/ai-models
wget https://huggingface.co/bartowski/mistralai_Magistral-Small-2506-GGUF/resolve/main/Magistral-Small-2506-Q4_K_M.gguf
```

**Alternative download methods:**
- Use `huggingface-hub`: `pip install huggingface-hub && huggingface-cli download unsloth/Magistral-Small-2506-GGUF Magistral-Small-2506-Q4_K_M.gguf`
- Download manually from [Unsloth](https://huggingface.co/unsloth/Magistral-Small-2506-GGUF) or [Bartowski](https://huggingface.co/bartowski/mistralai_Magistral-Small-2506-GGUF)

### Step 3: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/slyfox1186/magistral-small-instruct-2506.git
cd magistral-small-instruct-2506

# Create environment file
cp .env.example .env  # Edit this file with your settings
```

### Step 4: Configure Environment

Create/edit the `.env` file in the project root:

```bash
# Model configuration (use the filename that matches your download)
MODEL_PATH=/home/yourusername/ai-models/Magistral-Small-2506-Q4_K_M.gguf
GPU_LAYERS=35  # Adjust based on your GPU VRAM
CONTEXT_SIZE=32768

# API settings (optional - for web search features)
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here

# Redis settings (usually defaults are fine)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Step 5: Install Dependencies

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Install Node.js dependencies
cd client && npm install && cd ../

# Start Redis database
cd docker && docker-compose up -d && cd ../
```

### Step 6: Start the System

```bash
# Start both frontend and backend (from project root)
python start.py
```

**First startup takes 2-5 minutes** as the AI model loads into GPU memory.

### Step 7: Access Your AI Assistant

- **Chat Interface**: http://localhost:4000
- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health

## 🎮 How to Use

### Basic Chat
1. Open http://localhost:4000 in your browser
2. Type a message and press Enter
3. The AI will respond in real-time
4. Your conversations are automatically saved and remembered

### Advanced Features
- **Memory**: Ask "What did we talk about yesterday?" 
- **Web Search**: "What's the latest news about AI?"
- **Crypto Prices**: "What's the current Bitcoin price?"
- **Stock Data**: "Show me Apple's stock performance"
- **Personal Info**: Tell it your name, preferences - it will remember

### API Usage
```bash
# Send a chat message via API
curl -X POST "http://localhost:8000/api/chat-stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "session_id": "test-session"
  }'
```

## 🔧 Configuration Options

### Model Settings (in .env)
- `GPU_LAYERS`: Number of layers to load on GPU (0-40)
- `CONTEXT_SIZE`: Maximum conversation length (8192-32768)
- `TEMPERATURE`: Response creativity (0.1-1.0)

### Performance Tuning
- **More GPU Layers**: Faster responses, uses more VRAM
- **Larger Context**: Longer memory, uses more RAM
- **Lower Temperature**: More focused responses

### Custom Ports
```bash
# Use different ports if 4000/8000 are in use
python start.py --frontend-port 3000 --backend-port 9000
```

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Web UI    │────▶│   FastAPI   │────▶│ Local LLM    │
│ (React TS)  │     │  Backend    │     │ (Magistral)  │
└─────────────┘     └──────┬──────┘     └──────────────┘
                           │
                    ┌──────┴──────┐
                    │    Redis    │
                    │  (Memory)   │
                    └─────────────┘
```

### Recent Improvements
- **Modular Architecture**: Broke down 3,421-line monolithic file into 12 focused modules
- **99% code reduction** in main entry point
- **Type safety** throughout with comprehensive error handling
- **GPU optimization** with intelligent batching
- **Memory consolidation** for long-term conversations

## 🛠️ Development Commands

### Backend (Python/FastAPI)
```bash
python start.py                                    # Start everything
python start.py --verbose                          # Debug mode
ruff check backend/ --fix                          # Lint and fix
ruff format backend/                               # Format code
```

### Frontend (React/TypeScript)
```bash
cd client && npm run dev                           # Dev server only
cd client && npm run build                         # Production build
cd client && npm run lint                          # Check code style
```

### Infrastructure
```bash
cd docker && docker-compose up -d && cd ../        # Start Redis
cd monitoring && docker-compose up -d && cd ../    # Start monitoring
```

## 📊 Monitoring & Health

### Access Points
- **Frontend**: http://localhost:4000
- **Backend API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prometheus**: http://localhost:9090 (if monitoring enabled)
- **Grafana**: http://localhost:3000 (admin/neural-consciousness)

### Performance Monitoring
- **GPU Usage**: `nvidia-smi` command
- **Logs**: Check `logs/backend.log`
- **Memory Stats**: http://localhost:8000/api/memory-stats

## 🚨 Troubleshooting

### Common Issues

**"Model file not found"**
```bash
# Check if model path in .env is correct
ls -la $MODEL_PATH
# Should show the .gguf file
```

**"CUDA out of memory"**
```bash
# Reduce GPU layers in .env
GPU_LAYERS=20  # Try lower values
```

**"Port already in use"**
```bash
# Use different ports
python start.py --frontend-port 3000 --backend-port 9000
```

**"Redis connection failed"**
```bash
# Restart Redis
cd docker && docker-compose restart && cd ../
```

**Slow responses**
- Increase `GPU_LAYERS` in .env (if you have VRAM)
- Ensure no other GPU processes are running
- Check `nvidia-smi` for GPU utilization

### Getting Help
- Check logs: `tail -f logs/backend.log`
- Test health: http://localhost:8000/health
- Monitor GPU: `watch nvidia-smi`

## 🎯 Why Use This?

### vs ChatGPT/Claude
- ✅ **100% Private** - Your conversations never leave your machine
- ✅ **No monthly fees** - Run unlimited conversations
- ✅ **Custom memory** - Remembers context across sessions
- ✅ **Offline capable** - Works without internet
- ✅ **Customizable** - Modify behavior and responses

### vs Other Local LLMs
- ✅ **Easy setup** - Automated installation and configuration
- ✅ **Web interface** - No command line required
- ✅ **Advanced features** - Memory, web search, data integration
- ✅ **Production ready** - Monitoring, logging, error handling

## 🧠 Memory System

### What It Remembers
- **Personal Information**: Name, preferences, important details
- **Conversation History**: Previous discussions and context
- **Learned Patterns**: Your communication style and interests
- **Important Facts**: Things you've taught it

### Memory Features
- **Semantic Search**: Find relevant past conversations
- **Automatic Consolidation**: Optimizes memory over time
- **Importance Scoring**: Prioritizes meaningful information
- **Core Memories**: Persistent facts about you

## 🔒 Security & Privacy

- **Local Only**: All processing happens on your machine
- **No Telemetry**: No data sent to external servers
- **Encrypted Storage**: Conversations stored securely
- **Network Isolation**: Can run completely offline
- **Access Control**: Only accessible from your machine by default

## 🤝 Contributing

### Code Quality Standards
- **Python**: Ruff linting, type hints, async/await patterns
- **TypeScript**: Strict mode, proper interfaces
- **Testing**: Add tests for new features
- **Documentation**: Update README for user-facing changes

### Development Setup
```bash
# Install dev dependencies
pip install -r backend/requirements-dev.txt
cd client && npm install && cd ../

# Run in development mode
python start.py --verbose
```

## 📄 License

MIT License - See LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/slyfox1186/magistral-small-instruct-2506/issues)
- **Discussions**: [GitHub Discussions](https://github.com/slyfox1186/magistral-small-instruct-2506/discussions)
- **Documentation**: Check the `docs/` folder for detailed guides

---

**Enjoy your private AI assistant!** 🤖✨