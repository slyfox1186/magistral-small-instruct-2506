#!/usr/bin/env python3
"""FastAPI backend for Magistral Small chat application.
Simplified modular architecture - imports all components.
"""

# Import configuration first
from main.config import *

# Import global variables and background processing
from main.globals import *

# Import helper functions
from main.helpers import *

# Import Pydantic models
from main.models import *

# Import system prompt
from main.system_prompt import *

# Import lifespan management
from main.lifespan import lifespan

# Create FastAPI app with CORS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Now setup all the route handlers by calling their setup functions
from main.health_routes import setup_health_routes
from main.memory_routes import setup_memory_routes
from main.chat_routes import setup_chat_routes
from main.trading_routes import setup_trading_routes

# Setup all the routes
setup_health_routes(app)
setup_memory_routes(app)
setup_chat_routes(app)
setup_trading_routes(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )