#!/usr/bin/env python3
"""FastAPI backend for Devstral Small chat application.

Simplified modular architecture - imports all components.
"""

# Import configuration first
import time
import uuid

# Create FastAPI app with CORS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import error handling
from error_handlers import setup_exception_handlers

# Import route setup functions
from modules.chat_routes import setup_chat_routes
from modules.config import API_CONFIG
from modules.crud_routes import setup_crud_routes
from modules.health_routes import setup_health_routes

# Import lifespan management
from modules.lifespan import lifespan
from modules.memory_routes import setup_memory_routes

# Import Pydantic models
# Import system prompt
from modules.trading_routes import setup_trading_routes

# Generate a unique server instance ID for cache busting
SERVER_INSTANCE_ID = str(uuid.uuid4())
SERVER_START_TIME = int(time.time())

app = FastAPI(
    lifespan=lifespan,
    title="Devstral Chat API",
    version=f"1.0.{SERVER_START_TIME}",  # Dynamic version for cache busting
)

# Setup comprehensive error handling
setup_exception_handlers(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware to force browser cache clearing
@app.middleware("http")
async def add_cache_control_headers(request, call_next):
    """Add headers to prevent browser caching and force fresh content."""
    response = await call_next(request)

    # Force no caching for all responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    # Add timestamp to force cache busting
    response.headers["X-Content-Version"] = str(int(time.time()))

    # Force browser to check for new versions
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["X-Accel-Buffering"] = "no"

    return response


# Setup all the routes
setup_health_routes(app)
setup_memory_routes(app)
setup_chat_routes(app)
setup_trading_routes(app)
setup_crud_routes(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost for security
        port=8000,
        reload=True,
        log_level="info",
    )
