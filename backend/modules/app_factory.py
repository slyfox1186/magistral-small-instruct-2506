#!/usr/bin/env python3
"""FastAPI app creation and configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_CONFIG
from .lifespan import lifespan


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    # Create FastAPI app with lifespan management
    app = FastAPI(lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup all route modules
    from .chat_routes import setup_chat_routes
    from .health_routes import setup_health_routes
    from .memory_routes import setup_memory_routes
    from .trading_routes import setup_trading_routes

    setup_health_routes(app)
    setup_memory_routes(app)
    setup_chat_routes(app)
    setup_trading_routes(app)

    return app


# Create the app instance
app = create_app()
