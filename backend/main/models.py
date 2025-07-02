#!/usr/bin/env python3
"""Pydantic models for request/response validation."""

from typing import Any

from pydantic import BaseModel, Field


# ===================== Chat Models =====================

class Message(BaseModel):
    """Chat message structure."""

    role: str = Field(..., max_length=20)
    content: str = Field(..., max_length=100000)  # 100KB limit


class ChatStreamRequest(BaseModel):
    """Chat streaming request model."""

    session_id: str = Field(..., max_length=100)
    messages: list[Message] = Field(..., max_items=100)


# ===================== Memory Models =====================

class VitalMemoryBRequest(BaseModel):
    """Vital memory storage request model."""

    memory: str
    importance: float = Field(..., ge=0.0, le=1.0)


class VitalMemoryResponse(BaseModel):
    """Response for vital memory operations."""

    success: bool
    memory_id: str | None = None
    message: str | None = None
    deleted_count: int | None = None


class ClearMemoriesResponse(BaseModel):
    """Response for clearing memories."""

    success: bool
    message: str


class ScrapedContent(BaseModel):
    """Model for scraped web content."""

    url: str
    content: str
    metadata: dict[str, Any] | None = None


class MemoryIngestion(BaseModel):
    """Model for general memory ingestion."""

    content: str
    conversation_id: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] | None = None


# ===================== Status Models =====================

class ServiceStatus(BaseModel):
    """Individual service status."""

    llm: str
    redis: str
    memory_manager: str
    token_manager: str


class LockStatus(BaseModel):
    """Lock status information."""

    locked: bool
    owner: tuple[str, str, str | None] | None = None


class StatusResponse(BaseModel):
    """Overall status response."""

    status: str
    services: ServiceStatus
    model_lock: LockStatus


# ===================== Trading API Models =====================

class CryptoQuoteRequest(BaseModel):
    """Request model for cryptocurrency quotes."""

    coin_ids: list[str] = Field(..., description="List of CoinGecko coin IDs")


class StockQuoteRequest(BaseModel):
    """Request model for stock quotes."""

    symbols: list[str] = Field(..., description="List of stock ticker symbols")


class CryptoDataResponse(BaseModel):
    """Response model for cryptocurrency data."""

    success: bool
    data: str | None = None
    sources: list[dict[str, str]] | None = None
    error: str | None = None


class StockDataResponse(BaseModel):
    """Response model for stock data."""

    success: bool
    data: str | None = None
    sources: list[dict[str, str]] | None = None
    error: str | None = None