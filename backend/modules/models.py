#!/usr/bin/env python3
"""Pydantic models for request/response validation."""

from datetime import datetime
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


# ===================== CRUD Models =====================


class ConversationBase(BaseModel):
    """Base model for conversations."""

    title: str = Field(..., max_length=200)
    tags: list[str] = Field(default_factory=list)
    archived: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationCreate(ConversationBase):
    """Model for creating new conversations."""

    pass


class ConversationUpdate(BaseModel):
    """Model for updating conversations."""

    title: str | None = Field(None, max_length=200)
    tags: list[str] | None = None
    archived: bool | None = None
    metadata: dict[str, Any] | None = None


class ConversationResponse(ConversationBase):
    """Response model for conversations."""

    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(default=0)


class MessageBase(BaseModel):
    """Base model for messages."""

    role: str = Field(..., max_length=20)
    content: str = Field(..., max_length=100000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageCreate(MessageBase):
    """Model for creating new messages."""

    conversation_id: str = Field(..., max_length=100)


class MessageUpdate(BaseModel):
    """Model for updating messages."""

    content: str | None = Field(None, max_length=100000)
    metadata: dict[str, Any] | None = None


class MessageResponse(MessageBase):
    """Response model for messages."""

    id: str
    conversation_id: str
    created_at: datetime
    updated_at: datetime


class UserSettingsBase(BaseModel):
    """Base model for user settings."""

    theme: str = Field(default="celestial-indigo", pattern="^(celestial-indigo|veridian-twilight|solaris-flare|hunters-vision|nebula|crimson-ember|cyberpunk-neon|obsidian-slate)$")
    ai_personality: str = Field(default="helpful", max_length=100)
    response_style: str = Field(default="balanced", max_length=100)
    memory_retention: bool = Field(default=True)
    auto_summarize: bool = Field(default=True)
    preferred_language: str = Field(default="en", max_length=10)
    custom_prompts: list[str] = Field(default_factory=list)


class UserSettingsCreate(UserSettingsBase):
    """Model for creating user settings."""

    user_id: str = Field(..., max_length=100)


class UserSettingsUpdate(BaseModel):
    """Model for updating user settings."""

    theme: str | None = Field(None, pattern="^(celestial-indigo|veridian-twilight|solaris-flare|hunters-vision|nebula|crimson-ember|cyberpunk-neon|obsidian-slate)$")
    ai_personality: str | None = Field(None, max_length=100)
    response_style: str | None = Field(None, max_length=100)
    memory_retention: bool | None = None
    auto_summarize: bool | None = None
    preferred_language: str | None = Field(None, max_length=10)
    custom_prompts: list[str] | None = None


class UserSettingsResponse(UserSettingsBase):
    """Response model for user settings."""

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response model for conversation listings."""

    conversations: list[ConversationResponse]
    total: int
    page: int
    page_size: int


class MessageListResponse(BaseModel):
    """Response model for message listings."""

    messages: list[MessageResponse]
    total: int
    page: int
    page_size: int
