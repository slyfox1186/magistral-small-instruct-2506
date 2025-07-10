#!/usr/bin/env python3
"""CRUD API routes for conversations, messages, and user settings."""


from fastapi import FastAPI, HTTPException, Query

from modules.models import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageListResponse,
    MessageResponse,
    MessageUpdate,
    UserSettingsCreate,
    UserSettingsResponse,
    UserSettingsUpdate,
)
from services.crud_service import CRUDService

# Global CRUD service instance
crud_service = CRUDService()


def setup_crud_routes(app: FastAPI) -> None:
    """Setup CRUD routes for the FastAPI app."""
    # ===================== Conversation Routes =====================

    @app.post("/api/conversations", response_model=ConversationResponse)
    async def create_conversation(conversation: ConversationCreate):
        """Create a new conversation."""
        try:
            return await crud_service.create_conversation(conversation)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create conversation: {e!s}")

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
    async def get_conversation(conversation_id: str):
        """Get a conversation by ID."""
        conversation = await crud_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    @app.get("/api/conversations", response_model=ConversationListResponse)
    async def list_conversations(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        archived: bool | None = Query(None, description="Filter by archived status")
    ):
        """List conversations with pagination and optional filtering."""
        try:
            return await crud_service.list_conversations(page=page, page_size=page_size, archived=archived)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list conversations: {e!s}")

    @app.put("/api/conversations/{conversation_id}", response_model=ConversationResponse)
    async def update_conversation(conversation_id: str, update_data: ConversationUpdate):
        """Update a conversation."""
        conversation = await crud_service.update_conversation(conversation_id, update_data)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        """Delete a conversation."""
        success = await crud_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}

    # ===================== Message Routes =====================

    @app.post("/api/messages", response_model=MessageResponse)
    async def create_message(message: MessageCreate):
        """Create a new message."""
        try:
            return await crud_service.create_message(message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create message: {e!s}")

    @app.get("/api/messages/{message_id}", response_model=MessageResponse)
    async def get_message(message_id: str):
        """Get a message by ID."""
        message = await crud_service.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return message

    @app.get("/api/conversations/{conversation_id}/messages", response_model=MessageListResponse)
    async def list_messages(
        conversation_id: str,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(50, ge=1, le=200, description="Items per page")
    ):
        """List messages for a conversation with pagination."""
        try:
            return await crud_service.list_messages(conversation_id=conversation_id, page=page, page_size=page_size)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list messages: {e!s}")

    @app.put("/api/messages/{message_id}", response_model=MessageResponse)
    async def update_message(message_id: str, update_data: MessageUpdate):
        """Update a message."""
        message = await crud_service.update_message(message_id, update_data)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return message

    @app.delete("/api/messages/{message_id}")
    async def delete_message(message_id: str):
        """Delete a message."""
        success = await crud_service.delete_message(message_id)
        if not success:
            raise HTTPException(status_code=404, detail="Message not found")
        return {"message": "Message deleted successfully"}

    # ===================== User Settings Routes =====================

    @app.post("/api/users/{user_id}/settings", response_model=UserSettingsResponse)
    async def create_user_settings(user_id: str, settings: UserSettingsCreate):
        """Create user settings."""
        # Ensure user_id matches
        settings.user_id = user_id
        try:
            return await crud_service.create_user_settings(settings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create user settings: {e!s}")

    @app.get("/api/users/{user_id}/settings", response_model=UserSettingsResponse)
    async def get_user_settings(user_id: str):
        """Get user settings."""
        settings = await crud_service.get_user_settings(user_id)
        if not settings:
            raise HTTPException(status_code=404, detail="User settings not found")
        return settings

    @app.put("/api/users/{user_id}/settings", response_model=UserSettingsResponse)
    async def update_user_settings(user_id: str, update_data: UserSettingsUpdate):
        """Update user settings."""
        settings = await crud_service.update_user_settings(user_id, update_data)
        if not settings:
            raise HTTPException(status_code=404, detail="User settings not found")
        return settings

    @app.delete("/api/users/{user_id}/settings")
    async def delete_user_settings(user_id: str):
        """Delete user settings."""
        success = await crud_service.delete_user_settings(user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User settings not found")
        return {"message": "User settings deleted successfully"}

    # ===================== Utility Routes =====================

    @app.get("/api/conversations/{conversation_id}/summary")
    async def get_conversation_summary(conversation_id: str):
        """Get a summary of a conversation."""
        conversation = await crud_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get recent messages for summary
        messages = await crud_service.list_messages(conversation_id, page=1, page_size=10)

        return {
            "conversation": conversation,
            "recent_messages": messages.messages[-5:] if messages.messages else [],
            "total_messages": messages.total
        }

    @app.post("/api/conversations/{conversation_id}/archive")
    async def archive_conversation(conversation_id: str):
        """Archive a conversation."""
        update_data = ConversationUpdate(archived=True)
        conversation = await crud_service.update_conversation(conversation_id, update_data)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation archived successfully", "conversation": conversation}

    @app.post("/api/conversations/{conversation_id}/unarchive")
    async def unarchive_conversation(conversation_id: str):
        """Unarchive a conversation."""
        update_data = ConversationUpdate(archived=False)
        conversation = await crud_service.update_conversation(conversation_id, update_data)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation unarchived successfully", "conversation": conversation}

    @app.get("/api/stats/conversations")
    async def get_conversation_stats():
        """Get conversation statistics."""
        try:
            # Get total conversations
            all_conversations = await crud_service.list_conversations(page=1, page_size=1)
            total_conversations = all_conversations.total

            # Get archived conversations
            archived_conversations = await crud_service.list_conversations(page=1, page_size=1, archived=True)
            total_archived = archived_conversations.total

            # Get active conversations
            active_conversations = total_conversations - total_archived

            return {
                "total_conversations": total_conversations,
                "active_conversations": active_conversations,
                "archived_conversations": total_archived
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get conversation stats: {e!s}")
