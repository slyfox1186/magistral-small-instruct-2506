import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Conversation, MessageData } from '@/utils/types';
import { crudApi } from '@/api/crud';
import { useAlerts } from '@/hooks/useAlerts';
import { ChatMessage } from '@/components/ChatMessage';
import ConfirmModal from '@/components/ui/ConfirmModal';

interface ConversationViewProps {
  conversation: Conversation;
  onConversationUpdate: (conversation: Conversation) => void;
}

export const ConversationView: React.FC<ConversationViewProps> = ({
  conversation,
  onConversationUpdate,
}) => {
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editTitle, setEditTitle] = useState(conversation.title);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [messageToDelete, setMessageToDelete] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { showAlert } = useAlerts();

  const loadMessages = useCallback(async () => {
    try {
      setLoading(true);
      const response = await crudApi.listMessages(conversation.id, 1, 100);
      setMessages(response.messages);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load messages');
      showAlert('Failed to load messages', 'error');
    } finally {
      setLoading(false);
    }
  }, [conversation.id, showAlert]);

  useEffect(() => {
    loadMessages();
  }, [conversation.id, loadMessages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };


  const handleUpdateTitle = async () => {
    if (editTitle.trim() === conversation.title) {
      setEditingTitle(false);
      return;
    }

    try {
      const updated = await crudApi.updateConversation(conversation.id, {
        title: editTitle.trim(),
      });
      onConversationUpdate(updated);
      setEditingTitle(false);
      showAlert('Title updated', 'success');
    } catch {
      showAlert('Failed to update title', 'error');
      setEditTitle(conversation.title);
    }
  };

  const handleDeleteMessage = (messageId: string) => {
    setMessageToDelete(messageId);
    setShowDeleteModal(true);
  };

  const confirmDeleteMessage = async () => {
    if (!messageToDelete) return;

    try {
      await crudApi.deleteMessage(messageToDelete);
      setMessages(prev => prev.filter(m => m.id !== messageToDelete));
      showAlert('Message deleted', 'success');
    } catch {
      showAlert('Failed to delete message', 'error');
    } finally {
      setShowDeleteModal(false);
      setMessageToDelete(null);
    }
  };

  const cancelDeleteMessage = () => {
    setShowDeleteModal(false);
    setMessageToDelete(null);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="conversation-view loading">
        <div className="loading-spinner">Loading conversation...</div>
      </div>
    );
  }

  return (
    <div className="conversation-view">
      <div className="conversation-header">
        {editingTitle ? (
          <div className="title-edit">
            <input
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleUpdateTitle()}
              onBlur={handleUpdateTitle}
              autoFocus
            />
          </div>
        ) : (
          <div className="conversation-title-display">
            <h2 onClick={() => setEditingTitle(true)} title="Click to edit">
              {conversation.title}
            </h2>
            <div className="conversation-meta">
              <span>Created: {formatDate(conversation.created_at)}</span>
              <span>Updated: {formatDate(conversation.updated_at)}</span>
              <span>{conversation.message_count} messages</span>
              {conversation.archived && <span className="archived-badge">Archived</span>}
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          {error}
          <button onClick={loadMessages} className="btn btn-sm">
            Retry
          </button>
        </div>
      )}

      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-messages">
            No messages yet. Start the conversation!
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`message-wrapper ${message.role}-message`}>
              <div className={`message-role-indicator ${message.role}`}>
                {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
              </div>
              <ChatMessage
                message={{
                  id: message.id,
                  role: message.role as 'user' | 'assistant' | 'error',
                  content: message.content,
                }}
                isStreaming={false}
              />
              <div className="message-actions">
                <span className="message-timestamp">
                  {formatDate(message.created_at)}
                </span>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => handleDeleteMessage(message.id)}
                  title="Delete message"
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <ConfirmModal
        isOpen={showDeleteModal}
        title="Delete Message"
        message="Are you sure you want to delete this message? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        onConfirm={confirmDeleteMessage}
        onCancel={cancelDeleteMessage}
        variant="danger"
      />
    </div>
  );
};