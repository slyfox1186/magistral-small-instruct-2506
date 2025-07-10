import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Conversation, MessageData, MessageCreate } from '@/utils/types';
import { crudApi } from '@/api/crud';
import { useAlerts } from '@/hooks/useAlerts';
import { ChatMessage } from '@/components/ChatMessage';

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
  const [newMessage, setNewMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editTitle, setEditTitle] = useState(conversation.title);
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

  const handleSendMessage = async () => {
    if (!newMessage.trim() || sending) return;

    try {
      setSending(true);
      
      // Create user message
      const userMessage: MessageCreate = {
        conversation_id: conversation.id,
        role: 'user',
        content: newMessage.trim(),
        metadata: { timestamp: new Date().toISOString() },
      };

      const createdMessage = await crudApi.createMessage(userMessage);
      setMessages(prev => [...prev, createdMessage]);
      setNewMessage('');

      // TODO: Integrate with existing chat streaming API
      // For now, we'll just add a placeholder assistant response
      const assistantMessage: MessageCreate = {
        conversation_id: conversation.id,
        role: 'assistant',
        content: 'This is a placeholder response. The actual AI response will be implemented when integrating with the existing chat system.',
        metadata: { timestamp: new Date().toISOString() },
      };

      setTimeout(async () => {
        try {
          const assistantResponse = await crudApi.createMessage(assistantMessage);
          setMessages(prev => [...prev, assistantResponse]);
        } catch {
          showAlert('Failed to get AI response', 'error');
        }
      }, 1000);

    } catch {
      showAlert('Failed to send message', 'error');
    } finally {
      setSending(false);
    }
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

  const handleDeleteMessage = async (messageId: string) => {
    if (!window.confirm('Are you sure you want to delete this message?')) {
      return;
    }

    try {
      await crudApi.deleteMessage(messageId);
      setMessages(prev => prev.filter(m => m.id !== messageId));
      showAlert('Message deleted', 'success');
    } catch {
      showAlert('Failed to delete message', 'error');
    }
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
            <div key={message.id} className="message-wrapper">
              <ChatMessage
                message={{
                  id: message.id,
                  role: message.role as 'user' | 'assistant' | 'error',
                  content: message.content,
                }}
                isStreaming={false}
              />
              <div className="message-actions">
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => handleDeleteMessage(message.id)}
                  title="Delete message"
                >
                  Delete
                </button>
                <span className="message-timestamp">
                  {formatDate(message.created_at)}
                </span>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="message-input">
        <div className="input-group">
          <textarea
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
            disabled={sending || conversation.archived}
            rows={3}
          />
          <button
            onClick={handleSendMessage}
            disabled={!newMessage.trim() || sending || conversation.archived}
            className="btn btn-primary"
          >
            {sending ? 'Sending...' : 'Send'}
          </button>
        </div>
        {conversation.archived && (
          <div className="archived-warning">
            This conversation is archived. Unarchive it to send new messages.
          </div>
        )}
      </div>
    </div>
  );
};