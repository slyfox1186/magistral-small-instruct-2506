import React, { useState, useEffect, useCallback } from 'react';
import { Conversation, ConversationCreate } from '@/utils/types';
import { crudApi } from '@/api/crud';
import { useAlerts } from '@/hooks/useAlerts';

interface ConversationListProps {
  onSelectConversation: (conversation: Conversation) => void;
  selectedConversationId?: string;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  onSelectConversation,
  selectedConversationId,
}) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [showNewConversationForm, setShowNewConversationForm] = useState(false);
  const [newConversationTitle, setNewConversationTitle] = useState('');
  const { showAlert } = useAlerts();

  const loadConversations = useCallback(async () => {
    try {
      setLoading(true);
      const response = await crudApi.listConversations(1, 50, showArchived);
      setConversations(response.conversations);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load conversations');
      showAlert('Failed to load conversations', 'error');
    } finally {
      setLoading(false);
    }
  }, [showArchived, showAlert]);

  useEffect(() => {
    loadConversations();
  }, [showArchived, loadConversations]);

  const handleCreateConversation = async () => {
    if (!newConversationTitle.trim()) {
      showAlert('Please enter a conversation title', 'warning');
      return;
    }

    try {
      const newConversation: ConversationCreate = {
        title: newConversationTitle.trim(),
        tags: [],
        metadata: {},
      };

      const created = await crudApi.createConversation(newConversation);
      setConversations(prev => [created, ...prev]);
      setNewConversationTitle('');
      setShowNewConversationForm(false);
      onSelectConversation(created);
      showAlert('Conversation created successfully', 'success');
    } catch {
      showAlert('Failed to create conversation', 'error');
    }
  };

  const handleArchiveConversation = async (conversation: Conversation, event: React.MouseEvent) => {
    event.stopPropagation();
    
    try {
      if (conversation.archived) {
        await crudApi.unarchiveConversation(conversation.id);
        showAlert('Conversation unarchived', 'success');
      } else {
        await crudApi.archiveConversation(conversation.id);
        showAlert('Conversation archived', 'success');
      }
      loadConversations();
    } catch {
      showAlert('Failed to update conversation', 'error');
    }
  };

  const handleDeleteConversation = async (conversation: Conversation, event: React.MouseEvent) => {
    event.stopPropagation();
    
    if (!window.confirm(`Are you sure you want to delete "${conversation.title}"? This action cannot be undone.`)) {
      return;
    }

    try {
      await crudApi.deleteConversation(conversation.id);
      setConversations(prev => prev.filter(c => c.id !== conversation.id));
      showAlert('Conversation deleted', 'success');
    } catch {
      showAlert('Failed to delete conversation', 'error');
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (loading) {
    return (
      <div className="conversation-list loading">
        <div className="loading-spinner">Loading conversations...</div>
      </div>
    );
  }

  return (
    <div className="conversation-list">
      <div className="conversation-list-header">
        <h3>Conversations</h3>
        <div className="conversation-list-controls">
          <button
            className="btn btn-primary btn-sm"
            onClick={() => setShowNewConversationForm(true)}
          >
            + New
          </button>
          <label className="archive-toggle">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(e) => setShowArchived(e.target.checked)}
            />
            Show Archived
          </label>
        </div>
      </div>

      {showNewConversationForm && (
        <div className="new-conversation-form">
          <input
            type="text"
            placeholder="Conversation title..."
            value={newConversationTitle}
            onChange={(e) => setNewConversationTitle(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleCreateConversation()}
            autoFocus
          />
          <div className="form-actions">
            <button className="btn btn-primary btn-sm" onClick={handleCreateConversation}>
              Create
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setShowNewConversationForm(false);
                setNewConversationTitle('');
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="error-message">
          {error}
          <button onClick={loadConversations} className="btn btn-sm">
            Retry
          </button>
        </div>
      )}

      <div className="conversation-items">
        {conversations.length === 0 ? (
          <div className="empty-state">
            {showArchived ? 'No archived conversations' : 'No conversations yet. Create your first one!'}
          </div>
        ) : (
          conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`conversation-item ${
                selectedConversationId === conversation.id ? 'selected' : ''
              } ${conversation.archived ? 'archived' : ''}`}
              onClick={() => onSelectConversation(conversation)}
            >
              <div className="conversation-info">
                <div className="conversation-title">{conversation.title}</div>
                <div className="conversation-meta">
                  <div className="conversation-meta-row">
                    <span className="meta-label">Messages:</span>
                    <span>{conversation.message_count}</span>
                  </div>
                  <div className="conversation-meta-row">
                    <span className="meta-label">Updated:</span>
                    <span>{formatDate(conversation.updated_at)}</span>
                  </div>
                </div>
                {conversation.tags.length > 0 && (
                  <div className="conversation-tags">
                    {conversation.tags.map((tag: string, index: number) => (
                      <span key={index} className="tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                <div className="conversation-actions">
                  <button
                    className="conversation-action-btn"
                    onClick={(e) => handleArchiveConversation(conversation, e)}
                    title={conversation.archived ? 'Unarchive' : 'Archive'}
                  >
                    {conversation.archived ? 'Unarchive' : 'Archive'}
                  </button>
                  <button
                    className="conversation-action-btn danger"
                    onClick={(e) => handleDeleteConversation(conversation, e)}
                    title="Delete"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};