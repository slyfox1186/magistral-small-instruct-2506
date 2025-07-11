import React, { useState } from 'react';
import { Conversation } from '@/utils/types';
import { ConversationList } from './ConversationList';
import { ConversationView } from './ConversationView';
import { UserSettingsComponent } from './UserSettings';

interface CrudInterfaceProps {
  userId: string;
  onBackToChat?: (conversationId?: string) => void;
}

type ActiveView = 'conversations' | 'settings';

const CrudInterface: React.FC<CrudInterfaceProps> = ({ userId, onBackToChat }) => {
  const [activeView, setActiveView] = useState<ActiveView>('conversations');
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);

  const handleSelectConversation = (conversation: Conversation) => {
    setSelectedConversation(conversation);
    setActiveView('conversations');
  };

  const handleConversationUpdate = (updatedConversation: Conversation) => {
    setSelectedConversation(updatedConversation);
  };

  return (
    <div className="crud-interface">
      <div className="crud-nav">
        {onBackToChat && (
          <button
            className="nav-button back-button"
            onClick={() => onBackToChat(selectedConversation?.id)}
            title="Back to Chat"
          >
            ← Back to Chat
          </button>
        )}
        <button
          className={`nav-button ${activeView === 'conversations' ? 'active' : ''}`}
          onClick={() => setActiveView('conversations')}
        >
          💬 Conversations
        </button>
        <button
          className={`nav-button ${activeView === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveView('settings')}
        >
          ⚙️ Settings
        </button>
      </div>

      <div className="crud-content">
        {activeView === 'conversations' && (
          <div className="conversations-layout">
            <div className="conversation-sidebar">
              <ConversationList
                onSelectConversation={handleSelectConversation}
                selectedConversationId={selectedConversation?.id}
              />
            </div>
            <div className="conversation-main">
              {selectedConversation ? (
                <ConversationView
                  conversation={selectedConversation}
                  onConversationUpdate={handleConversationUpdate}
                />
              ) : (
                <div className="no-conversation-selected">
                  <div className="empty-state">
                    <h3>Select a conversation</h3>
                    <p>Choose a conversation from the sidebar to view and manage messages.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeView === 'settings' && (
          <div className="settings-layout">
            <UserSettingsComponent userId={userId} />
          </div>
        )}
      </div>
    </div>
  );
};

export default CrudInterface;