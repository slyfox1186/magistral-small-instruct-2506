import React, { useRef, useEffect } from 'react';
import { TransitionGroup, CSSTransition } from 'react-transition-group';
import { AppMessage } from '../utils/messageUtils';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';

const ThinkingDots = () => (
  <div className="thinking-dots">
    <span></span>
    <span></span>
    <span></span>
  </div>
);

interface ChatContainerProps {
  messages: AppMessage[];
  isStreaming: boolean;
  streamingContent: string;
  onSendMessage: (message: string) => void;
  onClearMessages: () => void;
  sessionId: string;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  messages,
  isStreaming,
  streamingContent,
  onSendMessage,
  onClearMessages,
  sessionId,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  // Check if user has scrolled up
  const isUserScrolledUp = () => {
    const container = chatContainerRef.current;
    if (!container) return false;
    
    const threshold = 100;
    const position = container.scrollTop + container.clientHeight;
    const height = container.scrollHeight;
    
    return height - position > threshold;
  };

  // Only auto-scroll if user hasn't manually scrolled up
  useEffect(() => {
    if (!isUserScrolledUp()) {
      scrollToBottom();
    }
  }, [streamingContent]);

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Neural Consciousness Chat</h1>
        <div className="chat-controls">
          <span className="session-id">Session: {sessionId.substring(0, 8)}...</span>
          <button
            onClick={onClearMessages}
            className="clear-button"
            disabled={messages.length === 0 || isStreaming}
            aria-label="Clear chat history"
          >
            Clear Chat
          </button>
        </div>
      </div>

      <div className="messages-container" ref={chatContainerRef}>
        <TransitionGroup>
          {messages.map((message) => (
            <CSSTransition
              key={message.id}
              timeout={300}
              classNames="message-transition"
            >
              <ChatMessage
                message={message}
                isStreaming={isStreaming && message.id === messages[messages.length - 1]?.id}
                streamingContent={streamingContent}
              />
            </CSSTransition>
          ))}
        </TransitionGroup>
        
        {isStreaming && messages.length > 0 && 
         messages[messages.length - 1].role === 'assistant' && 
         !streamingContent && (
          <div className="thinking-indicator">
            <ThinkingDots />
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <ChatInput
        onSendMessage={onSendMessage}
        isStreaming={isStreaming}
      />
    </div>
  );
};