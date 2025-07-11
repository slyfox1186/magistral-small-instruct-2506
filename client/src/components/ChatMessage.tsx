import React, { useState, useCallback } from 'react';
import { AppMessage } from '../utils/messageUtils';
import { COPY_FEEDBACK_TIMEOUT } from '../constants';

// Import the markdown renderer directly to avoid lazy loading issues
import MarkdownItRenderer from './MarkdownItRenderer';

const CopyIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
    <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
  </svg>
);

interface ChatMessageProps {
  message: AppMessage;
  isStreaming?: boolean;
  streamingContent?: string;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  isStreaming = false,
  streamingContent = '',
}) => {
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [copyFeedbackTimer, setCopyFeedbackTimer] = useState<ReturnType<typeof setTimeout> | null>(null);

  const handleCopy = useCallback(async () => {
    const textToCopy = message.role === 'assistant' && isStreaming 
      ? streamingContent 
      : message.content;

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopyFeedback('Copied!');
      
      // Clear any existing timer
      if (copyFeedbackTimer) {
        clearTimeout(copyFeedbackTimer);
      }
      
      // Set new timer
      const timer = setTimeout(() => {
        setCopyFeedback(null);
        setCopyFeedbackTimer(null);
      }, COPY_FEEDBACK_TIMEOUT);
      
      setCopyFeedbackTimer(timer);
    } catch (err) {
      console.error('Failed to copy text:', err);
      setCopyFeedback('Failed to copy');
      
      const timer = setTimeout(() => {
        setCopyFeedback(null);
        setCopyFeedbackTimer(null);
      }, COPY_FEEDBACK_TIMEOUT);
      
      setCopyFeedbackTimer(timer);
    }
  }, [message, isStreaming, streamingContent, copyFeedbackTimer]);

  // Cleanup timer on unmount
  React.useEffect(() => {
    return () => {
      if (copyFeedbackTimer) {
        clearTimeout(copyFeedbackTimer);
      }
    };
  }, [copyFeedbackTimer]);

  const renderContent = () => {
    const content = message.role === 'assistant' && isStreaming 
      ? streamingContent 
      : message.content;

    if (message.role === 'assistant') {
      return <MarkdownItRenderer markdown={content} />;
    }

    // For user messages, also use markdown rendering to maintain consistency
    return <MarkdownItRenderer markdown={content} />;
  };

  return (
    <div className={`message ${message.role}`}>
      <div className="message-header">
        <span className="message-role">
          {message.role === 'user' ? 'You' : 'Assistant'}
        </span>
        <span className="message-timestamp">
          {new Date().toLocaleTimeString()}
        </span>
      </div>
      <div className="message-content">
        {renderContent()}
      </div>
      {message.role === 'assistant' && (
        <button
          className="copy-button"
          onClick={handleCopy}
          title="Copy message"
          aria-label="Copy message"
        >
          {copyFeedback || <CopyIcon />}
        </button>
      )}
    </div>
  );
};