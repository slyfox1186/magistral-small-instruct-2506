import React, { useState, useCallback, useRef, useEffect } from 'react';

const SendIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="22" y1="2" x2="11" y2="13"></line>
    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
  </svg>
);

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isStreaming: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isStreaming,
  disabled = false,
  placeholder = "Type your message here...",
}) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const newHeight = Math.min(textarea.scrollHeight, 300);
      textarea.style.height = `${newHeight}px`;
    }
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [input, adjustTextareaHeight]);

  const handleSubmit = useCallback((e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (input.trim() && !isStreaming && !disabled) {
      onSendMessage(input.trim());
      setInput('');
      
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [input, isStreaming, disabled, onSendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter without shift
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  }, []);

  return (
    <form onSubmit={handleSubmit} className="chat-input-form">
      <div className="chat-input-container">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isStreaming || disabled}
          className="chat-input"
          rows={1}
          aria-label="Chat message input"
        />
        <button
          type="submit"
          disabled={!input.trim() || isStreaming || disabled}
          className="send-button"
          aria-label="Send message"
        >
          <SendIcon />
        </button>
      </div>
      {isStreaming && (
        <div className="streaming-indicator">
          AI is thinking...
        </div>
      )}
    </form>
  );
};