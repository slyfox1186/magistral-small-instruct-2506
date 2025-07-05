import { useState, useCallback, useRef, useEffect } from 'react';
import { streamChatResponse } from '../api/chat';
import { logger, performanceMonitor } from '../utils';
import {
  AppMessage,
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  formatMessagesForAPI,
} from '../utils/messageUtils';
import {
  isTokenChunk,
  isDoneChunk,
  isErrorChunk,
  safeGetFromLocalStorage,
  isAppMessageArray,
} from '../utils/typeGuards';
import { resetTokenBuffer } from '../utils/markdownUtils';

interface UseChatOptions {
  sessionId: string;
  onStreamingStateChange?: (isStreaming: boolean) => void;
  onError?: (error: string) => void;
}

export function useChat({
  sessionId,
  onStreamingStateChange,
  onError,
}: UseChatOptions) {
  // Load messages from localStorage on initial render
  const [messages, setMessages] = useState<AppMessage[]>(() => {
    const savedMessages = safeGetFromLocalStorage('chat_messages');
    if (savedMessages && isAppMessageArray(savedMessages)) {
      logger.info(`Loaded ${savedMessages.length} messages from local storage`);
      return savedMessages;
    }
    return [];
  });

  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  
  const currentRequestRef = useRef<AbortController | null>(null);
  const tokenBufferRef = useRef<string[]>([]);
  const isProcessingRef = useRef(false);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem('chat_messages', JSON.stringify(messages));
      } catch (error) {
        logger.error('Failed to save messages to localStorage:', error);
      }
    }
  }, [messages]);

  // Update streaming state
  useEffect(() => {
    onStreamingStateChange?.(isStreaming);
  }, [isStreaming, onStreamingStateChange]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    localStorage.removeItem('chat_messages');
    logger.info('Chat messages cleared');
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isStreaming) {
      return;
    }

    // Cancel any ongoing request
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
    }

    const userMessage = createUserMessage(content);
    const apiMessages = formatMessagesForAPI([...messages, userMessage]);

    setMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);
    setStreamingContent('');
    tokenBufferRef.current = [];
    resetTokenBuffer();

    const assistantMessage = createAssistantMessage('');
    const assistantMessageId = assistantMessage.id;
    setMessages(prev => [...prev, assistantMessage]);

    const eventTarget = new EventTarget();
    const abortController = new AbortController();
    currentRequestRef.current = abortController;

    try {
      const startTime = performance.now();
      
      // Handle token chunks
      eventTarget.addEventListener('token', ((event: CustomEvent) => {
        if (abortController.signal.aborted) return;
        
        const chunk = event.detail;
        if (isTokenChunk(chunk)) {
          tokenBufferRef.current.push(chunk.content);
          const newContent = tokenBufferRef.current.join('');
          setStreamingContent(newContent);
          
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: newContent }
                : msg
            )
          );
        }
      }) as EventListener);

      // Handle completion
      eventTarget.addEventListener('done', ((event: CustomEvent) => {
        if (abortController.signal.aborted) return;
        
        const chunk = event.detail;
        if (isDoneChunk(chunk)) {
          const endTime = performance.now();
          const duration = endTime - startTime;
          
          performanceMonitor.recordMetric('chat_response_time', duration);
          logger.info(`Chat response completed in ${duration.toFixed(2)}ms`);
          
          setIsStreaming(false);
          setStreamingContent('');
          tokenBufferRef.current = [];
          currentRequestRef.current = null;
        }
      }) as EventListener);

      // Handle errors
      eventTarget.addEventListener('error', ((event: CustomEvent) => {
        const chunk = event.detail;
        if (isErrorChunk(chunk)) {
          logger.error('Streaming error:', chunk.error);
          onError?.(chunk.error);
          
          const errorMessage = createErrorMessage(chunk.error);
          setMessages(prev => [...prev, errorMessage]);
          
          setIsStreaming(false);
          setStreamingContent('');
          tokenBufferRef.current = [];
          currentRequestRef.current = null;
        }
      }) as EventListener);

      // Start streaming
      await streamChatResponse(apiMessages, sessionId, eventTarget, abortController);
      
    } catch (error) {
      logger.error('Chat error:', error);
      const errorText = error instanceof Error ? error.message : 'Unknown error occurred';
      onError?.(errorText);
      
      const errorMessage = createErrorMessage(errorText);
      setMessages(prev => [...prev, errorMessage]);
      
      setIsStreaming(false);
      setStreamingContent('');
      tokenBufferRef.current = [];
      currentRequestRef.current = null;
    }
  }, [messages, isStreaming, sessionId, onError]);

  const stopStreaming = useCallback(() => {
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
      setIsStreaming(false);
      setStreamingContent('');
      tokenBufferRef.current = [];
      logger.info('Streaming stopped by user');
    }
  }, []);

  return {
    messages,
    isStreaming,
    streamingContent,
    sendMessage,
    stopStreaming,
    clearMessages,
  };
}