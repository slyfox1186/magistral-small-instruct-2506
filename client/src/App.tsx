import React, { useState, useCallback, useRef, useEffect, lazy, Suspense } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';
import { MEMORY_SESSION_ID } from './api/chat';
import { CacheManager } from './utils/cacheUtils';
import './styles/App.css';
import './styles/StatusAnimations.css';
// Import status system components
import { AlertProvider, useAlert } from './contexts/AlertContext';
import { AlertContainer } from './components/alerts/AlertContainer';
import { StatusIndicator } from './components/status/StatusIndicator';
import { Status } from './types/status';
// Import configuration
import config from './utils/config';
// Import constants
import { MAX_STREAMING_BUFFER_SIZE, COPY_FEEDBACK_TIMEOUT } from './constants';
// Import message utilities
import {
  AppMessage,
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  formatMessagesForAPI,
} from './utils/messageUtils';
// Import utilities
import { logger, networkService, performanceMonitor } from './utils';
// Import type guards for safe JSON parsing
import {
  isAppMessageArray,
  isTokenChunk,
  isDoneChunk,
  isErrorChunk,
  safeGetFromLocalStorage,
  safeJsonParse,
} from './utils/typeGuards';
// Import markdown utilities including resetTokenBuffer
import { resetTokenBuffer } from './utils/markdownUtils';

// Lazy load the new markdown-it component with improved dynamic import
const MarkdownItRenderer = lazy(() =>
  import('./components/MarkdownItRenderer')
    .then((module) => {
      logger.debug('MarkdownItRenderer component loaded');
      return { default: module.default };
    })
    .catch((error) => {
      logger.error('Failed to load MarkdownItRenderer:', error);
      // Return a fallback component instead of throwing
      return {
        default: ({ markdown }: { markdown: string }) => (
          <div className="markdown-fallback">
            <pre>{markdown}</pre>
          </div>
        )
      };
    })
);

// Fix timeout type compatibility
type TimeoutType = ReturnType<typeof setTimeout>;

// Safe wrapper component for MarkdownItRenderer
const SafeMarkdownRenderer: React.FC<{
  markdown: string;
  isStreaming?: boolean;
  className?: string;
}> = ({ markdown, isStreaming = false, className = '' }) => {
  return (
    <Suspense 
      fallback={
        <div className="markdown-loading">
          <StatusIndicator status="thinking" size="small" />
          <span>Loading markdown...</span>
        </div>
      }
    >
      <MarkdownItRenderer 
        markdown={markdown} 
        isStreaming={isStreaming} 
        className={className} 
      />
    </Suspense>
  );
};

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


function App() {
  // Get alert functions
  const alert = useAlert();
  // Use fixed session ID from environment or generated ID
  const sessionId = MEMORY_SESSION_ID;

  // State declarations (must be before useEffects that reference them)
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<AppMessage[]>(() => {
    const savedMessages = safeGetFromLocalStorage(
      `chat_history_${MEMORY_SESSION_ID}`,
      isAppMessageArray,
      []
    );

    // Add nodeRef to each message since it can't be serialized
    return savedMessages.map((msg) => ({
      ...msg,
      nodeRef: React.createRef(),
    }));
  });
  const [inputMessage, setInputMessage] = useState('');
  const [currentResponseRaw, setCurrentResponseRaw] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [currentFinalResponse, setCurrentFinalResponse] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [streamingComplete, setStreamingComplete] = useState(false);
  const [currentMessageStatus, setCurrentMessageStatus] = useState<Status>('thinking');

  // Refs for managing state during streaming
  const accumulatedResponseRef = useRef<string>('');
  const safetyTimerRef = useRef<TimeoutType | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamingCompleteRef = useRef<boolean>(false);
  const isAccumulatingMessageRef = useRef<boolean>(false);
  const isFinalCompletionRef = useRef<boolean>(false);
  const currentAbortControllerRef = useRef<AbortController | null>(null);

  // Initialize cache management for development
  useEffect(() => {
    if (import.meta.env.DEV) {
      CacheManager.clearDevelopmentCaches();
      console.log('🔄 App initialized with fresh cache');
    }
    
    // Check if app was previously in a streaming state and reset if needed
    const checkPreviousStreamingState = () => {
      try {
        const lastCleanup = localStorage.getItem('app_cleanup_timestamp');
        const streamingState = localStorage.getItem('streaming_state');
        
        if (streamingState && !lastCleanup) {
          console.log('🔄 Detected previous streaming state, resetting...');
          // Force reset all streaming state
          setIsLoading(false);
          setCurrentResponseRaw('');
          setCurrentThinking('');
          setCurrentFinalResponse('');
          setStreamingComplete(true);
          setCurrentMessageStatus('success');
          
          // Reset all refs
          accumulatedResponseRef.current = '';
          streamingCompleteRef.current = true;
          isAccumulatingMessageRef.current = false;
          isFinalCompletionRef.current = true;
          
          // Clear the streaming state
          localStorage.removeItem('streaming_state');
        }
        
        // Clear cleanup timestamp (it's served its purpose)
        if (lastCleanup) {
          localStorage.removeItem('app_cleanup_timestamp');
        }
      } catch (e) {
        console.warn('Failed to check previous streaming state:', e);
      }
    };
    
    checkPreviousStreamingState();
  }, []);

  // Handle browser refresh/close - stop streaming and reset state
  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      // Send stop signal to backend if currently streaming
      if (isLoading || !streamingCompleteRef.current) {
        try {
          const stopSignal = new Blob([JSON.stringify({ session_id: sessionId })], { type: 'application/json' });
          navigator.sendBeacon(`${config.API_URL}/stop-stream`, stopSignal);
          console.log('🛑 SENT STOP SIGNAL to backend');
        } catch (e) {
          console.warn('Failed to send stop signal:', e);
        }
      }
      
      // IMMEDIATELY abort any active streaming
      if (currentAbortControllerRef.current) {
        console.log('🔄 ABORTING STREAM: Page refresh detected');
        currentAbortControllerRef.current.abort();
        currentAbortControllerRef.current = null;
      }
      
      // Reset all streaming state immediately
      setIsLoading(false);
      setStreamingComplete(true);
      streamingCompleteRef.current = true;
      isAccumulatingMessageRef.current = false;
      isFinalCompletionRef.current = true;
      accumulatedResponseRef.current = '';
      setCurrentResponseRaw('');
      setCurrentMessageStatus('success');
      
      // Clear all timeouts immediately
      if (streamCompletionTimerRef.current) {
        clearTimeout(streamCompletionTimerRef.current);
        streamCompletionTimerRef.current = null;
      }
      if (safetyTimerRef.current) {
        clearTimeout(safetyTimerRef.current);
        safetyTimerRef.current = null;
      }
      if (checkBottomTimeoutRef.current) {
        clearTimeout(checkBottomTimeoutRef.current);
        checkBottomTimeoutRef.current = null;
      }
      if (pasteTimeoutRef.current) {
        clearTimeout(pasteTimeoutRef.current);
        pasteTimeoutRef.current = null;
      }
      
      // Reset all refs immediately (these work even during unload)
      accumulatedResponseRef.current = '';
      streamingCompleteRef.current = true;
      isAccumulatingMessageRef.current = false;
      isFinalCompletionRef.current = true;
      lastUserInteractionRef.current = 0;
      
      // Clear localStorage streaming state
      try {
        localStorage.removeItem('streaming_state');
        localStorage.setItem('app_cleanup_timestamp', Date.now().toString());
      } catch (e) {
        console.warn('Failed to clear streaming state from localStorage:', e);
      }
      
      console.log('🔄 Page unload cleanup completed');
    };

    // Also handle page visibility change for mobile/tab switching
    const handleVisibilityChange = () => {
      if (document.hidden && (isLoading || !streamingCompleteRef.current)) {
        console.log('🔄 Page hidden during streaming, cleaning up...');
        
        // Cancel ongoing requests
        if (currentAbortControllerRef.current) {
          currentAbortControllerRef.current.abort();
          currentAbortControllerRef.current = null;
        }
        
        // Reset streaming flags
        streamingCompleteRef.current = true;
        isAccumulatingMessageRef.current = false;
        isFinalCompletionRef.current = true;
        accumulatedResponseRef.current = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isLoading]); // Add isLoading dependency so it updates

  // Additional refs not declared above
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pasteTimeoutRef = useRef<TimeoutType | null>(null);
  const inactivityTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamingRef = useRef<HTMLDivElement>(null);
  const streamCompletionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // State for copy feedback
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  // Helper function to parse thinking vs response content
  const parseThinkingContent = useCallback((fullContent: string) => {
    if (fullContent.includes('<think>')) {
      const thinkStartMatch = fullContent.match(/<think>([\s\S]*)/i);

      if (thinkStartMatch) {
        const thinkContent = thinkStartMatch[1];
        const thinkCompleteMatch = thinkContent.match(/^([\s\S]*?)<\/think>([\s\S]*)/i);

        if (thinkCompleteMatch) {
          // Complete thinking section found
          const thinking = thinkCompleteMatch[1].trim();
          const response = thinkCompleteMatch[2].trim();
          setCurrentThinking(thinking);
          setCurrentFinalResponse(response);
        } else {
          // Partial thinking - still streaming thinking content
          setCurrentThinking(thinkContent.trim());
          setCurrentFinalResponse('');
        }
      }
    } else {
      // No thinking tags - treat as regular response
      setCurrentThinking('');
      setCurrentFinalResponse(fullContent);
    }
  }, []);

  // Helper function to copy message text
  const handleCopyMessage = useCallback(async (messageContent: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(messageContent);
      setCopiedMessageId(messageId);
      // Clear the feedback after configured timeout
      setTimeout(() => setCopiedMessageId(null), COPY_FEEDBACK_TIMEOUT);
    } catch (error) {
      logger.error('Failed to copy message:', error);
    }
  }, []);

  // Helper function to focus the input field
  const focusInput = useCallback(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [inputRef]);

  // Helper function to manage streaming buffer size and prevent memory exhaustion
  const manageBufferSize = useCallback((currentBuffer: string): string => {
    if (currentBuffer.length > MAX_STREAMING_BUFFER_SIZE) {
      // Keep only the last portion that fits within the limit
      const trimmedBuffer = currentBuffer.slice(-MAX_STREAMING_BUFFER_SIZE);
      logger.info(
        `🔧 Buffer trimmed from ${currentBuffer.length} to ${trimmedBuffer.length} bytes (${(MAX_STREAMING_BUFFER_SIZE / 1024 / 1024).toFixed(1)}MB limit)`
      );
      return trimmedBuffer;
    }
    return currentBuffer;
  }, []);

  // Function to clear chat history
  const clearChatHistory = useCallback(() => {
    if (isLoading) {
      logger.warn('Cannot clear chat history while loading a response');
      return;
    }

    logger.info(`🗑️ Clearing chat history (${messages.length} messages)`);

    // Clear messages from state
    setMessages([]);

    // Clear from localStorage
    try {
      localStorage.removeItem(`chat_history_${MEMORY_SESSION_ID}`);
      logger.info('✅ Chat history cleared from both state and localStorage');
    } catch (error) {
      logger.error('Failed to clear chat history from localStorage:', error);
    }

    focusInput(); // Focus input after clearing history
  }, [isLoading, focusInput, messages.length]);

  // Utility function to clear additional state beyond basic chat history
  const clearAdditionalState = useCallback(() => {
    setCurrentResponseRaw('');
    setCurrentThinking('');
    setCurrentFinalResponse('');
    accumulatedResponseRef.current = '';
    setStreamingComplete(false);
    setError(null);

    // DO NOT generate new session ID - keep using the fixed session ID
    logger.info(`Keeping fixed session ID: ${sessionId}`);
  }, [sessionId]);

  // Function to clear both assistant memory AND chat history
  const [clearingMemory, setClearingMemory] = useState(false);
  const handleClearMemoryAndHistory = useCallback(async () => {
    if (clearingMemory) {
      logger.warn('Already clearing memory');
      return;
    }

    setClearingMemory(true);
    setError(null); // Clear previous errors before starting

    try {
      logger.info(`🧹 Attempting to clear memories via ${config.API_URL}/clear-vital-memories`);

      const response = await fetch(`${config.API_URL}/clear-vital-memories`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      logger.info(`📡 Clear memory API response: ${response.status} ${response.statusText}`);

      if (response.ok) {
        try {
          const data = await response.json();
          logger.info(`✅ Successfully cleared ${data.deleted_count} memories from the assistant`);

          // On successful API call, now clear the local state
          clearChatHistory();
          clearAdditionalState();

          // Show success message using alert system
          alert.success(`Successfully cleared ${data.deleted_count} memories`);
        } catch (jsonError) {
          logger.error('Failed to parse JSON response:', jsonError);
          setError('Memory cleared but response format was unexpected');
        }
      } else {
        const errorText = await response.text();
        logger.error(`❌ Failed to clear assistant memory: ${errorText}`);
        setError(`Failed to clear memory: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      logger.error('Error clearing assistant memory:', error);
      setError(`Error clearing memory: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setClearingMemory(false);
    }
  }, [clearingMemory, clearChatHistory, clearAdditionalState]);

  // Refs for direct DOM manipulation for streaming
  const streamingContentRef = useRef<HTMLDivElement | null>(null);

  // Add timestamp tracking for token arrivals
  const lastTokenTimestampRef = useRef<number>(Date.now());

  // --- 🚀 ULTRATHINK FIXED SCROLL CONTROL ---
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const lastUserInteractionRef = useRef<number>(0);
  const checkBottomTimeoutRef = useRef<TimeoutType | null>(null);
  // --- End of ULTRATHINK scroll control ---

  // Focus input on component mount and set initial textarea height
  useEffect(() => {
    focusInput();
    // Set initial textarea height using the same calculation as handleTextareaInput
    if (inputRef.current) {
      const textarea = inputRef.current;
      const defaultHeightRem = config.DEFAULT_TEXTAREA_HEIGHT_REM;

      // Temporarily set to default height to measure content
      textarea.style.height = `${defaultHeightRem}rem`;

      // Measure scroll height for empty textarea
      const contentScrollHeightPx = textarea.scrollHeight;
      const contentHeightRem = contentScrollHeightPx / config.FONT_SIZE_BASE_PX;

      // Use the calculated height, but ensure it's at least the default
      const finalHeightRem = Math.max(defaultHeightRem, contentHeightRem);
      textarea.style.height = `${finalHeightRem}rem`;
      textarea.style.overflowY = 'hidden'; // Hide scrollbar initially
    }
  }, [focusInput]);

  // Simplified effect to handle stream completion
  useEffect(() => {
    if (streamingComplete && streamingCompleteRef.current === false) {
      // Set flag to prevent duplicate processing
      streamingCompleteRef.current = true;

      // Clear any existing timer
      if (streamCompletionTimerRef.current) {
        clearTimeout(streamCompletionTimerRef.current);
      }

      // Set a short debounce timeout
      streamCompletionTimerRef.current = setTimeout(() => {
        // Only finalize and reset if this is a confirmed final completion
        // or if there's been enough time with no new tokens
        if (isFinalCompletionRef.current || Date.now() - lastTokenTimestampRef.current > 2000) {
          // Process the complete message
          if (accumulatedResponseRef.current && accumulatedResponseRef.current.length > 0) {
            // Finished accumulating this message
            isAccumulatingMessageRef.current = false;

            // Create a new assistant message with the COMPLETE accumulated response
            const newMessage = createAssistantMessage(accumulatedResponseRef.current);

            // Add the complete message without flushSync to prevent layout shifts
            setMessages((prevMessages) => [...prevMessages, newMessage]);

            // Show success status briefly
            setCurrentMessageStatus('success');
            alert.success('Message received successfully', 3000);

            // Use requestAnimationFrame to ensure DOM has updated before resetting state
            requestAnimationFrame(() => {
              setIsLoading(false);
              setCurrentResponseRaw('');
              setStreamingComplete(false);
              streamingCompleteRef.current = false;
              isFinalCompletionRef.current = false;
              
              // Clear streaming state from localStorage (successful completion)
              try {
                localStorage.removeItem('streaming_state');
              } catch (e) {
                console.warn('Failed to clear streaming state on completion:', e);
              }
            });

            // Clean up after DOM is guaranteed to be updated
            accumulatedResponseRef.current = '';

            // Reset the markdown token buffer when completing a message
            resetTokenBuffer();

            // Clear safety timer
            if (safetyTimerRef.current) clearTimeout(safetyTimerRef.current);
            safetyTimerRef.current = null;
          } else {
            // No content - just clean up
            setIsLoading(false);
            setCurrentResponseRaw('');
            setStreamingComplete(false);
            streamingCompleteRef.current = false;
            isFinalCompletionRef.current = false;

            accumulatedResponseRef.current = '';

            // Reset the markdown token buffer even when there's no content
            resetTokenBuffer();

            if (safetyTimerRef.current) clearTimeout(safetyTimerRef.current);
            safetyTimerRef.current = null;
          }
        } else {
          // This might be a temporary pause - don't reset accumulated content
          // Allow streaming to continue if more tokens arrive
          streamingCompleteRef.current = false;
          setStreamingComplete(false);
        }
      }, 50); // Short delay is sufficient
    }

    return () => {
      if (streamCompletionTimerRef.current) {
        clearTimeout(streamCompletionTimerRef.current);
      }
    };
  }, [streamingComplete]);

  // --- 🚀 ULTRATHINK FIXED SCROLL HANDLER ---
  // Detect user scroll interaction with streaming-aware autoscroll logic
  const handleUserScrollInteraction = useCallback(() => {
    lastUserInteractionRef.current = Date.now();

    const container = scrollContainerRef.current;
    if (!container) return;

    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // Make it easier for users to break out of auto-scroll
    // Lower threshold for more responsive scroll control
    const isNearBottom = distanceFromBottom < 30; // Reduced threshold for easier break-out

    if (!isNearBottom) {
      // User has scrolled away from bottom - immediately disable autoscroll
      if (autoScrollEnabled === true) {
        setAutoScrollEnabled(false);
        setShowScrollButton(true);
        
        // Clear any existing timeout that might re-enable autoscroll
        if (checkBottomTimeoutRef.current) {
          clearTimeout(checkBottomTimeoutRef.current);
          checkBottomTimeoutRef.current = null;
        }
      }
      return;
    }

    // User is near bottom - re-enable autoscroll if it was disabled
    if (autoScrollEnabled === false) {
      // Use a very short delay to prevent rapid toggling
      if (checkBottomTimeoutRef.current) {
        clearTimeout(checkBottomTimeoutRef.current);
      }
      
      checkBottomTimeoutRef.current = setTimeout(() => {
        // Double-check we're still at bottom after delay
        const currentDistanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
        if (currentDistanceFromBottom < 30) {
          setAutoScrollEnabled(true);
          setShowScrollButton(false);
        }
        checkBottomTimeoutRef.current = null;
      }, 150); // Short delay to prevent flicker
    }
  }, [autoScrollEnabled]);
  // --- End of ULTRATHINK scroll handler ---

  // 🚀 ULTRATHINK Scroll to bottom button handler
  const scrollToBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (container) {
      // Clear any pending timeout
      if (checkBottomTimeoutRef.current) {
        clearTimeout(checkBottomTimeoutRef.current);
        checkBottomTimeoutRef.current = null;
      }

      // Immediately re-enable autoscroll and hide button
      setAutoScrollEnabled(true);
      setShowScrollButton(false);

      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, []);

  // --- 🚀 ULTRATHINK FIXED EVENT LISTENERS ---
  // Comprehensive user interaction detection
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    // Add listeners for ALL possible user scroll interactions
    container.addEventListener('wheel', handleUserScrollInteraction, { passive: true });
    container.addEventListener('touchstart', handleUserScrollInteraction, { passive: true });
    container.addEventListener('touchmove', handleUserScrollInteraction, { passive: true });
    container.addEventListener('keydown', handleUserScrollInteraction, { passive: true });
    container.addEventListener('mousedown', handleUserScrollInteraction, { passive: true });

    return () => {
      container.removeEventListener('wheel', handleUserScrollInteraction);
      container.removeEventListener('touchstart', handleUserScrollInteraction);
      container.removeEventListener('touchmove', handleUserScrollInteraction);
      container.removeEventListener('keydown', handleUserScrollInteraction);
      container.removeEventListener('mousedown', handleUserScrollInteraction);

      // Cleanup timeout
      if (checkBottomTimeoutRef.current) {
        clearTimeout(checkBottomTimeoutRef.current);
        checkBottomTimeoutRef.current = null;
      }
    };
  }, [handleUserScrollInteraction]);

  // Single simplified autoscroll effect with improved streaming detection
  useEffect(() => {
    if (autoScrollEnabled === false || scrollContainerRef.current === null) return;

    const container = scrollContainerRef.current;

    // Check if user has manually scrolled away from bottom recently
    const timeSinceLastInteraction = Date.now() - lastUserInteractionRef.current;
    const hasRecentUserInteraction = timeSinceLastInteraction < 500; // Reduced to 0.5 seconds for more responsive re-engagement

    // CRITICAL FIX: Always autoscroll when actively streaming
    // Detect streaming by either loading state or current response content
    const isActivelyStreaming = isLoading || currentResponseRaw.length > 0;

    // Skip autoscroll only if user has interacted recently AND we're not actively streaming
    if (hasRecentUserInteraction && !isActivelyStreaming) {
      return; // Don't autoscroll during user interaction when not streaming
    }

    // Allow autoscroll when:
    // 1. Actively streaming (isLoading || currentResponseRaw) - ALWAYS (overrides user interaction)
    // 2. Messages array changes (new message added)
    // 3. Not actively streaming but no recent user interaction

    // Only autoscroll if explicitly enabled
    const scrollToBottom = () => {
      // Always use instant scroll to prevent bouncing
      container.scrollTop = container.scrollHeight;
    };

    // Use requestAnimationFrame to avoid conflicts with user scrolling
    const rafId = requestAnimationFrame(scrollToBottom);
    return () => cancelAnimationFrame(rafId);
  }, [messages, currentResponseRaw, autoScrollEnabled, isLoading]);
  // --- End of ULTRATHINK autoscroll ---

  // Helper function to reset textarea height to default
  const resetTextareaHeight = useCallback(() => {
    if (inputRef.current) {
      const textarea = inputRef.current;
      const defaultHeightRem = config.DEFAULT_TEXTAREA_HEIGHT_REM;

      // Force reset to default height (don't calculate based on content)
      textarea.style.height = `${defaultHeightRem}rem`;
      textarea.style.overflowY = 'hidden'; // Hide scrollbar on reset
    }
  }, []); // Config values are stable, no dependencies needed

  useEffect(() => {
    if (isLoading === false) {
      focusInput(); // Focus input when loading state changes
      resetTextareaHeight(); // Reset height when loading state changes
    }
  }, [isLoading, focusInput, resetTextareaHeight]);

  useEffect(() => {
    if (safetyTimerRef.current) clearTimeout(safetyTimerRef.current);
    safetyTimerRef.current = null;

    if (isLoading) {
      logger.debug('(Safety Timer) Setting safety timeout');
      safetyTimerRef.current = setTimeout(() => {
        logger.debug('(Safety Timer) Safety timeout triggered: Forcing reset');

        // Check if there's content but streaming got stuck
        if (accumulatedResponseRef.current.length > 0) {
          logger.debug('(Safety Timer) Content exists but streaming stuck. Completing response.');
          setStreamingComplete(true);
        } else {
          setIsLoading(false);
          setStreamingComplete(false);
          setCurrentResponseRaw('');
          accumulatedResponseRef.current = '';
        }

        logger.debug('(Safety Timer) UI state reset');
      }, config.SAFETY_TIMEOUT_MS);
    }
    return () => {
      if (safetyTimerRef.current) clearTimeout(safetyTimerRef.current);
    };
  }, [isLoading]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      try {
        // Create a version of messages without nodeRef (can't be serialized)
        const serializableMessages = messages.map(({ nodeRef: _nodeRef, ...rest }) => rest);
        localStorage.setItem(
          `chat_history_${MEMORY_SESSION_ID}`,
          JSON.stringify(serializableMessages)
        );
        logger.debug(`Saved ${messages.length} messages to localStorage`);
      } catch (error) {
        logger.error('Failed to save chat history to localStorage:', error);
      }
    }
  }, [messages]);

  useEffect(() => {
    logger.debug(`State update - isLoading: ${isLoading}, streamingComplete: ${streamingComplete}`);
  }, [isLoading, streamingComplete]);

  // Cleanup all timers and abort controllers on component unmount
  useEffect(() => {
    return () => {
      // Cleanup all timers
      if (pasteTimeoutRef.current) {
        clearTimeout(pasteTimeoutRef.current);
        pasteTimeoutRef.current = null;
      }
      if (checkBottomTimeoutRef.current) {
        clearTimeout(checkBottomTimeoutRef.current);
        checkBottomTimeoutRef.current = null;
      }
      if (safetyTimerRef.current) {
        clearTimeout(safetyTimerRef.current);
        safetyTimerRef.current = null;
      }
      if (streamCompletionTimerRef.current) {
        clearTimeout(streamCompletionTimerRef.current);
        streamCompletionTimerRef.current = null;
      }
      if (inactivityTimerRef.current) {
        clearInterval(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }
      // Abort any ongoing requests
      if (currentAbortControllerRef.current) {
        currentAbortControllerRef.current.abort();
        currentAbortControllerRef.current = null;
      }
    };
  }, []);

  const handleTextareaInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.currentTarget.value);

    const textarea = e.currentTarget;
    const defaultHeightRem = config.DEFAULT_TEXTAREA_HEIGHT_REM;
    const maxHeightRem = config.MAX_TEXTAREA_HEIGHT_REM;

    // Store current height before measuring
    const currentHeight = textarea.style.height;

    // Use auto to get natural height, then measure
    textarea.style.height = 'auto';
    const contentScrollHeightPx = textarea.scrollHeight;
    const contentHeightRem = contentScrollHeightPx / config.FONT_SIZE_BASE_PX;

    // Calculate new height with bounds
    const newHeightRem = Math.max(defaultHeightRem, Math.min(contentHeightRem, maxHeightRem));
    const newHeight = `${newHeightRem}rem`;

    // Only update if height actually changed significantly (avoid micro-adjustments)
    const currentHeightNum = parseFloat(currentHeight) || defaultHeightRem;
    const heightDifference = Math.abs(newHeightRem - currentHeightNum);

    if (heightDifference > 0.1) {
      // Only change if difference is significant
      textarea.style.height = newHeight;
    } else {
      // Restore current height if no significant change needed
      textarea.style.height = currentHeight || `${defaultHeightRem}rem`;
    }

    // Show scrollbar only when content exceeds max height
    if (contentHeightRem > maxHeightRem) {
      textarea.style.overflowY = 'auto';
    } else {
      textarea.style.overflowY = 'hidden';
    }
  };

  // New function to handle paste events
  const handlePaste = useCallback((event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    event.preventDefault();
    const pastedText = event.clipboardData.getData('text/plain');
    let cleanedText = pastedText;

    // Remove a single trailing newline character if present
    if (cleanedText.endsWith('\n')) {
      cleanedText = cleanedText.substring(0, cleanedText.length - 1);
    }

    // Insert the cleaned text at the current cursor position or append
    const target = event.currentTarget;
    const start = target.selectionStart;
    const end = target.selectionEnd;

    const currentText = target.value;
    const newText = currentText.substring(0, start) + cleanedText + currentText.substring(end);

    setInputMessage(newText);

    // Clear any existing timeout before setting a new one
    if (pasteTimeoutRef.current) {
      clearTimeout(pasteTimeoutRef.current);
    }

    // Adjust textarea height after paste
    // Use a slight delay to ensure the new value is rendered before calculating height
    pasteTimeoutRef.current = setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.style.height = 'auto'; // Reset height
        const scrollHeightPx = inputRef.current.scrollHeight; // scrollHeight is in pixels
        const maxHeightRem = config.MAX_TEXTAREA_HEIGHT_REM;
        const scrollHeightRem = scrollHeightPx / config.FONT_SIZE_BASE_PX; // Convert to rem

        inputRef.current.style.height = `${Math.min(scrollHeightRem, maxHeightRem)}rem`;

        // Show scrollbar only when content exceeds max height
        if (scrollHeightRem > maxHeightRem) {
          inputRef.current.style.overflowY = 'auto';
        } else {
          inputRef.current.style.overflowY = 'hidden';
        }
      }
      pasteTimeoutRef.current = null; // Clear the ref after timeout completes
    }, 0);
  }, []); // Config values are stable, no dependencies needed

  const handleSend = useCallback(async () => {
    // Guard against empty messages or loading state
    if (inputMessage.trim() === '') {
      logger.warn('Attempted to send an empty message.');
      return;
    }
    if (isLoading) {
      logger.warn('Attempted to send while a message is already loading.');
      return;
    }

    performanceMonitor.startMark('messageSendTotal');
    setError(null); // Clear previous errors
    setIsLoading(true);
    setCurrentMessageStatus('thinking'); // Start with thinking status
    isAccumulatingMessageRef.current = false; // Reset accumulation flag
    isFinalCompletionRef.current = false; // Reset final completion flag
    resetTokenBuffer(); // Reset token buffer for new message
    setCurrentResponseRaw(''); // Clear any stale raw response
    accumulatedResponseRef.current = ''; // Clear accumulated response
    streamingCompleteRef.current = false; // Reset streaming complete flag

    // Track streaming state in localStorage for refresh detection
    try {
      localStorage.setItem('streaming_state', JSON.stringify({
        isStreaming: true,
        timestamp: Date.now(),
        sessionId: sessionId
      }));
    } catch (e) {
      console.warn('Failed to save streaming state to localStorage:', e);
    }

    // CRITICAL: Enable auto-scroll when starting a new message
    setAutoScrollEnabled(true);
    setShowScrollButton(false);

    const userMessageText = inputMessage.trim();

    // --- Chat Message Sending Logic ---
    const newUserMessage = createUserMessage(userMessageText);

    // Clear previous state
    setCurrentResponseRaw('');
    setCurrentThinking('');
    setCurrentFinalResponse('');
    setError(null);
    setStreamingComplete(false);
    accumulatedResponseRef.current = ''; // Reset accumulator at the start

    // Reset the markdown token buffer when starting a new chat
    resetTokenBuffer();

    // Add user message and set loading state
    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage('');

    // Reset textarea height immediately and again after DOM updates
    if (inputRef.current) {
      inputRef.current.style.height = `${config.DEFAULT_TEXTAREA_HEIGHT_REM}rem`;
      inputRef.current.style.overflowY = 'hidden';
    }

    setIsLoading(true);

    // Reset textarea height multiple times to ensure it sticks
    setTimeout(() => {
      resetTextareaHeight();
    }, 0);

    // Additional reset to ensure it works even after React updates
    setTimeout(() => {
      if (inputRef.current && inputRef.current.value === '') {
        inputRef.current.style.height = `${config.DEFAULT_TEXTAREA_HEIGHT_REM}rem`;
        inputRef.current.style.overflowY = 'hidden';
      }
    }, 10);

    // Set the accumulating flag to true
    isAccumulatingMessageRef.current = true;

    let controller: AbortController | null = null;

    try {
      // Start performance monitoring
      performanceMonitor.startMark('chatApiCall');

      // Create abort controller for cancellation
      controller = new AbortController(); // Initialize controller

      // Handle previous request cancellation
      if (currentAbortControllerRef.current) {
        logger.debug('(handleSend) Cancelling previous request');
        currentAbortControllerRef.current.abort();
      }
      currentAbortControllerRef.current = controller;

      // Use the networkService to fetch the stream with retries
      const response = await networkService.streamRequest('/api/chat-stream', {
        method: 'POST',
        body: {
          session_id: sessionId,
          messages: formatMessagesForAPI(messages, inputMessage),
        },
        signal: controller.signal, // Pass the abort controller signal
        retries: 2, // Add retries to handle transient network issues
        retryDelay: 500, // 500ms between retries
      });

      if (response.ok === false) {
        // NetworkService's streamRequest already threw an error for non-ok status,
        // but we keep this check as a safeguard or if the implementation changes.
        // The error thrown by streamRequest is likely more informative.
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();

      // Token counting removed - no longer needed
      let lastActivityTime = Date.now();
      const MAX_INACTIVITY_MS = 10000; // 10 seconds max without activity

      // Safety check timer - store in ref for proper cleanup
      inactivityTimerRef.current = setInterval(() => {
        const now = Date.now();
        const inactiveTime = now - lastActivityTime;

        if (inactiveTime > MAX_INACTIVITY_MS) {
          logger.warn(`Stream inactive for ${inactiveTime}ms, forcing completion`);
          if (inactivityTimerRef.current) {
            clearInterval(inactivityTimerRef.current);
            inactivityTimerRef.current = null;
          }
          if (!streamingComplete) {
            setStreamingComplete(true);
          }
        }
      }, 2000);

      try {
        // Main stream processing loop
        let isProcessing = true;
        while (isProcessing) {
          let text = '';
          try {
            const { done, value } = await reader.read();

            // Update activity timestamp on any data or done signal
            lastActivityTime = Date.now();

            if (done) {
              if (inactivityTimerRef.current) {
                clearInterval(inactivityTimerRef.current);
                inactivityTimerRef.current = null;
              }
              if (!streamingComplete) {
                logger.debug('Stream ended naturally (reader done).');
                setStreamingComplete(true);
              }
              break;
            }

            if (value) {
              text = decoder.decode(value, { stream: true });
            } else {
              continue;
            }
          } catch (readError) {
            // Catch errors specifically from reader.read()
            logger.error(`Error reading from stream: ${readError}`);
            if (inactivityTimerRef.current) {
              clearInterval(inactivityTimerRef.current);
              inactivityTimerRef.current = null;
            }
            if (!streamingComplete) {
              setStreamingComplete(true);
              setError('Error reading stream data.'); // Set specific error
            }
            break; // Exit the loop on read error
          }

          // Process SSE data format
          const lines = text.split('\n');
          for (const line of lines) {
            if (line.startsWith('data:')) {
              const jsonStr = line.substring(5).trim();

              // CRITICAL FIX: Only accept exact [DONE] message, not partial matches
              if (jsonStr === '[DONE]') {
                isFinalCompletionRef.current = true; // Mark as confirmed final completion
                setStreamingComplete(true);
                logger.debug('(handleSend) Received [DONE] signal.');
                // Break inner loop once DONE is received
                isProcessing = false;
                break;
              }
              // Only proceed if it's NOT the DONE signal
              else if (jsonStr) {
                // Check if jsonStr has content
                // Log problematic data for debugging
                if (jsonStr.length > 1000) {
                  logger.debug('Large SSE data received:', jsonStr.length, 'characters');
                }

                // Use safe JSON parsing with type guards
                const data = safeJsonParse(jsonStr, (_): _ is unknown => true, null);

                if (data === null) {
                  logger.error('Failed to parse JSON chunk:', jsonStr.substring(0, 100) + '...');
                  continue;
                }

                // Check if this is the completion signal
                if (isDoneChunk(data)) {
                  try {
                    isFinalCompletionRef.current = true;
                    setStreamingComplete(true);
                    logger.debug('(handleSend) Received JSON done signal.');
                    
                    // Clean up any pending state safely
                    setCurrentMessageStatus('success');
                    
                    // Final update to ensure response is rendered
                    if (accumulatedResponseRef.current) {
                      setCurrentResponseRaw(accumulatedResponseRef.current);
                    }
                    
                    break;
                  } catch (completionError) {
                    logger.error('Error during stream completion:', completionError);
                    setError('Error completing response');
                    break;
                  }
                }

                // Check if this is an error
                if (isErrorChunk(data)) {
                  logger.error('Stream error received:', data.error);
                  setError(data.error);
                  break;
                }

                // Check if this is a token chunk
                if (isTokenChunk(data)) {
                  // ***** POINT A: Token successfully parsed *****
                  logger.debug(
                    `[${Date.now()}] App.tsx: Received token:`,
                    JSON.stringify(data.token.text)
                  );
                  let tokenText = data.token.text;

                  // Fix potential backtick issues
                  if (tokenText.includes('````')) {
                    tokenText = tokenText.replace(/`{4,}(\w+)/g, '```$1');
                  }

                  // Basic filter for control markers - skip tokens with these
                  if (
                    ['<eos>', '<end_of_turn>', '[DONE]', 'EOF'].some((marker) =>
                      tokenText.includes(marker)
                    )
                  ) {
                    continue;
                  }

                  // Add token to accumulator
                  accumulatedResponseRef.current += tokenText;

                  // Update status to streaming once tokens start arriving
                  if (currentMessageStatus === 'thinking') {
                    setCurrentMessageStatus('streaming');
                  }

                  // Manage buffer size to prevent memory exhaustion
                  accumulatedResponseRef.current = manageBufferSize(accumulatedResponseRef.current);

                  // Update current response for rendering
                  setCurrentResponseRaw(accumulatedResponseRef.current);

                  // Parse thinking vs response content
                  parseThinkingContent(accumulatedResponseRef.current);

                  // Update last token timestamp
                  lastTokenTimestampRef.current = Date.now();

                  // Auto-scroll handled by unified autoscroll effect

                  // Token counter updated (for debugging purposes only)
                } else {
                  logger.warn('Received unknown data chunk:', data);
                }
              } // End of else if (jsonStr)
            } // End of if (line.startsWith('data:'))

            // If streamingComplete was set by [DONE] in the inner loop, break the outer loop too
            if (streamingComplete) break;
          } // End of for...of lines loop

          // If streamingComplete was set by [DONE], break the outer while loop
          if (streamingComplete) break;
        } // End of while loop
      } catch (streamProcessingError) {
        logger.error(`Error during stream processing loop: ${streamProcessingError}`);
        if (inactivityTimerRef.current) {
          clearInterval(inactivityTimerRef.current);
          inactivityTimerRef.current = null;
        }
        if (!streamingComplete) {
          setStreamingComplete(true);
          setError('Error processing stream.');
        }
      }

      // CRITICAL: Ensure final state update after loop completes
      setCurrentResponseRaw(accumulatedResponseRef.current);
    } catch (err) {
      // Handle aborted requests (like from page refresh)
      if (controller && controller.signal.aborted) {
        console.log('🔄 STREAM ABORTED: Cleaning up and resetting for new stream');
        
        // COMPLETE reset - ready for new stream
        setIsLoading(false);
        setStreamingComplete(true);
        streamingCompleteRef.current = true;
        isAccumulatingMessageRef.current = false;
        isFinalCompletionRef.current = true;
        setCurrentResponseRaw('');
        accumulatedResponseRef.current = '';
        setCurrentMessageStatus('success');
        
        // Clear streaming state from localStorage
        try {
          localStorage.removeItem('streaming_state');
        } catch (e) {
          console.warn('Failed to clear streaming state on abort:', e);
        }
        
        console.log('🔄 STREAM CLEANUP COMPLETE: Ready for new stream');
        return;
      }

      logger.error('Error during chat request:', err);

      // Regular error handling
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
      setCurrentMessageStatus('error');
      alert.error(`Chat error: ${errorMsg}`, 5000);

      // Clean up timers
      if (safetyTimerRef.current) clearTimeout(safetyTimerRef.current);
      if (inactivityTimerRef.current) {
        clearInterval(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }

      // Reset the markdown token buffer on error
      resetTokenBuffer();

      // Create and add error message
      const errorMessage = createErrorMessage(errorMsg);

      // Clean up state
      setIsLoading(false);
      setCurrentResponseRaw('');
      accumulatedResponseRef.current = '';
      setStreamingComplete(false);
      
      // Clear streaming state from localStorage
      try {
        localStorage.removeItem('streaming_state');
      } catch (e) {
        console.warn('Failed to clear streaming state on error:', e);
      }

      // Add error message to chat - ensuring it's added even on network error
      setMessages((prev) => {
        // Avoid adding duplicate user messages if the error happened before the AI responded
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.role === 'user' && lastMessage.content === inputMessage) {
          // User message already exists, just add the error
          return [...prev, errorMessage];
        } else {
          // Add the user message (if needed) and the error
          // This path might not be strictly necessary depending on when errors occur,
          // but adding the error message itself is key.
          // Consider if the user message needs re-adding here based on specific error points.
          // For simplicity, we assume the user message was added before the error.
          return [...prev, errorMessage];
        }
      });
    } finally {
      // Outer finally block
      // Clear timer if it hasn't been cleared already (e.g., error before loop)
      if (inactivityTimerRef.current) {
        clearInterval(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }
      // Ensure the loading state is reset even if errors occur before reader setup
      if (isLoading) {
        setIsLoading(false);
        
        // Clear streaming state from localStorage (final cleanup)
        try {
          localStorage.removeItem('streaming_state');
        } catch (e) {
          console.warn('Failed to clear streaming state in finally block:', e);
        }
      }
      // Reset abort controller ref if it belongs to this request
      if (currentAbortControllerRef.current === controller) {
        currentAbortControllerRef.current = null;
      }
      performanceMonitor.endMark('chatApiCall'); // End performance monitoring when streaming completes
      focusInput(); // Focus input after message sending/receiving is complete
    }
  }, [
    inputMessage,
    messages,
    sessionId,
    focusInput,
    resetTextareaHeight,
    isLoading,
    streamingComplete,
    parseThinkingContent,
    manageBufferSize,
    currentMessageStatus,
    alert,
  ]); // Essential dependencies only

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    } else if (e.key === 'Enter' && e.shiftKey) {
      // Allow Shift+Enter to create new line, then trigger resize
      setTimeout(() => {
        if (inputRef.current) {
          // Trigger the same resize logic as handleTextareaInput
          const textarea = inputRef.current;
          textarea.style.height = 'auto';

          const defaultHeightRem = config.DEFAULT_TEXTAREA_HEIGHT_REM;
          const contentScrollHeightPx = textarea.scrollHeight;
          const contentHeightRem = contentScrollHeightPx / config.FONT_SIZE_BASE_PX;

          if (textarea.value.trim() === '') {
            textarea.style.height = `${defaultHeightRem}rem`;
          } else {
            const newHeightRem = Math.min(contentHeightRem, config.MAX_TEXTAREA_HEIGHT_REM);
            textarea.style.height = `${newHeightRem}rem`;
          }
        }
      }, 0); // Execute after the newline is inserted
    }
  };

  const animationTimeout = config.ANIMATION_TIMEOUT_MS;

  // Effect to store the streaming content div reference
  useEffect(() => {
    if (isLoading && streamingRef.current) {
      // Find and store the content div when streaming starts
      const contentDiv = streamingRef.current.querySelector('.message-content');
      if (contentDiv instanceof HTMLDivElement) {
        streamingContentRef.current = contentDiv;
      }
    } else {
      streamingContentRef.current = null;
    }
  }, [isLoading]);

  // Cleanup effect to prevent memory leaks on component unmount
  useEffect(() => {
    return () => {
      // Cleanup any active timers
      if (inactivityTimerRef.current) {
        clearInterval(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }
      if (pasteTimeoutRef.current) {
        clearTimeout(pasteTimeoutRef.current);
        pasteTimeoutRef.current = null;
      }
      if (streamCompletionTimerRef.current) {
        clearTimeout(streamCompletionTimerRef.current);
        streamCompletionTimerRef.current = null;
      }
      // Cancel any active requests
      if (currentAbortControllerRef.current) {
        currentAbortControllerRef.current.abort();
        currentAbortControllerRef.current = null;
      }
    };
  }, []);

  return (
    <div className="app-layout-container">
      <div className="corner-controls">
        <button
          onClick={handleClearMemoryAndHistory}
          disabled={isLoading || clearingMemory}
          className="clear-memory-button"
          title="Clear assistant's memory AND chat history"
        >
          {clearingMemory ? 'Clearing...' : 'Clear Memory & History'}
        </button>
        <button
          onClick={clearChatHistory}
          disabled={isLoading || messages.length === 0}
          className="clear-history-button"
          title="Clear chat history"
        >
          Clear History
        </button>
      </div>
      <div className="chat-container">
        {/* Messages container with autoscroll control */}
        <div ref={scrollContainerRef} className="messages-container">
          <TransitionGroup component={null}>
            {messages.map((message) => (
              <CSSTransition
                key={message.id}
                nodeRef={message.nodeRef}
                timeout={animationTimeout}
                classNames="message-anim"
                mountOnEnter
                unmountOnExit={false} // Keep messages in DOM for exit animation (if any)
              >
                <div ref={message.nodeRef} className={`message ${message.role}`}>
                  <div className="message-role">{message.role === 'user' ? 'YOU' : 'AI'}</div>
                  <div className="message-bubble">
                    <button
                      className={`copy-button ${copiedMessageId === message.id ? 'copied' : ''}`}
                      onClick={() => handleCopyMessage(message.content ?? '', message.id)}
                      title={copiedMessageId === message.id ? 'Copied!' : 'Copy message'}
                      aria-label="Copy message text"
                    >
                      {copiedMessageId === message.id ? (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      ) : (
                        <CopyIcon />
                      )}
                    </button>
                    <div className="message-content">
                      {/* Render final messages using SafeMarkdownRenderer */}
                      <SafeMarkdownRenderer markdown={message.content ?? ''} />
                    </div>
                  </div>
                </div>
              </CSSTransition>
            ))}
          </TransitionGroup>

          {/* Streaming message container - Keep structure absolutely stable */}
          {isLoading && (
            <div
              ref={streamingRef}
              key="streaming-message-container-stable"
              className="message assistant"
            >
              <div className="message-role">AI</div>
              <div className="message-bubble">
                {/* New Status Indicator */}
                <StatusIndicator status={currentMessageStatus} size="small" className="message-status" />
                {currentResponseRaw && (
                  <button
                    className="copy-button"
                    onClick={() => handleCopyMessage(currentResponseRaw, 'streaming')}
                    title="Copy message"
                    aria-label="Copy message text"
                  >
                    <CopyIcon />
                  </button>
                )}
                <div className="message-content">
                  {/* Show status indicator when no content yet */}
                  {!currentResponseRaw ? (
                    <div className="loading">
                      <StatusIndicator status="thinking" size="medium" />
                    </div>
                  ) : (
                    <>
                      {/* Show thinking content if available */}
                      {currentThinking && (
                        <div className="assistant-thinking">
                          <div className="thinking-header">🤔 Thinking...</div>
                          <div className="thinking-content">
                            <SafeMarkdownRenderer
                              markdown={currentThinking}
                              isStreaming={true}
                              className="thinking-markdown"
                            />
                          </div>
                        </div>
                      )}
                      {/* Show final response if available */}
                      {currentFinalResponse && (
                        <div className="assistant-response">
                          <SafeMarkdownRenderer
                            markdown={currentFinalResponse}
                            isStreaming={true}
                            className="response-markdown"
                          />
                        </div>
                      )}
                      {/* Fallback to show raw content if no parsing happened */}
                      {!currentThinking && !currentFinalResponse && (
                        <SafeMarkdownRenderer 
                          markdown={currentResponseRaw} 
                          isStreaming={true} 
                        />
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Display errors */}
          {error && (
            <div className="message assistant">
              <div className="message-role">ERROR</div>
              <div className="message-bubble error-message">
                <div className="message-content">
                  <p>{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Element to scroll to */}
          <div ref={messagesEndRef} />
        </div>

        {/* 🚀 ULTRATHINK Scroll to bottom button */}
        {showScrollButton && (
          <button
            onClick={scrollToBottom}
            className="scroll-to-bottom-btn"
            aria-label="Scroll to bottom"
            title="Scroll to bottom"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              width="20"
              height="20"
            >
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <polyline points="19 12 12 19 5 12"></polyline>
            </svg>
          </button>
        )}

        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onInput={handleTextareaInput}
              onPaste={handlePaste} // Add the new paste handler
              onKeyDown={handleKeyDown}
              placeholder="Type your message here..."
              disabled={isLoading}
              rows={1}
              className="message-input"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !inputMessage.trim()}
              className="send-button"
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Wrap App with AlertProvider
const AppWithAlerts = () => {
  return (
    <AlertProvider>
      <App />
      <AlertContainer />
    </AlertProvider>
  );
};

export default AppWithAlerts;
