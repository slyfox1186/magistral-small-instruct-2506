import React, { useState, useCallback, useRef, useEffect, lazy, Suspense } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';
import { MEMORY_SESSION_ID } from './api/chat';
import { CacheManager } from './utils/cacheUtils';
import './styles/App.css';
import './styles/StatusAnimations.css';
import './styles/themes.scss';
import './styles/Crud.css';
import './styles/ChatSettings.css';
// Import status system components
import { AlertProvider } from './contexts/AlertContext';
import { useAlert } from './hooks/useAlerts';
import { AlertContainer } from './components/alerts/AlertContainer';
import { StatusIndicator } from './components/status/StatusIndicator';
import { Status } from './types/status';
// Import theme system
import { ThemeProvider } from './contexts/ThemeContext';
import { ThemeToggleButton } from './components/ThemeToggleButton';
import ErrorBoundary from './components/ErrorBoundary';
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
// Import chat integration
import { chatIntegration } from './utils/chatIntegration';
// Import connection status
import ConnectionStatus from './components/ConnectionStatus';
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
      return module;
    })
    .catch((error) => {
      logger.error('Failed to load MarkdownItRenderer:', error);
      throw error;
    })
);

// Import CRUD interface directly to avoid Suspense/hook issues
import CrudInterface from './components/crud/CrudInterface';
// Import ChatSettings component
import ChatSettings from './components/settings/ChatSettings';
// Import the new tab system
import MainAppTabs from './components/tabs/MainAppTabs';

// Fix timeout type compatibility
type TimeoutType = ReturnType<typeof setTimeout>;

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

const AttachmentIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
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

const ThinkingDots = () => (
  <div className="thinking-dots">
    <span></span>
    <span></span>
    <span></span>
  </div>
);

const MenuIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="3" y1="6" x2="21" y2="6"></line>
    <line x1="3" y1="12" x2="21" y2="12"></line>
    <line x1="3" y1="18" x2="21" y2="18"></line>
  </svg>
);

const SettingsIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="12" cy="12" r="3"></circle>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
  </svg>
);

function App() {
  // Get alert functions
  const alert = useAlert();
  // Use fixed session ID from environment or generated ID
  const sessionId = MEMORY_SESSION_ID;
  // Separate API session ID for conversation switching
  const [apiSessionId, setApiSessionId] = useState(MEMORY_SESSION_ID);
  // Interface mode state - now includes analytics and system
  const [interfaceMode, setInterfaceMode] = useState<'chat' | 'crud' | 'analytics' | 'system'>('chat');
  // Settings panel state
  const [showSettings, setShowSettings] = useState(false);
  // Sidebar state
  const [showSidebar, setShowSidebar] = useState(false);
  // Conversation loading state
  const [loadingConversation, setLoadingConversation] = useState(false);
  // Backend health state  
  const [,] = useState<boolean | null>(null);
  // State machine for chat integration lifecycle
  const [chatIntegrationState, setChatIntegrationState] = useState<
    'uninitialized' | 'initializing' | 'initialized' | 'switching_conversation'
  >('uninitialized');

  // Initialize cache management for development
  useEffect(() => {
    if (import.meta.env.DEV) {
      CacheManager.clearDevelopmentCaches();
      logger.info('🔄 App initialized with fresh cache');
    }
  }, []);

  // Load messages from localStorage on initial render with type safety
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
  // This state now holds the raw Markdown being streamed
  // CORRECTED: Atomic state machine - single source of truth
  type ChatState = 
    | { status: 'idle' }
    | { status: 'thinking' }  
    | { status: 'streaming'; accumulatedContent: string }
    | { status: 'completing'; accumulatedContent: string }
    | { status: 'error'; error: string };

  const [chatState, setChatState] = useState<ChatState>({ status: 'idle' });
  
  // Legacy state for compatibility - will be removed
  const [currentResponseRaw, setCurrentResponseRaw] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [currentFinalResponse, setCurrentFinalResponse] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [currentMessageStatus, setCurrentMessageStatus] = useState<Status>('thinking');
  
  // Derived state - computed, never set directly
  const isLoading = chatState.status !== 'idle' && chatState.status !== 'error';
  const streamingComplete = chatState.status === 'completing';
  
  // File attachment state
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // File size limits
  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB per file
  const MAX_TOTAL_SIZE = 20 * 1024 * 1024; // 20MB total

  // Store the accumulated response for final message creation
  const accumulatedResponseRef = useRef<string>('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Ref for current abort controller
  const currentAbortControllerRef = useRef<AbortController | null>(null);
  // Ref for paste timeout cleanup
  const pasteTimeoutRef = useRef<TimeoutType | null>(null);

  // Ref for inactivity timer cleanup to prevent memory leaks
  const inactivityTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Refs for streaming UI elements - simplified as we render Markdown directly
  const streamingRef = useRef<HTMLDivElement>(null);

  // REMOVED: Complex ref flags replaced by atomic state machine

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

  // File attachment handlers
  const handleAttachFile = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    setFileError(null); // Clear previous errors
    
    if (!files || files.length === 0) {
      return;
    }

    const fileArray = Array.from(files);
    
    // Check individual file sizes
    const oversizedFiles = fileArray.filter(file => file.size > MAX_FILE_SIZE);
    if (oversizedFiles.length > 0) {
      const fileNames = oversizedFiles.map(f => f.name).join(', ');
      setFileError(`Files too large (max 5MB): ${fileNames}`);
      return;
    }
    
    // Check total size including existing files
    const currentTotalSize = attachedFiles.reduce((sum, file) => sum + file.size, 0);
    const newTotalSize = fileArray.reduce((sum, file) => sum + file.size, 0);
    if (currentTotalSize + newTotalSize > MAX_TOTAL_SIZE) {
      setFileError('Total file size would exceed 20MB limit');
      return;
    }

    // Filter for text-based files
    const textFiles = fileArray.filter(file => {
      const fileType = file.type;
      const fileName = file.name.toLowerCase();
      return (
        fileType.startsWith('text/') ||
        fileType === 'application/json' ||
        fileType === 'application/javascript' ||
        fileType === 'application/typescript' ||
        fileName.endsWith('.md') ||
        fileName.endsWith('.txt') ||
        fileName.endsWith('.js') ||
        fileName.endsWith('.ts') ||
        fileName.endsWith('.tsx') ||
        fileName.endsWith('.jsx') ||
        fileName.endsWith('.py') ||
        fileName.endsWith('.sh') ||
        fileName.endsWith('.sql') ||
        fileName.endsWith('.css') ||
        fileName.endsWith('.scss') ||
        fileName.endsWith('.json') ||
        fileName.endsWith('.xml') ||
        fileName.endsWith('.yaml') ||
        fileName.endsWith('.yml') ||
        fileName.endsWith('.toml') ||
        fileName.endsWith('.ini') ||
        fileName.endsWith('.conf') ||
        fileName.endsWith('.config')
      );
    });
    
    const rejectedFiles = fileArray.filter(file => !textFiles.includes(file));
    if (rejectedFiles.length > 0) {
      const rejectedNames = rejectedFiles.map(f => f.name).join(', ');
      const errorMsg = `Unsupported file types: ${rejectedNames}. Please select text-based files only.`;
      setFileError(errorMsg);
      alert.error(errorMsg, 5000);
      if (textFiles.length === 0) {
        return;
      }
    }
    
    if (textFiles.length > 0) {
      setAttachedFiles(prev => [...prev, ...textFiles]);
      logger.info(`Attached ${textFiles.length} file(s)`);
      alert.success(`Attached ${textFiles.length} file(s)`, 3000);
    }
    
    // Clear the input so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [attachedFiles, MAX_FILE_SIZE, MAX_TOTAL_SIZE, alert]);

  const handleRemoveFile = useCallback((index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
    setFileError(null); // Clear any file errors when removing files
  }, []);

  const readFileContent = useCallback(async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      // Additional validation before reading
      if (file.size > MAX_FILE_SIZE) {
        reject(new Error(`File ${file.name} exceeds size limit (${Math.round(MAX_FILE_SIZE / 1024 / 1024)}MB)`));
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          if (typeof content !== 'string') {
            reject(new Error(`Failed to read file content: ${file.name}`));
            return;
          }
          resolve(content);
        } catch {
          reject(new Error(`Error processing file content: ${file.name}`));
        }
      };
      reader.onerror = () => {
        reject(new Error(`Failed to read file: ${file.name}`));
      };
      reader.onabort = () => {
        reject(new Error(`File reading aborted: ${file.name}`));
      };
      
      try {
        reader.readAsText(file, 'UTF-8');
      } catch {
        reject(new Error(`Failed to start reading file: ${file.name}`));
      }
    });
  }, [MAX_FILE_SIZE]);

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
    
    // Clear attached files and errors
    setAttachedFiles([]);
    setFileError(null);
    
    // Reset API session ID to default for new conversations
    setApiSessionId(MEMORY_SESSION_ID);
    logger.info(`Reset API session ID to default: ${MEMORY_SESSION_ID}`);

    // Clear from localStorage
    try {
      localStorage.removeItem(`chat_history_${MEMORY_SESSION_ID}`);
      logger.info('✅ Chat history cleared from both state and localStorage');
    } catch (error) {
      logger.error('Failed to clear chat history from localStorage:', error);
    }

    // Archive current conversation in CRUD database
    try {
      chatIntegration.archiveCurrentConversation();
      logger.info('Archived current conversation in CRUD database');
    } catch (error) {
      logger.error('Failed to archive conversation in CRUD database:', error);
      // Don't block the chat flow if CRUD archiving fails
    }

    focusInput(); // Focus input after clearing history
  }, [isLoading, focusInput, messages.length]);

  // Utility function to clear additional state beyond basic chat history
  const clearAdditionalState = useCallback(() => {
    setCurrentResponseRaw('');
    setCurrentThinking('');
    setCurrentFinalResponse('');
    accumulatedResponseRef.current = '';
    setChatState({ status: 'idle' });
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
      logger.info(`🧹 Attempting to clear conversation memories via ${config.API_URL}/clear-conversation-memories/${apiSessionId}`);

      const response = await fetch(`${config.API_URL}/clear-conversation-memories/${apiSessionId}`, {
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

          // Show success message briefly
          setError(`✅ Successfully cleared ${data.deleted_count} memories`);
          setTimeout(() => setError(null), 3000);
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

  // Load conversation messages from CRUD database
  const loadConversationMessages = useCallback(async (conversationId: string) => {
    try {
      setLoadingConversation(true);
      setChatIntegrationState('switching_conversation');
      logger.info(`Loading messages for conversation: ${conversationId}`);
      
      // Import the CRUD API
      const { crudApi } = await import('@/api/crud');
      
      // Load messages from the conversation
      const response = await crudApi.listMessages(conversationId, 1, 100);
      
      // Convert CRUD messages to AppMessage format
      const convertedMessages: AppMessage[] = response.messages.map((msg: any) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: new Date(msg.created_at).toLocaleTimeString(),
        status: 'success' as const,
        nodeRef: React.createRef(), // Add nodeRef for consistency
      }));
      
      // CRITICAL: Clear localStorage to prevent conflicts with conversation switching
      try {
        localStorage.removeItem(`chat_history_${MEMORY_SESSION_ID}`);
        logger.info('Cleared localStorage to prevent conflicts with conversation switching');
      } catch (error) {
        logger.warn('Failed to clear localStorage:', error);
      }
      
      // Update the chat state with loaded messages
      setMessages(convertedMessages);
      
      // CRITICAL: Update API session ID to conversation ID for memory context
      setApiSessionId(conversationId);
      
      // Update chat integration to use this conversation
      await chatIntegration.setCurrentConversation(conversationId);
      
      logger.info(`Successfully loaded ${convertedMessages.length} messages from conversation ${conversationId}`);
      logger.info(`Updated API session ID to: ${conversationId} for memory context`);
      
      // Set state to initialized after successful conversation switch
      setChatIntegrationState('initialized');
      
    } catch (error) {
      logger.error('Failed to load conversation messages:', error);
      alert.error('Failed to load conversation messages');
      setChatIntegrationState('initialized'); // Reset to stable state
    } finally {
      setLoadingConversation(false);
    }
  }, [alert]);

  // Handle back to chat with optional conversation switching
  const handleBackToChat = useCallback(async (conversationId?: string) => {
    if (conversationId) {
      await loadConversationMessages(conversationId);
    }
    setInterfaceMode('chat');
  }, [loadConversationMessages]);

  // Handle tab mode changes for the new tab system
  const handleTabModeChange = useCallback((mode: 'chat' | 'crud' | 'analytics' | 'system') => {
    setInterfaceMode(mode);
  }, []);

  // Removed isFinalCompletionRef - replaced with atomic state machine

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

  // Initialize chat integration using explicit state machine
  useEffect(() => {
    const initializeChatIntegration = async () => {
      try {
        // ONLY initialize if in uninitialized state AND have messages from localStorage
        if (chatIntegrationState === 'uninitialized' && messages.length > 0) {
          setChatIntegrationState('initializing');
          
          // Initialize conversation with first user message
          const firstUserMessage = messages.find(m => m.role === 'user');
          await chatIntegration.initializeConversation(
            apiSessionId, 
            firstUserMessage?.content || undefined
          );
          
          // Save existing messages to CRUD database (ONLY for localStorage messages)
          await chatIntegration.saveMessages(messages);
          logger.info('Synced existing chat messages to CRUD database');
          
          setChatIntegrationState('initialized');
        }
      } catch (error) {
        logger.error('Failed to initialize chat integration:', error);
        setChatIntegrationState('initialized'); // Reset to stable state
      }
    };

    initializeChatIntegration();
  }, [messages, apiSessionId, chatIntegrationState]); // Explicit state dependency

  // CORRECTED: Simple atomic completion effect
  useEffect(() => {
    if (chatState.status === 'completing') {
      logger.warn('VERBOSE: Processing completion - creating final message');
      logger.warn('DEBUG: chatState.accumulatedContent length:', chatState.accumulatedContent?.length || 0);
      logger.warn('DEBUG: accumulatedResponseRef.current length:', accumulatedResponseRef.current?.length || 0);
      logger.warn('DEBUG: currentResponseRaw length:', currentResponseRaw?.length || 0);
      
      // Try multiple sources for content - use fallback logic
      const finalContent = chatState.accumulatedContent || accumulatedResponseRef.current || currentResponseRaw || '';
      
      if (finalContent && finalContent.length > 0) {
        logger.warn('VERBOSE: Creating final message with content length:', finalContent.length);
        // Create final assistant message
        const newMessage = createAssistantMessage(finalContent);
        setMessages((prevMessages) => [...prevMessages, newMessage]);

        // Save to database asynchronously
        (async () => {
          try {
            await chatIntegration.saveMessage(newMessage);
            setMessages((currentMessages) => {
              if (currentMessages.length === 2) {
                chatIntegration.autoUpdateTitle(currentMessages).catch(console.error);
              }
              return currentMessages;
            });
          } catch (error) {
            logger.error('Failed to save message:', error);
          }
        })();

        // Reset markdown buffer
        resetTokenBuffer();
      } else {
        logger.error('CRITICAL: No content available for final message creation!');
        logger.error('DEBUG: All content sources empty - this should not happen');
        // Create an error message to indicate the problem
        const errorMessage = createErrorMessage('Assistant response was empty. Please try again.');
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
      }

      // Atomic transition back to idle - unlocks input immediately
      logger.warn('VERBOSE: Transitioning to idle - input should unlock now');
      setChatState({ status: 'idle' });
      setCurrentResponseRaw('');
      setCurrentMessageStatus('success');
    }
  }, [chatState.status, currentResponseRaw]);

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

    // Only disable autoscroll if user scrolls significantly away from bottom
    // Be more forgiving to prevent accidental disabling
    const isNearBottom = distanceFromBottom < 100; // Increased threshold

    if (!isNearBottom) {
      // User has scrolled away from bottom - immediately disable autoscroll
      setAutoScrollEnabled(false);
      setShowScrollButton(true);

      // Clear any existing timeout that might re-enable autoscroll
      if (checkBottomTimeoutRef.current) {
        clearTimeout(checkBottomTimeoutRef.current);
        checkBottomTimeoutRef.current = null;
      }
      return;
    }

    // User is at bottom - immediately re-enable autoscroll
    if (autoScrollEnabled === false) {
      setAutoScrollEnabled(true);
      setShowScrollButton(false);
    }

    // Clear any existing timeout since we're at bottom
    if (checkBottomTimeoutRef.current) {
      clearTimeout(checkBottomTimeoutRef.current);
      checkBottomTimeoutRef.current = null;
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
    const hasRecentUserInteraction = timeSinceLastInteraction < 1000; // Reduced to 1 second

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


  // Save messages to localStorage whenever they change (only for default session, not switched conversations)
  useEffect(() => {
    // Only save to localStorage if we're in the default session (not a switched conversation)
    if (messages.length > 0 && apiSessionId === MEMORY_SESSION_ID) {
      try {
        // Create a version of messages without nodeRef (can't be serialized)
        const serializableMessages = messages.map(({ nodeRef: _nodeRef, ...rest }) => rest);
        localStorage.setItem(
          `chat_history_${MEMORY_SESSION_ID}`,
          JSON.stringify(serializableMessages)
        );
        logger.debug(`Saved ${messages.length} messages to localStorage for default session`);
      } catch (error) {
        logger.error('Failed to save chat history to localStorage:', error);
      }
    } else if (apiSessionId !== MEMORY_SESSION_ID) {
      logger.debug(`Skipping localStorage save for conversation session: ${apiSessionId}`);
    }
  }, [messages, apiSessionId]);

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
      // Removed streamCompletionTimerRef - no longer needed with state machine
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

    // CRITICAL FIX: If textarea is empty, immediately reset to default height
    if (e.currentTarget.value.trim() === '') {
      textarea.style.height = `${defaultHeightRem}rem`;
      textarea.style.overflowY = 'hidden';
      return;
    }

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
    setChatState({ status: 'thinking' });
    setCurrentMessageStatus('thinking'); // Start with thinking status
    // Removed isAccumulatingMessageRef and isFinalCompletionRef - no longer needed with state machine
    resetTokenBuffer(); // Reset token buffer for new message
    setCurrentResponseRaw(''); // Clear any stale raw response
    accumulatedResponseRef.current = ''; // Clear accumulated response
    // Removed streamingCompleteRef - no longer needed with state machine

    // CRITICAL: Enable auto-scroll when starting a new message
    setAutoScrollEnabled(true);
    setShowScrollButton(false);

    const userMessageText = inputMessage.trim();

    // --- Chat Message Sending Logic ---
    let apiMessageText = userMessageText;
    
    // Include attached files content for API (but not in displayed message)
    if (attachedFiles.length > 0) {
      try {
        // Show progress for large file operations
        if (attachedFiles.some(file => file.size > 1024 * 1024)) { // 1MB threshold
          alert.info('Reading large files, please wait...', 3000);
        }
        
        const fileContents = await Promise.all(
          attachedFiles.map(async (file) => {
            const content = await readFileContent(file);
            return `\n\n**Attached File: ${file.name}**\n\`\`\`\n${content}\n\`\`\``;
          })
        );
        // Only add file contents to API message, not displayed message
        apiMessageText = userMessageText + fileContents.join('');
        
        // Clear attached files after including them in message
        setAttachedFiles([]);
        setFileError(null); // Clear any file errors
        logger.info(`Included ${attachedFiles.length} attached file(s) in message context`);
        alert.success(`Included ${attachedFiles.length} file(s) in context`, 2000);
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Failed to read one or more attached files';
        logger.error('Error reading attached files:', error);
        setError(errorMsg);
        alert.error(errorMsg, 5000);
        setChatState({ status: 'error', error: errorMsg });
        return;
      }
    }

    // Create user message with only the original text (no file contents in UI)
    const newUserMessage = createUserMessage(userMessageText);

    // Clear previous state
    setCurrentResponseRaw('');
    setCurrentThinking('');
    setCurrentFinalResponse('');
    setError(null);
    setChatState({ status: 'thinking' });
    accumulatedResponseRef.current = ''; // Reset accumulator at the start

    // Reset the markdown token buffer when starting a new chat
    resetTokenBuffer();

    // Add user message and set loading state
    setMessages((prev) => [...prev, newUserMessage]);
    setInputMessage('');

    // Initialize conversation and save user message to CRUD database
    try {
      await chatIntegration.initializeConversation(apiSessionId, userMessageText);
      await chatIntegration.saveMessage(newUserMessage);
    } catch (error) {
      logger.error('Failed to save user message to CRUD database:', error);
      // Don't block the chat flow if CRUD saving fails
    }

    // Reset textarea height immediately and again after DOM updates
    if (inputRef.current) {
      inputRef.current.style.height = `${config.DEFAULT_TEXTAREA_HEIGHT_REM}rem`;
      inputRef.current.style.overflowY = 'hidden';
    }

    setChatState({ status: 'streaming', accumulatedContent: '' });

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

    // Removed isAccumulatingMessageRef - no longer needed with state machine

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
          session_id: apiSessionId,
          messages: formatMessagesForAPI(messages, apiMessageText),
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
      const MAX_INACTIVITY_MS = 30000; // 30 seconds max without activity for LLM generation
      
      // Main stream processing loop variable - defined outside timer for access
      let isProcessing = true;
      logger.warn('VERBOSE: Starting stream processing loop');

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
          // Force break the stream loop instead of setting completed
          isProcessing = false;
        }
        
      }, 2000);

      try {
        // Main stream processing loop
        while (isProcessing) {
          let text = '';
          try {
            // Add timeout protection to prevent hanging on stalled streams
            const READER_TIMEOUT_MS = 10000; // 10 second timeout for LLM generation
            const readResult = await Promise.race([
              reader.read(),
              new Promise<{ done: true; value?: undefined }>((_, reject) =>
                setTimeout(() => reject(new Error('Reader timeout')), READER_TIMEOUT_MS)
              ),
            ]);
            
            const { done, value } = readResult;

            // Update activity timestamp on any data or done signal
            lastActivityTime = Date.now();
            
            if (value && value.length > 0) {
              logger.debug(`App.tsx: Received ${value.length} bytes from reader`);
            }

            if (done) {
              if (inactivityTimerRef.current) {
                clearInterval(inactivityTimerRef.current);
                inactivityTimerRef.current = null;
              }
              logger.debug('Stream ended naturally (reader done).');
              // Don't set completed here - let the loop finish and set it at the end
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
            if (chatState.status !== 'error') {
              setChatState({ status: 'error', error: 'Error reading stream data.' });
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
                // CORRECTED: Atomic state transition to completing
                logger.warn('COMPLETION: Received [DONE] string signal - transitioning to completing');
                setChatState({ status: 'completing', accumulatedContent: accumulatedResponseRef.current });
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
                  // CORRECTED: Atomic state transition to completing
                  logger.warn('VERBOSE: JSON done signal detected!', JSON.stringify(data));
                  logger.warn('VERBOSE: Original jsonStr was:', jsonStr);
                  logger.warn('VERBOSE: Transitioning chatState to completing');
                  setChatState({ status: 'completing', accumulatedContent: accumulatedResponseRef.current });
                  logger.warn('VERBOSE: Breaking stream processing loop');
                  isProcessing = false;
                  break;
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
                  logger.warn('VERBOSE: Processing token chunk:', data.token.text.substring(0, 50));
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
                  logger.debug('DEBUG: Token added, accumulator now has length:', accumulatedResponseRef.current.length);

                  // Update status to streaming once tokens start arriving
                  if (currentMessageStatus === 'thinking') {
                    setCurrentMessageStatus('streaming');
                    logger.warn('DEBUG: Status changed from thinking to streaming');
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

            // If isProcessing was set to false, break the outer loop too
            if (!isProcessing) break;
          } // End of for...of lines loop

          // If isProcessing was set to false, break the outer while loop
          if (!isProcessing) break;
        } // End of while loop
      } catch (streamProcessingError) {
        logger.error(`Error during stream processing loop: ${streamProcessingError}`);
        if (inactivityTimerRef.current) {
          clearInterval(inactivityTimerRef.current);
          inactivityTimerRef.current = null;
        }
        if (chatState.status !== 'error') {
          setChatState({ status: 'error', error: 'Error processing stream.' });
          setError('Error processing stream.');
        }
      }

      // CRITICAL: Ensure final state update after loop completes
      setCurrentResponseRaw(accumulatedResponseRef.current);
      
      // State transition now happens inside the loop when done signal is received
      logger.warn(`VERBOSE: Stream loop ended - current chatState: ${chatState.status}`);
    } catch (err) {
      // Handle aborted requests
      if (controller && controller.signal.aborted) {
        // Just clean up, don't show error for intentional abort
        setChatState({ status: 'idle' });
        setCurrentResponseRaw('');
        accumulatedResponseRef.current = '';
        return;
      }

      logger.error('Error during chat request:', err);

      // Regular error handling
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
      setCurrentMessageStatus('error');
      alert.error(`Chat error: ${errorMsg}`, 5000);

      // Clean up timers
      if (inactivityTimerRef.current) {
        clearInterval(inactivityTimerRef.current);
        inactivityTimerRef.current = null;
      }

      // Reset the markdown token buffer on error
      resetTokenBuffer();

      // Create and add error message
      const errorMessage = createErrorMessage(errorMsg);

      // Clean up state
      setChatState({ status: 'error', error: errorMsg });
      setCurrentResponseRaw('');
      accumulatedResponseRef.current = '';

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
      // Ensure the stream state is reset even if errors occur before reader setup
      if ((chatState as ChatState).status === 'thinking' || (chatState as ChatState).status === 'streaming') {
        setChatState({ status: 'idle' });
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
    attachedFiles,
    readFileContent,
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
      // Removed streamCompletionTimerRef - no longer needed with state machine
      // Cancel any active requests
      if (currentAbortControllerRef.current) {
        currentAbortControllerRef.current.abort();
        currentAbortControllerRef.current = null;
      }
    };
  }, []);

  // Handle keyboard shortcuts for sidebar
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && showSidebar) {
        setShowSidebar(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [showSidebar]);

  return (
    <div className="app-layout-container">
      {/* Sidebar Toggle Button - at app level */}
      <button
        className="sidebar-toggle"
        onClick={() => setShowSidebar(!showSidebar)}
        title="Toggle menu"
        aria-label="Toggle sidebar menu"
      >
        <MenuIcon />
      </button>
      
      {/* Collapsible Sidebar - at app level */}
      <div className={`sidebar ${showSidebar ? 'sidebar--open' : ''}`}>
        <div className="sidebar-content">
          <ConnectionStatus className="connection-status--compact" />
          <ThemeToggleButton />
          <button
            onClick={() => setShowSettings(true)}
            className="settings-button"
            title="Chat Settings"
          >
            <SettingsIcon />
          </button>
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
      </div>
      
      {/* Backdrop to close sidebar when clicking outside */}
      {showSidebar && (
        <div 
          className="sidebar-backdrop" 
          onClick={() => setShowSidebar(false)}
          aria-hidden="true"
        />
      )}

      {/* Simple Tab Navigation */}
      <div className="simple-tab-nav">
        <button 
          className={`simple-tab ${interfaceMode === 'chat' ? 'active' : ''}`}
          onClick={() => setInterfaceMode('chat')}
        >
          💬 Chat
        </button>
        <button 
          className={`simple-tab ${interfaceMode === 'crud' ? 'active' : ''}`}
          onClick={() => setInterfaceMode('crud')}
        >
          📁 Data Management
        </button>
        <button 
          className={`simple-tab ${interfaceMode === 'analytics' ? 'active' : ''}`}
          onClick={() => setInterfaceMode('analytics')}
        >
          📊 Analytics
        </button>
        <button 
          className={`simple-tab ${interfaceMode === 'system' ? 'active' : ''}`}
          onClick={() => setInterfaceMode('system')}
        >
          ⚙️ System
        </button>
      </div>

      {/* Content based on interface mode */}
      {interfaceMode === 'chat' && (
        <div className="chat-container">
          {/* Conversation loading indicator */}
          {loadingConversation && (
            <div className="loading-conversation">
              <div className="loading-spinner"></div>
              <span>Loading conversation...</span>
            </div>
          )}
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
                      {/* Render final messages using MarkdownItRenderer */}
                      <Suspense fallback={<div>Loading markdown...</div>}>
                        <MarkdownItRenderer markdown={message.content ?? ''} />
                      </Suspense>
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
                  {/* Show thinking dots when no content yet */}
                  {!currentResponseRaw ? (
                    <div className="loading">
                      <ThinkingDots />
                    </div>
                  ) : (
                    <>
                      {/* Show thinking content if available */}
                      {currentThinking && (
                        <div className="assistant-thinking">
                          <div className="thinking-header">🤔 Thinking...</div>
                          <div className="thinking-content">
                            <Suspense
                              fallback={
                                <div className="loading">
                                  <ThinkingDots />
                                </div>
                              }
                            >
                              <MarkdownItRenderer
                                markdown={currentThinking}
                                isStreaming={true}
                                className="thinking-markdown"
                              />
                            </Suspense>
                          </div>
                        </div>
                      )}
                      {/* Show final response if available */}
                      {currentFinalResponse && (
                        <div className="assistant-response">
                          <Suspense
                            fallback={
                              <div className="loading">
                                <ThinkingDots />
                              </div>
                            }
                          >
                            <MarkdownItRenderer
                              markdown={currentFinalResponse}
                              isStreaming={true}
                              className="response-markdown"
                            />
                          </Suspense>
                        </div>
                      )}
                      {/* Fallback to show raw content if no parsing happened */}
                      {!currentThinking && !currentFinalResponse && (
                        <Suspense
                          fallback={
                            <div className="loading">
                              <ThinkingDots />
                            </div>
                          }
                        >
                          <MarkdownItRenderer markdown={currentResponseRaw} isStreaming={true} />
                        </Suspense>
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
          {/* File Error Display */}
          {fileError && (
            <div className="file-error-container">
              <div className="file-error-message">
                ⚠️ {fileError}
                <button
                  onClick={() => setFileError(null)}
                  className="file-error-dismiss"
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              </div>
            </div>
          )}
          
          {/* Attached Files Display */}
          {attachedFiles.length > 0 && (
            <div className="attached-files-container">
              <div className="attached-files-header">
                <span>Attached Files ({attachedFiles.length})</span>
                <span className="total-size">
                  {Math.round(attachedFiles.reduce((sum, file) => sum + file.size, 0) / 1024)}KB total
                </span>
              </div>
              <div className="attached-files-list">
                {attachedFiles.map((file, index) => (
                  <div key={index} className="attached-file-item">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">({Math.round(file.size / 1024)}KB)</span>
                    <button
                      onClick={() => handleRemoveFile(index)}
                      className="remove-file-button"
                      aria-label={`Remove ${file.name}`}
                      title="Remove file"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          <div className="input-wrapper">
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              multiple
              accept=".txt,.md,.js,.ts,.tsx,.jsx,.py,.sh,.sql,.css,.scss,.json,.xml,.yaml,.yml,.toml,.ini,.conf,.config,text/*,application/json,application/javascript,application/typescript"
              style={{ display: 'none' }}
            />
            
            {/* File attachment button */}
            <button
              onClick={handleAttachFile}
              disabled={isLoading}
              className="attachment-button"
              aria-label="Attach text-based files"
              aria-describedby="file-input-help"
              title="Attach text-based files (scripts, configs, etc.) - Max 5MB per file, 20MB total"
            >
              <AttachmentIcon />
            </button>
            
            {/* Screen reader help text */}
            <span id="file-input-help" className="sr-only">
              Select text-based files such as scripts, configuration files, or documentation. 
              Maximum 5MB per file, 20MB total.
            </span>
            
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
      )}

      {interfaceMode === 'crud' && (
        <ErrorBoundary>
          <CrudInterface 
            userId={sessionId} 
            onBackToChat={handleBackToChat}
          />
        </ErrorBoundary>
      )}

      {interfaceMode === 'analytics' && (
        <div className="analytics-content" style={{ padding: '2rem', height: '100%', overflow: 'auto' }}>
          <div style={{ 
            padding: '2rem', 
            textAlign: 'center', 
            color: 'var(--text-secondary, #cbd5e1)',
            background: 'var(--bg-secondary, #252730)',
            borderRadius: '8px',
            margin: '2rem',
            minHeight: '200px'
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📊</div>
            <h2 style={{ margin: '1rem 0', color: 'var(--text-primary, #f8fafc)' }}>
              Analytics Dashboard
            </h2>
            <p>Conversation insights, usage patterns, and performance metrics will be displayed here.</p>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '1rem',
              marginTop: '2rem'
            }}>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#a78bfa', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Total Conversations</strong>
                <div style={{ fontSize: '2rem', marginTop: '0.5rem', fontWeight: '600', color: 'var(--text-primary)' }}>247</div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#34d399', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Avg Response Time</strong>
                <div style={{ fontSize: '2rem', marginTop: '0.5rem', fontWeight: '600', color: 'var(--text-primary)' }}>1.2s</div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#60a5fa', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Active Users</strong>
                <div style={{ fontSize: '2rem', marginTop: '0.5rem', fontWeight: '600', color: 'var(--text-primary)' }}>12</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {interfaceMode === 'system' && (
        <div className="system-content" style={{ padding: '2rem', height: '100%', overflow: 'auto' }}>
          <div style={{ 
            padding: '2rem', 
            textAlign: 'center', 
            color: 'var(--text-secondary, #cbd5e1)',
            background: 'var(--bg-secondary, #252730)',
            borderRadius: '8px',
            margin: '2rem',
            minHeight: '200px'
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚙️</div>
            <h2 style={{ margin: '1rem 0', color: 'var(--text-primary, #f8fafc)' }}>
              System Health
            </h2>
            <p>System monitoring, health checks, and configuration options.</p>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
              gap: '1.5rem',
              marginTop: '2rem'
            }}>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#34d399', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Backend Status</strong>
                <div style={{ 
                  fontSize: '1.1rem', 
                  marginTop: '0.8rem',
                  color: '#34d399',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#34d399' }}></div>
                  Online
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginTop: '0.5rem' }}>
                  Uptime: 99.9%
                </div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#34d399', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Database</strong>
                <div style={{ 
                  fontSize: '1.1rem', 
                  marginTop: '0.8rem',
                  color: '#34d399',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#34d399' }}></div>
                  Connected
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginTop: '0.5rem' }}>
                  Latency: 12ms
                </div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#f59e0b', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Memory Usage</strong>
                <div style={{ 
                  fontSize: '1.8rem', 
                  marginTop: '0.8rem',
                  color: 'var(--text-primary)',
                  fontWeight: '600'
                }}>
                  78%
                </div>
                <div style={{ 
                  width: '100%', 
                  height: '4px', 
                  background: 'var(--bg-primary)', 
                  borderRadius: '2px',
                  marginTop: '0.8rem',
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: '78%', 
                    height: '100%', 
                    background: 'linear-gradient(90deg, #f59e0b, #f97316)',
                    borderRadius: '2px'
                  }}></div>
                </div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#60a5fa', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>CPU Usage</strong>
                <div style={{ 
                  fontSize: '1.8rem', 
                  marginTop: '0.8rem',
                  color: 'var(--text-primary)',
                  fontWeight: '600'
                }}>
                  24%
                </div>
                <div style={{ 
                  width: '100%', 
                  height: '4px', 
                  background: 'var(--bg-primary)', 
                  borderRadius: '2px',
                  marginTop: '0.8rem',
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: '24%', 
                    height: '100%', 
                    background: 'linear-gradient(90deg, #60a5fa, #3b82f6)',
                    borderRadius: '2px'
                  }}></div>
                </div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#a78bfa', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Active Sessions</strong>
                <div style={{ 
                  fontSize: '1.8rem', 
                  marginTop: '0.8rem',
                  color: 'var(--text-primary)',
                  fontWeight: '600'
                }}>
                  8
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', marginTop: '0.5rem' }}>
                  Peak: 15 today
                </div>
              </div>
              <div style={{ 
                padding: '1.5rem', 
                background: 'var(--bg-tertiary, #2f3349)', 
                borderRadius: '8px',
                border: '1px solid var(--border-color, #475569)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <strong style={{ color: '#10b981', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Disk Usage</strong>
                <div style={{ 
                  fontSize: '1.8rem', 
                  marginTop: '0.8rem',
                  color: 'var(--text-primary)',
                  fontWeight: '600'
                }}>
                  45%
                </div>
                <div style={{ 
                  width: '100%', 
                  height: '4px', 
                  background: 'var(--bg-primary)', 
                  borderRadius: '2px',
                  marginTop: '0.8rem',
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: '45%', 
                    height: '100%', 
                    background: 'linear-gradient(90deg, #10b981, #059669)',
                    borderRadius: '2px'
                  }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Chat Settings Panel */}
      <ChatSettings 
        userId={sessionId}
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
}

// Wrap App with ThemeProvider and AlertProvider
const AppWithProviders = () => {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <AlertProvider>
          <App />
          <AlertContainer />
        </AlertProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default AppWithProviders;
