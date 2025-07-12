import { logger } from '../utils';

// Define the structure for a single message in the chat history
export interface Message {
  role: 'user' | 'assistant' | 'error';
  content: string;
  // Add image_base64 if you plan to send images from the frontend
  // image_base64?: string | null;
}

// Define the structure for the request body sent to the backend
export interface ChatStreamRequest {
  session_id: string;
  messages: Message[];
  // system_prompt is handled backend-side for now
}

// --- Response Chunk Types ---

// Represents a chunk containing a generated token
export interface TokenChunk {
  token: {
    text: string;
  };
  error?: never; // Ensure error is not present
  done?: never; // Ensure done is not present
}

// Represents an error chunk from the stream
export interface ErrorChunk {
  error: string;
  token?: never;
  done?: never;
}

// Represents the done signal
export interface DoneChunk {
  done: true;
  token?: never;
  error?: never;
}

// Union type for possible valid JSON chunks from the stream
export type ChatStreamChunk = TokenChunk | ErrorChunk | DoneChunk;

// Type for the callback function that processes each chunk
export type StreamChunkCallback = (chunk: ChatStreamChunk) => void;

// Define a type for Vite's import.meta.env
interface ImportMeta {
  env: {
    VITE_BACKEND_URL?: string;
    [key: string]: string | undefined;
  };
}

// Backend API base URL (adjust if your backend runs elsewhere)
// Use import.meta.env for Vite projects
const BACKEND_URL = (import.meta as ImportMeta).env.VITE_BACKEND_URL || 'http://localhost:8000';

// Generate or use a unique session ID for memory persistence
// Can be overridden by VITE_MEMORY_SESSION_ID environment variable
// Stored in sessionStorage to persist across page refreshes but not browser sessions
const getOrCreateSessionId = (): string => {
  const envSessionId = (import.meta as ImportMeta).env.VITE_MEMORY_SESSION_ID;
  if (envSessionId) return envSessionId;

  const storageKey = 'chat_session_id';
  let sessionId = sessionStorage.getItem(storageKey);

  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem(storageKey, sessionId);
  }

  return sessionId;
};

export const MEMORY_SESSION_ID = getOrCreateSessionId();

/**
 * Calls the backend chat stream API and processes the SSE response.
 *
 * @param messages The list of messages in the conversation history.
 * @param onChunk A callback function to handle each received chunk (token, error, or done signal).
 * @param onError A callback function to handle fetch errors or non-200 responses.
 * @param sessionId Optional session ID for the chat. Defaults to MEMORY_SESSION_ID.
 */
export async function streamChatResponse(
  messages: Message[],
  onChunk: StreamChunkCallback,
  onError: (error: Error) => void,
  sessionId: string = MEMORY_SESSION_ID
): Promise<void> {
  const requestBody: ChatStreamRequest = {
    session_id: sessionId, // Use the provided sessionId or fall back to the default
    messages: messages,
  };

  logger.debug('Sending request to backend:', requestBody);

  try {
    // Add connection keep-alive and timeout headers
    const response = await fetch(`${BACKEND_URL}/api/chat-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
      body: JSON.stringify(requestBody),
      // Add keepalive to prevent connection drops
      keepalive: true,
    });

    logger.debug('Received response status:', response.status);

    if (!response.ok) {
      const errorBody = await response.text();
      logger.error('Backend error response:', errorBody);
      throw new Error(
        `Backend request failed: ${response.status} ${response.statusText}. Body: ${errorBody}`
      );
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    // Process the stream data with timeout protection
    let isStreamActive = true;
    const STREAM_TIMEOUT_MS = 10000; // 10 second timeout per chunk for LLM generation
    
    while (isStreamActive) {
      try {
        // Add timeout to prevent hanging on stalled streams
        const readResult = await Promise.race([
          reader.read(),
          new Promise<{ done: true; value?: undefined }>((_, reject) =>
            setTimeout(() => reject(new Error('Stream read timeout')), STREAM_TIMEOUT_MS)
          ),
        ]);
        
        const { done, value } = readResult;
        if (done) {
          logger.debug('Stream finished.');
          isStreamActive = false;
          break;
        }
        
        // Validate that we have data
        if (!value || value.length === 0) {
          continue; // Skip empty chunks
        }
        
        logger.debug(`Received ${value.length} bytes from stream`);

        // Decode the chunk and add it to the buffer
        const chunkText = decoder.decode(value, { stream: true });
        buffer += chunkText;

        // Process buffer line by line for SSE messages
        let eolIndex;
        while ((eolIndex = buffer.indexOf('\n\n')) >= 0) {
          const message = buffer.slice(0, eolIndex).trim();
          buffer = buffer.slice(eolIndex + 2); // Skip the \n\n

          if (message.startsWith('data: ')) {
            const dataContent = message.substring(6).trim(); // Get content after "data: "

            if (dataContent === '[DONE]') {
              logger.debug('Received [DONE] signal.');
              onChunk({ done: true });
              isStreamActive = false;
              // Optionally break here if [DONE] truly signifies the absolute end
              break; // Break out of the inner loop
            } else {
              try {
                const jsonData = JSON.parse(dataContent) as ChatStreamChunk;
                logger.debug('Parsed SSE data:', jsonData); // Log parsed data
                onChunk(jsonData);
              } catch (parseError) {
                logger.error('Failed to parse SSE data chunk:', dataContent, parseError);
              }
            }
          } else if (message) {
            // Ignore empty lines or lines not starting with 'data: '
            logger.debug('Ignoring non-data SSE line:', message);
          }
        }
      } catch (error) {
        if (error instanceof Error && error.message === 'Stream read timeout') {
          logger.error('Stream timed out waiting for data');
          onError(new Error('Stream connection timeout - no data received within 10 seconds'));
          isStreamActive = false;
          break;
        }
        throw error; // Re-throw other errors
      }
    }

    // Handle any remaining data in the buffer after the stream ends (unlikely for SSE but good practice)
    if (buffer.trim().startsWith('data: ')) {
      const dataContent = buffer.trim().substring(6).trim();
      if (dataContent === '[DONE]') {
        logger.debug('Received [DONE] signal at very end.');
        onChunk({ done: true });
      } else if (dataContent) {
        try {
          const jsonData = JSON.parse(dataContent) as ChatStreamChunk;
          logger.debug('Parsed final SSE data:', jsonData);
          onChunk(jsonData);
        } catch (parseError) {
          logger.error('Failed to parse final SSE data chunk:', dataContent, parseError);
        }
      }
    }
  } catch (error) {
    logger.error('Error during chat stream:', error);
    onError(error instanceof Error ? error : new Error(String(error)));
    // Ensure the stream is signalled as 'done' even on error to stop loading indicators etc.
    // But check if onChunk was already called with done:true before potentially calling it again
    onChunk({ done: true });
  }
}
