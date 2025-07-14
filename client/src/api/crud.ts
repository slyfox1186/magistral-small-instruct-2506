import { API_URL } from '@/utils/config';
import {
  Conversation,
  ConversationCreate,
  ConversationUpdate,
  ConversationListResponse,
  MessageData,
  MessageCreate,
  MessageUpdate,
  MessageListResponse,
  UserSettings,
  UserSettingsCreate,
  UserSettingsUpdate,
} from '@/utils/types';

// In development, use empty string to leverage Vite proxy
// In production, use the full API URL
const BASE_URL = import.meta.env.DEV ? '' : (API_URL || 'http://localhost:8000');

class CrudApiClient {
  private readonly maxRetries = 3;
  private readonly baseDelay = 1000; // 1 second

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    retryCount = 0
  ): Promise<T> {
    const url = `${BASE_URL}${endpoint}`;
    const defaultHeaders = {
      'Content-Type': 'application/json',
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
        // Add timeout for requests
        signal: AbortSignal.timeout(10000), // 10 second timeout
      });

      if (!response.ok) {
        const errorText = await response.text();
        
        // Handle special case: backend starting up (503 from Vite proxy)
        if (response.status === 503) {
          try {
            const errorData = JSON.parse(errorText);
            if (errorData.code === 'BACKEND_STARTING') {
              throw new Error('BACKEND_STARTING: Backend is starting up, please wait...');
            }
          } catch (parseError) {
            // If parsing fails, treat as regular 503 error
          }
        }
        
        throw new Error(`API request failed: ${response.status} ${errorText}`);
      }

      return response.json();
    } catch (error) {
      // Check if it's a connection error and we should retry
      if (retryCount < this.maxRetries && this.isRetryableError(error)) {
        const delay = this.baseDelay * Math.pow(2, retryCount); // Exponential backoff
        console.warn(`API request failed, retrying in ${delay}ms... (attempt ${retryCount + 1}/${this.maxRetries})`, error);
        await this.sleep(delay);
        return this.makeRequest<T>(endpoint, options, retryCount + 1);
      }
      
      // If it's a connection error, provide a more user-friendly message
      if (this.isConnectionError(error)) {
        throw new Error('Cannot connect to server. Please ensure the backend is running and try again.');
      }
      
      throw error;
    }
  }

  private isRetryableError(error: unknown): boolean {
    if (error instanceof Error) {
      // Retry on connection errors, timeouts, or server errors (5xx)
      return (
        error.message.includes('ECONNREFUSED') ||
        error.message.includes('fetch failed') ||
        error.message.includes('NetworkError') ||
        error.message.includes('timeout') ||
        error.name === 'TimeoutError' ||
        error.message.includes('BACKEND_STARTING') ||
        (error.message.includes('API request failed: 5')) // 5xx errors
      );
    }
    return false;
  }

  private isConnectionError(error: unknown): boolean {
    if (error instanceof Error) {
      return (
        error.message.includes('ECONNREFUSED') ||
        error.message.includes('fetch failed') ||
        error.message.includes('NetworkError') ||
        error.name === 'TimeoutError'
      );
    }
    return false;
  }

  // ===================== Health Check =====================

  async healthCheck(): Promise<boolean> {
    const timestamp = new Date().toISOString();
    const url = `${BASE_URL}/api/health`;
    
    console.log(`[CRUD-API ${timestamp}] 🩺 Starting health check...`);
    console.log(`[CRUD-API ${timestamp}] 🌐 URL: ${url}`);
    console.log(`[CRUD-API ${timestamp}] 🏠 BASE_URL: ${BASE_URL}`);
    console.log(`[CRUD-API ${timestamp}] 🔧 import.meta.env.DEV: ${import.meta.env.DEV}`);
    
    const startTime = Date.now();
    
    try {
      console.log(`[CRUD-API ${timestamp}] 📡 Sending fetch request...`);
      
      const response = await fetch(url, {
        method: 'GET',
        signal: AbortSignal.timeout(5000), // 5 second timeout for health check
      });
      
      const duration = Date.now() - startTime;
      
      console.log(`[CRUD-API ${timestamp}] 📨 Response received in ${duration}ms:`, {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries()),
        url: response.url
      });
      
      if (response.status === 503) {
        const responseText = await response.text();
        try {
          const errorData = JSON.parse(responseText);
          if (errorData.code === 'BACKEND_STARTING') {
            console.log(`[CRUD-API ${timestamp}] 🚀 Backend is starting up...`);
            return false; // Backend not ready yet
          }
        } catch (parseError) {
          // If parsing fails, treat as regular 503 error
        }
      }
      
      const result = response.ok;
      console.log(`[CRUD-API ${timestamp}] ${result ? '✅' : '❌'} Health check result: ${result}`);
      
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      
      console.log(`[CRUD-API ${timestamp}] ❌ Health check FAILED after ${duration}ms:`, {
        error: error instanceof Error ? error.message : String(error),
        errorType: error?.constructor?.name,
        cause: error instanceof Error ? error.cause : undefined,
        stack: error instanceof Error ? error.stack?.split('\n').slice(0, 5) : undefined
      });
      
      return false;
    }
  }

  // ===================== Conversation Methods =====================

  async createConversation(data: ConversationCreate): Promise<Conversation> {
    return this.makeRequest<Conversation>('/api/conversations', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getConversation(id: string): Promise<Conversation> {
    return this.makeRequest<Conversation>(`/api/conversations/${id}`);
  }

  async listConversations(
    page: number = 1,
    pageSize: number = 20,
    archived?: boolean
  ): Promise<ConversationListResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    });

    if (archived !== undefined) {
      params.set('archived', archived.toString());
    }

    return this.makeRequest<ConversationListResponse>(
      `/api/conversations?${params.toString()}`
    );
  }

  async updateConversation(
    id: string,
    data: ConversationUpdate
  ): Promise<Conversation> {
    return this.makeRequest<Conversation>(`/api/conversations/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteConversation(id: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/api/conversations/${id}`, {
      method: 'DELETE',
    });
  }

  async archiveConversation(id: string): Promise<{ message: string; conversation: Conversation }> {
    return this.makeRequest<{ message: string; conversation: Conversation }>(
      `/api/conversations/${id}/archive`,
      { method: 'POST' }
    );
  }

  async unarchiveConversation(id: string): Promise<{ message: string; conversation: Conversation }> {
    return this.makeRequest<{ message: string; conversation: Conversation }>(
      `/api/conversations/${id}/unarchive`,
      { method: 'POST' }
    );
  }

  // ===================== Message Methods =====================

  async createMessage(data: MessageCreate): Promise<MessageData> {
    return this.makeRequest<MessageData>('/api/messages', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getMessage(id: string): Promise<MessageData> {
    return this.makeRequest<MessageData>(`/api/messages/${id}`);
  }

  async listMessages(
    conversationId: string,
    page: number = 1,
    pageSize: number = 50
  ): Promise<MessageListResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    });

    return this.makeRequest<MessageListResponse>(
      `/api/conversations/${conversationId}/messages?${params.toString()}`
    );
  }

  async updateMessage(id: string, data: MessageUpdate): Promise<MessageData> {
    return this.makeRequest<MessageData>(`/api/messages/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteMessage(id: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/api/messages/${id}`, {
      method: 'DELETE',
    });
  }

  // ===================== User Settings Methods =====================

  async createUserSettings(userId: string, data: UserSettingsCreate): Promise<UserSettings> {
    return this.makeRequest<UserSettings>(`/api/users/${userId}/settings`, {
      method: 'POST',
      body: JSON.stringify({ ...data, user_id: userId }),
    });
  }

  async getUserSettings(userId: string): Promise<UserSettings> {
    return this.makeRequest<UserSettings>(`/api/users/${userId}/settings`);
  }

  async updateUserSettings(userId: string, data: UserSettingsUpdate): Promise<UserSettings> {
    return this.makeRequest<UserSettings>(`/api/users/${userId}/settings`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteUserSettings(userId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>(`/api/users/${userId}/settings`, {
      method: 'DELETE',
    });
  }

  // ===================== Utility Methods =====================

  async getConversationSummary(id: string): Promise<{
    conversation: Conversation;
    recent_messages: MessageData[];
    total_messages: number;
  }> {
    return this.makeRequest(`/api/conversations/${id}/summary`);
  }

  async getConversationStats(): Promise<{
    total_conversations: number;
    active_conversations: number;
    archived_conversations: number;
  }> {
    return this.makeRequest('/api/stats/conversations');
  }
}

export const crudApi = new CrudApiClient();
export default crudApi;