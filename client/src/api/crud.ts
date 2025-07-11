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
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${BASE_URL}${endpoint}`;
    const defaultHeaders = {
      'Content-Type': 'application/json',
    };

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} ${errorText}`);
    }

    return response.json();
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