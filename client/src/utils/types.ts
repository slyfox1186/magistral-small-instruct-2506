// Logger types
export enum LogLevel {
  NONE = 0,
  ERROR = 1,
  WARN = 2,
  INFO = 3,
  DEBUG = 4,
  TRACE = 5,
}

export interface LoggerOptions {
  level: LogLevel;
  prefix?: string;
  timestampEnabled?: boolean;
}

export type LogParams = unknown[];

// Network types
export interface RequestOptions {
  method: string;
  headers?: Record<string, string>;
  body?: Record<string, unknown> | FormData;
  signal?: AbortSignal;
  retries?: number;
  retryDelay?: number;
}

export interface StreamRequestOptions extends RequestOptions {
  retries?: number;
  retryDelay?: number;
}

export interface FetchResponse<T = unknown> {
  data: T;
  status: number;
  statusText: string;
  headers: Headers;
}

export type ApiErrorHandler = (error: Error) => void;

// Performance monitoring types
export interface PerformanceMark {
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  metadata?: Record<string, unknown>;
}

export interface PerformanceMarkOptions {
  [key: string]: string | number | boolean | null | undefined;
}

export interface PerformanceMonitorOptions {
  enabled: boolean;
  sampleRate: number;
  maxMarks: number;
}

// AppMessage is defined in messageUtils.ts - import from there if needed

// CRUD Types
export interface Conversation {
  id: string;
  title: string;
  tags: string[];
  archived: boolean;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationCreate {
  title: string;
  tags?: string[];
  archived?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ConversationUpdate {
  title?: string;
  tags?: string[];
  archived?: boolean;
  metadata?: Record<string, unknown>;
}

export interface MessageData {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface MessageCreate {
  conversation_id: string;
  role: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface MessageUpdate {
  content?: string;
  metadata?: Record<string, unknown>;
}

// Import Theme type from theme context for consistency
export type Theme = 'celestial-indigo' | 'veridian-twilight' | 'solaris-flare' | 'hunters-vision' | 'nebula' | 'crimson-ember' | 'cyberpunk-neon' | 'obsidian-slate';

export interface UserSettings {
  id: string;
  user_id: string;
  theme: Theme;
  ai_personality: string;
  response_style: string;
  memory_retention: boolean;
  auto_summarize: boolean;
  preferred_language: string;
  custom_prompts: string[];
  created_at: string;
  updated_at: string;
}

export interface UserSettingsCreate {
  user_id: string;
  theme?: Theme;
  ai_personality?: string;
  response_style?: string;
  memory_retention?: boolean;
  auto_summarize?: boolean;
  preferred_language?: string;
  custom_prompts?: string[];
}

export interface UserSettingsUpdate {
  theme?: Theme;
  ai_personality?: string;
  response_style?: string;
  memory_retention?: boolean;
  auto_summarize?: boolean;
  preferred_language?: string;
  custom_prompts?: string[];
}

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
  page: number;
  page_size: number;
}

export interface MessageListResponse {
  messages: MessageData[];
  total: number;
  page: number;
  page_size: number;
}
