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

// Chat message types
export interface AppMessage {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: Date;
  nodeRef?: React.RefObject<HTMLDivElement>;
}
