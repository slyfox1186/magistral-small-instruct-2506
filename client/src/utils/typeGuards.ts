/**
 * Type guards for safe JSON parsing and validation.
 * Eliminates 'any' types and provides runtime type safety.
 */

import { AppMessage } from './types';

/**
 * Type guard for saved message data from localStorage
 */
export function isAppMessageArray(data: unknown): data is Omit<AppMessage, 'nodeRef'>[] {
  if (!Array.isArray(data)) return false;

  return data.every(
    (item) =>
      typeof item === 'object' &&
      item !== null &&
      'id' in item &&
      'role' in item &&
      'content' in item &&
      typeof item.id === 'string' &&
      typeof item.content === 'string' &&
      (item.role === 'user' || item.role === 'assistant' || item.role === 'system')
  );
}

/**
 * Type guard for streaming token data
 */
export interface StreamTokenChunk {
  token: {
    text: string;
  };
}

export function isTokenChunk(data: unknown): data is StreamTokenChunk {
  if (typeof data !== 'object' || data === null) return false;

  const obj = data as Record<string, unknown>;
  if (!('token' in obj)) return false;

  const token = obj.token;
  if (typeof token !== 'object' || token === null) return false;

  const tokenObj = token as Record<string, unknown>;
  return 'text' in tokenObj && typeof tokenObj.text === 'string';
}

/**
 * Type guard for stream completion signal
 */
export interface StreamDoneChunk {
  done: true;
}

export function isDoneChunk(data: unknown): data is StreamDoneChunk {
  if (typeof data !== 'object' || data === null) return false;

  const obj = data as Record<string, unknown>;
  return 'done' in obj && obj.done === true && Object.keys(obj).length === 1;
}

/**
 * Type guard for error chunks
 */
export interface StreamErrorChunk {
  error: string;
}

export function isErrorChunk(data: unknown): data is StreamErrorChunk {
  if (typeof data !== 'object' || data === null) return false;

  const obj = data as Record<string, unknown>;
  return 'error' in obj && typeof obj.error === 'string';
}

/**
 * Safe JSON parser with type validation
 */
export function safeJsonParse<T>(
  jsonString: string,
  typeGuard: (data: unknown) => data is T,
  fallback: T
): T {
  try {
    const parsed = JSON.parse(jsonString);
    if (typeGuard(parsed)) {
      return parsed;
    } else {
      console.warn('JSON data failed type validation:', parsed);
      return fallback;
    }
  } catch (error) {
    console.error('JSON parse error:', error);
    return fallback;
  }
}

/**
 * Safe localStorage getter with type validation
 */
export function safeGetFromLocalStorage<T>(
  key: string,
  typeGuard: (data: unknown) => data is T,
  fallback: T
): T {
  try {
    const item = localStorage.getItem(key);
    if (item === null) return fallback;

    return safeJsonParse(item, typeGuard, fallback);
  } catch (error) {
    console.error(`Error reading from localStorage key "${key}":`, error);
    return fallback;
  }
}
