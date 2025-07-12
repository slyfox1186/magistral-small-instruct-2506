/**
 * Application configuration
 *
 * This file centralizes all configuration values and provides
 * sensible defaults when environment variables are not set.
 */

// Helper function to parse numeric environment variables
const parseNumericEnv = (value: string | undefined, defaultValue: number): number => {
  if (!value) return defaultValue;
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
};

// Helper function to parse boolean environment variables
const parseBooleanEnv = (value: string | undefined, defaultValue: boolean): boolean => {
  if (!value) return defaultValue;
  return value.toLowerCase() === 'true';
};

// Helper function to parse string environment variables
const parseStringEnv = (value: string | undefined, defaultValue: string): string => {
  return value || defaultValue;
};

// Helper function to parse float environment variables
const parseFloatEnv = (value: string | undefined, defaultValue: number): number => {
  if (!value) return defaultValue;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

/**
 * Logging levels enum
 */
export enum LogLevel {
  NONE = 0,
  ERROR = 1,
  WARN = 2,
  INFO = 3,
  DEBUG = 4,
  TRACE = 5,
}

// Application metadata
export const APP_NAME = parseStringEnv(import.meta.env.VITE_APP_NAME, 'Aria Chat');
export const APP_VERSION = parseStringEnv(import.meta.env.VITE_APP_VERSION, '1.0.0');

// Backend configuration
export const BACKEND_URL = parseStringEnv(import.meta.env.VITE_BACKEND_URL, '');
export const API_URL = parseStringEnv(
  import.meta.env.VITE_API_URL,
  BACKEND_URL ? `${BACKEND_URL}/api` : '/api'
);

// Stream settings
export const MIN_RESPONSE_LENGTH = parseNumericEnv(import.meta.env.VITE_MIN_RESPONSE_LENGTH, 10);
export const COMPLETION_DEBOUNCE_MS = parseNumericEnv(
  import.meta.env.VITE_COMPLETION_DEBOUNCE_MS,
  500
);

// UI settings
export const ANIMATION_TIMEOUT_MS = parseNumericEnv(import.meta.env.VITE_ANIMATION_TIMEOUT_MS, 300);
export const MAX_TEXTAREA_HEIGHT_REM = parseFloatEnv(
  import.meta.env.VITE_MAX_TEXTAREA_HEIGHT_REM,
  6
); // ~4 lines (96px / 16px)
export const DEFAULT_TEXTAREA_HEIGHT_REM = parseFloatEnv(
  import.meta.env.VITE_DEFAULT_TEXTAREA_HEIGHT_REM,
  3.0
); // ~1 line with proper padding and line-height (48px / 16px)

// Feature toggles
export const ENABLE_DEBUG_LOGGING = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_DEBUG_LOGGING,
  false
);
export const ENABLE_MEMORY_FEATURES = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_MEMORY_FEATURES,
  true
);

// API and Network settings
export const API_TIMEOUT_MS = parseNumericEnv(import.meta.env.VITE_API_TIMEOUT_MS, 30000);
export const MAX_RETRIES = parseNumericEnv(import.meta.env.VITE_MAX_RETRIES, 3);
export const RETRY_DELAY_MS = parseNumericEnv(import.meta.env.VITE_RETRY_DELAY_MS, 1000);

// Message settings
export const MAX_MESSAGE_HISTORY = parseNumericEnv(import.meta.env.VITE_MAX_MESSAGE_HISTORY, 100);
export const MAX_MESSAGE_LENGTH = parseNumericEnv(import.meta.env.VITE_MAX_MESSAGE_LENGTH, 10000);
export const MESSAGE_TRUNCATION_SUFFIX = parseStringEnv(
  import.meta.env.VITE_MESSAGE_TRUNCATION_SUFFIX,
  '...(truncated)'
);

// Error handling
export const ERROR_DISPLAY_DURATION_MS = parseNumericEnv(
  import.meta.env.VITE_ERROR_DISPLAY_DURATION_MS,
  5000
);
export const ENABLE_ERROR_REPORTING = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_ERROR_REPORTING,
  false
);

// Theming
export const THEME_PRIMARY_COLOR = parseStringEnv(
  import.meta.env.VITE_THEME_PRIMARY_COLOR,
  '#4a5568'
);
export const THEME_SECONDARY_COLOR = parseStringEnv(
  import.meta.env.VITE_THEME_SECONDARY_COLOR,
  '#e53e3e'
);
export const THEME_BACKGROUND_COLOR = parseStringEnv(
  import.meta.env.VITE_THEME_BACKGROUND_COLOR,
  '#f7fafc'
);
export const THEME_TEXT_COLOR = parseStringEnv(import.meta.env.VITE_THEME_TEXT_COLOR, '#1a202c');
export const ENABLE_DARK_MODE = parseBooleanEnv(import.meta.env.VITE_ENABLE_DARK_MODE, true);
export const SYNTAX_HIGHLIGHT_THEME = parseStringEnv(
  import.meta.env.VITE_SYNTAX_HIGHLIGHT_THEME,
  'vscode-dark'
);

// Performance
export const ENABLE_VIRTUALIZATION = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_VIRTUALIZATION,
  false
);
export const DEBOUNCE_USER_INPUT_MS = parseNumericEnv(
  import.meta.env.VITE_DEBOUNCE_USER_INPUT_MS,
  50
);
export const THROTTLE_SCROLL_MS = parseNumericEnv(import.meta.env.VITE_THROTTLE_SCROLL_MS, 100);
export const RENDER_BATCH_SIZE = parseNumericEnv(import.meta.env.VITE_RENDER_BATCH_SIZE, 10);

// Accessibility
export const ENABLE_SCREEN_READER_SUPPORT = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_SCREEN_READER_SUPPORT,
  true
);
export const MIN_CONTRAST_RATIO = parseFloatEnv(import.meta.env.VITE_MIN_CONTRAST_RATIO, 4.5);
export const FONT_SIZE_BASE_PX = parseNumericEnv(import.meta.env.VITE_FONT_SIZE_BASE_PX, 16);

// Debugging
// Force TRACE level for debugging, bypassing environment variable
export const LOGGING_LEVEL = LogLevel.TRACE;
export const ENABLE_PERFORMANCE_MONITORING = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_PERFORMANCE_MONITORING,
  true
);
export const SHOW_DEVELOPER_TOOLS = parseBooleanEnv(
  import.meta.env.VITE_SHOW_DEVELOPER_TOOLS,
  true
);

export default {
  // Application metadata
  APP_NAME,
  APP_VERSION,

  // Backend configuration
  BACKEND_URL,
  API_URL,

  // Stream settings
  MIN_RESPONSE_LENGTH,
  COMPLETION_DEBOUNCE_MS,

  // UI settings
  ANIMATION_TIMEOUT_MS,
  MAX_TEXTAREA_HEIGHT_REM,
  DEFAULT_TEXTAREA_HEIGHT_REM,

  // Feature toggles
  ENABLE_DEBUG_LOGGING,
  ENABLE_MEMORY_FEATURES,

  // API and Network settings
  API_TIMEOUT_MS,
  MAX_RETRIES,
  RETRY_DELAY_MS,

  // Message settings
  MAX_MESSAGE_HISTORY,
  MAX_MESSAGE_LENGTH,
  MESSAGE_TRUNCATION_SUFFIX,

  // Error handling
  ERROR_DISPLAY_DURATION_MS,
  ENABLE_ERROR_REPORTING,

  // Theming
  THEME_PRIMARY_COLOR,
  THEME_SECONDARY_COLOR,
  THEME_BACKGROUND_COLOR,
  THEME_TEXT_COLOR,
  ENABLE_DARK_MODE,
  SYNTAX_HIGHLIGHT_THEME,

  // Performance
  ENABLE_VIRTUALIZATION,
  DEBOUNCE_USER_INPUT_MS,
  THROTTLE_SCROLL_MS,
  RENDER_BATCH_SIZE,

  // Accessibility
  ENABLE_SCREEN_READER_SUPPORT,
  MIN_CONTRAST_RATIO,
  FONT_SIZE_BASE_PX,

  // Debugging
  LOGGING_LEVEL,
  ENABLE_PERFORMANCE_MONITORING,
  SHOW_DEVELOPER_TOOLS,
};
