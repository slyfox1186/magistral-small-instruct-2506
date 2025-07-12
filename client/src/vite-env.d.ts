/// <reference types="vite/client" />

// Ensure DOM types are available
/// <reference lib="dom" />
/// <reference lib="dom.iterable" />

// Explicit global declarations for ESLint
declare global {
  const File: typeof globalThis.File;
  const FileReader: typeof globalThis.FileReader;
  const HTMLInputElement: typeof globalThis.HTMLInputElement;
  const FileList: typeof globalThis.FileList;
  const JSX: typeof globalThis.JSX;
  const SVGPathElement: typeof globalThis.SVGPathElement;
  const NodeListOf: typeof globalThis.NodeListOf;
  const HTMLLinkElement: typeof globalThis.HTMLLinkElement;
}

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL: string;
  readonly VITE_APP_NAME: string;
  readonly VITE_APP_VERSION: string;

  // Stream settings
  readonly VITE_MIN_RESPONSE_LENGTH: string;
  readonly VITE_COMPLETION_DEBOUNCE_MS: string;

  // UI settings
  readonly VITE_ANIMATION_TIMEOUT_MS: string;
  readonly VITE_MAX_TEXTAREA_HEIGHT_PX: string;
  readonly VITE_DEFAULT_TEXTAREA_HEIGHT_PX: string;

  // Features toggles
  readonly VITE_ENABLE_DEBUG_LOGGING: string;
  readonly VITE_ENABLE_MEMORY_FEATURES: string;

  // API and Network settings
  readonly VITE_API_TIMEOUT_MS: string;
  readonly VITE_MAX_RETRIES: string;
  readonly VITE_RETRY_DELAY_MS: string;

  // Message settings
  readonly VITE_MAX_MESSAGE_HISTORY: string;
  readonly VITE_MAX_MESSAGE_LENGTH: string;
  readonly VITE_MESSAGE_TRUNCATION_SUFFIX: string;

  // Error handling
  readonly VITE_ERROR_DISPLAY_DURATION_MS: string;
  readonly VITE_ENABLE_ERROR_REPORTING: string;

  // Theming
  readonly VITE_THEME_PRIMARY_COLOR: string;
  readonly VITE_THEME_SECONDARY_COLOR: string;
  readonly VITE_THEME_BACKGROUND_COLOR: string;
  readonly VITE_THEME_TEXT_COLOR: string;
  readonly VITE_ENABLE_DARK_MODE: string;
  readonly VITE_SYNTAX_HIGHLIGHT_THEME: string;

  // Performance
  readonly VITE_ENABLE_VIRTUALIZATION: string;
  readonly VITE_DEBOUNCE_USER_INPUT_MS: string;
  readonly VITE_THROTTLE_SCROLL_MS: string;
  readonly VITE_RENDER_BATCH_SIZE: string;

  // Accessibility
  readonly VITE_ENABLE_SCREEN_READER_SUPPORT: string;
  readonly VITE_MIN_CONTRAST_RATIO: string;
  readonly VITE_FONT_SIZE_BASE_PX: string;

  // Debugging
  readonly VITE_LOGGING_LEVEL: string;
  readonly VITE_ENABLE_PERFORMANCE_MONITORING: string;
  readonly VITE_SHOW_DEVELOPER_TOOLS: string;

  // more env variables...
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

