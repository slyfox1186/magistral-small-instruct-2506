import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Helper function to check if a string contains a substring (TypeScript-friendly)
function contains(source: string, searchString: string): boolean {
  return source.indexOf(searchString) !== -1;
}

// Helper function to check if an array contains a value (TypeScript-friendly)
function arrayContains<T>(arr: T[], item: T): boolean {
  return arr.indexOf(item) !== -1;
}

// Helper function to group language files
function getLangGroup(lang: string): string {
  // Common language groups
  const webLanguages = ['html', 'css', 'javascript', 'typescript', 'jsx', 'tsx', 'json', 'xml'];
  const scriptingLanguages = ['python', 'ruby', 'perl', 'php', 'shell', 'bash', 'powershell'];
  const systemLanguages = ['c', 'cpp', 'csharp', 'java', 'kotlin', 'swift', 'go', 'rust'];
  const dataLanguages = ['sql', 'yaml', 'toml', 'ini', 'csv'];
  const markupLanguages = ['markdown', 'latex', 'asciidoc', 'restructuredtext'];

  if (arrayContains(webLanguages, lang)) return 'web';
  if (arrayContains(scriptingLanguages, lang)) return 'scripting';
  if (arrayContains(systemLanguages, lang)) return 'system';
  if (arrayContains(dataLanguages, lang)) return 'data';
  if (arrayContains(markupLanguages, lang)) return 'markup';

  // Use the first letter for other languages
  return 'misc';
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 4000,
    // Force cache busting for CSS files during development
    force: true,
    // Proxy API requests to the backend during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        // Add timeout and retry configuration to handle backend startup timing
        timeout: 10000, // 10 second timeout
        // Configure proxy to be more resilient during backend startup
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('[VITE-PROXY] Backend connection error (backend may still be starting):', err.code);
            // Send a 503 instead of crashing
            if (!res.headersSent) {
              res.writeHead(503, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
              });
              res.end(JSON.stringify({ 
                error: 'Backend starting up, please wait...', 
                code: 'BACKEND_STARTING',
                retry: true 
              }));
            }
          });
          
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log(`[VITE-PROXY] ${req.method} ${req.url} -> ${options.target}${req.url}`);
          });
        },
      },
    },
  },
  css: {
    // Force CSS cache invalidation
    devSourcemap: true,
    preprocessorOptions: {
      scss: {
        // Use modern Sass API to avoid deprecation warnings
        api: 'modern-compiler',
        // Silently ignore deprecation warnings if any remain
        silenceDeprecations: ['legacy-js-api'],
      },
    },
  },
  build: {
    sourcemap: false, // Disable sourcemaps in production for smaller files
    minify: 'terser', // Use terser for better minification
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.logs in production
        drop_debugger: true,
      },
    },
    // Configure code splitting and chunking
    rollupOptions: {
      output: {
        // Ensure assets are properly named for better caching
        assetFileNames: 'assets/[name]-[hash][extname]',
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        manualChunks: (id: string) => {
          // Create separate chunks for major dependencies
          if (contains(id, 'node_modules')) {
            // Core React packages
            if (
              contains(id, 'react/') ||
              contains(id, 'react-dom/') ||
              contains(id, 'scheduler/') ||
              contains(id, 'react-transition-group/')
            ) {
              return 'vendor-react';
            }

            // Markdown-it and DOMPurify
            if (contains(id, 'markdown-it') || contains(id, 'dompurify')) {
              return 'vendor-markdown-it';
            }

            // Split highlight.js core from languages
            if (contains(id, 'highlight.js')) {
              if (contains(id, 'highlight.js/lib/languages/')) {
                // Group language files into smaller chunks
                const lang = id.split('languages/')[1].split('.')[0];
                const langGroup = getLangGroup(lang);
                return `highlight-lang-${langGroup}`;
              }
              if (contains(id, 'highlight.js/lib/core')) {
                return 'highlight-core';
              }
              if (contains(id, 'highlight.js/styles')) {
                return 'highlight-styles';
              }
              return 'highlight-other';
            }

            // UI/utility packages
            if (contains(id, 'uuid') || contains(id, 'axios') || contains(id, 'lodash')) {
              return 'vendor-utils';
            }

            // All other node_modules
            return 'vendor-other';
          }

          // Split application code by major features
          if (contains(id, '/components/')) {
            return 'app-components';
          }

          if (contains(id, '/utils/')) {
            return 'app-utils';
          }

          // API and state management
          if (contains(id, '/api/') || contains(id, '/store/')) {
            return 'app-state';
          }
        },
      },
    },
    // Set cache busting strategy
    cssCodeSplit: true,
    // Increase the warning limit to avoid unnecessary warnings
    chunkSizeWarningLimit: 1200,
  },
});
