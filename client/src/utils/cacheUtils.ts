/**
 * Cache management utilities for the Neural Consciousness Chat System
 */

import { logger } from './index';

export class CacheManager {
  private static readonly CACHE_VERSION_KEY = 'neural_chat_cache_version';
  private static readonly CURRENT_VERSION = Date.now().toString();

  /**
   * Force reload CSS files with cache busting
   */
  static forceCSSReload(): void {
    if (import.meta.env.DEV) {
      const links = document.querySelectorAll('link[rel="stylesheet"]') as NodeListOf<HTMLLinkElement>;
      
      links.forEach(link => {
        const href = link.href;
        if (href) {
          // Create new URL with timestamp to bust cache
          const url = new URL(href);
          url.searchParams.set('v', Date.now().toString());
          url.searchParams.set('cache_bust', 'true');
          
          // Clone the link element
          const newLink = link.cloneNode() as HTMLLinkElement;
          newLink.href = url.toString();
          
          // Replace old link with new one
          link.parentNode?.insertBefore(newLink, link.nextSibling);
          
          // Remove old link after new one loads
          newLink.addEventListener('load', () => {
            link.remove();
          });
          
          logger.debug(`🔄 CSS cache busted: ${href} → ${newLink.href}`);
        }
      });
    }
  }

  /**
   * Clear all development caches
   */
  static clearDevelopmentCaches(): void {
    if (import.meta.env.DEV) {
      // Clear localStorage items related to caching
      const cacheKeys = Object.keys(localStorage).filter(key => 
        key.includes('vite') || 
        key.includes('cache') || 
        key.includes('css') ||
        key.includes('dev') ||
        key.startsWith('__vite')
      );
      
      cacheKeys.forEach(key => {
        localStorage.removeItem(key);
        logger.debug(`🗑️ Cleared localStorage: ${key}`);
      });
      
      // Clear sessionStorage items
      const sessionKeys = Object.keys(sessionStorage).filter(key => 
        key.includes('vite') || key.includes('cache')
      );
      
      sessionKeys.forEach(key => {
        sessionStorage.removeItem(key);
        logger.debug(`🗑️ Cleared sessionStorage: ${key}`);
      });
      
      // Force reload CSS
      this.forceCSSReload();
      
      logger.debug('🧹 Development caches cleared');
    }
  }

  /**
   * Check if cache version has changed and clear if needed
   */
  static checkCacheVersion(): void {
    if (import.meta.env.DEV) {
      const storedVersion = localStorage.getItem(this.CACHE_VERSION_KEY);
      
      if (!storedVersion || storedVersion !== this.CURRENT_VERSION) {
        logger.debug('🔄 Cache version changed, clearing development caches...');
        this.clearDevelopmentCaches();
        localStorage.setItem(this.CACHE_VERSION_KEY, this.CURRENT_VERSION);
      }
    }
  }

  /**
   * Force browser to reload with cache bypass
   */
  static hardReload(): void {
    if (import.meta.env.DEV) {
      logger.debug('🔄 Performing hard reload with cache bypass...');
      window.location.reload();
    }
  }

  /**
   * Initialize cache management for development
   */
  static initDevelopment(): void {
    if (import.meta.env.DEV) {
      logger.debug('🚀 Cache management initialized for development');
      
      // Check cache version on load
      this.checkCacheVersion();
      
      // Add global cache control functions for debugging
      interface WindowWithDebugFunctions extends Window {
        __clearCache?: () => void;
        __hardReload?: () => void;
        __forceCSSReload?: () => void;
      }
      
      const debugWindow = window as WindowWithDebugFunctions;
      debugWindow.__clearCache = () => this.clearDevelopmentCaches();
      debugWindow.__hardReload = () => this.hardReload();
      debugWindow.__forceCSSReload = () => this.forceCSSReload();
      
      logger.debug('💡 Debug functions available: __clearCache(), __hardReload(), __forceCSSReload()');
    }
  }
}

// Auto-initialize in development
if (import.meta.env.DEV) {
  CacheManager.initDevelopment();
}

export default CacheManager;