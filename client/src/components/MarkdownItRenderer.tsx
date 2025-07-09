import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-dark.css'; // Import the theme CSS
import DOMPurify from 'dompurify';
import '../styles/MarkdownIt.css';
import { logger } from '../utils';

// Component interface
export interface MarkdownItRendererProps {
  markdown: string;
  isStreaming?: boolean;
  className?: string;
}

/**
 * Component for rendering markdown content with syntax highlighting
 */
const MarkdownItRenderer: React.FC<MarkdownItRendererProps> = ({
  markdown = '',
  isStreaming = false,
  className = '',
}) => {
  const prevMarkdownRef = useRef<string>(markdown); // Store previous markdown

  // DOM reference for streaming updates
  const contentRef = useRef<HTMLDivElement>(null);

  // Image loading state
  const [imagesLoaded, setImagesLoaded] = useState(true);

  // Parse and render errors
  const [renderError, setRenderError] = useState<string | null>(null);

  // Create markdown-it instance with configuration
  const md = useMemo(() => {
    const markdownIt = new MarkdownIt({
      html: true,
      xhtmlOut: true,
      breaks: true,
      linkify: true,
      typographer: true,
      highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
          try {
            return (
              '<pre class="hljs"><code>' +
              hljs.highlight(str, { language: lang }).value +
              '</code></pre>'
            );
          } catch (error) {
            // Log error but continue with unhighlighted code
            logger.warn('Syntax highlighting failed for language:', lang, error);
          }
        }
        return '<pre class="hljs"><code>' + markdownIt.utils.escapeHtml(str) + '</code></pre>';
      },
    });

    // Add link attributes for security
    const defaultRender =
      markdownIt.renderer.rules.link_open ||
      ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));

    markdownIt.renderer.rules.link_open = function (tokens, idx, options, _env, self) {
      tokens[idx].attrPush(['target', '_blank']);
      tokens[idx].attrPush(['rel', 'noopener noreferrer']);

      // Extract category from title and add data-link-category attribute
      const titleAttr = tokens[idx].attrs.find((attr) => attr[0] === 'title');
      if (titleAttr && typeof titleAttr[1] === 'string' && titleAttr[1].startsWith('category:')) {
        const categoryName = titleAttr[1].substring('category:'.length);
        if (categoryName) {
          // Ensure categoryName is not empty
          tokens[idx].attrPush(['data-link-category', categoryName]);
        }
      }

      return defaultRender(tokens, idx, options, _env, self);
    };

    // Wrap tables in a container for horizontal scrolling
    const defaultTableOpen =
      markdownIt.renderer.rules.table_open ||
      ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));

    markdownIt.renderer.rules.table_open = function (tokens, idx, options, _env, self) {
      return '<div class="table-container">' + defaultTableOpen(tokens, idx, options, _env, self);
    };

    const defaultTableClose =
      markdownIt.renderer.rules.table_close ||
      ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));

    markdownIt.renderer.rules.table_close = function (tokens, idx, options, _env, self) {
      return defaultTableClose(tokens, idx, options, _env, self) + '</div>';
    };

    return markdownIt;
  }, []); // Remove markdown dependency - md instance doesn't need recreation

  /**
   * Performs a full re-render of the markdown, sanitizes, and updates the DOM.
   */
  const performFullRenderAndDOMUpdate = useCallback(
    (currentMarkdown: string) => {
      if (!contentRef.current) return;
      try {
        const rawHtml = md.render(currentMarkdown);
        const sanitizedHtml = DOMPurify.sanitize(rawHtml, {
          USE_PROFILES: { html: true },
          ALLOWED_TAGS: [
            'p',
            'br',
            'b',
            'i',
            'em',
            'strong',
            'a',
            'ul',
            'ol',
            'li',
            'code',
            'pre',
            'hr',
            'h1',
            'h2',
            'h3',
            'h4',
            'h5',
            'h6',
            'blockquote',
            'img',
            'table',
            'thead',
            'tbody',
            'tr',
            'th',
            'td',
            'del',
            'sup',
            'sub',
            'span',
            'u',
            'div', // Added div for table container
          ],
          ALLOWED_ATTR: [
            'href',
            'target',
            'rel',
            'class',
            'alt',
            'src',
            'loading',
            'onerror',
            'onload',
            'width',
            'height',
            'style',
            'data-link-category', // Add data-link-category here
          ],
          ALLOWED_CLASSES: {
            div: ['table-container'], // Only allow table-container class on divs
          },
        });

        contentRef.current.innerHTML = sanitizedHtml;
        prevMarkdownRef.current = currentMarkdown;

        const images = contentRef.current.querySelectorAll('img');
        if (images.length > 0) {
          setImagesLoaded(false);
          let loadedCount = 0;
          const handleImageLoad = () => {
            loadedCount++;
            if (loadedCount === images.length) {
              setImagesLoaded(true);
            }
          };
          const cleanupListeners: Array<() => void> = [];

          images.forEach((img) => {
            if (img.complete) {
              handleImageLoad();
            } else {
              img.addEventListener('load', handleImageLoad);
              img.addEventListener('error', handleImageLoad);

              // Store cleanup functions
              cleanupListeners.push(() => {
                img.removeEventListener('load', handleImageLoad);
                img.removeEventListener('error', handleImageLoad);
              });
            }
          });

          // Return cleanup function
          return () => {
            cleanupListeners.forEach((cleanup) => cleanup());
          };
        } else {
          setImagesLoaded(true);
        }
      } catch (error) {
        logger.error('Error during full render and DOM update:', error);
        setRenderError('Failed to render markdown content (full)');
        if (contentRef.current) {
          try {
            contentRef.current.textContent = currentMarkdown; // Fallback
          } catch {
            contentRef.current.innerHTML = '';
          }
        }
      }
    },
    [md]
  ); // md is stable

  /**
   * Renders markdown to HTML - optimized for streaming performance
   */
  useEffect(() => {
    setRenderError(null); // Reset error on new render attempt

    if (isStreaming) {
      // During streaming, use requestAnimationFrame to batch renders
      const frameId = requestAnimationFrame(() => {
        performFullRenderAndDOMUpdate(markdown);
      });

      return () => {
        cancelAnimationFrame(frameId);
      };
    } else {
      // For completed messages, render immediately
      performFullRenderAndDOMUpdate(markdown);
    }
  }, [markdown, performFullRenderAndDOMUpdate, isStreaming]);

  return (
    <div
      // No key prop needed here
      ref={contentRef}
      className={`markdown-content ${isStreaming ? 'streaming' : ''} ${imagesLoaded ? 'images-loaded' : 'images-loading'} ${renderError ? 'has-render-error' : ''} ${className.trim()}`}
    />
  );
};

export default MarkdownItRenderer;
