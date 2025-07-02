/**
 * Utility functions for working with markdown streams
 */

/**
 * Fixes sentence spacing issues in text
 * (adds spaces after periods, exclamation points, question marks when followed by capital letters)
 */
export function fixSentenceSpacing(text: string): string {
  // Only apply the spacing fix for periods, question marks, and exclamation points
  // This regex adds a space after these punctuation marks if they're followed by a letter or number
  return text.replace(/([.?!…])([A-Za-z0-9])/g, '$1 $2');
}

/**
 * A buffer that stores recent tokens to help detect markdown structures during streaming
 */
class MarkdownTokenBuffer {
  private buffer: string = '';
  private readonly maxBufferSize: number = 2000; // Maximum number of characters to store

  constructor(initialContent: string = '') {
    this.buffer = initialContent;
  }

  /**
   * Add a new token to the buffer
   */
  addToken(token: string): void {
    this.buffer += token;

    // Trim buffer if it gets too large
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer = this.buffer.slice(-this.maxBufferSize);
    }
  }

  /**
   * Get current buffer content
   */
  getContent(): string {
    return this.buffer;
  }

  /**
   * Check if the buffer contains a language-specific code block opening
   */
  hasLanguageCodeBlockOpening(): boolean {
    // Match pattern like ```python, ```javascript, etc.
    const codeBlockRegex = /```[a-zA-Z0-9-]+/g;
    return codeBlockRegex.test(this.buffer);
  }

  /**
   * Detect if we have unbalanced code blocks
   */
  hasUnbalancedCodeBlocks(): boolean {
    const codeBlockOpenings = this.buffer.match(/```(?:[a-zA-Z0-9-]*)/g) || [];
    const codeBlockClosings = this.buffer.match(/```(?!\w)/g) || [];
    return codeBlockOpenings.length > codeBlockClosings.length;
  }

  /**
   * Get the language of the last opened code block
   */
  getLastCodeBlockLanguage(): string | null {
    const matches = this.buffer.match(/```([a-zA-Z0-9-]+)(?!.*```[a-zA-Z0-9-]+)/);
    return matches ? matches[1] : null;
  }
}

// Singleton pattern for a global token buffer
let globalTokenBuffer = new MarkdownTokenBuffer();

/**
 * Process a single token during streaming
 * Use this for each token as it arrives from the backend
 */
export function processStreamingToken(token: string): void {
  globalTokenBuffer.addToken(token);
}

/**
 * Reset the token buffer
 */
export function resetTokenBuffer(): void {
  globalTokenBuffer = new MarkdownTokenBuffer();
}

/**
 * Balances unclosed markdown elements during streaming
 */
export function balanceMarkdownForStreaming(text: string): string {
  // Use the global buffer to help with detecting code block patterns
  // but still analyze the provided text for balancing
  let result = text;

  // Fix for unclosed code blocks - detect both plain ``` and language-specific blocks like ```python
  // First find all code block markers, including those with language specifiers
  const codeBlockOpenings = text.match(/```(?:[a-zA-Z0-9-]*)/g) || [];
  const codeBlockClosings = text.match(/```(?!\w)/g) || []; // Only count ``` not followed by a word char as closing

  // Check if the code block has proper content or is incomplete
  if (codeBlockOpenings.length > codeBlockClosings.length) {
    // Get the last opening fence
    const lastFencePattern = /```(?:[a-zA-Z0-9-]*)/g;
    let lastMatch;
    let lastFenceIndex = -1;
    let lastFenceMatch = '';

    // Find the last code fence opening
    while ((lastMatch = lastFencePattern.exec(text)) !== null) {
      lastFenceIndex = lastMatch.index;
      lastFenceMatch = lastMatch[0];
    }

    if (lastFenceIndex >= 0) {
      const textAfterLastFence = text.substring(lastFenceIndex + lastFenceMatch.length);

      // Fix hanging space after opening fence, which can cause rendering issues
      if (lastFenceMatch.endsWith(' ')) {
        // Remove the trailing space and ensure language is properly attached
        const languagePart = lastFenceMatch.trim();
        result = result.replace(lastFenceMatch, languagePart);
      }

      // Only add closing fence if there's at least some content
      // or if it's clearly an incomplete code block
      if (
        textAfterLastFence.includes('\n') ||
        textAfterLastFence.length > 10 ||
        /\bfunction\b|\bdef\b|\bclass\b|\bconst\b|\blet\b|\bvar\b/.test(textAfterLastFence)
      ) {
        // Simply close the code block without any language-specific implementation
        result += '\n```';
      }
    }
  }

  // Check and balance unclosed inline code ticks
  const singleTickCount = (text.match(/`(?!``)/g) || []).length;
  if (singleTickCount % 2 !== 0) {
    // Only add the closing tick if it's not immediately after an opening triple tick
    if (!text.endsWith('```') && !text.endsWith('``')) {
      result += '`';
    }
  }

  // Check and balance unclosed parentheses (could be part of links)
  const openParenCount = (text.match(/\(/g) || []).length;
  const closeParenCount = (text.match(/\)/g) || []).length;
  if (openParenCount > closeParenCount) {
    result += ')'.repeat(openParenCount - closeParenCount);
  }

  // Check and balance unclosed square brackets (could be part of links)
  const openBracketCount = (text.match(/\[/g) || []).length;
  const closeBracketCount = (text.match(/\]/g) || []).length;
  if (openBracketCount > closeBracketCount) {
    result += ']'.repeat(openBracketCount - closeBracketCount);
  }

  // Handle unclosed bold/italic markers
  const asteriskCount = (text.match(/\*/g) || []).length;
  if (asteriskCount % 2 !== 0) {
    result += '*';
  }

  // Handle unclosed double asterisks (bold)
  const doubleAsteriskCount = (text.match(/\*\*/g) || []).length;
  if (doubleAsteriskCount % 2 !== 0) {
    result += '**';
  }

  // Handle unclosed underscores (italic)
  const underscoreCount = (text.match(/_/g) || []).length;
  if (underscoreCount % 2 !== 0) {
    result += '_';
  }

  // Handle unclosed double underscores (bold)
  const doubleUnderscoreCount = (text.match(/__/g) || []).length;
  if (doubleUnderscoreCount % 2 !== 0) {
    result += '__';
  }

  // Handle unclosed HTML tags (this is a simple heuristic and won't catch all cases)
  const lastOpenTag = text.match(/<([a-zA-Z0-9]+)[^>]*>(?![^<]*<\/\1>)/);
  if (lastOpenTag) {
    result += `</${lastOpenTag[1]}>`;
  }

  // Handle incomplete list items (prevents broken rendering during streaming)
  // Check if the last line is only a list marker without content
  const lines = result.split('\n');
  if (lines.length > 0) {
    const lastLine = lines[lines.length - 1];
    // Regex to detect incomplete list items: "- ", "* ", "1. ", "  - ", "    * ", etc.
    // Also handles task lists: "- [ ] ", "- [x] "
    const incompleteListRegex = /^\s*([-*+]|(\d+\.))\s*(\[[ xX]\])?\s*$/;

    if (incompleteListRegex.test(lastLine)) {
      // Remove the incomplete list item to prevent broken rendering
      // It will be added back when more content arrives
      const remainingLines = lines.slice(0, -1);
      result = remainingLines.join('\n');

      // If we removed the only line, ensure we don't return empty string
      if (remainingLines.length === 0 && lines.length === 1) {
        result = ''; // Return empty instead of just the list marker
      }
    }
  }

  return result;
}

/**
 * Decodes HTML entities in markdown content
 */
export function decodeHtmlEntities(text: string): string {
  return text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x2F;/g, '/')
    .replace(/&mdash;/g, '—')
    .replace(/&ndash;/g, '–')
    .replace(/&lsquo;/g, "'")
    .replace(/&rsquo;/g, "'")
    .replace(/&ldquo;/g, '"')
    .replace(/&rdquo;/g, '"')
    .replace(/&hellip;/g, '…')
    .replace(/&amp;lt;/g, '<') // Nested HTML entities
    .replace(/&amp;gt;/g, '>') // Nested HTML entities
    .replace(/&amp;quot;/g, '"') // Nested HTML entities
    .replace(/&amp;amp;/g, '&') // Nested HTML entities
    .replace(/&#x(\d+);/g, (_: string, hex: string) => String.fromCharCode(parseInt(hex, 16)));
}

/**
 * Ensures code blocks preserve their exact formatting by protecting them
 * from unwanted transformations during markdown processing
 */
export function preserveCodeFormatting(text: string): string {
  // Already a perfect copy - text is returned as-is to maintain formatting
  // The goal is to make sure the markdown parser will preserve code blocks exactly
  // The CSS fixes we've added should now ensure proper display
  return text;
}
