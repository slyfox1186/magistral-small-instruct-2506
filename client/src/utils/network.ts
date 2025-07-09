import config from './config';
import { getLogger } from './logger';
import { performanceMonitor } from './performance';
import { RequestOptions, StreamRequestOptions } from './types';

// Create a dedicated logger for network operations
const logger = getLogger('Network');

// Available HTTP methods
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

// Response interface for HTTP requests
export interface ApiResponse<T> {
  data: T | null;
  status: number;
  headers: Headers;
  error?: Error | null;
}

/**
 * Creates a controlled delay using Promise
 * @param ms Time to delay in milliseconds
 * @returns A Promise that resolves after the specified time
 */
const delay = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * Parses response based on content type
 * @param response Fetch Response object
 * @returns Parsed response data
 */
const parseResponse = async <T>(response: Response): Promise<T> => {
  const contentType = response.headers.get('Content-Type') || '';

  if (contentType.includes('application/json')) {
    return response.json() as Promise<T>;
  }
  if (contentType.includes('text/')) {
    return response.text() as unknown as Promise<T>;
  }
  if (contentType.includes('form')) {
    return response.formData() as unknown as Promise<T>;
  }
  if (contentType.includes('video/') || contentType.includes('audio/')) {
    return response.blob() as unknown as Promise<T>;
  }

  return response.text() as unknown as Promise<T>; // Default to text
};

/**
 * Network service for handling API requests with retry logic,
 * timeout handling, and error management
 */
export class NetworkService {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseUrl: string = config.BACKEND_URL) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };

    logger.info(`NetworkService initialized with baseUrl: ${this.baseUrl}`);
  }

  /**
   * Make a request to the API with retry logic and timeout handling
   * @param endpoint API endpoint (will be appended to baseUrl)
   * @param requestConfig Request configuration
   * @returns ApiResponse object with data and metadata
   */
  async request<T>(
    endpoint: string,
    requestConfig: RequestOptions = { method: 'GET' }
  ): Promise<ApiResponse<T>> {
    const {
      method = 'GET',
      headers = {},
      body, // body type comes from RequestOptions, which is Record<string, unknown> | FormData
      signal,
      retries = config.MAX_RETRIES,
      retryDelay = config.RETRY_DELAY_MS,
    } = requestConfig;

    const timeout = requestConfig.retryDelay || config.API_TIMEOUT_MS;

    // Build full URL (handle both absolute and relative URLs)
    const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;

    // Combine default headers with provided headers
    const requestHeadersInit: Record<string, string> = {
      ...this.defaultHeaders,
      ...headers,
    };

    // Build request options
    const options: RequestInit = {
      method,
      // headers will be set after body processing
      signal,
    };

    // Add body if provided
    if (body) {
      if (body instanceof FormData) {
        options.body = body;
        // For FormData, remove 'Content-Type' so browser can set it with boundary
        delete requestHeadersInit['Content-Type'];
      } else {
        options.body = typeof body === 'string' ? body : JSON.stringify(body);
        // Ensure Content-Type is application/json if not FormData and not otherwise specified
        if (!requestHeadersInit['Content-Type']) {
          requestHeadersInit['Content-Type'] = 'application/json';
        }
      }
    }
    options.headers = requestHeadersInit;

    // Create an abort controller for the timeout if a signal wasn't provided
    let timeoutId: number | undefined;
    let localAbortController: AbortController | undefined;

    if (!signal && timeout) {
      localAbortController = new AbortController();
      timeoutId = window.setTimeout(() => {
        if (localAbortController) {
          localAbortController.abort();
          logger.warn(`Request to ${url} aborted due to timeout (${timeout}ms)`);
        }
      }, timeout);

      options.signal = localAbortController.signal;
    }

    let attempts = 0;
    let lastError: Error | null = null;

    logger.debug(`Making ${method} request to ${url}`, {
      method,
      headers: requestHeadersInit,
      timeout,
      retries,
    });

    // Start performance measurement
    const markName = `api-${method}-${endpoint.replace(/[^a-zA-Z0-9]/g, '-')}`;
    performanceMonitor.startMark(markName, { method, url });

    try {
      // Attempt the request with retries
      while (attempts <= retries) {
        attempts++;

        try {
          const response = await fetch(url, options);

          // End performance measurement
          performanceMonitor.endMark(markName, {
            status: response.status,
            attempt: attempts,
          });

          // Clear timeout if we set one
          if (timeoutId !== undefined) {
            clearTimeout(timeoutId);
          }

          // If the response is not ok (status outside 200-299), throw an error
          if (!response.ok) {
            const errorData = await parseResponse(response).catch(() => null);
            const error = new Error(`HTTP Error: ${response.status} ${response.statusText}`);

            // Add additional context to the error using a type assertion
            const extendedError = error as Error & {
              status: number;
              statusText: string;
              data: unknown;
            };

            extendedError.status = response.status;
            extendedError.statusText = response.statusText;
            extendedError.data = errorData;

            // Log the error
            logger.error(`API error: ${response.status} ${response.statusText}`, {
              url,
              method,
              errorData,
            });

            throw extendedError;
          }

          // Parse and return successful response
          const data = await parseResponse<T>(response);

          logger.debug(`Request successful: ${url}`, {
            status: response.status,
            attempts,
          });

          return {
            data,
            status: response.status,
            headers: response.headers,
            error: null,
          };
        } catch (error) {
          lastError = error as Error;

          // Don't retry if the request was aborted
          if (error instanceof DOMException && error.name === 'AbortError') {
            performanceMonitor.endMark(markName, {
              error: 'aborted',
              attempt: attempts,
            });

            logger.warn(`Request aborted: ${url}`);

            throw new Error('Request aborted: timeout or manual cancellation');
          }

          // If this was the last retry, throw the error
          if (attempts > retries) {
            performanceMonitor.endMark(markName, {
              error: 'max_retries_exceeded',
              attempt: attempts,
            });

            logger.error(`Request failed after ${attempts} attempts: ${url}`, error);
            break;
          }

          // Log the retry
          logger.warn(`Retrying request (${attempts}/${retries + 1}): ${url}`, error);

          // Wait before retrying (with exponential backoff)
          const backoffDelay = retryDelay * Math.pow(2, attempts - 1);
          await delay(backoffDelay);
        }
      }

      // If we get here, all retries have failed
      throw lastError;
    } catch (error) {
      // End performance measurement if not already ended
      performanceMonitor.endMark(markName, { error: true });

      // Clear timeout if we set one
      if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
      }

      // Return error response with type safety
      const typedError = error as Error & { status?: number };

      return {
        data: null,
        status: typedError?.status || 0,
        headers: new Headers(),
        error: error as Error,
      };
    }
  }

  /**
   * Make a GET request
   * @param endpoint API endpoint
   * @param config Request configuration
   * @returns ApiResponse object
   */
  async get<T>(
    endpoint: string,
    config: Omit<RequestOptions, 'method' | 'body'> = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...config, method: 'GET' });
  }

  /**
   * Make a POST request
   * @param endpoint API endpoint
   * @param body Request body
   * @param config Request configuration
   * @returns ApiResponse object
   */
  async post<T>(
    endpoint: string,
    body: Record<string, unknown> | FormData,
    config: Omit<RequestOptions, 'method' | 'body'> = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'POST',
      body,
    });
  }

  /**
   * Make a PUT request
   * @param endpoint API endpoint
   * @param body Request body
   * @param config Request configuration
   * @returns ApiResponse object
   */
  async put<T>(
    endpoint: string,
    body: Record<string, unknown> | FormData,
    config: Omit<RequestOptions, 'method' | 'body'> = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'PUT',
      body,
    });
  }

  /**
   * Make a PATCH request
   * @param endpoint API endpoint
   * @param body Request body
   * @param config Request configuration
   * @returns ApiResponse object
   */
  async patch<T>(
    endpoint: string,
    body: Record<string, unknown> | FormData,
    config: Omit<RequestOptions, 'method' | 'body'> = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...config,
      method: 'PATCH',
      body,
    });
  }

  /**
   * Make a DELETE request
   * @param endpoint API endpoint
   * @param config Request configuration
   * @returns ApiResponse object
   */
  async delete<T>(
    endpoint: string,
    config: Omit<RequestOptions, 'method'> = {}
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...config, method: 'DELETE' });
  }

  /**
   * Make a request and return the raw Response object for streaming.
   * Handles basic setup, headers, and signal, but leaves stream processing to the caller.
   * @param endpoint API endpoint
   * @param requestConfig Request configuration (method, headers, body, signal)
   * @returns Promise resolving to the raw Fetch Response object
   * @throws Throws error on fetch failure or non-ok response status before streaming starts.
   */
  async streamRequest(
    endpoint: string,
    requestConfig: StreamRequestOptions = { method: 'POST' }
  ): Promise<Response> {
    const {
      method = 'POST',
      headers = {},
      body,
      signal,
      retries = requestConfig.retries || 2, // Default to 2 retries for streams
      retryDelay = requestConfig.retryDelay || 500, // Default 500ms between retries
    } = requestConfig;

    const timeout = requestConfig.retryDelay || config.API_TIMEOUT_MS;

    const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;

    // Enhanced headers for better stream handling
    const requestHeaders = {
      ...this.defaultHeaders,
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      Pragma: 'no-cache',
      Expires: '0',
      Connection: 'keep-alive',
      ...headers,
    };

    // Setup fetch options
    const getRequestOptions = (): RequestInit => ({
      method,
      headers: requestHeaders,
      body: body ? (typeof body === 'string' ? body : JSON.stringify(body)) : undefined,
      signal,
      mode: 'cors',
      credentials: 'include', // Important for cross-origin requests with cookies
      cache: 'no-store',
      redirect: 'follow',
    });

    // Timeout handling
    let timeoutId: number | undefined;
    let localAbortController: AbortController | undefined;

    if (!signal && timeout) {
      localAbortController = new AbortController();
      const effectiveSignal = localAbortController.signal;
      const options = getRequestOptions();
      options.signal = effectiveSignal;

      timeoutId = window.setTimeout(() => {
        localAbortController!.abort();
        logger.warn(`Stream request to ${url} aborted due to timeout (${timeout}ms)`);
      }, timeout);
    }

    logger.debug(`Initiating stream request (${method}) to ${url} with retries=${retries}`);

    // Implement retry logic for streams
    let attempts = 0;
    let lastError: Error | null = null;

    while (attempts <= retries) {
      attempts++;

      try {
        // Force cache invalidation
        const timestamp = Date.now();
        const cacheInvalidator = `_nocache=${timestamp}`;
        const finalUrl = url.includes('?')
          ? `${url}&${cacheInvalidator}`
          : `${url}?${cacheInvalidator}`;

        // Get fresh options for each attempt
        const options = getRequestOptions();
        if (localAbortController && !signal) {
          options.signal = localAbortController.signal;
        }

        // Attempt fetch
        const response = await fetch(finalUrl, options);

        // Clear timeout once connection is established
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
          timeoutId = undefined;
        }

        // Check response status
        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Could not read error details');
          const error = new Error(`HTTP Error: ${response.status} ${response.statusText}`);
          Object.assign(error, {
            status: response.status,
            statusText: response.statusText,
            data: errorText,
          });
          throw error;
        }

        // Verify stream capabilities
        if (!response.body) {
          logger.warn('Stream response missing ReadableStream body');
        } else {
          logger.debug(`Stream established successfully (attempt ${attempts})`);
        }

        // Return successful response
        return response;
      } catch (error) {
        // Clear timeout if active
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
          timeoutId = undefined;
        }

        // Handle specific abort case
        if (error instanceof DOMException && error.name === 'AbortError') {
          logger.warn(`Stream request aborted: ${url}`);
          throw new Error('Stream request aborted by user or timeout');
        }

        // Track error for potential retry
        lastError = error as Error;
        logger.warn(
          `Stream request failed (attempt ${attempts}/${retries + 1}): ${(error as Error).message || String(error)}`
        );

        // Stop retrying if we've hit the limit
        if (attempts > retries) {
          logger.error(`Stream request failed after ${attempts} attempts`);
          break;
        }

        // Wait before retrying with increasing backoff
        const backoffMs = retryDelay * Math.pow(1.5, attempts - 1);
        logger.debug(`Retrying in ${backoffMs}ms (attempt ${attempts}/${retries + 1})`);
        await delay(backoffMs);
      }
    }

    // If we reach here, all retries failed
    throw lastError || new Error('Stream request failed after all retry attempts');
  }
}

// Export a singleton instance
export const networkService = new NetworkService();
export default networkService;
