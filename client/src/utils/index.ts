/**
 * Utils index file - exports all utility functions
 *
 * Import from here instead of individual files to get all utilities:
 * import { logger, config, networkService } from './utils';
 */

// Import utilities
import config, { LogLevel } from './config';
import logger, { getLogger } from './logger';
import networkService, { NetworkService, HttpMethod, ApiResponse } from './network';
import performanceMonitor, { measureMarkdownRenderTime } from './performance';
import messageUtils, {
  AppMessage,
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  formatMessagesForAPI,
} from './messageUtils';
import { RequestOptions } from './types';
import connectionManager, { ConnectionStatus } from './connectionManager';

// Export configuration
export { config, LogLevel };

// Export logger
export { logger, getLogger };

// Export network utilities
export { networkService, NetworkService, type HttpMethod, type RequestOptions, type ApiResponse };

// Export performance monitoring
export { performanceMonitor, measureMarkdownRenderTime };

// Export message utilities
export {
  messageUtils,
  type AppMessage,
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  formatMessagesForAPI,
};

// Export connection management
export { connectionManager, type ConnectionStatus };

// Export a default object with all the utilities
export default {
  config,
  logger,
  getLogger,
  networkService,
  performanceMonitor,
  messageUtils,
};
