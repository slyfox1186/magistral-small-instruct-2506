import config, { LogLevel } from '../utils/config';
import { LogParams } from './types';

/**
 * Logger class with configurable log levels
 * and context-based logging
 */
class Logger {
  private context: string;
  private loggingLevel: LogLevel = config.LOGGING_LEVEL;

  /**
   * Create a new logger instance
   * @param context An optional context name for the logger
   */
  constructor(context: string = 'App') {
    this.context = context;
  }

  /**
   * Check if a given log level is enabled
   * @param level The log level to check
   * @returns True if the level is enabled
   */
  private isLevelEnabled(level: LogLevel): boolean {
    return level <= this.loggingLevel;
  }

  /**
   * Format a log entry with timestamp and context
   * @param level The log level as a string
   * @param message The message to log
   * @returns The formatted log entry
   */
  private formatLogEntry(level: string, message: string): string {
    const timestamp = new Date().toISOString();
    return `[${timestamp}] [${level}] [${this.context}] ${message}`;
  }

  /**
   * Log an error message
   * @param message The message to log
   * @param args Optional additional arguments
   */
  error(message: string, ...args: LogParams): void {
    if (this.isLevelEnabled(LogLevel.ERROR)) {
      console.error(this.formatLogEntry('ERROR', message), ...args);

      if (config.ENABLE_ERROR_REPORTING) {
        // Here you would integrate with your error reporting service
        // e.g., Sentry, LogRocket, etc.
        this.reportError(message);
      }
    }
  }

  /**
   * Log a warning message
   * @param message The message to log
   * @param args Optional additional arguments
   */
  warn(message: string, ...args: LogParams): void {
    if (this.isLevelEnabled(LogLevel.WARN)) {
      console.warn(this.formatLogEntry('WARN', message), ...args);
    }
  }

  /**
   * Log an info message
   * @param message The message to log
   * @param args Optional additional arguments
   */
  info(message: string, ...args: LogParams): void {
    if (this.isLevelEnabled(LogLevel.INFO)) {
      // Note: Only console.warn and console.error are allowed per project guidelines
      // Info messages are treated as warnings for visibility
      console.warn(this.formatLogEntry('INFO', message), ...args);
    }
  }

  /**
   * Log a debug message
   * @param message The message to log
   * @param args Optional additional arguments
   */
  debug(message: string, ...args: LogParams): void {
    if (this.isLevelEnabled(LogLevel.DEBUG)) {
      // Note: Only console.warn and console.error are allowed per project guidelines
      // Debug messages are silently dropped in production
      if (this.loggingLevel === LogLevel.DEBUG) {
        console.warn(this.formatLogEntry('DEBUG', message), ...args);
      }
    }
  }

  /**
   * Log a trace message
   * @param message The message to log
   * @param args Optional additional arguments
   */
  trace(message: string, ...args: LogParams): void {
    if (this.isLevelEnabled(LogLevel.TRACE)) {
      // Note: Only console.warn and console.error are allowed per project guidelines
      // Trace messages are silently dropped in production
      if (this.loggingLevel === LogLevel.TRACE) {
        console.warn(this.formatLogEntry('TRACE', message), ...args);
      }
    }
  }

  /**
   * Create a child logger with a sub-context
   * @param subContext The sub-context for the child logger
   * @returns A new logger instance with the combined context
   */
  child(subContext: string): Logger {
    return new Logger(`${this.context}:${subContext}`);
  }

  /**
   * Report an error to an error tracking service
   * @param message The error message
   */
  private reportError(message: string): void {
    // Placeholder for error reporting integration

    // For now, just log that we would report this
    // Note: Only console.warn and console.error are allowed per project guidelines
    console.warn(`[Error Reporting] Would report error to service: ${message}`);
  }
}

// Export a default logger instance
export const logger = new Logger();

// Export a function to create context-specific loggers
export function getLogger(context: string): Logger {
  return new Logger(context);
}

export default logger;
