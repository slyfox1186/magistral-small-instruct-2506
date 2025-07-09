import config from './config';
import { getLogger } from './logger';
import { PerformanceMark, PerformanceMarkOptions } from './types';

// Create a dedicated logger for performance monitoring
const logger = getLogger('Performance');

/**
 * Performance monitoring utility for tracking component render times,
 * network requests, and other performance metrics
 */
class PerformanceMonitor {
  private enabled: boolean;
  private marks: Map<string, PerformanceMark>;
  private measures: Map<string, PerformanceMark>;

  constructor() {
    this.enabled = config.ENABLE_PERFORMANCE_MONITORING;
    this.marks = new Map();
    this.measures = new Map();

    if (this.enabled) {
      logger.info('Performance monitoring initialized');
      this.setupObservers();
    }
  }

  /**
   * Sets up performance observers for long tasks, resources, etc.
   */
  private setupObservers(): void {
    // Check if browser supports PerformanceObserver
    if (typeof PerformanceObserver !== 'undefined') {
      try {
        // Only register longtask observer if supported
        if (
          'supportedEntryTypes' in PerformanceObserver &&
          Array.isArray(PerformanceObserver.supportedEntryTypes) &&
          PerformanceObserver.supportedEntryTypes.includes('longtask')
        ) {
          // Observe long tasks (tasks that block the main thread for >50ms)
          const longTaskObserver = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              logger.warn(`Long task detected: ${entry.duration.toFixed(2)}ms`, entry);
              if (entry.duration > 100) {
                this.reportPerformanceIssue('long-task', {
                  duration: entry.duration,
                  entryType: entry.entryType,
                  startTime: entry.startTime,
                });
              }
            }
          });
          longTaskObserver.observe({ entryTypes: ['longtask'] });
        } else {
          logger.info('Longtask PerformanceObserver not supported in this browser');
        }

        // Observe resource loading
        const resourceObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.duration > 1000) {
              // Log slow resource loads (>1s)
              logger.warn(`Slow resource load: ${entry.name} - ${entry.duration.toFixed(2)}ms`);
            }
          }
        });
        resourceObserver.observe({ entryTypes: ['resource'] });

        logger.info('Performance observers set up successfully');
      } catch (e) {
        logger.error('Failed to setup performance observers', e);
      }
    }
  }

  /**
   * Start timing an operation
   * @param markName A unique identifier for this timing operation
   * @param metadata Optional metadata about what's being timed
   */
  startMark(markName: string, metadata?: PerformanceMarkOptions): void {
    if (!this.enabled) return;

    this.marks.set(markName, {
      name: markName,
      startTime: performance.now(),
      metadata,
    });

    if (metadata) {
      logger.debug(`Started timing ${markName}`, metadata);
    } else {
      logger.debug(`Started timing ${markName}`);
    }
  }

  /**
   * End timing an operation and record the measurement
   * @param markName The identifier used in startMark
   * @param additionalMetadata Any additional metadata to add to the measure
   * @returns The duration in milliseconds, or undefined if the mark wasn't found
   */
  endMark(markName: string, additionalMetadata?: PerformanceMarkOptions): number | undefined {
    if (!this.enabled) return undefined;

    const mark = this.marks.get(markName);
    if (!mark) {
      logger.warn(`Attempted to end timing for unknown mark: ${markName}`);
      return undefined;
    }

    const endTime = performance.now();
    const duration = endTime - mark.startTime;

    // Combine original metadata with additional metadata
    const metadata = {
      ...mark.metadata,
      ...additionalMetadata,
    };

    const completedMark: PerformanceMark = {
      name: markName,
      startTime: mark.startTime,
      endTime,
      duration,
      metadata,
    };

    this.measures.set(markName, completedMark);
    this.marks.delete(markName);

    if (metadata) {
      logger.debug(`Ended timing ${markName}: ${duration.toFixed(2)}ms`, metadata);
    } else {
      logger.debug(`Ended timing ${markName}: ${duration.toFixed(2)}ms`);
    }

    return duration;
  }

  /**
   * Measure the execution time of a function
   * @param fn The function to measure
   * @param markName A name for this measurement
   * @param metadata Optional metadata about what's being measured
   * @returns The result of the function
   */
  measureFn<T>(fn: () => T, markName: string, metadata?: PerformanceMarkOptions): T {
    if (!this.enabled) return fn();

    this.startMark(markName, metadata);
    const result = fn();
    this.endMark(markName);
    return result;
  }

  /**
   * Measure the execution time of an async function
   * @param fn The async function to measure
   * @param markName A name for this measurement
   * @param metadata Optional metadata about what's being measured
   * @returns A promise that resolves to the result of the function
   */
  async measureAsyncFn<T>(
    fn: () => Promise<T>,
    markName: string,
    metadata?: PerformanceMarkOptions
  ): Promise<T> {
    if (!this.enabled) return fn();

    this.startMark(markName, metadata);
    try {
      const result = await fn();
      this.endMark(markName);
      return result;
    } catch (error) {
      this.endMark(markName, { error: true });
      throw error;
    }
  }

  /**
   * Get all recorded measurements
   * @returns A map of all measurements
   */
  getMeasurements(): Map<string, PerformanceMark> {
    return this.measures;
  }

  /**
   * Clear all measurements and marks
   */
  clearMeasurements(): void {
    this.marks.clear();
    this.measures.clear();
    logger.debug('Cleared all performance measurements');
  }

  /**
   * Report a performance issue to monitoring services (if enabled)
   * @param issueType The type of performance issue
   * @param details Details about the issue
   */
  private reportPerformanceIssue(issueType: string, details: Record<string, unknown>): void {
    if (!this.enabled) return;

    // Placeholder for sending to performance monitoring service

    logger.warn(
      `Performance issue (${issueType}) would be reported to monitoring service`,
      details
    );
  }
}

export const performanceMonitor = new PerformanceMonitor();
export default performanceMonitor;

// Add markdown rendering performance monitoring
export const measureMarkdownRenderTime = (label: string, callback: () => void): void => {
  const start = performance.now();
  callback();
  const end = performance.now();
  const duration = end - start;
  console.warn(`Markdown Render [${label}]: ${duration.toFixed(2)}ms`);
};
