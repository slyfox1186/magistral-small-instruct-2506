/**
 * Connection Manager - Handles backend connectivity with graceful startup
 * 
 * This utility provides:
 * - Backend readiness detection
 * - Connection status monitoring
 * - Graceful handling of startup timing issues
 */

import { crudApi } from '@/api/crud';
import { logger } from '@/utils';

export interface ConnectionStatus {
  connected: boolean;
  ready: boolean;
  lastCheck: Date;
  error?: string;
}

class ConnectionManager {
  private status: ConnectionStatus = {
    connected: false,
    ready: false,
    lastCheck: new Date(),
  };

  private checkInterval: number | null = null;
  private readonly CHECK_INTERVAL = 5000; // 5 seconds
  private readonly STARTUP_CHECK_INTERVAL = 1000; // 1 second during startup
  private listeners: Array<(status: ConnectionStatus) => void> = [];
  private isCheckingStartup = true;

  /**
   * Start monitoring connection status
   */
  start(): void {
    this.checkConnectionStatus();
    this.scheduleNextCheck();
  }

  /**
   * Stop monitoring connection status
   */
  stop(): void {
    if (this.checkInterval) {
      clearTimeout(this.checkInterval);
      this.checkInterval = null;
    }
  }

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return { ...this.status };
  }

  /**
   * Check if backend is ready for API calls
   */
  isReady(): boolean {
    return this.status.connected && this.status.ready;
  }

  /**
   * Subscribe to connection status changes
   */
  onStatusChange(callback: (status: ConnectionStatus) => void): () => void {
    this.listeners.push(callback);
    
    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * Wait for backend to be ready
   */
  async waitForReady(timeoutMs = 30000): Promise<boolean> {
    if (this.isReady()) {
      return true;
    }

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        unsubscribe();
        resolve(false);
      }, timeoutMs);

      const unsubscribe = this.onStatusChange((status) => {
        if (status.connected && status.ready) {
          clearTimeout(timeout);
          unsubscribe();
          resolve(true);
        }
      });
    });
  }

  /**
   * Force a connection check
   */
  async checkNow(): Promise<ConnectionStatus> {
    await this.checkConnectionStatus();
    return this.getStatus();
  }

  private async checkConnectionStatus(): Promise<void> {
    const checkStartTime = Date.now();
    const timestamp = new Date().toISOString();
    
    console.log(`[CONNECTION-MGR ${timestamp}] 🔍 Starting connection check...`);
    console.log(`[CONNECTION-MGR ${timestamp}] 📊 Current status:`, {
      connected: this.status.connected,
      ready: this.status.ready,
      isCheckingStartup: this.isCheckingStartup,
      lastCheck: this.status.lastCheck,
      error: this.status.error
    });
    
    try {
      console.log(`[CONNECTION-MGR ${timestamp}] 🌐 Calling crudApi.healthCheck()...`);
      
      const healthCheckStartTime = Date.now();
      const isHealthy = await crudApi.healthCheck();
      const healthCheckDuration = Date.now() - healthCheckStartTime;
      
      console.log(`[CONNECTION-MGR ${timestamp}] ✅ Health check completed in ${healthCheckDuration}ms:`, {
        isHealthy,
        url: 'http://localhost:8000/api/health',
        duration: `${healthCheckDuration}ms`
      });
      
      const newStatus: ConnectionStatus = {
        connected: isHealthy,
        ready: isHealthy,
        lastCheck: new Date(),
        error: isHealthy ? undefined : 'Backend health check failed',
      };

      console.log(`[CONNECTION-MGR ${timestamp}] 📋 New status created:`, newStatus);

      // If we successfully connected during startup phase, switch to normal intervals
      if (isHealthy && this.isCheckingStartup) {
        this.isCheckingStartup = false;
        console.log(`[CONNECTION-MGR ${timestamp}] 🎯 SUCCESS! Backend connection established, switching to normal monitoring`);
        logger.info('Backend connection established, switching to normal monitoring');
      }

      console.log(`[CONNECTION-MGR ${timestamp}] 📤 Updating status and notifying ${this.listeners.length} listeners...`);
      this.updateStatus(newStatus);
      
      const totalDuration = Date.now() - checkStartTime;
      console.log(`[CONNECTION-MGR ${timestamp}] ⏱️  Total check completed in ${totalDuration}ms`);
      
    } catch (error) {
      const checkDuration = Date.now() - checkStartTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown connection error';
      
      console.log(`[CONNECTION-MGR ${timestamp}] ❌ Health check FAILED after ${checkDuration}ms:`, {
        error: errorMessage,
        errorType: error?.constructor?.name,
        stack: error instanceof Error ? error.stack?.split('\n').slice(0, 3) : undefined
      });
      
      const newStatus: ConnectionStatus = {
        connected: false,
        ready: false,
        lastCheck: new Date(),
        error: errorMessage,
      };

      console.log(`[CONNECTION-MGR ${timestamp}] 📋 Error status created:`, newStatus);
      console.log(`[CONNECTION-MGR ${timestamp}] 📤 Updating status and notifying ${this.listeners.length} listeners...`);
      this.updateStatus(newStatus);
      
      if (this.isCheckingStartup) {
        // Check if it's a backend starting message
        if (errorMessage.includes('BACKEND_STARTING')) {
          console.log(`[CONNECTION-MGR ${timestamp}] 🚀 STARTUP PHASE: Backend is starting up (received from Vite proxy)`);
          logger.debug(`Backend starting up: ${errorMessage}`);
        } else {
          console.log(`[CONNECTION-MGR ${timestamp}] 🔄 STARTUP PHASE: Backend not ready yet: ${errorMessage}`);
          logger.debug(`Backend not ready yet: ${errorMessage}`);
        }
      } else {
        console.log(`[CONNECTION-MGR ${timestamp}] ⚠️  NORMAL PHASE: Backend connection lost: ${errorMessage}`);
        logger.warn(`Backend connection lost: ${errorMessage}`);
      }
    }
  }

  private updateStatus(newStatus: ConnectionStatus): void {
    const timestamp = new Date().toISOString();
    const wasConnected = this.status.connected;
    const oldStatus = { ...this.status };
    
    console.log(`[CONNECTION-MGR ${timestamp}] 🔄 Updating status:`, {
      from: oldStatus,
      to: newStatus,
      wasConnected,
      listenersCount: this.listeners.length
    });
    
    this.status = newStatus;

    // Notify listeners of status change
    console.log(`[CONNECTION-MGR ${timestamp}] 📢 Notifying ${this.listeners.length} listeners...`);
    this.listeners.forEach((callback, index) => {
      try {
        console.log(`[CONNECTION-MGR ${timestamp}] 📤 Calling listener ${index + 1}/${this.listeners.length}...`);
        callback(this.status);
        console.log(`[CONNECTION-MGR ${timestamp}] ✅ Listener ${index + 1} called successfully`);
      } catch (error) {
        console.log(`[CONNECTION-MGR ${timestamp}] ❌ Error in listener ${index + 1}:`, error);
        logger.error('Error in connection status listener:', error);
      }
    });

    // Log significant status changes
    if (!wasConnected && newStatus.connected) {
      console.log(`[CONNECTION-MGR ${timestamp}] 🎉 CONNECTION ESTABLISHED! Backend is now online!`);
      logger.info('Backend connection established');
    } else if (wasConnected && !newStatus.connected) {
      console.log(`[CONNECTION-MGR ${timestamp}] 💔 CONNECTION LOST! Backend went offline!`);
      logger.warn('Backend connection lost');
    } else if (wasConnected === newStatus.connected) {
      console.log(`[CONNECTION-MGR ${timestamp}] 🔄 Status updated but connection state unchanged (${newStatus.connected ? 'connected' : 'disconnected'})`);
    }
  }

  private scheduleNextCheck(): void {
    const timestamp = new Date().toISOString();
    
    if (this.checkInterval) {
      console.log(`[CONNECTION-MGR ${timestamp}] 🔄 Clearing existing check interval...`);
      clearTimeout(this.checkInterval);
    }

    const interval = this.isCheckingStartup ? 
      this.STARTUP_CHECK_INTERVAL : 
      this.CHECK_INTERVAL;

    console.log(`[CONNECTION-MGR ${timestamp}] ⏰ Scheduling next check in ${interval}ms (${this.isCheckingStartup ? 'STARTUP' : 'NORMAL'} mode)`);

    this.checkInterval = window.setTimeout(() => {
      console.log(`[CONNECTION-MGR ${timestamp}] 🚀 Timer fired! Starting next check cycle...`);
      this.checkConnectionStatus().then(() => {
        console.log(`[CONNECTION-MGR ${timestamp}] ✅ Check cycle completed, scheduling next...`);
        this.scheduleNextCheck();
      }).catch((error) => {
        console.log(`[CONNECTION-MGR ${timestamp}] ❌ Check cycle failed:`, error);
        this.scheduleNextCheck(); // Still schedule next check even if this one failed
      });
    }, interval);
    
    console.log(`[CONNECTION-MGR ${timestamp}] ✅ Next check scheduled with timeout ID:`, this.checkInterval);
  }
}

// Export singleton instance
export const connectionManager = new ConnectionManager();
export default connectionManager;