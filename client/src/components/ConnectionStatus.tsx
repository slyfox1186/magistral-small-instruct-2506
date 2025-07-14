import React, { useState, useEffect } from 'react';
import { connectionManager, ConnectionStatus as ConnectionStatusType } from '@/utils/connectionManager';
import { logger } from '@/utils';

interface ConnectionStatusProps {
  className?: string;
  showWhenConnected?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ 
  className = '', 
  showWhenConnected = false 
}) => {
  const [status, setStatus] = useState<ConnectionStatusType>(connectionManager.getStatus());

  useEffect(() => {
    // Start connection monitoring
    connectionManager.start();

    // Subscribe to status changes
    const unsubscribe = connectionManager.onStatusChange((newStatus) => {
      setStatus(newStatus);
    });

    // Initial status check
    connectionManager.checkNow().catch((error) => {
      logger.error('Failed to check initial connection status:', error);
    });

    return () => {
      unsubscribe();
      connectionManager.stop();
    };
  }, []);

  // Don't show anything if connected and showWhenConnected is false
  if (status.connected && !showWhenConnected) {
    return null;
  }

  const getStatusColor = () => {
    if (status.connected && status.ready) return 'success';
    if (status.connected && !status.ready) return 'warning';
    return 'error';
  };

  const getStatusText = () => {
    if (status.connected && status.ready) return 'Connected';
    if (status.connected && !status.ready) return 'Backend Starting...';
    return 'Connecting to Backend...';
  };

  const getStatusIcon = () => {
    if (status.connected && status.ready) return '🟢';
    if (status.connected && !status.ready) return '🟡';
    return '🔴';
  };

  return (
    <div className={`connection-status connection-status--${getStatusColor()} ${className}`}>
      <span className="connection-status__icon" aria-label={getStatusText()}>
        {getStatusIcon()}
      </span>
      <span className="connection-status__text">
        {getStatusText()}
      </span>
      {status.error && !status.connected && (
        <span className="connection-status__error" title={status.error}>
          ⚠️
        </span>
      )}
    </div>
  );
};

export default ConnectionStatus;