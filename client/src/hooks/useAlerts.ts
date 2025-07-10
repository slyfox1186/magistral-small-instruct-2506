import { useContext, useCallback } from 'react';
import { AlertContext } from '../contexts/AlertContextDefinition';

// Custom hook for using the alert context
export const useAlerts = () => {
  const context = useContext(AlertContext);
  
  if (!context) {
    throw new Error('useAlerts must be used within an AlertProvider');
  }
  
  return context;
};

// Convenience functions for different alert types
export const useAlert = () => {
  const { addAlert } = useAlerts();

  return {
    success: useCallback((message: string, duration?: number) => {
      addAlert({ type: 'success', message, duration });
    }, [addAlert]),
    
    error: useCallback((message: string, duration?: number) => {
      addAlert({ type: 'error', message, duration });
    }, [addAlert]),
    
    warning: useCallback((message: string, duration?: number) => {
      addAlert({ type: 'warning', message, duration });
    }, [addAlert]),
    
    info: useCallback((message: string, duration?: number) => {
      addAlert({ type: 'info', message, duration });
    }, [addAlert]),
  };
};