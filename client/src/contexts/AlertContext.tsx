import React, { 
  useReducer, 
  useCallback, 
  useMemo,
  ReactNode 
} from 'react';
import { AlertContext, alertReducer, initialAlertState } from './AlertContextDefinition';
import { Alert } from '../types/status';

// Provider component
export const AlertProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(alertReducer, initialAlertState);

  // Memoized function to add alerts
  const addAlert = useCallback((alert: Omit<Alert, 'id' | 'timestamp'>) => {
    dispatch({ type: 'ADD_ALERT', payload: alert });
  }, []);

  // Memoize the provider value to prevent unnecessary re-renders
  const value = useMemo(() => ({
    state,
    dispatch,
    addAlert
  }), [state, addAlert]);

  return (
    <AlertContext.Provider value={value}>
      {children}
    </AlertContext.Provider>
  );
};

