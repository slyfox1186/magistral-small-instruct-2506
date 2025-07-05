import React, { 
  createContext, 
  useContext, 
  useReducer, 
  useCallback, 
  useMemo,
  ReactNode 
} from 'react';
import { Alert, AlertState, AlertAction, AlertType } from '../types/status';

interface AlertContextType {
  state: AlertState;
  dispatch: React.Dispatch<AlertAction>;
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp'>) => void;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

// Reducer for managing alert state
const alertReducer = (state: AlertState, action: AlertAction): AlertState => {
  switch (action.type) {
    case 'ADD_ALERT': {
      const newAlert: Alert = {
        ...action.payload,
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        timestamp: Date.now(),
      };
      
      return {
        ...state,
        alerts: [...state.alerts, newAlert],
      };
    }

    case 'DISMISS_ALERT': {
      return {
        ...state,
        alerts: state.alerts.filter(alert => alert.id !== action.payload.id),
      };
    }

    case 'CLEAR_ALERTS': {
      return {
        ...state,
        alerts: [],
      };
    }

    default:
      return state;
  }
};

// Initial state
const initialState: AlertState = {
  alerts: [],
};

// Provider component
export const AlertProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(alertReducer, initialState);

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