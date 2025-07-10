import { createContext } from 'react';
import { Alert, AlertState, AlertAction } from '../types/status';

export interface AlertContextType {
  state: AlertState;
  dispatch: React.Dispatch<AlertAction>;
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp'>) => void;
  showAlert: (message: string, type: 'success' | 'error' | 'warning' | 'info') => void;
}

export const AlertContext = createContext<AlertContextType | undefined>(undefined);

// Reducer for managing alert state
export const alertReducer = (state: AlertState, action: AlertAction): AlertState => {
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
export const initialAlertState: AlertState = {
  alerts: [],
};