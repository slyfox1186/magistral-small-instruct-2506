// Status system types for the chat application

export type Status = 'thinking' | 'streaming' | 'processing' | 'success' | 'error' | 'warning' | 'info';

export type AlertType = 'success' | 'error' | 'warning' | 'info';

export interface Alert {
  id: string;
  type: AlertType;
  message: string;
  duration?: number; // Duration in ms before auto-dismiss
  timestamp: number;
}

export interface AlertState {
  alerts: Alert[];
}

export type AlertAction =
  | { type: 'ADD_ALERT'; payload: Omit<Alert, 'id' | 'timestamp'> }
  | { type: 'DISMISS_ALERT'; payload: { id: string } }
  | { type: 'CLEAR_ALERTS' };

export interface StatusIndicatorProps {
  status: Status;
  size?: 'small' | 'medium' | 'large';
  className?: string;
}

export interface NotificationToastProps extends Alert {
  onDismiss: () => void;
}