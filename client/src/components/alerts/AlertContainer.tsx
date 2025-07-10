import React from 'react';
import { useAlerts } from '../../hooks/useAlerts';
import { NotificationToast } from './NotificationToast';

export const AlertContainer: React.FC = () => {
  const { state, dispatch } = useAlerts();

  const handleDismiss = (id: string) => {
    dispatch({ type: 'DISMISS_ALERT', payload: { id } });
  };

  return (
    <div className="alert-container" aria-live="polite" aria-atomic="true">
      {state.alerts.map((alert) => (
        <NotificationToast
          key={alert.id}
          {...alert}
          onDismiss={() => handleDismiss(alert.id)}
        />
      ))}
    </div>
  );
};