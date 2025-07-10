import React, { useCallback, useEffect, useRef, useState } from 'react';
import { NotificationToastProps, AlertType } from '../../types/status';

// Icons for different alert types
const icons: Record<AlertType, JSX.Element> = {
  success: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  ),
  error: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  ),
  warning: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  info: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  ),
};

export const NotificationToast: React.FC<NotificationToastProps> = ({
  id: _id,
  type,
  message,
  duration = 5000,
  onDismiss,
}) => {
  const [isExiting, setIsExiting] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  // Handle dismissal with exit animation
  const handleDismiss = useCallback(() => {
    setIsExiting(true);
    // Wait for exit animation to complete before removing
    setTimeout(() => {
      onDismiss();
    }, 300); // Match this with CSS exit animation duration
  }, [onDismiss]);

  // Auto-dismiss effect with proper cleanup
  useEffect(() => {
    if (duration && duration > 0) {
      timerRef.current = setTimeout(() => {
        handleDismiss();
      }, duration);

      // Cleanup function to clear timeout
      return () => {
        if (timerRef.current) {
          clearTimeout(timerRef.current);
          timerRef.current = null;
        }
      };
    }
  }, [duration, handleDismiss]); // Only re-run if duration or handleDismiss changes

  // ARIA live region attributes
  const ariaLive = type === 'error' ? 'assertive' : 'polite';
  const role = type === 'error' ? 'alert' : 'status';

  return (
    <div
      className={`notification-toast toast-${type} ${isExiting ? 'toast-exit' : 'toast-enter'}`}
      role={role}
      aria-live={ariaLive}
      aria-atomic="true"
    >
      {/* Animated gradient border */}
      <div className="toast-border" />
      
      {/* Content */}
      <div className="toast-content">
        <div className="toast-icon">
          {icons[type]}
        </div>
        
        <div className="toast-message">
          {message}
        </div>
        
        <button
          className="toast-close"
          onClick={handleDismiss}
          aria-label="Dismiss notification"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
      
      {/* Progress bar for timed dismissal */}
      {duration && duration > 0 && (
        <div 
          ref={progressRef}
          className="toast-progress" 
          style={{ animationDuration: `${duration}ms` }}
        />
      )}
    </div>
  );
};