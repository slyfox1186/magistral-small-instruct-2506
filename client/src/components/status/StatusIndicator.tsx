import React, { useEffect, useRef } from 'react';
import { Status, StatusIndicatorProps } from '../../types/status';

// Size configurations
const sizes = {
  small: 20,
  medium: 24,
  large: 32,
};

// Status colors aligned with Veridian Twilight theme
const statusColors: Record<Status, string> = {
  thinking: 'var(--accent-info)',      // Sage green
  streaming: 'var(--accent-primary)',  // Dusty mauve
  processing: 'var(--accent-info)',    // Sage green
  success: 'var(--accent-success)',    // Bright green
  error: 'var(--accent-danger)',       // Red
  warning: 'var(--accent-warning)',    // Amber
  info: 'var(--accent-info)',          // Sage green
};

// All paths have identical structure for smooth morphing
// Base shape is an 8-point circle that morphs into other shapes
const statusPaths: Record<Status, string> = {
  // Base: 8-point circle
  thinking:   "M12 4 C7.58 4 4 7.58 4 12 C4 16.42 7.58 20 12 20 C16.42 20 20 16.42 20 12 C20 7.58 16.42 4 12 4 Z",
  streaming:  "M12 4 C7.58 4 4 7.58 4 12 C4 16.42 7.58 20 12 20 C16.42 20 20 16.42 20 12 C20 7.58 16.42 4 12 4 Z",
  processing: "M12 4 C7.58 4 4 7.58 4 12 C4 16.42 7.58 20 12 20 C16.42 20 20 16.42 20 12 C20 7.58 16.42 4 12 4 Z",
  
  // Checkmark - morphed from circle
  success: "M6 12 C6 12 9 15 9 15 C9 15 18 6 18 6 C18 6 18 6 18 6 C18 6 9 15 9 15 C9 15 6 12 6 12 C6 12 6 12 6 12 Z",
  
  // X shape - a single path that traces the X
  error: "M6 6 C9 9 15 15 18 18 C15 9 9 15 6 12 C9 9 15 15 12 12 C12 12 9 9 6 6 C9 9 15 15 18 18 C15 9 9 15 6 12 Z",
  
  // Warning - a rounded rectangle (pill shape)
  warning: "M8 4 C6 4 4 6 4 8 L4 16 C4 18 6 20 8 20 L16 20 C18 20 20 18 20 16 L20 8 C20 6 18 4 16 4 Z",
  
  // Info - a square
  info: "M4 4 C4 4 20 4 20 4 C20 4 20 20 20 20 C20 20 4 20 4 20 C4 20 4 4 4 4 C4 4 4 4 4 4 C4 4 4 4 4 4 Z",
};

// ARIA labels for accessibility
const statusLabels: Record<Status, string> = {
  thinking: 'AI is thinking',
  streaming: 'Message is streaming',
  processing: 'Processing request',
  success: 'Operation successful',
  error: 'An error occurred',
  warning: 'Warning',
  info: 'Information',
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ 
  status, 
  size = 'medium',
  className = '' 
}) => {
  const pathRef = useRef<SVGPathElement>(null);
  const prevStatusRef = useRef<Status>(status);
  const particleContainerRef = useRef<HTMLDivElement>(null);

  // Trigger particle effect on status changes to success or error
  useEffect(() => {
    if (prevStatusRef.current !== status) {
      if (status === 'success' || status === 'error') {
        // Add particle emitter class
        if (particleContainerRef.current) {
          particleContainerRef.current.classList.add('particle-emitter', `particle-${status}`);
          
          // Remove the class after animation completes
          setTimeout(() => {
            if (particleContainerRef.current) {
              particleContainerRef.current.classList.remove('particle-emitter', `particle-${status}`);
            }
          }, 1000);
        }
      }
      prevStatusRef.current = status;
    }
  }, [status]);

  const svgSize = sizes[size];
  
  return (
    <div 
      className={`status-indicator status-${status} size-${size} ${className}`}
      role="status"
      aria-label={statusLabels[status]}
    >
      <div ref={particleContainerRef} className="particle-container" />
      
      <svg 
        width={svgSize} 
        height={svgSize} 
        viewBox="0 0 24 24" 
        fill="none"
        className="status-svg"
      >
        <path
          ref={pathRef}
          d={statusPaths[status]}
          stroke={statusColors[status]}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="status-path"
        />
        
        {/* Additional elements for specific statuses */}
        {status === 'thinking' && (
          <g className="thinking-dots">
            <circle cx="8" cy="12" r="1.5" fill={statusColors[status]} />
            <circle cx="12" cy="12" r="1.5" fill={statusColors[status]} />
            <circle cx="16" cy="12" r="1.5" fill={statusColors[status]} />
          </g>
        )}
        
        {status === 'streaming' && (
          <g className="streaming-wave">
            <path
              d="M4 12 Q8 6, 12 12 T20 12"
              stroke={statusColors[status]}
              strokeWidth="1.5"
              fill="none"
              opacity="0.6"
            />
          </g>
        )}
      </svg>
    </div>
  );
};