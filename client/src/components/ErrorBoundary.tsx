import { Component, ErrorInfo, ReactNode } from 'react';
import { logger } from '../utils';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error to our logging service
    logger.error('React Error Boundary caught an error:', error);
    logger.error('Error component stack:', errorInfo.componentStack);

    // Update state with error info for debugging
    this.setState({
      error,
      errorInfo,
    });

    // You could also send error reports to an error reporting service here
    // Example: reportErrorToService(error, errorInfo)
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div
          style={{
            padding: '20px',
            margin: '20px',
            border: '1px solid #ff6b6b',
            borderRadius: '8px',
            backgroundColor: '#ffe0e0',
            color: '#721c24',
          }}
        >
          <h2>🚨 Something went wrong</h2>
          <p>
            An unexpected error occurred in the application. Please try refreshing the page or
            contact support if the problem persists.
          </p>

          <div style={{ marginTop: '16px' }}>
            <button
              onClick={this.handleReset}
              style={{
                padding: '8px 16px',
                marginRight: '8px',
                backgroundColor: '#ff6b6b',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Try Again
            </button>

            <button
              onClick={() => window.location.reload()}
              style={{
                padding: '8px 16px',
                backgroundColor: '#868e96',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Refresh Page
            </button>
          </div>

          {/* Show error details in development */}
          {import.meta.env.DEV && this.state.error && (
            <details style={{ marginTop: '16px' }}>
              <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
                Error Details (Development Only)
              </summary>
              <pre
                style={{
                  marginTop: '8px',
                  padding: '12px',
                  backgroundColor: '#f8f9fa',
                  border: '1px solid #dee2e6',
                  borderRadius: '4px',
                  fontSize: '12px',
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                }}
              >
                <strong>Error:</strong> {this.state.error.message}
                {'\n\n'}
                <strong>Stack:</strong>
                {'\n'}
                {this.state.error.stack}
                {this.state.errorInfo && (
                  <>
                    {'\n\n'}
                    <strong>Component Stack:</strong>
                    {'\n'}
                    {this.state.errorInfo.componentStack}
                  </>
                )}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
