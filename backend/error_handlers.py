#!/usr/bin/env python3
"""Centralized error handling for the application.

This module provides structured error handling with proper logging,
error categorization, and recovery strategies.
"""

import logging
import sys
import traceback
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for better handling and monitoring."""
    
    NETWORK = "network"
    DATABASE = "database"
    MODEL = "model"
    MEMORY = "memory"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class ApplicationError(Exception):
    """Base exception for application-specific errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.details = details or {}


class NetworkError(ApplicationError):
    """Network-related errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message,
            ErrorCategory.NETWORK,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            details
        )


class DatabaseError(ApplicationError):
    """Database-related errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message,
            ErrorCategory.DATABASE,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            details
        )


class ModelError(ApplicationError):
    """Model/LLM-related errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message,
            ErrorCategory.MODEL,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            details
        )


class ValidationError(ApplicationError):
    """Input validation errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message,
            ErrorCategory.VALIDATION,
            status.HTTP_400_BAD_REQUEST,
            details
        )


class RateLimitError(ApplicationError):
    """Rate limiting errors."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(
            message,
            ErrorCategory.RATE_LIMIT,
            status.HTTP_429_TOO_MANY_REQUESTS,
            details
        )


def handle_error_with_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable[..., T]:
    """Decorator for handling errors with retry logic.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch and retry
    """
    import asyncio
    import time
    from functools import wraps
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> T:
        last_exception = None
        delay = retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(
                        f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                    )
        
        if last_exception:
            raise last_exception
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> T:
        last_exception = None
        delay = retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(
                        f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                    )
        
        if last_exception:
            raise last_exception
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class ErrorHandler:
    """Centralized error handler for the application."""
    
    @staticmethod
    def handle_exception(
        e: Exception,
        context: Optional[str] = None
    ) -> dict[str, Any]:
        """Convert exception to structured error response.
        
        Args:
            e: The exception to handle
            context: Optional context about where the error occurred
            
        Returns:
            Dictionary with error details
        """
        error_id = f"{id(e):x}"
        
        # Log the error with full traceback
        logger.error(
            f"Error {error_id} in {context or 'unknown context'}: {type(e).__name__}: {e}",
            exc_info=True
        )
        
        # Handle specific error types
        if isinstance(e, ApplicationError):
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "category": e.category.value,
                "details": e.details,
                "error_id": error_id
            }
        
        elif isinstance(e, HTTPException):
            return {
                "error": e.detail,
                "error_type": "HTTPException",
                "category": ErrorCategory.UNKNOWN.value,
                "status_code": e.status_code,
                "error_id": error_id
            }
        
        elif isinstance(e, ValueError):
            return {
                "error": str(e),
                "error_type": "ValueError",
                "category": ErrorCategory.VALIDATION.value,
                "error_id": error_id
            }
        
        elif isinstance(e, KeyError):
            return {
                "error": f"Missing required key: {e}",
                "error_type": "KeyError",
                "category": ErrorCategory.VALIDATION.value,
                "error_id": error_id
            }
        
        elif isinstance(e, MemoryError):
            return {
                "error": "System out of memory",
                "error_type": "MemoryError",
                "category": ErrorCategory.MEMORY.value,
                "error_id": error_id,
                "details": {"critical": True}
            }
        
        elif isinstance(e, TimeoutError):
            return {
                "error": "Operation timed out",
                "error_type": "TimeoutError",
                "category": ErrorCategory.NETWORK.value,
                "error_id": error_id
            }
        
        elif isinstance(e, ConnectionError):
            return {
                "error": "Connection failed",
                "error_type": "ConnectionError",
                "category": ErrorCategory.NETWORK.value,
                "error_id": error_id
            }
        
        else:
            # Generic error handling
            return {
                "error": "An unexpected error occurred",
                "error_type": type(e).__name__,
                "category": ErrorCategory.UNKNOWN.value,
                "error_id": error_id,
                "details": {"message": str(e)} if str(e) else {}
            }
    
    @staticmethod
    async def handle_request_error(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """FastAPI exception handler for requests.
        
        Args:
            request: The FastAPI request
            exc: The exception that occurred
            
        Returns:
            JSON response with error details
        """
        error_response = ErrorHandler.handle_exception(
            exc,
            context=f"{request.method} {request.url.path}"
        )
        
        # Determine status code
        if isinstance(exc, ApplicationError):
            status_code = exc.status_code
        elif isinstance(exc, HTTPException):
            status_code = exc.status_code
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )


def setup_exception_handlers(app):
    """Setup global exception handlers for FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    # Handle our custom application errors
    app.add_exception_handler(ApplicationError, ErrorHandler.handle_request_error)
    
    # Handle standard Python errors
    app.add_exception_handler(ValueError, ErrorHandler.handle_request_error)
    app.add_exception_handler(KeyError, ErrorHandler.handle_request_error)
    app.add_exception_handler(TimeoutError, ErrorHandler.handle_request_error)
    app.add_exception_handler(ConnectionError, ErrorHandler.handle_request_error)
    app.add_exception_handler(MemoryError, ErrorHandler.handle_request_error)
    
    # Catch-all handler
    app.add_exception_handler(Exception, ErrorHandler.handle_request_error)


def log_unhandled_exception(exc_type, exc_value, exc_traceback):
    """Handler for unhandled exceptions.
    
    This ensures all crashes are properly logged.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow keyboard interrupt to work normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


# Install the unhandled exception handler
sys.excepthook = log_unhandled_exception


# Example usage
if __name__ == "__main__":
    # Demo the error handling
    
    @handle_error_with_retry(
        max_retries=3,
        exceptions=(ConnectionError, TimeoutError)
    )
    async def flaky_network_call():
        """Simulated network call that might fail."""
        import random
        if random.random() < 0.7:
            raise ConnectionError("Network unreachable")
        return "Success!"
    
    # Test error categorization
    errors = [
        NetworkError("Failed to connect to server"),
        DatabaseError("Connection pool exhausted"),
        ModelError("GPU out of memory"),
        ValidationError("Invalid input format", {"field": "username"}),
        RateLimitError("Too many requests", retry_after=60),
    ]
    
    for error in errors:
        response = ErrorHandler.handle_exception(error)
        print(f"\n{error.__class__.__name__}:")
        print(f"  Category: {response['category']}")
        print(f"  Message: {response['error']}")
        print(f"  Details: {response.get('details', {})}")