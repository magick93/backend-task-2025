"""
Error handling and custom exceptions for the text analysis microservice.

This module defines a hierarchy of custom exceptions for different error types,
error response schemas, and validation error handling compatible with API Gateway.

Key components:
- Base exception class with consistent error formatting
- Specific exception types for different error scenarios
- Error response formatting for API Gateway compatibility
- Validation error handling utilities
"""

from typing import Dict, Any, Optional, List, Union
from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import traceback
import logging
from datetime import datetime

from .schemas import ErrorResponse

# Configure logger
logger = logging.getLogger(__name__)


class TextAnalysisError(Exception):
    """
    Base exception class for all text analysis errors.
    
    Provides consistent error formatting and logging capabilities.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        status_code: HTTP status code for API responses
        details: Optional additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    
    def to_error_response(self) -> ErrorResponse:
        """Convert exception to ErrorResponse schema."""
        return ErrorResponse(
            error=self.error_code,
            message=self.message,
            status_code=self.status_code,
            details=self.details
        )


class ValidationError(TextAnalysisError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str = "Invalid input data",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class AuthenticationError(TextAnalysisError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationError(TextAnalysisError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Authorization failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class ResourceNotFoundError(TextAnalysisError):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self,
        resource_type: str = "resource",
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"{resource_type} not found"
        if resource_id:
            message = f"{resource_type} with ID '{resource_id}' not found"
        
        details = details or {}
        details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class ProcessingError(TextAnalysisError):
    """Raised when text processing fails."""
    
    def __init__(
        self,
        message: str = "Text processing failed",
        processing_stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if processing_stage:
            details["processing_stage"] = processing_stage
        
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class ModelLoadingError(TextAnalysisError):
    """Raised when ML model loading fails."""
    
    def __init__(
        self,
        model_name: str,
        message: str = "Model loading failed",
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["model_name"] = model_name
        
        super().__init__(
            message=f"{message}: {model_name}",
            error_code="MODEL_LOADING_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class RateLimitError(TextAnalysisError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if limit:
            details["limit"] = limit
        if reset_time:
            details["reset_time"] = reset_time.isoformat()
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


class ServiceUnavailableError(TextAnalysisError):
    """Raised when a required service is unavailable."""
    
    def __init__(
        self,
        service_name: str,
        message: str = "Service unavailable",
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["service_name"] = service_name
        
        super().__init__(
            message=f"{message}: {service_name}",
            error_code="SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class ConfigurationError(TextAnalysisError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        config_key: str,
        message: str = "Configuration error",
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["config_key"] = config_key
        
        super().__init__(
            message=f"{message}: {config_key}",
            error_code="CONFIGURATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


def handle_validation_error(
    validation_error: Union[RequestValidationError, ValidationError]
) -> ErrorResponse:
    """
    Handle Pydantic validation errors and format them consistently.
    
    Args:
        validation_error: Pydantic validation error
        
    Returns:
        Formatted ErrorResponse
    """
    error_details = []
    
    if hasattr(validation_error, 'errors'):
        # Handle Pydantic ValidationError
        for error in validation_error.errors():
            field = " -> ".join(str(loc) for loc in error.get("loc", []))
            error_details.append({
                "field": field,
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown")
            })
    elif hasattr(validation_error, 'detail'):
        # Handle FastAPI RequestValidationError
        if isinstance(validation_error.detail, list):
            for error in validation_error.detail:
                field = " -> ".join(str(loc) for loc in error.get("loc", []))
                error_details.append({
                    "field": field,
                    "message": error.get("msg", "Validation error"),
                    "type": error.get("type", "unknown")
                })
    
    return ErrorResponse(
        error="VALIDATION_ERROR",
        message="Invalid input data",
        status_code=status.HTTP_400_BAD_REQUEST,
        details={"field_errors": error_details}
    )


def handle_http_exception(http_exception: HTTPException) -> ErrorResponse:
    """
    Handle FastAPI HTTPException and format it consistently.
    
    Args:
        http_exception: FastAPI HTTPException
        
    Returns:
        Formatted ErrorResponse
    """
    details = {}
    if hasattr(http_exception, 'detail') and isinstance(http_exception.detail, dict):
        details = http_exception.detail
    
    return ErrorResponse(
        error="HTTP_ERROR",
        message=str(http_exception.detail) if http_exception.detail else "HTTP error occurred",
        status_code=http_exception.status_code,
        details=details
    )


def handle_generic_exception(exception: Exception) -> ErrorResponse:
    """
    Handle generic exceptions and format them consistently.
    
    Args:
        exception: Any exception
        
    Returns:
        Formatted ErrorResponse
    """
    # Log the full traceback for debugging
    logger.error(f"Unhandled exception: {str(exception)}", exc_info=True)
    
    # Don't expose internal details in production
    error_message = "An internal server error occurred"
    error_code = "INTERNAL_SERVER_ERROR"
    
    # In development/staging, include more details
    import os
    if os.getenv("ENVIRONMENT", "production") in ["development", "staging"]:
        error_message = str(exception)
        error_details = {
            "exception_type": type(exception).__name__,
            "traceback": traceback.format_exc()
        }
    else:
        error_details = {}
    
    return ErrorResponse(
        error=error_code,
        message=error_message,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details=error_details
    )


def create_error_response(
    error: Union[TextAnalysisError, Exception, str],
    status_code: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """
    Create a standardized error response from various error types.
    
    Args:
        error: Error object, exception, or error message string
        status_code: Optional HTTP status code (inferred from error if not provided)
        details: Optional additional error details
        
    Returns:
        Formatted ErrorResponse
    """
    if isinstance(error, TextAnalysisError):
        return error.to_error_response()
    
    elif isinstance(error, HTTPException):
        return handle_http_exception(error)
    
    elif isinstance(error, (RequestValidationError, ValidationError)):
        return handle_validation_error(error)
    
    elif isinstance(error, Exception):
        # Generic exception
        error_message = str(error)
        error_code = type(error).__name__.upper()
        
        return ErrorResponse(
            error=error_code,
            message=error_message,
            status_code=status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or {}
        )
    
    else:
        # String error message
        return ErrorResponse(
            error="UNKNOWN_ERROR",
            message=str(error),
            status_code=status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or {}
        )


def setup_exception_handlers(app):
    """
    Set up exception handlers for a FastAPI application.
    
    This function should be called during application setup to register
    exception handlers for consistent error responses.
    
    Args:
        app: FastAPI application instance
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse
    
    @app.exception_handler(TextAnalysisError)
    async def text_analysis_error_handler(
        request: Request,
        exc: TextAnalysisError
    ) -> JSONResponse:
        """Handle TextAnalysisError exceptions."""
        error_response = exc.to_error_response()
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        error_response = handle_validation_error(exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        error_response = handle_http_exception(exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle all other exceptions."""
        error_response = handle_generic_exception(exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )
    
    logger.info("Exception handlers registered successfully")


# Export all exceptions and utilities
__all__ = [
    "TextAnalysisError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "ProcessingError",
    "ModelLoadingError",
    "RateLimitError",
    "ServiceUnavailableError",
    "ConfigurationError",
    "handle_validation_error",
    "handle_http_exception",
    "handle_generic_exception",
    "create_error_response",
    "setup_exception_handlers"
]