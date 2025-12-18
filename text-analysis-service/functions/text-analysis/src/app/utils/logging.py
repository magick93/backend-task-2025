"""
Logging configuration for the text analysis microservice.

Provides consistent logging setup across all modules with:
- Structured logging format
- Configurable log levels
- AWS Lambda integration
- Performance logging
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# JSON log format for structured logging
JSON_LOG_FORMAT = {
    'timestamp': '%(asctime)s',
    'logger': '%(name)s',
    'level': '%(levelname)s',
    'message': '%(message)s',
    'module': '%(module)s',
    'function': '%(funcName)s',
    'line': '%(lineno)d'
}


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for logging.
    
    Converts log records to JSON format for easier parsing in cloud environments.
    """
    
    def __init__(self, fmt_dict: Optional[Dict[str, str]] = None):
        """
        Initialize the structured formatter.
        
        Args:
            fmt_dict: Dictionary mapping field names to format strings
        """
        super().__init__()
        self.fmt_dict = fmt_dict or JSON_LOG_FORMAT
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string representation of the log record
        """
        log_dict = {}
        
        for key, fmt in self.fmt_dict.items():
            if key == 'timestamp':
                log_dict[key] = datetime.fromtimestamp(record.created).strftime(DEFAULT_DATE_FORMAT)
            else:
                log_dict[key] = fmt % record.__dict__
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_dict.update(record.extra)
        
        return json.dumps(log_dict)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_type: str = 'text',
    stream: Optional[Any] = None
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: 'text' for human-readable, 'json' for structured logging
        stream: Output stream (defaults to sys.stdout)
        
    Returns:
        Configured logger instance
    """
    # Determine log level
    if level is None:
        # Use environment variable or default to INFO
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Convert string level to logging constant
    log_level = getattr(logging, level, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create handler
    if stream is None:
        stream = sys.stdout
    
    handler = logging.StreamHandler(stream)
    handler.setLevel(log_level)
    
    # Create formatter based on format type
    if format_type.lower() == 'json':
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Add AWS Lambda context if available
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        logger = add_lambda_context(logger)
    
    return logger


def add_lambda_context(logger: logging.Logger) -> logging.Logger:
    """
    Add AWS Lambda context to logger for Lambda execution.
    
    Args:
        logger: Logger to enhance
        
    Returns:
        Enhanced logger with Lambda context
    """
    # This would be enhanced with actual Lambda context in runtime
    # For now, just add a flag
    logger.lambda_context = True
    
    # Create a filter to add Lambda metadata
    class LambdaFilter(logging.Filter):
        def filter(self, record):
            record.aws_region = os.getenv('AWS_REGION', 'unknown')
            record.lambda_function = os.getenv('AWS_LAMBDA_FUNCTION_NAME', 'unknown')
            record.lambda_version = os.getenv('AWS_LAMBDA_FUNCTION_VERSION', 'unknown')
            return True
    
    logger.addFilter(LambdaFilter())
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Convenience function that uses setup_logger with defaults.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return setup_logger(name)


class PerformanceLogger:
    """
    Performance logging utility for tracking execution times.
    
    Provides context manager for timing code blocks and
    structured performance logging.
    """
    
    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize performance logger.
        
        Args:
            logger: Logger instance to use
            operation: Name of the operation being timed
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results when exiting context."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() * 1000  # ms
        
        if exc_type is None:
            self.logger.info(
                f"Completed operation: {self.operation}",
                extra={
                    'operation': self.operation,
                    'duration_ms': duration,
                    'status': 'success'
                }
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation}",
                extra={
                    'operation': self.operation,
                    'duration_ms': duration,
                    'status': 'failed',
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                }
            )
    
    def log_duration(self, duration_ms: float, status: str = 'success'):
        """
        Log operation duration.
        
        Args:
            duration_ms: Duration in milliseconds
            status: Operation status ('success', 'failed', 'warning')
        """
        self.logger.info(
            f"Operation: {self.operation}",
            extra={
                'operation': self.operation,
                'duration_ms': duration_ms,
                'status': status
            }
        )


def log_execution_time(logger: logging.Logger, operation: str):
    """
    Decorator to log execution time of a function.
    
    Args:
        logger: Logger instance
        operation: Operation name (defaults to function name)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceLogger(logger, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Module-level logger for logging module itself
_logger = setup_logger(__name__)


def configure_root_logger(
    level: str = 'INFO',
    format_type: str = 'text',
    propagate: bool = True
):
    """
    Configure the root logger for the application.
    
    Args:
        level: Log level
        format_type: 'text' or 'json'
        propagate: Whether to propagate to root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    
    if format_type.lower() == 'json':
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Configure propagation
    root_logger.propagate = propagate
    
    _logger.info(f"Root logger configured with level={level}, format={format_type}")


# Default logger for import
default_logger = get_logger(__name__)