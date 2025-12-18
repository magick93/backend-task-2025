"""
Timing utilities for performance measurement and profiling.

Provides:
- High-resolution timing for code blocks
- Performance profiling decorators
- Execution time tracking
- Memory usage monitoring (optional)
"""

import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
import sys

from .logging import setup_logger

logger = setup_logger(__name__)


class Timer:
    """
    High-resolution timer for measuring code execution time.
    
    Supports context manager usage and manual start/stop.
    """
    
    def __init__(self, name: Optional[str] = None, auto_log: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Timer name for logging
            auto_log: Whether to automatically log results on exit
        """
        self.name = name or "unnamed_timer"
        self.auto_log = auto_log
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
        
    def __enter__(self):
        """Start timer when entering context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log results when exiting context."""
        self.stop()
        if self.auto_log and exc_type is None:
            self.log_result()
    
    def start(self) -> 'Timer':
        """
        Start the timer.
        
        Returns:
            Self for method chaining
        """
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed = None
        return self
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        if self.elapsed is None:
            if self.start_time is None:
                return 0.0
            # Timer is still running
            return (time.perf_counter() - self.start_time) * 1000
        return self.elapsed * 1000
    
    def elapsed_seconds(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self.elapsed is None:
            if self.start_time is None:
                return 0.0
            # Timer is still running
            return time.perf_counter() - self.start_time
        return self.elapsed
    
    def log_result(self, level: str = 'debug'):
        """
        Log timer results.
        
        Args:
            level: Log level ('debug', 'info', 'warning')
        """
        elapsed_ms = self.elapsed_ms()
        
        log_method = getattr(logger, level, logger.debug)
        log_method(
            f"Timer '{self.name}': {elapsed_ms:.2f} ms",
            extra={
                'timer_name': self.name,
                'elapsed_ms': elapsed_ms,
                'elapsed_seconds': elapsed_ms / 1000
            }
        )
    
    def get_result(self) -> Dict[str, Any]:
        """
        Get timer results as dictionary.
        
        Returns:
            Dictionary with timer results
        """
        return {
            'name': self.name,
            'elapsed_ms': self.elapsed_ms(),
            'elapsed_seconds': self.elapsed_seconds(),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'is_running': self.start_time is not None and self.end_time is None
        }


def timeit(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        name: Custom timer name (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            timer_name = name or f.__name__
            with Timer(timer_name, auto_log=True):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


class PerformanceMonitor:
    """
    Monitor performance metrics across multiple operations.
    
    Tracks execution times, counts, and aggregates statistics.
    """
    
    def __init__(self, name: str = "performance_monitor"):
        """
        Initialize performance monitor.
        
        Args:
            name: Monitor name
        """
        self.name = name
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
    def start_operation(self, operation_name: str) -> str:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation ID for stopping
        """
        op_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        with self.lock:
            self.operations[op_id] = {
                'name': operation_name,
                'start_time': time.perf_counter(),
                'end_time': None,
                'elapsed': None
            }
        
        return op_id
    
    def stop_operation(self, op_id: str) -> float:
        """
        Stop timing an operation.
        
        Args:
            op_id: Operation ID from start_operation
            
        Returns:
            Elapsed time in seconds
        """
        with self.lock:
            if op_id not in self.operations:
                raise KeyError(f"Operation {op_id} not found")
            
            op = self.operations[op_id]
            op['end_time'] = time.perf_counter()
            op['elapsed'] = op['end_time'] - op['start_time']
            
            return op['elapsed']
    
    @contextmanager
    def operation(self, operation_name: str):
        """
        Context manager for timing an operation.
        
        Args:
            operation_name: Name of the operation
        """
        op_id = self.start_operation(operation_name)
        try:
            yield
        finally:
            self.stop_operation(op_id)
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for an operation type.
        
        Args:
            operation_name: Operation name
            
        Returns:
            Dictionary with operation statistics
        """
        with self.lock:
            ops = [
                op for op in self.operations.values()
                if op['name'] == operation_name and op['elapsed'] is not None
            ]
        
        if not ops:
            return {
                'name': operation_name,
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0
            }
        
        elapsed_times = [op['elapsed'] for op in ops]
        
        return {
            'name': operation_name,
            'count': len(ops),
            'total_time': sum(elapsed_times),
            'avg_time': sum(elapsed_times) / len(ops),
            'min_time': min(elapsed_times),
            'max_time': max(elapsed_times),
            'total_time_ms': sum(elapsed_times) * 1000,
            'avg_time_ms': (sum(elapsed_times) / len(ops)) * 1000
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all operation types.
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        with self.lock:
            operation_names = set(op['name'] for op in self.operations.values())
        
        stats = {}
        for name in operation_names:
            stats[name] = self.get_operation_stats(name)
        
        return stats
    
    def log_summary(self, level: str = 'info'):
        """
        Log performance summary.
        
        Args:
            level: Log level
        """
        stats = self.get_all_stats()
        
        if not stats:
            logger.debug(f"Performance monitor '{self.name}': No operations recorded")
            return
        
        log_method = getattr(logger, level, logger.info)
        
        summary_lines = [f"Performance summary for '{self.name}':"]
        for name, stat in stats.items():
            summary_lines.append(
                f"  {name}: {stat['count']} calls, "
                f"avg {stat['avg_time_ms']:.2f} ms, "
                f"total {stat['total_time_ms']:.2f} ms"
            )
        
        log_method("\n".join(summary_lines), extra={'performance_stats': stats})
    
    def reset(self):
        """Reset all recorded operations."""
        with self.lock:
            self.operations.clear()


# Global performance monitor for easy access
_global_monitor = PerformanceMonitor("global")


def get_global_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    return _global_monitor


def track_performance(operation_name: Optional[str] = None):
    """
    Decorator to track function performance in global monitor.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            with _global_monitor.operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class ThroughputCalculator:
    """
    Calculate throughput (operations per second) for repetitive operations.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize throughput calculator.
        
        Args:
            window_size: Number of recent operations to consider
        """
        self.window_size = window_size
        self.timestamps: list = []
        self.lock = threading.Lock()
    
    def record_operation(self):
        """Record an operation timestamp."""
        with self.lock:
            self.timestamps.append(time.time())
            # Keep only the most recent timestamps
            if len(self.timestamps) > self.window_size:
                self.timestamps.pop(0)
    
    def get_throughput(self) -> float:
        """
        Calculate current throughput (operations per second).
        
        Returns:
            Throughput in operations per second
        """
        with self.lock:
            if len(self.timestamps) < 2:
                return 0.0
            
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span <= 0:
                return 0.0
            
            return (len(self.timestamps) - 1) / time_span
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get throughput statistics.
        
        Returns:
            Dictionary with throughput statistics
        """
        throughput = self.get_throughput()
        
        with self.lock:
            return {
                'throughput_ops_per_second': throughput,
                'window_size': len(self.timestamps),
                'max_window_size': self.window_size,
                'oldest_timestamp': self.timestamps[0] if self.timestamps else None,
                'newest_timestamp': self.timestamps[-1] if self.timestamps else None
            }


def estimate_remaining_time(
    completed: int,
    total: int,
    elapsed_seconds: float
) -> Optional[float]:
    """
    Estimate remaining time based on completion rate.
    
    Args:
        completed: Number of items completed
        total: Total number of items
        elapsed_seconds: Time elapsed so far in seconds
        
    Returns:
        Estimated remaining time in seconds, or None if cannot estimate
    """
    if completed <= 0:
        return None
    
    if completed >= total:
        return 0.0
    
    # Calculate time per item
    time_per_item = elapsed_seconds / completed
    
    # Estimate remaining time
    remaining_items = total - completed
    return remaining_items * time_per_item


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


# Convenience functions for common timing operations
def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function call.
    
    Args:
        func: Function to call
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (function result, execution time in seconds)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


@contextmanager
def measure_block(name: str = "code_block"):
    """
    Context manager to measure execution time of a code block.
    
    Args:
        name: Name of the code block for logging
        
    Yields:
        Timer instance
    """
    with Timer(name, auto_log=True) as timer:
        yield timer


# Module initialization
logger.debug("Timing utilities initialized")