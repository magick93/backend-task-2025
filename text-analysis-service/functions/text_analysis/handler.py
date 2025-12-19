"""
Lambda handler wrapper for the text analysis microservice.

This module is a thin wrapper that re-exports the actual lambda handler
from the app package (src/app/handler.py) to maintain compatibility
with AWS SAM template (Handler: app.lambda_handler).
"""

from opentelemetry import trace
from src.app.handler import lambda_handler as core_handler
from src.app.utils.telemetry import setup_telemetry

# Setup telemetry once at module level (cold start)
setup_telemetry()
tracer = trace.get_tracer(__name__)

def lambda_handler(event, context):
    """
    Instrumented Lambda handler.
    """
    with tracer.start_as_current_span("text-analysis-handler"):
        return core_handler(event, context)

# Re-export the handler
__all__ = ["lambda_handler"]
