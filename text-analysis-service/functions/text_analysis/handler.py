"""
Lambda handler wrapper for the text analysis microservice.

This module is a thin wrapper that re-exports the actual lambda handler
from the app package (src/app/handler.py) to maintain compatibility
with AWS SAM template (Handler: app.lambda_handler).
"""

import os
from opentelemetry import trace
from src.app.handler import lambda_handler as core_handler
from src.app.utils.telemetry import setup_telemetry

# AWS X-Ray SDK initialization for tracing
try:
    from aws_xray_sdk.core import xray_recorder, patch_all
    from aws_xray_sdk.core.context import Context
    
    # Initialize X-Ray recorder
    xray_recorder.configure(
        service='text-analysis-service',
        context=Context(),
        sampling=False  # Let Lambda handle sampling
    )
    
    # Patch all supported libraries
    patch_all()
    
    XRAY_ENABLED = True
except ImportError:
    # X-Ray SDK not installed, continue without it
    XRAY_ENABLED = False

# Setup OpenTelemetry telemetry once at module level (cold start)
setup_telemetry()
tracer = trace.get_tracer(__name__)

def lambda_handler(event, context):
    """
    Instrumented Lambda handler.
    """
    # Start X-Ray segment if enabled
    if XRAY_ENABLED:
        xray_recorder.begin_segment('text-analysis-handler')
        try:
            with tracer.start_as_current_span("text-analysis-handler"):
                return core_handler(event, context)
        finally:
            xray_recorder.end_segment()
    else:
        with tracer.start_as_current_span("text-analysis-handler"):
            return core_handler(event, context)

# Re-export the handler
__all__ = ["lambda_handler"]
