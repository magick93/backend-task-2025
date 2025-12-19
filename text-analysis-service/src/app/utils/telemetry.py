import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

def setup_telemetry():
    """
    Configures OpenTelemetry with OTLP exporter for Honeycomb.
    Reads configuration from environment variables:
    - OTEL_SERVICE_NAME
    - OTEL_EXPORTER_OTLP_HEADERS
    - OTEL_EXPORTER_OTLP_ENDPOINT
    """
    api_key_header = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    
    # Graceful degradation: Disable if key is missing or is the dummy value
    if not api_key_header or "replace-me" in api_key_header:
        logger.info("Telemetry disabled: Missing or dummy Honeycomb API Key.")
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "text-analysis-service")

    try:
        resource = Resource(attributes={
            "service.name": service_name
        })

        provider = TracerProvider(resource=resource)
        
        # OTLPSpanExporter automatically reads endpoint and headers from 
        # OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS env vars
        exporter = OTLPSpanExporter()

        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        logger.info(f"Telemetry enabled for service: {service_name}")
        
    except Exception as e:
        logger.error(f"Failed to setup telemetry: {e}")
