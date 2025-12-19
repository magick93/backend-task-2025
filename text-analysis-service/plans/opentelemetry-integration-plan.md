# OpenTelemetry Integration Plan for Text Analysis Service

## Current Architecture Analysis

### Project Structure
- **Lambda Handler**: [`functions/text_analysis/handler.py`](functions/text_analysis/handler.py:1) (thin wrapper) → [`src/app/handler.py`](src/app/handler.py:1) (main entry point)
- **Pipeline Components**: [`src/app/pipeline/orchestrator.py`](src/app/pipeline/orchestrator.py:1) coordinates:
  - Preprocessing (`preprocessing.py`)
  - Embedding (`embedding.py`) 
  - Clustering (`clustering.py`)
  - Sentiment Analysis (`sentiment.py`)
  - Insights Generation (`insights.py`)
  - Comparison Analysis (`comparison.py`)
- **Logging**: [`src/app/utils/logging.py`](src/app/utils/logging.py:1) provides structured JSON logging with Lambda context
- **Deployment**: AWS SAM with `template.yaml`, Python 3.13 runtime

### Current Dependencies
- **Core**: numpy, scikit-learn, pydantic, torch, sentence-transformers, vaderSentiment
- **No existing observability**: No OpenTelemetry or APM dependencies

## OpenTelemetry Integration Strategy

### 1. Dependencies Required

```python
# Add to requirements.txt and setup.py
opentelemetry-api>=1.24.0
opentelemetry-sdk>=1.24.0
opentelemetry-exporter-otlp-proto-http>=1.24.0
opentelemetry-instrumentation>=0.45b0
opentelemetry-instrumentation-boto3>=0.45b0  # For AWS SDK instrumentation
opentelemetry-semantic-conventions>=1.24.0
```

### 2. Configuration Approach

#### Environment Variables (add to `template.yaml`):
```yaml
Environment:
  Variables:
    # Existing
    LOG_LEVEL: INFO
    
    # OpenTelemetry Configuration
    OTEL_SERVICE_NAME: text-analysis-service
    OTEL_EXPORTER_OTLP_ENDPOINT: https://api.honeycomb.io
    OTEL_EXPORTER_OTLP_HEADERS: x-honeycomb-team=${HONEYCOMB_API_KEY}
    OTEL_EXPORTER_OTLP_PROTOCOL: http/protobuf
    OTEL_RESOURCE_ATTRIBUTES: deployment.environment=${DEPLOYMENT_ENV},service.version=${SERVICE_VERSION}
    
    # Honeycomb-specific
    HONEYCOMB_API_KEY: ${HoneycombApiKey}  # SAM parameter
    HONEYCOMB_DATASET: text-analysis-${DEPLOYMENT_ENV}
    
    # Feature flags
    ENABLE_OPENTELEMETRY: true
```

#### SAM Parameters (add to `template.yaml`):
```yaml
Parameters:
  HoneycombApiKey:
    Type: String
    NoEcho: true
    Description: Honeycomb API key for observability
  DeploymentEnvironment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]
    Description: Deployment environment
  ServiceVersion:
    Type: String
    Default: 0.1.0
    Description: Service version for tracking
```

### 3. Instrumentation Design

#### A. Global OpenTelemetry Setup (`src/app/utils/telemetry.py`)
```python
"""
OpenTelemetry configuration and utilities for the text analysis service.
"""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

def configure_opentelemetry():
    """Configure OpenTelemetry SDK with Honeycomb exporter."""
    if not os.getenv("ENABLE_OPENTELEMETRY", "true").lower() == "true":
        return None
    
    # Create resource with service attributes
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "text-analysis-service"),
        ResourceAttributes.SERVICE_VERSION: os.getenv("SERVICE_VERSION", "0.1.0"),
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENVIRONMENT", "dev"),
        "aws.region": os.getenv("AWS_REGION", "unknown"),
    })
    
    # Configure tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure Honeycomb exporter
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://api.honeycomb.io")
    
    span_exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=dict(item.split("=") for item in headers.split(",") if "=" in item)
    )
    
    # Add batch processor
    span_processor = BatchSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    return tracer_provider
```

#### B. Lambda Handler Instrumentation (`src/app/handler.py`)
```python
# Add to imports
from opentelemetry import trace
from .utils.telemetry import configure_opentelemetry

# Initialize OpenTelemetry at module level
tracer_provider = configure_opentelemetry()
tracer = trace.get_tracer(__name__) if tracer_provider else None

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entrypoint with OpenTelemetry instrumentation."""
    # Create root span for the Lambda invocation
    with tracer.start_as_current_span("lambda_invocation") if tracer else nullcontext():
        # Add attributes
        span = trace.get_current_span()
        span.set_attributes({
            "aws.lambda.function_name": context.function_name if context else "unknown",
            "aws.lambda.function_version": context.function_version if context else "unknown",
            "aws.lambda.invoked_arn": context.invoked_function_arn if context else "unknown",
            "request.type": "comparison" if "comparison" in event.get("body", event) else "standalone",
        })
        
        # Original handler logic with additional spans for key operations
        return _lambda_handler_internal(event, context)
```

#### C. Pipeline Component Instrumentation (`src/app/pipeline/orchestrator.py`)
```python
# Add to imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class PipelineOrchestrator:
    def process_standalone(self, sentences: List[Dict[str, str]], job_id: str) -> Dict[str, Any]:
        """Process standalone analysis with OpenTelemetry spans."""
        with tracer.start_as_current_span("process_standalone") as span:
            span.set_attributes({
                "job.id": job_id,
                "sentence.count": len(sentences),
                "analysis.type": "standalone",
            })
            
            try:
                # Instrument each major step
                with tracer.start_as_current_span("preprocessing"):
                    preprocessed_sentences = self.preprocessor.preprocess_batch(sentence_texts)
                
                with tracer.start_as_current_span("embedding_clustering"):
                    embeddings = self.embedding_model.embed(preprocessed_sentences)
                    labels = self.cluster_analyzer.cluster(embeddings)
                
                with tracer.start_as_current_span("sentiment_analysis"):
                    sentiment_results = self.sentiment_analyzer.analyze_batch(preprocessed_sentences)
                
                # ... rest of the method
                
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("cluster.count", len(clusters))
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
```

#### D. Individual Component Instrumentation
Each pipeline component should have its own spans:
- `EmbeddingModel.embed()`: Track embedding generation time, model used
- `ClusterAnalyzer.cluster()`: Track clustering algorithm, cluster count, noise points
- `SentimentAnalyzer.analyze_batch()`: Track sentiment distribution
- `InsightGenerator.generate_cluster_insights()`: Track insight generation time

### 4. Metrics Collection

#### Key Metrics to Track:
1. **Request Metrics**:
   - `text_analysis.request.duration` (histogram)
   - `text_analysis.request.count` (counter)
   - `text_analysis.request.errors` (counter)

2. **Pipeline Metrics**:
   - `text_analysis.pipeline.embedding.duration` (histogram)
   - `text_analysis.pipeline.clustering.duration` (histogram)
   - `text_analysis.pipeline.sentiment.duration` (histogram)

3. **Business Metrics**:
   - `text_analysis.sentences.processed` (counter)
   - `text_analysis.clusters.generated` (histogram)
   - `text_analysis.sentiment.distribution` (gauge)

#### Metrics Implementation:
```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

meter_provider = MeterProvider(
    metric_readers=[
        PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                headers=headers
            )
        )
    ]
)

meter = meter_provider.get_meter(__name__)

# Define metrics
request_duration = meter.create_histogram(
    name="text_analysis.request.duration",
    description="Duration of text analysis requests",
    unit="ms"
)

sentences_processed = meter.create_counter(
    name="text_analysis.sentences.processed",
    description="Total sentences processed",
    unit="1"
)
```

### 5. Log Correlation

Enhance existing logging to include trace context:

```python
# In src/app/utils/logging.py
import opentelemetry.trace as trace

def add_trace_context(record):
    """Add OpenTelemetry trace context to log records."""
    current_span = trace.get_current_span()
    if current_span and current_span.get_span_context().is_valid:
        span_context = current_span.get_span_context()
        record.trace_id = format(span_context.trace_id, '032x')
        record.span_id = format(span_context.span_id, '016x')
        record.trace_flags = str(span_context.trace_flags)
    return True

# Add filter to logger
logger.addFilter(add_trace_context)
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. Add OpenTelemetry dependencies to `requirements.txt` and `setup.py`
2. Create `src/app/utils/telemetry.py` with basic configuration
3. Update `template.yaml` with environment variables and parameters
4. Instrument Lambda handler with root span

### Phase 2: Pipeline Instrumentation (Week 2)
1. Instrument `PipelineOrchestrator` methods with spans
2. Add spans to key pipeline components (embedding, clustering, sentiment)
3. Implement basic metrics collection
4. Update logging to include trace context

### Phase 3: Advanced Features (Week 3)
1. Implement comprehensive metrics collection
2. Add custom attributes for business context
3. Configure sampling strategies
4. Add error tracking and alerting attributes

### Phase 4: Testing & Validation (Week 4)
1. Write unit tests for telemetry utilities
2. Create integration tests with mocked exporters
3. Validate Honeycomb integration in staging
4. Performance testing with telemetry enabled

## Testing Strategy

### Unit Tests
- Test telemetry configuration with different environment variables
- Verify span creation and attribute setting
- Test metrics collection

### Integration Tests
- Mock OTLP exporter to verify data format
- Test trace context propagation through pipeline
- Validate log-trace correlation

### Performance Impact
- Baseline performance measurements before/after instrumentation
- Memory usage monitoring
- Cold start impact assessment

## Success Criteria

1. **Observability Coverage**: 100% of Lambda invocations generate traces
2. **Pipeline Visibility**: Each major pipeline component has dedicated spans
3. **Metric Collection**: Key business and performance metrics exported to Honeycomb
4. **Log Correlation**: All logs include trace IDs for correlation
5. **Performance**: <5% performance impact from instrumentation
6. **Reliability**: Telemetry system doesn't affect core functionality

## Risk Mitigation

1. **Performance Impact**: Use batch exporters, configure appropriate sampling
2. **Dependency Failure**: Graceful degradation if Honeycomb is unavailable
3. **Cost Control**: Configure sampling rates, limit custom attributes
4. **Security**: API keys stored in AWS Secrets Manager, not in code

## Maintenance Considerations

1. **Version Updates**: Regular updates to OpenTelemetry packages
2. **Configuration Management**: Centralized configuration in `telemetry.py`
3. **Documentation**: Keep instrumentation guidelines updated
4. **Monitoring**: Monitor the observability system itself

This plan provides comprehensive observability while maintaining the service's performance and reliability characteristics.