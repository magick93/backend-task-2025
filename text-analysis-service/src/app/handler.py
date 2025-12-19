"""
Lambda handler for the text analysis microservice.

This module contains the AWS Lambda entrypoint that:
1. Parses incoming requests using AWS Lambda Powertools
2. Calls the pipeline orchestrator
3. Formats responses using Pydantic models
4. Handles errors
"""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import random

from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.event_handler.openapi.config import OpenAPIConfig
from aws_lambda_powertools.event_handler.openapi.params import Body
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.logging import correlation_paths

from .pipeline.orchestrator import PipelineOrchestrator
from .utils.logging import setup_logger
from .utils.timing import Timer
from .api.schemas import (
    StandaloneInput, 
    ComparisonInput, 
    ErrorResponse, 
    StandaloneOutput, 
    ComparisonOutput
)

# Initialize Powertools
logger = Logger(service="text-analysis-service")
tracer = Tracer(service="text-analysis-service")
metrics = Metrics(namespace="TextAnalysis")

app = APIGatewayRestResolver(
    enable_validation=True,
    openapi_config=OpenAPIConfig(
        title="Text Analysis Service API",
        version="1.0.0",
        description="API for text analysis, including clustering, sentiment analysis, and comparative insights.",
        extensions={
            "x-tagGroups": [
                {"name": "Core Services", "tags": ["Analysis"]},
                {"name": "Admin Operations", "tags": ["Admin Operations"]}
            ],
            "x-logo": {
                "url": "https://example.com/logo.png",
                "altText": "Text Analysis Service Logo"
            }
        }
    )
)

def _generate_job_id() -> str:
    """Generate a unique job identifier."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_suffix = random.randint(1000, 9999)
    return f"job_{timestamp}_{random_suffix}"

@app.post(
    "/analyze", 
    tags=["Analysis"],
    summary="Analyze text data",
    description="Analyze text data using standalone or comparison mode.",
    response_description="Analysis results including clusters and insights.",
    responses={
        200: {"description": "Successful analysis", "model": Union[ComparisonOutput, StandaloneOutput]},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    openapi_extensions={
        "x-codeSamples": [
            {
                "lang": "Python",
                "label": "Python (Requests)",
                "source": "import requests\n\nurl = 'https://api.example.com/analyze'\npayload = {\n    'surveyTitle': 'Example',\n    'theme': 'Feedback',\n    'baseline': [{'sentence': 'Great service', 'id': '1'}]\n}\nresponse = requests.post(url, json=payload)\nprint(response.json())"
            }
        ]
    }
)
@tracer.capture_method
def analyze(payload: Union[ComparisonInput, StandaloneInput] = Body(...)) -> Dict[str, Any]:
    """
    Process text analysis request.
    
    Accepts either StandaloneInput or ComparisonInput.
    Returns StandaloneOutput or ComparisonOutput.
    """
    timer = Timer()
    job_id = _generate_job_id()
    
    logger.info(f"Starting text analysis job {job_id}")
    logger.append_keys(job_id=job_id)
    
    try:
        orchestrator = PipelineOrchestrator()
        
        # Determine request type based on payload type
        if isinstance(payload, ComparisonInput):
            logger.info("Processing comparative analysis request")
            result = orchestrator.process_comparison(
                baseline=[{"sentence": s.sentence, "id": s.id} for s in payload.baseline],
                comparison=[{"sentence": s.sentence, "id": s.id} for s in payload.comparison],
                job_id=job_id
            )
        else:
            logger.info("Processing standalone analysis request")
            result = orchestrator.process_standalone(
                sentences=[{"sentence": s.sentence, "id": s.id} for s in payload.baseline],
                job_id=job_id
            )

        processing_time_ms = timer.elapsed_ms()
        logger.info(f"Completed job {job_id} in {processing_time_ms}ms")
        
        # Add metadata
        if "metadata" not in result:
            result["metadata"] = {}
        
        result["metadata"].update({
            "jobId": job_id,
            "processingTimeMs": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        return result

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        # In a real app we might want to map specific exceptions to 400
        raise

@app.get("/health", tags=["Admin Operations"], summary="Health check")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
@metrics.log_metrics
def lambda_handler(event: Dict[str, Any], context: LambdaContext) -> Dict[str, Any]:
    return app.resolve(event, context)
