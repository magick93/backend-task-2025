"""
Lambda handler for the text analysis microservice.

This module contains the AWS Lambda entrypoint that:
1. Parses incoming requests
2. Calls the pipeline orchestrator
3. Formats responses
4. Handles errors

No ML logic should be here - only request/response handling.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .pipeline.orchestrator import PipelineOrchestrator
from .utils.logging import setup_logger
from .utils.timing import Timer
from .api.schemas import StandaloneInput, ComparisonInput, ErrorResponse

logger = setup_logger(__name__)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entrypoint for text analysis requests.
    
    Args:
        event: Lambda event containing request data
        context: Lambda context object
        
    Returns:
        Response dictionary with analysis results or error
    """
    timer = Timer()
    job_id = _generate_job_id()
    
    try:
        logger.info(f"Starting text analysis job {job_id}")
        
        # Parse request
        request_data = _parse_request(event)
        logger.debug(f"Parsed request: {request_data}")
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Execute pipeline based on request type
        if request_data["type"] == "comparison":
            logger.info("Processing comparative analysis request")
            result = orchestrator.process_comparison(
                baseline=request_data["baseline"],
                comparison=request_data["comparison"],
                job_id=job_id
            )
        else:  # standalone
            logger.info("Processing standalone analysis request")
            # For standalone, we pass the baseline sentences
            result = orchestrator.process_standalone(
                sentences=request_data["baseline"],
                job_id=job_id
            )
        
        # Format successful response
        processing_time_ms = timer.elapsed_ms()
        response = _format_success_response(result, job_id, processing_time_ms)
        
        logger.info(f"Completed job {job_id} in {processing_time_ms}ms")
        return response
        
    except Exception as e:
        # Handle errors gracefully
        processing_time_ms = timer.elapsed_ms()
        error_response = _format_error_response(e, job_id, processing_time_ms)
        
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        return error_response


def _parse_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate the incoming Lambda event using Pydantic schemas.
    
    Args:
        event: Lambda event
    
    Returns:
        Parsed request data with keys:
        - 'type': 'standalone' or 'comparison'
        - 'surveyTitle': string
        - 'theme': string
        - 'baseline': list of Sentence objects (dict with 'sentence' and 'id')
        - 'comparison': list of Sentence objects (only for comparison type)
    
    Raises:
        ValueError: If request is malformed or missing required fields
    """
    try:
        # Check if this is an API Gateway request
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event
        
        # Determine request type based on presence of 'comparison' field
        if "comparison" in body:
            # Comparative analysis request
            input_model = ComparisonInput(**body)
            return {
                "type": "comparison",
                "surveyTitle": input_model.surveyTitle,
                "theme": input_model.theme,
                "baseline": [{"sentence": s.sentence, "id": s.id} for s in input_model.baseline],
                "comparison": [{"sentence": s.sentence, "id": s.id} for s in input_model.comparison]
            }
        else:
            # Standalone analysis request
            input_model = StandaloneInput(**body)
            return {
                "type": "standalone",
                "surveyTitle": input_model.surveyTitle,
                "theme": input_model.theme,
                "baseline": [{"sentence": s.sentence, "id": s.id} for s in input_model.baseline]
            }
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request: {str(e)}")
    except Exception as e:
        # Re-raise validation errors as ValueError
        raise ValueError(f"Invalid request data: {str(e)}")


def _format_success_response(
    result: Dict[str, Any], 
    job_id: str, 
    processing_time_ms: int
) -> Dict[str, Any]:
    """
    Format a successful response.
    
    Args:
        result: Pipeline result
        job_id: Job identifier
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        Formatted response dictionary
    """
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "X-Job-ID": job_id
        },
        "body": json.dumps({
            "status": "success",
            "jobId": job_id,
            "processingTimeMs": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result": result
        }, ensure_ascii=False)
    }


def _format_error_response(
    error: Exception, 
    job_id: str, 
    processing_time_ms: int
) -> Dict[str, Any]:
    """
    Format an error response.
    
    Args:
        error: Exception that occurred
        job_id: Job identifier
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        Formatted error response dictionary
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    status_code = 400 if isinstance(error, ValueError) else 500
    
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "X-Job-ID": job_id
        },
        "body": json.dumps({
            "status": "error",
            "jobId": job_id,
            "processingTimeMs": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": {
                "type": error_type,
                "message": error_message
            }
        }, ensure_ascii=False)
    }


def _generate_job_id() -> str:
    """
    Generate a unique job identifier.
    
    Returns:
        Job ID string
    """
    # TODO: Implement more robust job ID generation
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    import random
    random_suffix = random.randint(1000, 9999)
    return f"job_{timestamp}_{random_suffix}"