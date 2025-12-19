"""
Unit tests for the Lambda handler (functions/text_analysis/app.py).

Updated for AWS Lambda Powertools refactoring.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import os

# Add the functions directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'text_analysis'))

from src.app.handler import lambda_handler, _generate_job_id

class TestLambdaHandler:
    """Test the main lambda_handler function."""

    def test_lambda_handler_standalone_success(self, lambda_event_standalone, test_context):
        """Test lambda_handler with standalone analysis request."""
        mock_orchestrator = MagicMock()
        mock_result = {
            "clusters": [
                {"title": "Test Cluster", "sentiment": "negative", "sentences": ["s1"], "keyInsights": ["insight"]}
            ],
            "metadata": {"existing": "data"}
        }
        mock_orchestrator.process_standalone.return_value = mock_result
        
        # We need to ensure the event has the correct path for the router
        lambda_event_standalone["path"] = "/analyze"
        lambda_event_standalone["httpMethod"] = "POST"
        
        with patch('src.app.handler.PipelineOrchestrator', return_value=mock_orchestrator):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer.return_value.elapsed_ms.return_value = 150
                with patch('src.app.handler._generate_job_id', return_value="job_123"):
                    response = lambda_handler(lambda_event_standalone, test_context)
        
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        
        # Verify body matches StandaloneOutput (clusters + metadata)
        assert body["clusters"] == mock_result["clusters"]
        assert body["metadata"]["existing"] == "data"
        assert body["metadata"]["jobId"] == "job_123"
        assert body["metadata"]["processingTimeMs"] == 150

    def test_lambda_handler_comparison_success(self, lambda_event_comparison, test_context):
        """Test lambda_handler with comparison analysis request."""
        mock_orchestrator = MagicMock()
        mock_result = {
            "clusters": [],
            "metadata": {}
        }
        mock_orchestrator.process_comparison.return_value = mock_result
        
        lambda_event_comparison["path"] = "/analyze"
        lambda_event_comparison["httpMethod"] = "POST"
        
        with patch('src.app.handler.PipelineOrchestrator', return_value=mock_orchestrator):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer.return_value.elapsed_ms.return_value = 200
                with patch('src.app.handler._generate_job_id', return_value="job_456"):
                    response = lambda_handler(lambda_event_comparison, test_context)
        
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["clusters"] == []
        assert body["metadata"]["jobId"] == "job_456"

    def test_lambda_handler_validation_error(self, test_context):
        """Test validation error (handled by Powertools)."""
        # Create invalid event (missing baseline)
        event = {
            "httpMethod": "POST",
            "path": "/analyze",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "surveyTitle": "Test",
                "theme": "Test"
            })
        }
        
        response = lambda_handler(event, test_context)
        # Powertools returns 422 for Pydantic validation errors
        assert response["statusCode"] == 422
        
    def test_health_check(self, test_context):
        """Test health check endpoint."""
        event = {
            "httpMethod": "GET",
            "path": "/health",
            "headers": {}
        }
        
        response = lambda_handler(event, test_context)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "healthy"

class TestGenerateJobId:
    """Test the _generate_job_id function."""

    def test_generate_job_id_format(self):
        """Test that job ID follows expected format."""
        job_id = _generate_job_id()
        assert job_id.startswith("job_")
        parts = job_id.split("_")
        assert len(parts) == 3

    def test_generate_job_id_uniqueness(self):
        """Test that job IDs are likely unique."""
        job_ids = [_generate_job_id() for _ in range(10)]
        assert len(set(job_ids)) == len(job_ids)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
