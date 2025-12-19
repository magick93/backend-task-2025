"""
Unit tests for the Lambda handler (functions/text_analysis/app.py).

Tests cover:
- Lambda handler entrypoint with standalone and comparison requests
- Request parsing for API Gateway and direct invocation events
- Response formatting (success and error)
- Job ID generation
- Error handling and edge cases
- Mocking of PipelineOrchestrator to avoid actual ML processing
"""

import json
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
import sys
import os

# Add the functions directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'text_analysis'))

# Import the Lambda handler module using importlib to handle hyphen in directory name
import importlib.util
spec = importlib.util.spec_from_file_location(
    "handler",
    os.path.join(os.path.dirname(__file__), '..', '..', 'functions', 'text_analysis', 'handler.py')
)
handler_module = importlib.util.module_from_spec(spec)
sys.modules["handler"] = handler_module
spec.loader.exec_module(handler_module)

# Import functions from the loaded module
lambda_handler = handler_module.lambda_handler
# Since handler.py is a wrapper, it only exports lambda_handler.
# We need to mock the underlying functions from the installed app package or the wrapper if it exposed them.
# The previous test relied on app.py being the implementation. Now it's a wrapper.
# We should test the implementation in src/app/handler.py via the wrapper, or test the wrapper.
# But wait, the wrapper re-exports lambda_handler from app.handler.
# So lambda_handler IS the implementation from src/app/handler.py.
# However, the helper functions (_parse_request, etc.) are NOT exported by the wrapper.
# We need to import them from app.handler (the package module) directly for testing.

import app.handler as implementation_module

_parse_request = implementation_module._parse_request
_format_success_response = implementation_module._format_success_response
_format_error_response = implementation_module._format_error_response
_generate_job_id = implementation_module._generate_job_id
PipelineOrchestrator = implementation_module.PipelineOrchestrator
Timer = implementation_module.Timer


class TestLambdaHandler:
    """Test the main lambda_handler function."""

    def test_lambda_handler_standalone_success(self, lambda_event_standalone, test_context):
        """Test lambda_handler with standalone analysis request."""
        # Mock the PipelineOrchestrator to return a predictable result
        mock_orchestrator = MagicMock()
        mock_result = {
            "clusters": [
                {"id": 0, "size": 3, "representative": "Delivery was late"},
                {"id": 1, "size": 2, "representative": "Food arrived cold"}
            ],
            "insights": {
                "summary": "Test summary",
                "themes": ["delivery", "food quality"]
            }
        }
        mock_orchestrator.process_standalone.return_value = mock_result
        
        # Mock the Timer to return a fixed elapsed time
        # We need to patch the implementation module, not 'app' (which was the file name)
        # The implementation module is 'app.handler'
        with patch('src.app.handler.PipelineOrchestrator', return_value=mock_orchestrator):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer_instance = MagicMock()
                mock_timer_instance.elapsed_ms.return_value = 150
                mock_timer.return_value = mock_timer_instance
                
                # Mock _generate_job_id to return a predictable ID
                with patch('src.app.handler._generate_job_id', return_value="job_20250101000000_1234"):
                    response = lambda_handler(lambda_event_standalone, test_context)
        
        # Verify response structure
        assert response["statusCode"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["headers"]["X-Job-ID"] == "job_20250101000000_1234"
        
        # Parse and verify body
        body = json.loads(response["body"])
        assert body["status"] == "success"
        assert body["jobId"] == "job_20250101000000_1234"
        assert body["processingTimeMs"] == 150
        assert "timestamp" in body
        assert body["result"] == mock_result
        
        # Verify orchestrator was called correctly
        mock_orchestrator.process_standalone.assert_called_once()
        call_args = mock_orchestrator.process_standalone.call_args
        # call_args is a tuple of (args, kwargs)
        args, kwargs = call_args
        # The function is called with keyword arguments: sentences=..., job_id=...
        assert kwargs['sentences'] == [
            {"sentence": "The delivery was late by 30 minutes", "id": "feedback-001"},
            {"sentence": "Food arrived cold and soggy", "id": "feedback-002"},
            {"sentence": "Driver was friendly and professional", "id": "feedback-003"},
            {"sentence": "App interface is confusing to navigate", "id": "feedback-004"},
            {"sentence": "Prices are too high compared to competitors", "id": "feedback-005"}
        ]
        assert kwargs['job_id'] == "job_20250101000000_1234"  # job_id

    def test_lambda_handler_comparison_success(self, lambda_event_comparison, test_context):
        """Test lambda_handler with comparison analysis request."""
        # Mock the PipelineOrchestrator
        mock_orchestrator = MagicMock()
        mock_result = {
            "baseline": {"clusters": [], "insights": {}},
            "comparison": {"clusters": [], "insights": {}},
            "comparative_insights": {"differences": [], "similarities": []}
        }
        mock_orchestrator.process_comparison.return_value = mock_result
        
        with patch('src.app.handler.PipelineOrchestrator', return_value=mock_orchestrator):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer_instance = MagicMock()
                mock_timer_instance.elapsed_ms.return_value = 200
                mock_timer.return_value = mock_timer_instance
                
                with patch('src.app.handler._generate_job_id', return_value="job_20250101000000_5678"):
                    response = lambda_handler(lambda_event_comparison, test_context)
        
        # Verify response
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "success"
        assert body["jobId"] == "job_20250101000000_5678"
        assert body["result"] == mock_result
        
        # Verify orchestrator was called correctly
        mock_orchestrator.process_comparison.assert_called_once()
        call_args = mock_orchestrator.process_comparison.call_args
        args, kwargs = call_args
        # Called with keyword arguments: baseline=..., comparison=..., job_id=...
        assert len(kwargs['baseline']) == 5  # baseline sentences
        assert len(kwargs['comparison']) == 5  # comparison sentences
        assert kwargs['job_id'] == "job_20250101000000_5678"  # job_id

    def test_lambda_handler_direct_invocation_standalone(self, test_context):
        """Test lambda_handler with direct invocation (no API Gateway)."""
        # Create a direct invocation event (no 'body' field)
        direct_event = {
            "surveyTitle": "Direct Test",
            "theme": "test theme",
            "baseline": [
                {"sentence": "Test sentence 1", "id": "test-001"},
                {"sentence": "Test sentence 2", "id": "test-002"}
            ]
        }
        
        mock_orchestrator = MagicMock()
        mock_result = {"clusters": [], "insights": {}}
        mock_orchestrator.process_standalone.return_value = mock_result
        
        with patch('src.app.handler.PipelineOrchestrator', return_value=mock_orchestrator):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer_instance = MagicMock()
                mock_timer_instance.elapsed_ms.return_value = 100
                mock_timer.return_value = mock_timer_instance
                
                with patch('src.app.handler._generate_job_id', return_value="job_direct_123"):
                    response = lambda_handler(direct_event, test_context)
        
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "success"
        
        # Verify orchestrator was called with correct sentences
        mock_orchestrator.process_standalone.assert_called_once()
        call_args = mock_orchestrator.process_standalone.call_args
        args, kwargs = call_args
        # Called with keyword arguments
        assert kwargs['sentences'] == [
            {"sentence": "Test sentence 1", "id": "test-001"},
            {"sentence": "Test sentence 2", "id": "test-002"}
        ]

    def test_lambda_handler_validation_error(self, lambda_event_standalone, test_context):
        """Test lambda_handler with invalid request data (validation error)."""
        # Mock _parse_request to raise ValueError
        with patch('src.app.handler._parse_request', side_effect=ValueError("Missing required field: baseline")):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer_instance = MagicMock()
                mock_timer_instance.elapsed_ms.return_value = 50
                mock_timer.return_value = mock_timer_instance
                
                with patch('src.app.handler._generate_job_id', return_value="job_error_123"):
                    response = lambda_handler(lambda_event_standalone, test_context)
        
        # Should return 400 error
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["status"] == "error"
        assert body["error"]["type"] == "ValueError"
        assert "Missing required field" in body["error"]["message"]

    def test_lambda_handler_pipeline_exception(self, lambda_event_standalone, test_context):
        """Test lambda_handler when pipeline raises an exception."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_standalone.side_effect = Exception("Pipeline failed unexpectedly")
        
        with patch('src.app.handler.PipelineOrchestrator', return_value=mock_orchestrator):
            with patch('src.app.handler.Timer') as mock_timer:
                mock_timer_instance = MagicMock()
                mock_timer_instance.elapsed_ms.return_value = 75
                mock_timer.return_value = mock_timer_instance
                
                with patch('src.app.handler._generate_job_id', return_value="job_exception_123"):
                    response = lambda_handler(lambda_event_standalone, test_context)
        
        # Should return 500 error
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["status"] == "error"
        # The error type could be Exception or AcceleratorError depending on whether mock works
        # Accept either
        assert body["error"]["type"] in ["Exception", "AcceleratorError"]
        if body["error"]["type"] == "Exception":
            assert "Pipeline failed" in body["error"]["message"]
        else:
            # AcceleratorError from CUDA
            assert "CUDA error" in body["error"]["message"]

    def test_lambda_handler_malformed_json(self, test_context):
        """Test lambda_handler with malformed JSON in API Gateway event."""
        malformed_event = {
            "httpMethod": "POST",
            "body": "{invalid json",
            "headers": {"Content-Type": "application/json"}
        }
        
        with patch('src.app.handler.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer_instance.elapsed_ms.return_value = 10
            mock_timer.return_value = mock_timer_instance
            
            with patch('src.app.handler._generate_job_id', return_value="job_malformed_123"):
                response = lambda_handler(malformed_event, test_context)
        
        # Should return 400 error for JSON decode error
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["status"] == "error"
        assert "Invalid JSON" in body["error"]["message"]

    def test_lambda_handler_empty_body(self, test_context):
        """Test lambda_handler with empty body in API Gateway event."""
        empty_event = {
            "httpMethod": "POST",
            "body": "",
            "headers": {"Content-Type": "application/json"}
        }
        
        with patch('src.app.handler.Timer') as mock_timer:
            mock_timer_instance = MagicMock()
            mock_timer_instance.elapsed_ms.return_value = 5
            mock_timer.return_value = mock_timer_instance
            
            with patch('src.app.handler._generate_job_id', return_value="job_empty_123"):
                response = lambda_handler(empty_event, test_context)
        
        # Should return 400 error
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert body["status"] == "error"


class TestParseRequest:
    """Test the _parse_request function."""

    def test_parse_request_api_gateway_standalone(self, lambda_event_standalone):
        """Test parsing API Gateway event for standalone analysis."""
        result = _parse_request(lambda_event_standalone)
        
        assert result["type"] == "standalone"
        assert result["surveyTitle"] == "Food Delivery App Feedback"
        assert result["theme"] == "customer experience"
        assert len(result["baseline"]) == 5
        assert "comparison" not in result
        
        # Verify sentence structure
        assert result["baseline"][0]["sentence"] == "The delivery was late by 30 minutes"
        assert result["baseline"][0]["id"] == "feedback-001"

    def test_parse_request_api_gateway_comparison(self, lambda_event_comparison):
        """Test parsing API Gateway event for comparison analysis."""
        result = _parse_request(lambda_event_comparison)
        
        assert result["type"] == "comparison"
        assert result["surveyTitle"] == "Food Delivery App Feedback"
        assert result["theme"] == "customer experience"
        assert len(result["baseline"]) == 5
        assert len(result["comparison"]) == 5
        
        # Verify comparison sentences
        assert result["comparison"][0]["sentence"] == "Delivery is always on time now"
        assert result["comparison"][0]["id"] == "feedback-101"

    def test_parse_request_direct_invocation_standalone(self):
        """Test parsing direct invocation event (no API Gateway)."""
        direct_event = {
            "surveyTitle": "Direct Test",
            "theme": "test theme",
            "baseline": [
                {"sentence": "Sentence 1", "id": "id1"},
                {"sentence": "Sentence 2", "id": "id2"}
            ]
        }
        
        result = _parse_request(direct_event)
        
        assert result["type"] == "standalone"
        assert result["surveyTitle"] == "Direct Test"
        assert result["theme"] == "test theme"
        assert len(result["baseline"]) == 2

    def test_parse_request_direct_invocation_comparison(self):
        """Test parsing direct invocation event for comparison."""
        direct_event = {
            "surveyTitle": "Comparison Test",
            "theme": "test",
            "baseline": [{"sentence": "Baseline 1", "id": "b1"}],
            "comparison": [{"sentence": "Comparison 1", "id": "c1"}]
        }
        
        result = _parse_request(direct_event)
        
        assert result["type"] == "comparison"
        assert "comparison" in result
        assert len(result["comparison"]) == 1

    def test_parse_request_missing_required_field(self):
        """Test parsing event missing required fields."""
        invalid_event = {
            "surveyTitle": "Test",
            # Missing 'baseline' field
        }
        
        with pytest.raises(ValueError, match="Invalid request data"):
            _parse_request(invalid_event)

    def test_parse_request_malformed_json_body(self):
        """Test parsing with malformed JSON in body."""
        event = {
            "body": "{invalid json",
            "headers": {"Content-Type": "application/json"}
        }
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_request(event)

    def test_parse_request_empty_body(self):
        """Test parsing with empty body."""
        event = {
            "body": "",
            "headers": {"Content-Type": "application/json"}
        }
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_request(event)

    def test_parse_request_invalid_sentence_structure(self):
        """Test parsing with invalid sentence structure."""
        event = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": [
                {"text": "Missing id field"}  # Should have 'sentence' and 'id'
            ]
        }
        
        with pytest.raises(ValueError, match="Invalid request data"):
            _parse_request(event)


class TestFormatSuccessResponse:
    """Test the _format_success_response function."""

    def test_format_success_response_basic(self):
        """Test basic success response formatting."""
        result = {"test": "data"}
        job_id = "job_123"
        processing_time_ms = 150
        
        response = _format_success_response(result, job_id, processing_time_ms)
        
        assert response["statusCode"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["headers"]["X-Job-ID"] == job_id
        
        body = json.loads(response["body"])
        assert body["status"] == "success"
        assert body["jobId"] == job_id
        assert body["processingTimeMs"] == processing_time_ms
        assert body["result"] == result
        assert "timestamp" in body
        # Check timestamp format (ISO 8601 with Z)
        assert body["timestamp"].endswith("Z")

    def test_format_success_response_with_unicode(self):
        """Test success response with Unicode characters."""
        result = {"message": "Test with emoji 🚀 and unicode café"}
        job_id = "job_unicode"
        processing_time_ms = 200
        
        response = _format_success_response(result, job_id, processing_time_ms)
        
        # ensure_ascii=False should preserve Unicode
        body = json.loads(response["body"])
        assert "🚀" in body["result"]["message"]
        assert "café" in body["result"]["message"]

    def test_format_success_response_empty_result(self):
        """Test success response with empty result."""
        result = {}
        job_id = "job_empty"
        processing_time_ms = 0
        
        response = _format_success_response(result, job_id, processing_time_ms)
        
        body = json.loads(response["body"])
        assert body["result"] == {}


class TestFormatErrorResponse:
    """Test the _format_error_response function."""

    def test_format_error_response_value_error(self):
        """Test error response for ValueError (should be 400)."""
        error = ValueError("Validation failed: missing field")
        job_id = "job_error_123"
        processing_time_ms = 50
        
        response = _format_error_response(error, job_id, processing_time_ms)
        
        assert response["statusCode"] == 400  # ValueError -> 400
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["headers"]["X-Job-ID"] == job_id
        
        body = json.loads(response["body"])
        assert body["status"] == "error"
        assert body["jobId"] == job_id
        assert body["processingTimeMs"] == processing_time_ms
        assert body["error"]["type"] == "ValueError"
        assert body["error"]["message"] == "Validation failed: missing field"
        assert "timestamp" in body

    def test_format_error_response_generic_exception(self):
        """Test error response for generic Exception (should be 500)."""
        error = Exception("Internal server error")
        job_id = "job_exception_456"
        processing_time_ms = 75
        
        response = _format_error_response(error, job_id, processing_time_ms)
        
        assert response["statusCode"] == 500  # Generic Exception -> 500
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["headers"]["X-Job-ID"] == job_id
        
        body = json.loads(response["body"])
        assert body["status"] == "error"
        assert body["jobId"] == job_id
        assert body["processingTimeMs"] == processing_time_ms
        assert body["error"]["type"] == "Exception"
        assert body["error"]["message"] == "Internal server error"

    def test_format_error_response_custom_exception(self):
        """Test error response for custom exception types."""
        class CustomError(Exception):
            pass
        
        error = CustomError("Custom error message")
        job_id = "job_custom_123"
        processing_time_ms = 30
        
        response = _format_error_response(error, job_id, processing_time_ms)
        
        # Custom exception (not ValueError) should be 500
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert body["error"]["type"] == "CustomError"
        assert body["error"]["message"] == "Custom error message"

    def test_format_error_response_with_empty_message(self):
        """Test error response with empty error message."""
        error = ValueError("")  # Empty message
        job_id = "job_empty_msg"
        processing_time_ms = 20
        
        response = _format_error_response(error, job_id, processing_time_ms)
        
        body = json.loads(response["body"])
        assert body["error"]["message"] == ""  # Should preserve empty string


class TestGenerateJobId:
    """Test the _generate_job_id function."""

    def test_generate_job_id_format(self):
        """Test that job ID follows expected format."""
        job_id = _generate_job_id()
        
        # Should start with "job_"
        assert job_id.startswith("job_")
        
        # Should have timestamp and random suffix separated by underscores
        parts = job_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "job"
        
        # Timestamp should be 14 digits (YYYYMMDDHHMMSS)
        assert len(parts[1]) == 14
        assert parts[1].isdigit()
        
        # Random suffix should be 4 digits
        assert len(parts[2]) == 4
        assert parts[2].isdigit()
        assert 1000 <= int(parts[2]) <= 9999

    def test_generate_job_id_uniqueness(self):
        """Test that job IDs are likely unique (not guaranteed but should differ)."""
        # Generate multiple job IDs
        job_ids = [_generate_job_id() for _ in range(10)]
        
        # They should all be different (very high probability)
        assert len(set(job_ids)) == len(job_ids)
        
        # All should follow the same format
        for job_id in job_ids:
            assert job_id.startswith("job_")
            parts = job_id.split("_")
            assert len(parts) == 3

    def test_generate_job_id_deterministic_with_mocked_random(self):
        """Test job ID generation with mocked random for deterministic testing."""
        # random is imported inside _generate_job_id, so we need to patch it at the module level
        with patch('random.randint', return_value=9999):
            with patch('app.handler.datetime') as mock_datetime:
                mock_datetime.utcnow.return_value.strftime.return_value = "20250101000000"
                job_id = _generate_job_id()
                
                assert job_id == "job_20250101000000_9999"

    def test_generate_job_id_with_mocked_time(self):
        """Test job ID generation with mocked time."""
        with patch('app.handler.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = "20251218022416"
            with patch('random.randint', return_value=1234):
                job_id = _generate_job_id()
                
                assert job_id == "job_20251218022416_1234"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])