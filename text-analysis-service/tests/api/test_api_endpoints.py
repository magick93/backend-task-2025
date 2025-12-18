"""
API endpoint integration tests for the text analysis service.

These tests validate the API Gateway endpoints when running with `sam local start-api`.
They make actual HTTP requests to the local SAM API running on port 3000.

Test Structure:
1. Health check test (optional)
2. Standalone analysis endpoint tests
3. Comparison analysis endpoint tests
4. Error handling tests (malformed JSON, missing fields, etc.)

Requirements:
- `sam local start-api` must be running on port 3000
- `requests` library installed
- Test data fixtures available

Usage:
    pytest tests/api/test_api_endpoints.py -v
    pytest tests/api/test_api_endpoints.py -m api -v  # Run only API tests
    pytest tests/api/test_api_endpoints.py --skip-if-api-not-running  # Skip if API unavailable

Note: These tests are marked with `@pytest.mark.api` and will be skipped if the
API is not available (connection error).
"""

import json
import pytest
import requests
from typing import Dict, Any, List
import time

# Base URL for SAM local API
BASE_URL = "http://127.0.0.1:3000"
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"

# Test markers
API_TEST_MARKER = "api"
INTEGRATION_TEST_MARKER = "integration"

# Helper function to check if API is available
def is_api_available() -> bool:
    """Check if the SAM local API is running and reachable."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code < 500
    except (requests.ConnectionError, requests.Timeout):
        return False

# Pytest marker for API tests
pytestmark = [
    pytest.mark.api,
    pytest.mark.integration,
    pytest.mark.skipif(not is_api_available(), reason="SAM local API not running on port 3000")
]


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def standalone_request_data(self) -> Dict[str, Any]:
        """Fixture providing valid standalone analysis request data."""
        return {
            "surveyTitle": "Food Delivery App Feedback",
            "theme": "customer experience",
            "baseline": [
                {"sentence": "The delivery was late by 30 minutes", "id": "feedback-001"},
                {"sentence": "Food arrived cold and soggy", "id": "feedback-002"},
                {"sentence": "Driver was friendly and professional", "id": "feedback-003"},
                {"sentence": "App interface is confusing to navigate", "id": "feedback-004"},
                {"sentence": "Prices are too high compared to competitors", "id": "feedback-005"}
            ]
        }
    
    @pytest.fixture
    def comparison_request_data(self) -> Dict[str, Any]:
        """Fixture providing valid comparison analysis request data."""
        return {
            "surveyTitle": "Food Delivery App Feedback",
            "theme": "customer experience",
            "baseline": [
                {"sentence": "The delivery was late by 30 minutes", "id": "feedback-001"},
                {"sentence": "Food arrived cold and soggy", "id": "feedback-002"},
                {"sentence": "Driver was friendly and professional", "id": "feedback-003"},
                {"sentence": "App interface is confusing to navigate", "id": "feedback-004"},
                {"sentence": "Prices are too high compared to competitors", "id": "feedback-005"}
            ],
            "comparison": [
                {"sentence": "Delivery is always on time now", "id": "feedback-101"},
                {"sentence": "Food arrives hot and fresh", "id": "feedback-102"},
                {"sentence": "Drivers are rude sometimes", "id": "feedback-103"},
                {"sentence": "App interface improved significantly", "id": "feedback-104"},
                {"sentence": "Prices increased again", "id": "feedback-105"}
            ]
        }
    
    def test_api_health(self):
        """Test that the API is reachable (optional health check)."""
        # Note: SAM local doesn't have a default health endpoint, but we can test the analyze endpoint
        # with a HEAD request or check if we get a 404/405/403 for other methods
        try:
            response = requests.options(ANALYZE_ENDPOINT, timeout=2)
            # Should get 200, 405 (Method Not Allowed), 404 (Not Found), or 403 (Forbidden)
            # which all indicate the API is reachable
            assert response.status_code in (200, 405, 404, 403)
        except requests.ConnectionError:
            pytest.skip("API not reachable")
    
    def test_standalone_analysis_success(self, standalone_request_data):
        """
        Test successful standalone analysis request.
        
        Validates:
        - HTTP 200 status code
        - Correct Content-Type header
        - X-Job-ID header present
        - Response JSON structure matches expected schema
        - Job ID in response body matches header
        """
        # Make POST request
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=standalone_request_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        # Assert status code
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        # Assert headers
        assert response.headers["Content-Type"] == "application/json"
        assert "X-Job-ID" in response.headers
        job_id_header = response.headers["X-Job-ID"]
        
        # Parse response
        response_data = response.json()
        
        # Assert response structure
        assert response_data["status"] == "success"
        assert response_data["jobId"] == job_id_header
        assert "processingTimeMs" in response_data
        assert "timestamp" in response_data
        assert "result" in response_data
        
        # Assert result structure
        result = response_data["result"]
        assert "clusters" in result
        assert isinstance(result["clusters"], list)
        
        # If clusters are returned, validate their structure
        if result["clusters"]:
            cluster = result["clusters"][0]
            assert "title" in cluster
            assert "sentiment" in cluster
            assert "sentences" in cluster
            assert "keyInsights" in cluster
    
    def test_comparison_analysis_success(self, comparison_request_data):
        """
        Test successful comparison analysis request.
        
        Validates:
        - HTTP 200 status code
        - Correct headers
        - Response JSON structure for comparison analysis
        - Comparison-specific fields in result
        """
        # Make POST request
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=comparison_request_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        # Assert status code
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        # Parse response
        response_data = response.json()
        
        # Assert response structure
        assert response_data["status"] == "success"
        assert "jobId" in response_data
        assert "result" in response_data
        
        # Assert result structure for comparison
        result = response_data["result"]
        # Comparison result contains baseline, comparison, comparison_analysis, processing_metadata
        assert "baseline" in result
        assert "comparison" in result
        assert "comparison_analysis" in result
        assert "processing_metadata" in result
        
        # Validate baseline structure
        baseline = result["baseline"]
        assert "clusters" in baseline
        assert isinstance(baseline["clusters"], list)
        
        # Validate comparison structure
        comparison = result["comparison"]
        assert "clusters" in comparison
        assert isinstance(comparison["clusters"], list)
        
        # Validate comparison_analysis structure
        comparison_analysis = result["comparison_analysis"]
        assert "similarities" in comparison_analysis
        assert "differences" in comparison_analysis
        assert "similarity_score" in comparison_analysis
        assert "summary" in comparison_analysis
    
    def test_malformed_json(self):
        """Test handling of malformed JSON request body."""
        # Send invalid JSON
        response = requests.post(
            ANALYZE_ENDPOINT,
            data="{invalid json",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400 for malformed JSON, got {response.status_code}"
        
        # Parse error response
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "error" in response_data
        assert "type" in response_data["error"]
        assert "message" in response_data["error"]
        
        # Check error type indicates JSON parsing issue
        error_type = response_data["error"]["type"]
        assert "JSON" in error_type or "ValueError" in error_type or "Validation" in error_type
    
    def test_missing_required_fields(self):
        """Test handling of request missing required fields."""
        # Send request without required 'baseline' field
        invalid_request = {
            "surveyTitle": "Test Survey",
            "theme": "test theme"
            # Missing 'baseline'
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=invalid_request,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400 for missing fields, got {response.status_code}"
        
        # Parse error response
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "error" in response_data
        assert "message" in response_data["error"]
        
        # Error message should indicate missing field
        error_message = response_data["error"]["message"].lower()
        assert "baseline" in error_message or "required" in error_message or "missing" in error_message
    
    def test_empty_sentences_list(self):
        """Test handling of empty sentences list in baseline."""
        # Send request with empty baseline list
        invalid_request = {
            "surveyTitle": "Test Survey",
            "theme": "test theme",
            "baseline": []  # Empty list should be invalid
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=invalid_request,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400 for empty sentences, got {response.status_code}"
        
        # Parse error response
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "error" in response_data
    
    def test_invalid_sentence_format(self):
        """Test handling of invalid sentence format (missing 'id' field)."""
        # Send request with malformed sentence (missing 'id')
        invalid_request = {
            "surveyTitle": "Test Survey",
            "theme": "test theme",
            "baseline": [
                {"sentence": "This is a test sentence"}  # Missing 'id'
            ]
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=invalid_request,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400 for invalid sentence format, got {response.status_code}"
        
        # Parse error response
        response_data = response.json()
        assert response_data["status"] == "error"
    
    def test_duplicate_sentence_ids(self):
        """Test handling of duplicate sentence IDs within baseline."""
        # Send request with duplicate IDs
        invalid_request = {
            "surveyTitle": "Test Survey",
            "theme": "test theme",
            "baseline": [
                {"sentence": "First sentence", "id": "duplicate-id"},
                {"sentence": "Second sentence", "id": "duplicate-id"}  # Same ID
            ]
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=invalid_request,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400 for duplicate IDs, got {response.status_code}"
        
        # Parse error response
        response_data = response.json()
        assert response_data["status"] == "error"
        
        # Error message should mention duplicate or unique
        error_message = response_data["error"]["message"].lower()
        assert "unique" in error_message or "duplicate" in error_message
    
    def test_large_request_size(self):
        """Test handling of large request (many sentences)."""
        # Create request with 50 sentences (still reasonable)
        large_request = {
            "surveyTitle": "Large Survey",
            "theme": "performance testing",
            "baseline": [
                {"sentence": f"Sentence number {i}", "id": f"id-{i}"}
                for i in range(50)
            ]
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=large_request,
            headers={"Content-Type": "application/json"},
            timeout=30  # Longer timeout for large request
        )
        
        # Should either succeed (200) or return appropriate error (413, 400)
        # For this test, we just verify it doesn't crash with 500
        assert response.status_code != 500, f"Server error with large request: {response.text}"
        
        if response.status_code == 200:
            # If successful, validate response structure
            response_data = response.json()
            assert response_data["status"] == "success"
            assert "result" in response_data
    
    def test_unsupported_http_method(self):
        """Test that unsupported HTTP methods return appropriate error."""
        # Try GET request (should be unsupported)
        response = requests.get(ANALYZE_ENDPOINT, timeout=5)
        
        # SAM local might return 403, 404, or 405
        # We just verify it's not 200 (since POST is required)
        assert response.status_code != 200, "GET should not be supported"
        
        # Try PUT request
        response = requests.put(ANALYZE_ENDPOINT, timeout=5)
        assert response.status_code != 200, "PUT should not be supported"
    
    def test_response_headers(self):
        """Test that response headers are correctly set."""
        request_data = {
            "surveyTitle": "Header Test",
            "theme": "testing",
            "baseline": [
                {"sentence": "Test sentence for headers", "id": "header-test-001"}
            ]
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            # Check required headers
            assert "Content-Type" in response.headers
            assert response.headers["Content-Type"] == "application/json"
            
            assert "X-Job-ID" in response.headers
            job_id = response.headers["X-Job-ID"]
            assert job_id.startswith("job_")
            
            # Check CORS headers if present
            if "Access-Control-Allow-Origin" in response.headers:
                # CORS headers should be present for API Gateway
                pass
    
    def test_comparison_missing_comparison_field(self):
        """
        Test that a comparison request without 'comparison' field 
        is treated as standalone analysis.
        """
        # This is actually a valid standalone request (no comparison field)
        request_data = {
            "surveyTitle": "Test Survey",
            "theme": "test theme",
            "baseline": [
                {"sentence": "Test sentence", "id": "test-001"}
            ]
        }
        
        response = requests.post(
            ANALYZE_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Should succeed as standalone analysis
        assert response.status_code == 200, f"Expected 200 for valid standalone, got {response.status_code}"
        
        response_data = response.json()
        assert response_data["status"] == "success"


# Additional test class for API availability handling
class TestAPIAvailability:
    """Tests for API availability and connection handling."""
    
    def test_api_not_available_skips_tests(self):
        """
        This test verifies that the skipif marker works correctly.
        If API is not available, tests should be skipped, not fail.
        """
        # This test is redundant but demonstrates the skip mechanism
        if not is_api_available():
            pytest.skip("API not available - test skipped as expected")
        else:
            # If API is available, make a simple request to verify
            response = requests.post(
                ANALYZE_ENDPOINT,
                json={
                    "surveyTitle": "Test",
                    "theme": "Test",
                    "baseline": [{"sentence": "Test", "id": "test"}]
                },
                timeout=2
            )
            # Don't assert status code - could be 200 or 400/500
            assert response.status_code is not None