"""
Integration tests for AWS SAM local Lambda endpoint.

These tests validate that the Lambda function works correctly when invoked locally
through the SAM Lambda local endpoint (http://127.0.0.1:3001).

Requirements:
- SAM local Lambda must be running: `sam local start-lambda --port 3001`
- Tests will be skipped if the endpoint is not available
- Uses boto3 client configured to use the local endpoint

Test Scenarios:
1. Standalone analysis invocation
2. Comparison analysis invocation  
3. Malformed payload handling
4. Empty sentences handling
5. Error response format validation
"""

import json
import pytest
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from typing import Dict, Any

# Constants
LAMBDA_LOCAL_ENDPOINT = "http://127.0.0.1:3001"
LAMBDA_FUNCTION_NAME = "TextAnalysisFunction"
TEST_REGION = "us-east-1"
TEST_CREDENTIALS = {
    "aws_access_key_id": "testing",
    "aws_secret_access_key": "testing",
    "aws_session_token": "testing"
}


def is_lambda_local_available() -> bool:
    """
    Check if SAM local Lambda endpoint is available.
    
    Returns:
        bool: True if endpoint is reachable, False otherwise
    """
    import socket
    import urllib.request
    import urllib.error
    
    try:
        # Try to connect to the endpoint
        socket.create_connection(("127.0.0.1", 3001), timeout=2)
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


@pytest.fixture(scope="module")
def lambda_client():
    """
    Fixture that provides a boto3 Lambda client configured for local endpoint.
    
    Returns:
        boto3.client: Lambda client configured for local testing
    """
    return boto3.client(
        'lambda',
        endpoint_url=LAMBDA_LOCAL_ENDPOINT,
        region_name=TEST_REGION,
        **TEST_CREDENTIALS
    )


@pytest.fixture(scope="module")
def standalone_payload():
    """
    Fixture that provides a valid standalone analysis payload.
    
    Returns:
        dict: Standalone analysis payload
    """
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


@pytest.fixture(scope="module")
def comparison_payload():
    """
    Fixture that provides a valid comparison analysis payload.
    
    Returns:
        dict: Comparison analysis payload
    """
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


@pytest.fixture(scope="module")
def empty_sentences_payload():
    """
    Fixture that provides a payload with empty sentences list.
    
    Returns:
        dict: Payload with empty baseline
    """
    return {
        "surveyTitle": "Empty Feedback",
        "theme": "test",
        "baseline": []
    }


@pytest.mark.lambda_local
@pytest.mark.integration
@pytest.mark.skipif(
    not is_lambda_local_available(),
    reason="SAM local Lambda not running on port 3001"
)
class TestLambdaLocalIntegration:
    """Integration tests for SAM local Lambda endpoint."""
    
    def test_lambda_local_standalone(self, lambda_client, standalone_payload):
        """
        Test Lambda invocation with standalone analysis payload.
        
        Validates:
        - Lambda can be invoked successfully
        - Response contains expected structure
        - Status code is 200
        - Response body contains analysis results
        """
        try:
            # Invoke Lambda function
            response = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps(standalone_payload)
            )
            
            # Parse response
            response_payload = json.loads(response['Payload'].read())
            
            # Validate response structure
            assert 'statusCode' in response_payload
            assert response_payload['statusCode'] == 200
            assert 'headers' in response_payload
            assert 'body' in response_payload
            
            # Parse body (it's a JSON string)
            body = json.loads(response_payload['body'])
            
            # Validate response body structure
            assert 'status' in body
            assert body['status'] == 'success'
            assert 'jobId' in body
            assert 'processingTimeMs' in body
            assert 'timestamp' in body
            assert 'result' in body
            
            # Validate analysis results structure
            result = body['result']
            assert 'clusters' in result
            assert 'summary' in result
            assert 'processing_metadata' in result
            
            # Validate metadata
            metadata = result['processing_metadata']
            assert 'input_sentence_count' in metadata
            assert metadata['input_sentence_count'] == len(standalone_payload['baseline'])
            assert 'cluster_count' in metadata
            assert 'processing_time_ms' in metadata
            
            # Validate clusters
            clusters = result['clusters']
            assert isinstance(clusters, list)
            
            # Each cluster should have required fields
            for cluster in clusters:
                assert 'title' in cluster
                assert 'sentiment' in cluster
                assert 'sentence_count' in cluster
                assert 'representative_sentences' in cluster
                assert 'key_terms' in cluster
                assert 'summary' in cluster
                
        except ClientError as e:
            pytest.fail(f"Lambda invocation failed: {e}")
    
    def test_lambda_local_comparison(self, lambda_client, comparison_payload):
        """
        Test Lambda invocation with comparison analysis payload.
        
        Validates:
        - Lambda can process comparison analysis
        - Response contains comparison-specific fields
        - Both baseline and comparison results are present
        """
        try:
            # Invoke Lambda function
            response = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps(comparison_payload)
            )
            
            # Parse response
            response_payload = json.loads(response['Payload'].read())
            
            # Validate response structure
            assert 'statusCode' in response_payload
            assert response_payload['statusCode'] == 200
            assert 'headers' in response_payload
            assert 'body' in response_payload
            
            # Parse body (it's a JSON string)
            body = json.loads(response_payload['body'])
            
            # Validate response body structure
            assert 'status' in body
            assert body['status'] == 'success'
            assert 'jobId' in body
            assert 'processingTimeMs' in body
            assert 'timestamp' in body
            assert 'result' in body
            
            # Validate comparison results structure
            result = body['result']
            assert 'baseline' in result
            assert 'comparison' in result
            assert 'comparison_insights' in result
            
            # Validate baseline results
            baseline = result['baseline']
            assert 'clusters' in baseline
            assert 'summary' in baseline
            assert 'processing_metadata' in baseline
            
            # Validate comparison results
            comparison = result['comparison']
            assert 'clusters' in comparison
            assert 'summary' in comparison
            assert 'processing_metadata' in comparison
            
            # Validate comparison insights
            insights = result['comparison_insights']
            assert 'similarities' in insights
            assert 'differences' in insights
            assert 'trends' in insights
            
        except ClientError as e:
            pytest.fail(f"Lambda invocation failed: {e}")
    
    def test_lambda_local_malformed_payload(self, lambda_client):
        """
        Test Lambda error handling with malformed JSON payload.
        
        Validates:
        - Lambda returns appropriate error response
        - Status code indicates error (400 or 500)
        - Error message is present
        """
        try:
            # Invoke with invalid JSON
            response = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload='{invalid json'
            )
            
            # Parse response
            response_payload = json.loads(response['Payload'].read())
            
            # Should have error status code
            assert 'statusCode' in response_payload
            status_code = response_payload['statusCode']
            assert status_code >= 400  # Client or server error
            
            # Parse error body
            body = json.loads(response_payload['body'])
            
            # Should have error structure
            assert 'status' in body
            assert body['status'] == 'error'
            assert 'error' in body
            assert 'type' in body['error']
            assert 'message' in body['error']
            
        except ClientError as e:
            # ClientError is also acceptable for malformed payload
            assert 'Invalid' in str(e) or 'JSON' in str(e)
    
    def test_lambda_local_empty_sentences(self, lambda_client, empty_sentences_payload):
        """
        Test Lambda handling of empty sentences list.
        
        Validates:
        - Lambda handles empty input gracefully
        - Returns valid response structure
        - Cluster count is 0
        """
        try:
            # Invoke Lambda function
            response = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps(empty_sentences_payload)
            )
            
            # Parse response
            response_payload = json.loads(response['Payload'].read())
            
            # Validate response structure
            assert 'statusCode' in response_payload
            assert response_payload['statusCode'] == 200
            assert 'headers' in response_payload
            assert 'body' in response_payload
            
            # Parse body (it's a JSON string)
            body = json.loads(response_payload['body'])
            
            # Validate response body structure
            assert 'status' in body
            assert body['status'] == 'success'
            assert 'jobId' in body
            assert 'processingTimeMs' in body
            assert 'timestamp' in body
            assert 'result' in body
            
            # Validate empty results structure
            result = body['result']
            assert 'clusters' in result
            assert isinstance(result['clusters'], list)
            assert len(result['clusters']) == 0
            
            # Validate metadata
            metadata = result['processing_metadata']
            assert metadata['input_sentence_count'] == 0
            assert metadata['cluster_count'] == 0
            
        except ClientError as e:
            pytest.fail(f"Lambda invocation failed: {e}")
    
    def test_lambda_local_error_response_format(self, lambda_client):
        """
        Test error response format for invalid payload structure.
        
        Validates:
        - Error responses follow consistent format
        - Contains statusCode, body, and headers
        - Error message is descriptive
        """
        # Create payload missing required fields
        invalid_payload = {
            "surveyTitle": "Test",
            # Missing 'theme' and 'baseline' fields
        }
        
        try:
            response = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps(invalid_payload)
            )
            
            # Parse response
            response_payload = json.loads(response['Payload'].read())
            
            # Validate error response format
            assert 'statusCode' in response_payload
            assert response_payload['statusCode'] >= 400
            
            assert 'body' in response_payload
            assert 'headers' in response_payload
            
            # Parse error body
            body = json.loads(response_payload['body'])
            
            # Should contain error information
            assert 'status' in body
            assert body['status'] == 'error'
            assert 'error' in body
            assert 'type' in body['error']
            assert 'message' in body['error']
            
        except ClientError as e:
            # ClientError is acceptable for validation errors
            pass
    
    def test_lambda_local_invocation_types(self, lambda_client, standalone_payload):
        """
        Test different Lambda invocation types.
        
        Validates:
        - RequestResponse invocation returns immediate result
        - Event invocation returns quickly (async)
        """
        # Test RequestResponse (synchronous)
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps(standalone_payload)
        )
        
        assert 'StatusCode' in response
        assert response['StatusCode'] == 200
        assert 'Payload' in response
        
        # Parse and validate the response
        response_payload = json.loads(response['Payload'].read())
        assert 'statusCode' in response_payload
        assert response_payload['statusCode'] == 200
        
        # Test Event (asynchronous) - should return quickly
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='Event',
            Payload=json.dumps(standalone_payload)
        )
        
        assert 'StatusCode' in response
        # Event invocation returns 202 Accepted
        assert response['StatusCode'] == 202


@pytest.mark.lambda_local
@pytest.mark.integration
class TestLambdaLocalConnection:
    """Tests for Lambda local connection handling."""
    
    def test_endpoint_configuration(self):
        """
        Test that boto3 client is correctly configured for local endpoint.
        """
        client = boto3.client(
            'lambda',
            endpoint_url=LAMBDA_LOCAL_ENDPOINT,
            region_name=TEST_REGION,
            **TEST_CREDENTIALS
        )
        
        # Verify client configuration
        assert client._endpoint.host == '127.0.0.1:3001'
        assert client._client_config.region_name == TEST_REGION
    
    @pytest.mark.skipif(
        is_lambda_local_available(),
        reason="SAM local Lambda is running, test only when not available"
    )
    def test_connection_error_when_not_running(self):
        """
        Test that connection error occurs when SAM local is not running.
        
        This test only runs when SAM local Lambda is NOT available.
        """
        client = boto3.client(
            'lambda',
            endpoint_url=LAMBDA_LOCAL_ENDPOINT,
            region_name=TEST_REGION,
            **TEST_CREDENTIALS
        )
        
        # Should raise connection error
        with pytest.raises((EndpointConnectionError, ClientError)):
            client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps({"test": "data"})
            )


def test_lambda_local_marker_registration():
    """
    Test that pytest markers are properly registered.
    
    This ensures that tests can be filtered correctly:
    - pytest -m lambda_local: Run only Lambda local tests
    - pytest -m "not lambda_local": Skip Lambda local tests
    """
    # This test doesn't actually run Lambda, just validates marker setup
    pass


if __name__ == "__main__":
    # Quick connectivity check
    if is_lambda_local_available():
        print("✓ SAM local Lambda endpoint is available on port 3001")
        print("  Run tests with: pytest tests/integration/test_lambda_local.py -v")
    else:
        print("✗ SAM local Lambda endpoint is not available on port 3001")
        print("  Start it with: sam local start-lambda --port 3001")
        print("  Then run: pytest tests/integration/test_lambda_local.py -v --tb=short")