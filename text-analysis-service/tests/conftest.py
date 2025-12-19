"""
Shared pytest configuration and fixtures for AWS SAM testing.

This conftest.py file provides:
1. AWS environment setup to prevent tests from hitting real AWS accounts
2. Shared fixtures for testing Lambda functions
3. Mock AWS services using moto
4. Test data loading from fixtures directory
"""

import os
import json
import pytest
from unittest.mock import patch
from typing import Dict, List, Any

# Set AWS environment variables before any tests run
def pytest_configure(config):
    """Configure pytest environment before test collection."""
    # Set fake AWS credentials to prevent accidental access to real AWS accounts
    os.environ["POWERTOOLS_TRACE_DISABLED"] = "true"
    os.environ["POWERTOOLS_SERVICE_NAME"] = "text-analysis-service"
    
    # Provide dummy credentials so botocore doesn't look for a profile
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_REGION"] = "us-east-1"
    
    # Disable AWS CLI profile to avoid using real credentials
    # Delete AWS_PROFILE to prevent boto3 from trying to load a profile
    os.environ.pop("AWS_PROFILE", None)
    
    # Set test-specific environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope='session')
def mock_aws_services():
    """
    Fixture that sets up moto mocks for AWS services.
    
    This fixture can be used to mock AWS services like S3, DynamoDB, etc.
    Currently, the SAM template doesn't use these services, but this fixture
    provides a foundation for future AWS service mocking.
    
    Returns a context manager that can be used to mock AWS services.
    """
    # Import moto here to avoid import issues if moto is not installed
    try:
        from moto import mock_aws
        return mock_aws()
    except ImportError:
        # If moto is not available, return a dummy context manager
        class DummyMock:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyMock()


@pytest.fixture(scope='session')
def test_sentences():
    """
    Fixture that provides sample sentence data for testing.
    
    Returns a list of sentence dictionaries with 'sentence' and 'id' fields.
    """
    return [
        {"sentence": "The delivery was late by 30 minutes", "id": "feedback-001"},
        {"sentence": "Food arrived cold and soggy", "id": "feedback-002"},
        {"sentence": "Driver was friendly and professional", "id": "feedback-003"},
        {"sentence": "App interface is confusing to navigate", "id": "feedback-004"},
        {"sentence": "Prices are too high compared to competitors", "id": "feedback-005"},
        {"sentence": "Great selection of restaurants in my area", "id": "feedback-006"},
        {"sentence": "Delivery was late and food was cold", "id": "feedback-007"},
        {"sentence": "Customer service took too long to respond", "id": "feedback-008"},
        {"sentence": "Love the new tracking feature", "id": "feedback-009"},
        {"sentence": "Too many notifications are annoying", "id": "feedback-010"},
    ]


@pytest.fixture(scope='session')
def lambda_event_standalone():
    """
    Fixture that provides a sample Lambda event for standalone analysis.
    
    This event mimics the API Gateway event that triggers the standalone
    text analysis Lambda function.
    
    Returns a dictionary representing a Lambda event.
    """
    return {
        "httpMethod": "POST",
        "path": "/analyze/standalone",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token"
        },
        "queryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        "requestContext": {
            "accountId": "123456789012",
            "apiId": "test-api-id",
            "protocol": "HTTP/1.1",
            "httpMethod": "POST",
            "path": "/analyze/standalone",
            "stage": "test",
            "requestId": "test-request-id",
            "requestTime": "18/Dec/2025:02:15:42 +0000",
            "requestTimeEpoch": 1734488142
        },
        "body": json.dumps({
            "surveyTitle": "Food Delivery App Feedback",
            "theme": "customer experience",
            "baseline": [
                {"sentence": "The delivery was late by 30 minutes", "id": "feedback-001"},
                {"sentence": "Food arrived cold and soggy", "id": "feedback-002"},
                {"sentence": "Driver was friendly and professional", "id": "feedback-003"},
                {"sentence": "App interface is confusing to navigate", "id": "feedback-004"},
                {"sentence": "Prices are too high compared to competitors", "id": "feedback-005"}
            ]
        }),
        "isBase64Encoded": False
    }


@pytest.fixture(scope='session')
def lambda_event_comparison():
    """
    Fixture that provides a sample Lambda event for comparison analysis.
    
    This event mimics the API Gateway event that triggers the comparison
    text analysis Lambda function.
    
    Returns a dictionary representing a Lambda event.
    """
    return {
        "httpMethod": "POST",
        "path": "/analyze/comparison",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token"
        },
        "queryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        "requestContext": {
            "accountId": "123456789012",
            "apiId": "test-api-id",
            "protocol": "HTTP/1.1",
            "httpMethod": "POST",
            "path": "/analyze/comparison",
            "stage": "test",
            "requestId": "test-request-id",
            "requestTime": "18/Dec/2025:02:15:42 +0000",
            "requestTimeEpoch": 1734488142
        },
        "body": json.dumps({
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
        }),
        "isBase64Encoded": False
    }


@pytest.fixture(scope='session')
def standalone_fixture_data():
    """
    Fixture that loads the standalone analysis fixture data from JSON file.
    
    Returns the parsed JSON content from tests/fixtures/input_example.json.
    """
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'input_example.json')
    with open(fixture_path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope='session')
def comparison_fixture_data():
    """
    Fixture that loads the comparison analysis fixture data from JSON file.
    
    Returns the parsed JSON content from tests/fixtures/input_comparison_example.json.
    """
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'input_comparison_example.json')
    with open(fixture_path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope='function')
def mock_env_vars():
    """
    Fixture that mocks environment variables for testing.
    
    This fixture can be used to temporarily set environment variables
    for a specific test. It automatically restores the original values
    after the test completes.
    
    Returns a function that can be used to set environment variables.
    """
    original_env = {}
    
    def set_env(**kwargs):
        for key, value in kwargs.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
    
    yield set_env
    
    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope='function')
def mock_time():
    """
    Fixture that mocks time-related functions for deterministic testing.
    
    This fixture patches time.time() and time.sleep() to provide
    deterministic behavior in tests.
    """
    import time
    with patch('time.time') as mock_time_func:
        with patch('time.sleep') as mock_sleep:
            mock_time_func.return_value = 1734488142.0  # Fixed timestamp
            yield {
                'time': mock_time_func,
                'sleep': mock_sleep
            }


@pytest.fixture(scope='session')
def test_context():
    """
    Fixture that provides a mock Lambda context for testing.
    
    Returns a mock Lambda context object with required attributes.
    """
    class MockContext:
        function_name = 'text-analysis-function'
        function_version = '$LATEST'
        invoked_function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:text-analysis-function'
        memory_limit_in_mb = '128'
        aws_request_id = 'test-request-id'
        log_group_name = '/aws/lambda/text-analysis-function'
        log_stream_name = '2025/12/18/[$LATEST]test-log-stream'
        identity = None
        client_context = None
        
        def get_remaining_time_in_millis(self):
            return 30000  # 30 seconds remaining
    
    return MockContext()


# Clean up any AWS environment variables after all tests
def pytest_unconfigure(config):
    """Clean up test environment after all tests have run."""
    # We could clean up environment variables here, but it's not strictly necessary
    # since the test environment is isolated
    pass