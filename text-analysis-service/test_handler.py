#!/usr/bin/env python3
"""
Test the Lambda handler with the new input/output format.
"""
import json
import sys
sys.path.insert(0, 'src')

from app.handler import lambda_handler

# Test event matching required input format
event = {
    "body": json.dumps({
        "surveyTitle": "Robinhood App Store",
        "theme": "account",
        "baseline": [
            {
                "sentence": "The app is easy to use and intuitive.",
                "id": "s1"
            },
            {
                "sentence": "I had trouble withdrawing my funds.",
                "id": "s2"
            },
            {
                "sentence": "Customer support was very helpful.",
                "id": "s3"
            }
        ]
    }),
    "httpMethod": "POST",
    "resource": "/analyze",
    "requestContext": {
        "requestId": "test-request-id"
    }
}

# Mock context
class MockContext:
    aws_request_id = "test-aws-request-id"
    function_name = "test-function"
    
    def get_remaining_time_in_millis(self):
        return 30000

context = MockContext()

print("Testing Lambda handler with new input format...")
try:
    response = lambda_handler(event, context)
    print("Response status code:", response.get('statusCode'))
    print("Response body:")
    body = json.loads(response.get('body', '{}'))
    print(json.dumps(body, indent=2))
    
    # Validate output structure
    if 'clusters' in body:
        print(f"Success! Found {len(body['clusters'])} clusters.")
        for cluster in body['clusters']:
            print(f"  - {cluster['title']} ({cluster['sentiment']})")
            print(f"    Sentences: {cluster['sentences']}")
    else:
        print("Warning: 'clusters' not found in output.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)