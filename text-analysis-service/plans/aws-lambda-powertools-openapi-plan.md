# AWS Lambda Powertools OpenAPI Documentation Implementation Plan

## Overview
This plan outlines the refactoring of the text-analysis-service to use AWS Lambda Powertools for OpenAPI documentation generation, request validation, and improved API Gateway integration.

## Current State Analysis

### Architecture
- **Lambda Function**: `TextAnalysisFunction` (Python 3.13)
- **API Gateway**: Single POST `/analyze` endpoint
- **Handler**: Manual request parsing in `src/app/handler.py`
- **Validation**: Pydantic models in `src/app/api/schemas.py`
- **Dependencies**: No AWS Lambda Powertools currently installed

### Key Findings
1. Single endpoint handles both standalone and comparative analysis via request body detection
2. Pydantic models are well-defined but lack OpenAPI field extensions
3. Manual error handling and response formatting
4. No OpenAPI documentation generation
5. No API Gateway validation integration

## Implementation Goals

1. **Add AWS Lambda Powertools** with OpenAPI support
2. **Refactor handler** to use `APIGatewayRestResolver`
3. **Generate comprehensive OpenAPI documentation** with extensions
4. **Maintain backward compatibility** with existing API
5. **Improve validation** through API Gateway integration

## Detailed Implementation Steps

### Phase 1: Dependency Management
1. **Add AWS Lambda Powertools to requirements**
   - Update `requirements.txt` and `functions/text_analysis/requirements.txt`
   - Add: `aws-lambda-powertools[parser]>=2.0.0`
   - Ensure compatibility with existing dependencies

2. **Update SAM template** (if needed)
   - No changes required to `template.yaml` as handler path remains the same
   - Ensure Lambda layer not needed (direct pip install)

### Phase 2: Handler Refactoring

#### 2.1 Create New Powertools Handler
- Create `src/app/handler_powertools.py` with:
  - `APIGatewayRestResolver` instance
  - OpenAPI configuration with extensions
  - Route definitions for `/analyze` endpoint

#### 2.2 Configure OpenAPI Extensions
```python
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, OpenAPIConfig
from aws_lambda_powertools.event_handler.openapi.models import Tag, TagGroup

config = OpenAPIConfig(
    title="Text Analysis Service",
    version="1.0.0",
    description="Serverless text analysis microservice for clustering and sentiment analysis",
    tags=[
        Tag(name="analysis", description="Text analysis operations"),
        Tag(name="health", description="Service health checks")
    ],
    tag_groups=[
        TagGroup(
            name="Core Operations",
            tags=["analysis"],
            description="Primary text analysis functionality"
        )
    ],
    external_docs={
        "description": "Project Documentation",
        "url": "https://github.com/your-org/text-analysis-service"
    },
    # x-logo extension
    extensions={
        "x-logo": {
            "url": "https://example.com/logo.png",
            "altText": "Text Analysis Service Logo",
            "backgroundColor": "#FFFFFF",
            "href": "https://example.com"
        }
    }
)
```

#### 2.3 Define Routes
- Single POST `/analyze` route with:
  - Request validation using Pydantic models
  - Response models for both standalone and comparative analysis
  - Error handling with proper HTTP status codes
  - OpenAPI operation documentation

#### 2.4 Integrate with Existing Logic
- Reuse `PipelineOrchestrator` from existing code
- Adapt request/response formatting to match current API contract
- Maintain same error response structure

### Phase 3: Pydantic Model Enhancements

#### 3.1 Add OpenAPI Field Extensions
Update all Pydantic models in `src/app/api/schemas.py` with:
- Comprehensive `Field` descriptions
- Examples for each field
- Validation error messages
- OpenAPI schema extensions

Example enhancement:
```python
class Sentence(BaseModel):
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The text content of the sentence",
        examples=["The user interface is very intuitive"],
        json_schema_extra={
            "x-order": 1,
            "x-examples": {
                "simple": "The user interface is very intuitive",
                "complex": "Despite some initial confusion, the overall user experience improved significantly after the tutorial"
            }
        }
    )
```

#### 3.2 Create Response Models for OpenAPI
- Define explicit response models for successful operations
- Define error response models with proper HTTP status codes
- Add response examples for documentation

### Phase 4: OpenAPI Documentation Endpoints

#### 4.1 Add OpenAPI Schema Endpoint
- GET `/openapi.json` - Returns OpenAPI specification
- GET `/docs` - Optional Swagger UI endpoint (if needed)

#### 4.2 Add Health Check Endpoint
- GET `/health` - Simple health check for load balancers and monitoring

### Phase 5: Testing and Validation

#### 5.1 Unit Tests
- Test new Powertools handler
- Test OpenAPI schema generation
- Test request validation
- Ensure backward compatibility

#### 5.2 Integration Tests
- Test with API Gateway events
- Validate OpenAPI documentation
- Test error scenarios

#### 5.3 Documentation Validation
- Verify OpenAPI spec is valid
- Test with Swagger UI or Redoc
- Validate extensions (`x-tagGroups`, `x-logo`)

### Phase 6: Deployment and Migration

#### 6.1 Gradual Deployment Strategy
1. Deploy new handler alongside existing one
2. Test with canary deployment
3. Update API Gateway to use new handler
4. Monitor for errors

#### 6.2 Rollback Plan
- Maintain old handler as backup
- Quick rollback to previous version if issues

## OpenAPI Extensions Implementation

### x-tagGroups Configuration
```python
tag_groups=[
    TagGroup(
        name="Text Analysis",
        tags=["analysis"],
        description="Core text analysis operations including clustering and sentiment analysis"
    ),
    TagGroup(
        name="Service Management", 
        tags=["health"],
        description="Service health and monitoring endpoints"
    )
]
```

### x-logo Configuration
```python
extensions={
    "x-logo": {
        "url": "https://raw.githubusercontent.com/your-org/logo/main/text-analysis-logo.png",
        "altText": "Text Analysis Service",
        "backgroundColor": "#F5F5F5",
        "href": "https://your-org.github.io/text-analysis-service"
    }
}
```

### Validation Extensions
- Add `x-validation` patterns for complex validation rules
- Add `x-examples` for comprehensive documentation
- Add `x-order` for field ordering in documentation

## File Structure Changes

```
src/app/
├── handler.py                    # Existing handler (keep for rollback)
├── handler_powertools.py         # New Powertools handler
├── api/
│   ├── schemas.py               # Enhanced Pydantic models
│   └── openapi.py               # OpenAPI configuration
└── utils/
    └── openapi_utils.py         # OpenAPI helper functions
```

## Dependencies to Add

```txt
# requirements.txt and functions/text_analysis/requirements.txt
aws-lambda-powertools[parser]>=2.0.0
```

## Testing Strategy

### Test Categories
1. **Unit Tests**: Handler logic, validation, OpenAPI generation
2. **Integration Tests**: API Gateway events, end-to-end flows
3. **Documentation Tests**: OpenAPI spec validity, example correctness
4. **Backward Compatibility**: Ensure existing clients continue to work

### Test Files to Create/Update
- `tests/unit/test_powertools_handler.py`
- `tests/unit/test_openapi_schema.py`
- `tests/integration/test_powertools_integration.py`
- Update existing handler tests to ensure compatibility

## Success Criteria

1. ✅ OpenAPI documentation available at `/openapi.json`
2. ✅ All existing API functionality preserved
3. ✅ `x-tagGroups` and `x-logo` extensions present in OpenAPI spec
4. ✅ Request validation through API Gateway
5. ✅ Comprehensive field documentation in schemas
6. ✅ Health check endpoint available
7. ✅ All tests passing
8. ✅ No breaking changes to existing API contract

## Risks and Mitigations

### Risk 1: Breaking Changes
- **Mitigation**: Maintain old handler, deploy side-by-side, thorough testing

### Risk 2: Dependency Conflicts
- **Mitigation**: Test in isolated environment, pin dependency versions

### Risk 3: Performance Impact
- **Mitigation**: Benchmark before/after, optimize cold starts

### Risk 4: Documentation Accuracy
- **Mitigation**: Validate OpenAPI spec with multiple tools, test examples

## Timeline and Phasing

### Week 1: Foundation
- Add dependencies and basic Powertools setup
- Create enhanced Pydantic models
- Write initial tests

### Week 2: Core Implementation
- Implement Powertools handler
- Configure OpenAPI extensions
- Add health check endpoint

### Week 3: Testing and Validation
- Comprehensive testing
- Documentation validation
- Performance benchmarking

### Week 4: Deployment
- Gradual deployment
- Monitoring and observability
- Final validation

## Next Steps

1. Review and approve this plan
2. Switch to Code mode for implementation
3. Begin with Phase 1 (Dependency Management)
4. Iterate through phases with regular testing
5. Final deployment and validation

## References

- [AWS Lambda Powertools Documentation](https://docs.powertools.aws.dev/lambda/python/latest/)
- [OpenAPI Specification](https://spec.openapis.org/oas/v3.1.0)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Existing Project Documentation](README.md)