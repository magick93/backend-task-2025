# Architecture & Infrastructure

## Overview
This service is designed as a serverless application utilizing **AWS Lambda** and **API Gateway**. The core philosophy emphasizes low operational overhead, cost-effectiveness, and scalability.

## Key Decisions

### Serverless Architecture (AWS Lambda)
*   **Choice**: We utilized AWS Lambda (Python 3.13) with 1024 MB memory.
*   **Reasoning**:
    *   **Cost**: "Pay for what you use" model is ideal for sporadic text analysis workloads.
    *   **Scale**: Automatic scaling handles bursty traffic without manual intervention.
    *   **Trade-offs**: Cold starts are a potential factor, but mitigated by the `all-MiniLM-L6-v2` model's efficiency and in-memory caching.
*   **Timeout**: Set to 900 seconds (15 minutes) to accommodate potentially heavy ML processing tasks.

### API Gateway Pattern
*   **Routing**: Used `APIGatewayRestResolver` from **AWS Lambda Powertools**.
*   **Pattern**: A "Monolithic Lambda" pattern where API Gateway acts as a proxy, forwarding all requests to a single Lambda function which then handles internal routing.
*   **Endpoints**:
    *   `POST /analyze`: Main analysis entry point.
    *   `GET /health`: Operational status check.
    *   `GET /openapi.json`: API documentation.

### Monolithic Lambda Approach
*   **Why Chosen**:
    *   **Simplified State Management**: The ML model (Sentence Transformer) is relatively heavy. Loading it once in a shared execution context is more efficient than cold-starting multiple distinct functions.
    *   **Deployment Ease**: Managing a single SAM resource simplifies the template and deployment pipeline.
    *   **Code Sharing**: Core pipeline logic is centralized, avoiding code duplication across multiple functions.

### Observability
*   **Tooling**: **AWS Lambda Powertools** is the backbone of our observability strategy.
*   **Components**:
    *   **Logging**: Structured JSON logging.
    *   **Tracing**: AWS X-Ray enabled for deep performance insights.
    *   **Metrics**: Custom metrics support.
*   **OpenTelemetry**: Integration configured via environment variables (`OTEL_SERVICE_NAME`) for future-proofing export capabilities.

## Future Considerations
*   **Async Processing**: The current `/analyze` endpoint is synchronous. For very large datasets, moving to an asynchronous pattern (API Gateway -> SQS -> Lambda) would improve reliability and prevent timeouts.
*   **Provisioned Concurrency**: If cold start latencies become an issue, enabling provisioned concurrency on the Lambda function is a viable optimization.
