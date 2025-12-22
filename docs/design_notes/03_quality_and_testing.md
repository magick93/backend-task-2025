# Quality & Testing

## Overview
Our testing strategy emphasizes a robust pyramid of Unit and Integration tests to ensure reliability in a serverless, nondeterministic ML environment.

## Testing Pyramid

### Unit Testing (`tests/unit/`)
*   **Focus**: Individual components (`orchestrator`, `clustering`, `embedding`, `sentiment`).
*   **Mocking**: Extensive use of `unittest.mock`. Since components are injected via Dependency Injection, we can easily mock complex ML services to test orchestration logic without loading heavy models.
*   **Coverage**: Validates edge cases, empty inputs, and exception handling.

### Integration Testing (`tests/integration/`)
*   **Focus**: End-to-End flow.
*   **Tooling**: `sam local start-lambda`.
*   **Strategy**: We verify the actual API contract by spinning up a local replica of the Lambda environment. This catches configuration issues in `template.yaml` and handler logic that unit tests might miss.
*   **Scenarios**:
    *   Valid standalone requests.
    *   Comparison requests.
    *   Malformed payloads.

## ML Testing Challenges
*   **Determinism**: ML outputs can sometimes be non-deterministic (though `all-MiniLM-L6-v2` is generally stable).
*   **Strategy**: Tests focus on structural validity and broad correctness (e.g., "did we get clusters?") rather than asserting exact floating-point vector values, unless specific seeds are set.

## Project Structure & Quality
*   **Dependency Injection**: The `PipelineOrchestrator` receives its dependencies at runtime. This is the cornerstone of our testability, allowing us to swap out the real `EmbeddingService` for a lightweight mock during unit tests.
*   **Type Safety**: **Pydantic** models (`src/app/api/schemas.py`) are used for all API inputs and outputs. This ensures a strict contract and automatic validation of incoming requests.
*   **Clean Architecture**: Separation of concerns is enforced via `interfaces.py`, ensuring that implementation details (like the specific clustering library) do not leak into the orchestration logic.
