# Text Analysis Service

A serverless text analysis microservice built with AWS SAM (Serverless Application Model). The service processes text inputs, performs clustering, sentiment analysis, and generates insights.

## Overview

This service provides two main endpoints:
- **Standalone Analysis**: Cluster and analyze a single set of sentences.
- **Comparative Analysis**: Compare two sets of sentences to identify similarities and differences.

The pipeline includes:
- Text preprocessing
- Sentence embedding (using sentence-transformers)
- Clustering (DBSCAN-based)
- Sentiment analysis (keyword-based)
- Insight generation (template-based)
- Comparative analysis

## Getting Started

### Prerequisites

- Python 3.8+
- AWS SAM CLI (`pip install aws-sam-cli`)
- Docker (for local SAM testing)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd text-analysis-service
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   cp local-env-vars.example.json local-env-vars.json
   # Edit local-env-vars.json as needed
   ```

### Running Locally with SAM

1. Start SAM local API and Lambda services:
   ```bash
   sam local start-api --port 3000 --template template.yaml --env-vars local-env-vars.json
   sam local start-lambda --port 3001 --template template.yaml --env-vars local-env-vars.json
   ```

2. Test the endpoints:
   - Health check: `curl http://localhost:3000/health`
   - Standalone analysis: `curl -X POST http://localhost:3000/analyze -H "Content-Type: application/json" -d @data/input_example.json`
   - Comparative analysis: `curl -X POST http://localhost:3000/compare -H "Content-Type: application/json" -d @data/input_comparison_example.json`

## Testing

The project includes a comprehensive testing suite with automated scripts for local development and CI/CD integration.

### Test Categories

Tests are organized into four categories using pytest markers:

- **Unit Tests** (`--unit`): Fast tests with no external dependencies. Test individual components (embedding, clustering, sentiment, insights).
- **API Tests** (`--api`): Tests that require SAM local API running on port 3000. Validate HTTP endpoints and request/response formats.
- **Lambda Local Tests** (`--lambda-local`): Tests that require SAM local Lambda running on port 3001. Test Lambda function invocations directly.
- **Integration Tests** (`--integration`): Integration tests requiring external services or full pipeline execution.

### Quick Start

Run all tests using the automation script:

```bash
./run_local_tests.sh
```

Run specific test categories:

```bash
# Unit tests only (fast)
./run_local_tests.sh --unit

# API tests only (requires SAM local API)
./run_local_tests.sh --api

# Lambda local tests only (requires SAM local Lambda)
./run_local_tests.sh --lambda-local

# Integration tests only
./run_local_tests.sh --integration
```

### Automation Script

The `run_local_tests.sh` script automates the entire testing workflow:

1. **Starts SAM local services** (API Gateway and Lambda) in the background
2. **Performs health checks** to ensure services are ready
3. **Runs selected test categories** with appropriate pytest markers
4. **Generates JUnit XML reports** for CI/CD integration (`test-results-*.xml`)
5. **Cleans up** background processes and logs

For detailed usage, see [TESTING.md](TESTING.md).

### Test Configuration

- **pytest.ini**: Defines markers, test discovery, and default options.
- **conftest.py**: Shared fixtures and configuration for all tests.
- **Makefile**: Provides shortcuts for common test commands.

### Example Commands

```bash
# Run unit tests with coverage
python -m pytest tests/unit/ -v --cov=src --cov-report=html

# Run API tests with SAM local already running
python -m pytest tests/api/ -m api -v

# Run a specific test file
python -m pytest tests/unit/test_sentiment.py -v
```

### CI/CD Integration

The testing suite is designed for seamless CI/CD integration:

- **JUnit XML output**: Test results are saved as `test-results-*.xml` for CI systems.
- **Exit codes**: The script returns appropriate exit codes for success/failure.
- **Log files**: SAM service logs are captured in `sam_api.log` and `sam_lambda.log`.

See [TESTING.md](TESTING.md) for CI/CD examples (GitHub Actions, Jenkins).

## Local Development using AWS SAM

### 1. Install SAM CLI

Follow the official guide: [AWS SAM CLI Installation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)

### 2. Build and Deploy Locally

```bash
# Build the SAM application
sam build

# Start local API
sam local start-api --port 3000

# Start local Lambda
sam local start-lambda --port 3001
```

### 3. Configuration

The service uses a three-tiered configuration approach:

1.  **`local-env-vars.json`**:
    *   Used for **local SAM environment variables** (e.g., overriding `LOG_LEVEL` or function-specific env vars).
    *   Format:
        ```json
        {
          "TextAnalysisFunction": {
            "LOG_LEVEL": "DEBUG"
          }
        }
        ```
    *   `TextAnalysisFunction` is the logical ID of the Lambda function in `template.yaml`.

2.  **`infra/parameters.json`**:
    *   Used for **non-secret deployment configuration** (e.g., service name, OTel endpoint).
    *   These are passed as template parameters.

3.  **Parameter Overrides**:
    *   Used for **secrets** like `HoneycombApiKey`.
    *   These should be passed via command line or CI/CD secrets, not committed to code.

4.  **`.env` file** (for local testing script):
    *   The `run_local_tests.sh` script supports reading secrets from a `.env` file.
    *   Create a `.env` file (gitignored) containing `HONEYCOMB_API_KEY=your_key` and it will be automatically picked up and passed as a parameter override.
    *   **Distinction**: Use `.env` for **script secrets** (passed to SAM CLI), while `local-env-vars.json` is for **Lambda environment variables** (injected into the container).

### 4. Debugging

- Use `--debug` flag with SAM commands for verbose output.
- Check logs in `.aws-sam/logs/`.
- Use the `--no-cleanup` flag with `run_local_tests.sh` to keep services running for manual testing.

## Observability

This service is instrumented with OpenTelemetry to provide distributed tracing and observability. It is configured to export telemetry data to Honeycomb.io.

### Configuration

To enable Honeycomb tracing, you need to provide your Honeycomb API Key.

**Pass Parameter during Deployment/Local Run**:
Pass the parameter key when running SAM commands:
```bash
sam local start-api --parameter-overrides HoneycombApiKey=your-api-key-here OtelServiceName=my-service
```

> **Note**: If the `HoneycombApiKey` is not provided or left as the default "replace-me", the OpenTelemetry export will effectively be disabled (or fail to authenticate), ensuring no invalid data is sent.

## Project Structure

```
text-analysis-service/
├── functions/text-analysis/     # Lambda function code
├── src/                         # Source code (shared with Lambda)
├── tests/                       # Test suites
│   ├── unit/                    # Unit tests
│   ├── api/                     # API tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test data
├── scripts/                     # Utility scripts
├── infra/                       # Infrastructure templates
├── data/                        # Example input data
├── template.yaml                # SAM template
├── pytest.ini                   # Pytest configuration
├── run_local_tests.sh           # Automated testing script
├── Makefile                     # Build/test shortcuts
└── TESTING.md                   # Detailed testing documentation
```

## License

This project is licensed under the terms of the MIT License.