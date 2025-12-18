# AWS SAM Local Testing Automation

This document describes the automated testing workflow for the AWS SAM application using the `run_local_tests.sh` script.

## Overview

The `run_local_tests.sh` script automates the entire testing workflow for AWS SAM applications by:

1. **Starting SAM local services** - API Gateway and Lambda functions
2. **Running tests** - Unit, API, Lambda local, and integration tests
3. **Cleaning up** - Properly stopping services and cleaning up processes

## Prerequisites

- **AWS SAM CLI** - Install via `pip install aws-sam-cli`
- **Python 3.8+** - With dependencies installed (`pip install -r requirements.txt`)
- **Bash shell** - Linux/macOS compatible (Windows users can use WSL or Git Bash)

## Installation

Make the script executable:

```bash
chmod +x run_local_tests.sh
```

## Usage

### Basic Usage

```bash
# Run all tests (default)
./run_local_tests.sh

# Run all tests with verbose output
./run_local_tests.sh --verbose --all

# Run only unit tests (fast, no SAM services required)
./run_local_tests.sh --unit

# Run only API tests (requires SAM local API)
./run_local_tests.sh --api

# Run only Lambda local tests (requires SAM local Lambda)
./run_local_tests.sh --lambda-local

# Run only integration tests
./run_local_tests.sh --integration
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--unit` | Run only unit tests (fast, no SAM services required) |
| `--api` | Run only API tests (requires SAM local API on port 3000) |
| `--lambda-local` | Run only Lambda local tests (requires SAM local Lambda on port 3001) |
| `--integration` | Run only integration tests |
| `--all` | Run all tests (default) |
| `--verbose` | Enable verbose output |
| `--no-cleanup` | Don't clean up SAM processes after tests (useful for debugging) |
| `--help`, `-h` | Show help message |

### Test Categories

Based on the `pytest.ini` configuration, tests are categorized with markers:

- **`unit`** - Fast tests with no external dependencies
- **`api`** - Tests that require SAM local API running on port 3000
- **`lambda_local`** - Tests that require SAM local Lambda running on port 3001
- **`integration`** - Integration tests requiring external services or full pipeline
- **`slow`** - Slow tests (skipped by default)
- **`smoke`** - Smoke tests for basic functionality
- **`regression`** - Regression tests
- **`performance`** - Performance tests

## How It Works

### 0. Build Step

Before starting services, the script runs `sam build` to ensure the Lambda function code is compiled and dependencies are resolved. This ensures the latest changes are reflected in the local environment.

### 1. Service Startup

The script starts two SAM local services:

- **SAM Local API** (`sam local start-api`) on port 3000
- **SAM Local Lambda** (`sam local start-lambda`) on port 3001

Services are started in the background with:
- Warm containers enabled for faster execution
- Debug logging captured to log files
- Environment variables loaded from `.env` file (must be in JSON format)

### 2. Health Checks

The script performs health checks to ensure services are ready:

- **Port checking** - Verifies ports 3000 and 3001 are listening
- **Timeout handling** - 30-second timeout with 2-second intervals
- **Graceful failure** - If services fail to start, the script exits cleanly

### 3. Test Execution

Tests are executed with `pytest` using appropriate markers:

- **JUnit XML output** - Test results saved as XML for CI/CD integration
- **Short tracebacks** - Clean output for easier debugging
- **Marker filtering** - Only relevant tests are executed based on selection

### 4. Cleanup

The script ensures proper cleanup:

- **Signal trapping** - Handles Ctrl+C (SIGINT), termination (SIGTERM), and exit
- **Process termination** - Kills background SAM processes
- **Log preservation** - Service logs are saved to `sam_api.log` and `sam_lambda.log`

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (all tests passed) |
| 1 | General error (script failure) |
| 2 | SAM CLI not found |
| 3 | SAM service failed to start |
| 4 | Health check timeout |
| 5 | Pytest tests failed |

## Log Files

The script creates the following log files:

- `sam_api.log` - Output from SAM local API service
- `sam_lambda.log` - Output from SAM local Lambda service
- `test-results-*.xml` - JUnit XML test results for CI/CD integration

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install aws-sam-cli
      - run: ./run_local_tests.sh --all
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh './run_local_tests.sh --all'
                junit 'test-results-*.xml'
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```
   [WARNING] Port 3000 is already in use
   ```
   **Solution**: Stop any existing SAM processes or change ports in the script.

2. **SAM CLI not found**
   ```
   [ERROR] AWS SAM CLI not found
   ```
   **Solution**: Install SAM CLI: `pip install aws-sam-cli`

3. **Health check timeout**
   ```
   [ERROR] Timeout waiting for SAM local API on port 3000 after 30 seconds
   ```
   **Solution**: Check SAM logs for startup errors, increase timeout in script.

4. **Test failures**
   ```
   [ERROR] Some tests failed
   ```
   **Solution**: Check test output, run specific test categories to isolate issues.

### Debug Mode

For debugging, use the `--verbose` flag and `--no-cleanup`:

```bash
./run_local_tests.sh --verbose --no-cleanup
```

This will:
- Show detailed debug output
- Keep SAM services running after tests
- Preserve all log files

## Manual Testing (Alternative)

If you prefer to run tests manually:

```bash
# Build the SAM application first
sam build

# Start SAM local API
sam local start-api --port 3000 --template template.yaml --env-vars .env

# In another terminal, start SAM local Lambda
sam local start-lambda --port 3001 --template template.yaml --env-vars .env

# Run tests
python -m pytest -m "api" tests/api/
python -m pytest -m "lambda_local" tests/integration/test_lambda_local.py
```

**Note**: The `.env` file must be in JSON format (as required by SAM CLI). Example:
```json
{
  "Parameters": {
    "EmbeddingModel": "sentence-transformers/all-MiniLM-L6-v2",
    "ClusterThreshold": 0.7,
    "MinClusterSize": 2
  }
}
```

## Best Practices

1. **Run unit tests frequently** - They're fast and don't require SAM services
2. **Use `--verbose` for CI failures** - Helps identify issues
3. **Check log files** - When tests fail, examine `sam_api.log` and `sam_lambda.log`
4. **Update timeout values** - If tests are slow, increase `HEALTH_CHECK_TIMEOUT`
5. **Add custom markers** - Extend the script for project-specific test categories

## Extending the Script

To add new test categories:

1. Update `pytest.ini` with new markers
2. Add new test functions in the script
3. Update argument parsing and test selection logic

Example for adding a `performance` test category:

```bash
# In the script, add:
run_performance_tests() {
    python -m pytest -m "performance" --tb=short tests/performance/
}

# Update argument parsing:
--performance)
    RUN_PERFORMANCE=true
    RUN_ALL=false
    shift
    ;;
```

## License

This script is part of the AWS SAM testing infrastructure and follows the same licensing as the main project.