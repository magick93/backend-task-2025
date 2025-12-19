#!/usr/bin/env bash

# =============================================================================
# AWS SAM Local Testing Automation Script
# =============================================================================
#
# This script automates the testing workflow for AWS SAM applications by:
# 1. Starting SAM local API and Lambda services in the background
# 2. Waiting for services to be ready with health checks
# 3. Running pytest tests with appropriate markers
# 4. Cleaning up background processes when tests complete
#
# Usage:
#   ./run_local_tests.sh [OPTIONS]
#
# Options:
#   --unit              Run only unit tests (fast, no SAM services required)
#   --api               Run only API tests (requires SAM local API on port 3000)
#   --lambda-local      Run only Lambda local tests (requires SAM local Lambda on port 3001)
#   --integration       Run only integration tests
#   --all               Run all tests (default)
#   --help              Show this help message
#   --verbose           Enable verbose output
#   --no-cleanup        Don't clean up SAM processes after tests (for debugging)
#
# Examples:
#   ./run_local_tests.sh --unit              # Run unit tests only
#   ./run_local_tests.sh --api               # Run API tests with SAM local API
#   ./run_local_tests.sh --all               # Run all tests (default)
#   ./run_local_tests.sh --verbose --all     # Run all tests with verbose output
#
# Exit Codes:
#   0 - Success (all tests passed)
#   1 - General error (script failure)
#   2 - SAM CLI not found
#   3 - SAM service failed to start
#   4 - Health check timeout
#   5 - Pytest tests failed
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Port configuration
API_PORT=3000
LAMBDA_PORT=3001

# Timeout configuration (in seconds)
HEALTH_CHECK_TIMEOUT=300
HEALTH_CHECK_INTERVAL=2

# Process IDs for cleanup
SAM_API_PID=""
SAM_LAMBDA_PID=""

# Test selection flags
RUN_UNIT=false
RUN_API=false
RUN_LAMBDA_LOCAL=false
RUN_INTEGRATION=false
RUN_ALL=true

# Script options
VERBOSE=false
CLEANUP=true
PARAMETER_OVERRIDES_FLAG=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $*"
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a port is in use
port_in_use() {
    local port=$1
    if command_exists nc; then
        nc -z localhost "$port" >/dev/null 2>&1
    elif command_exists ss; then
        ss -tuln | grep -q ":$port "
    elif command_exists netstat; then
        netstat -tuln | grep -q ":$port "
    else
        # Fallback: try to connect to the port
        timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null
    fi
}

# Wait for a service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local timeout=$HEALTH_CHECK_TIMEOUT
    local interval=$HEALTH_CHECK_INTERVAL
    local elapsed=0
    
    log_info "Waiting for $service_name on port $port..."
    
    while [ $elapsed -lt $timeout ]; do
        if port_in_use "$port"; then
            log_success "$service_name is ready on port $port"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log_debug "Waited $elapsed seconds for $service_name..."
    done
    
    log_error "Timeout waiting for $service_name on port $port after $timeout seconds"
    return 1
}

# Clean up background processes
cleanup() {
    if [ "$CLEANUP" = false ]; then
        log_warning "Skipping cleanup (--no-cleanup flag set)"
        return
    fi
    
    log_info "Cleaning up background processes..."
    
    # Kill SAM API process if running
    if [ -n "$SAM_API_PID" ] && kill -0 "$SAM_API_PID" 2>/dev/null; then
        log_debug "Killing SAM API process (PID: $SAM_API_PID)"
        kill "$SAM_API_PID" 2>/dev/null || true
        wait "$SAM_API_PID" 2>/dev/null || true
    fi
    
    # Kill SAM Lambda process if running
    if [ -n "$SAM_LAMBDA_PID" ] && kill -0 "$SAM_LAMBDA_PID" 2>/dev/null; then
        log_debug "Killing SAM Lambda process (PID: $SAM_LAMBDA_PID)"
        kill "$SAM_LAMBDA_PID" 2>/dev/null || true
        wait "$SAM_LAMBDA_PID" 2>/dev/null || true
    fi
    
    # Clean up any other SAM processes
    pkill -f "sam local start-api" 2>/dev/null || true
    pkill -f "sam local start-lambda" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Setup trap for cleanup on script exit
setup_trap() {
    trap 'cleanup; exit' INT TERM EXIT
    log_debug "Signal traps set up for cleanup"
}

# Configure environment variables and overrides
configure_environment() {
    log_info "Configuring environment..."
    
    local overrides=""
    
    # Source .env file if it exists
    if [ -f .env ]; then
        log_info "Sourcing .env file..."
        set -a
        source .env
        set +a
    fi
    
    # Check for Honeycomb API Key
    if [ -n "${HONEYCOMB_API_KEY:-}" ]; then
        overrides="HoneycombApiKey=$HONEYCOMB_API_KEY"
    elif [ -n "${HoneycombApiKey:-}" ]; then
        overrides="HoneycombApiKey=$HoneycombApiKey"
    fi
    
    if [ -n "$overrides" ]; then
        log_info "Applying parameter overrides: $overrides"
        PARAMETER_OVERRIDES_FLAG="--parameter-overrides $overrides"
    fi
}

# =============================================================================
# SAM Functions
# =============================================================================

# Check if SAM CLI is installed
check_sam_cli() {
    log_info "Checking for AWS SAM CLI..."
    
    if ! command_exists sam; then
        log_error "AWS SAM CLI not found. Please install it:"
        log_error "  pip install aws-sam-cli"
        log_error "  or visit: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
        return 1
    fi
    
    local sam_version
    sam_version=$(sam --version 2>/dev/null || echo "unknown")
    log_success "AWS SAM CLI found: $sam_version"
    return 0
}

# Build SAM application
build_sam() {
    log_info "Building SAM application..."
    
    if ! sam build \
        --template template.yaml \
        --build-dir .aws-sam/build \
        --use-container \
        --cached \
        --parallel; then
        log_error "SAM build failed"
        return 1
    fi
    
    log_success "SAM build completed successfully"
    return 0
}

# Ensure SAM application is built
ensure_sam_built() {
    local built_template=".aws-sam/build/template.yaml"
    
    if [ -f "$built_template" ]; then
        log_debug "SAM application already built"
        return 0
    fi
    
    log_info "SAM application not built, building now..."
    if ! build_sam; then
        return 1
    fi
}

# Start SAM local API
start_sam_api() {
    log_info "Starting SAM local API on port $API_PORT..."
    
    # Ensure SAM is built
    if ! ensure_sam_built; then
        log_error "Failed to build SAM application"
        return 1
    fi
    
    # Check if port is already in use
    if port_in_use "$API_PORT"; then
        log_warning "Port $API_PORT is already in use"
        return 1
    fi
    
    # Start SAM local API in background using built template
    sam local start-api \
        --port "$API_PORT" \
        --template .aws-sam/build/template.yaml \
        --env-vars local-env-vars.json \
        $PARAMETER_OVERRIDES_FLAG \
        --warm-containers EAGER \
        --debug 2>&1 | tee "$PROJECT_ROOT/sam_api.log" &
    
    SAM_API_PID=$!
    log_debug "SAM API started with PID: $SAM_API_PID"
    
    # Wait for API to be ready
    if ! wait_for_service "$API_PORT" "SAM local API"; then
        log_error "Failed to start SAM local API"
        return 1
    fi
    
    log_success "SAM local API is ready on http://localhost:$API_PORT"
    return 0
}

# Start SAM local Lambda
start_sam_lambda() {
    log_info "Starting SAM local Lambda on port $LAMBDA_PORT..."
    
    # Ensure SAM is built
    if ! ensure_sam_built; then
        log_error "Failed to build SAM application"
        return 1
    fi
    
    # Check if port is already in use
    if port_in_use "$LAMBDA_PORT"; then
        log_warning "Port $LAMBDA_PORT is already in use"
        return 1
    fi
    
    # Start SAM local Lambda in background using built template
    sam local start-lambda \
        --port "$LAMBDA_PORT" \
        --template .aws-sam/build/template.yaml \
        --env-vars local-env-vars.json \
        $PARAMETER_OVERRIDES_FLAG \
        --debug 2>&1 | tee "$PROJECT_ROOT/sam_lambda.log" &
    
    SAM_LAMBDA_PID=$!
    log_debug "SAM Lambda started with PID: $SAM_LAMBDA_PID"
    
    # Wait for Lambda to be ready
    if ! wait_for_service "$LAMBDA_PORT" "SAM local Lambda"; then
        log_error "Failed to start SAM local Lambda"
        return 1
    fi
    
    log_success "SAM local Lambda is ready on http://localhost:$LAMBDA_PORT"
    return 0
}

# =============================================================================
# Test Functions
# =============================================================================

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    local pytest_args=()
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    # Try running with unit marker first
    python -m pytest \
        -m "unit" \
        --tb=short \
        --junitxml="$PROJECT_ROOT/test-results-unit.xml" \
        "${pytest_args[@]}" \
        tests/unit/
    
    local exit_code=$?
    
    # Exit code 5 means no tests were collected
    if [ $exit_code -eq 5 ]; then
        log_warning "No tests marked with 'unit', running all tests in tests/unit/ directory"
        if ! python -m pytest \
            --tb=short \
            --junitxml="$PROJECT_ROOT/test-results-unit.xml" \
            "${pytest_args[@]}" \
            tests/unit/; then
            log_error "Unit tests failed"
            return 1
        fi
    elif [ $exit_code -ne 0 ]; then
        log_error "Unit tests failed"
        return 1
    fi
    
    log_success "Unit tests passed"
    return 0
}

# Run API tests
run_api_tests() {
    log_info "Running API tests..."
    
    local pytest_args=()
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    if ! python -m pytest \
        -m "api" \
        --tb=short \
        --junitxml="$PROJECT_ROOT/test-results-api.xml" \
        "${pytest_args[@]}" \
        tests/api/; then
        log_error "API tests failed"
        return 1
    fi
    
    log_success "API tests passed"
    return 0
}

# Run Lambda local tests
run_lambda_local_tests() {
    log_info "Running Lambda local tests..."
    
    local pytest_args=()
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    if ! python -m pytest \
        -m "lambda_local" \
        --tb=short \
        --junitxml="$PROJECT_ROOT/test-results-lambda-local.xml" \
        "${pytest_args[@]}" \
        tests/integration/test_lambda_local.py; then
        log_error "Lambda local tests failed"
        return 1
    fi
    
    log_success "Lambda local tests passed"
    return 0
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    local pytest_args=()
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    if ! python -m pytest \
        -m "integration" \
        --tb=short \
        --junitxml="$PROJECT_ROOT/test-results-integration.xml" \
        "${pytest_args[@]}" \
        tests/integration/; then
        log_error "Integration tests failed"
        return 1
    fi
    
    log_success "Integration tests passed"
    return 0
}

# Run all tests
run_all_tests() {
    log_info "Running all tests..."
    
    local pytest_args=()
    if [ "$VERBOSE" = true ]; then
        pytest_args+=("-v")
    fi
    
    # Run tests with appropriate markers
    local markers=()
    if [ "$RUN_UNIT" = true ]; then
        markers+=("unit")
    fi
    if [ "$RUN_API" = true ]; then
        markers+=("api")
    fi
    if [ "$RUN_LAMBDA_LOCAL" = true ]; then
        markers+=("lambda_local")
    fi
    if [ "$RUN_INTEGRATION" = true ]; then
        markers+=("integration")
    fi
    
    if [ ${#markers[@]} -eq 0 ]; then
        # Run all tests except slow ones by default
        markers=("not slow")
    fi
    
    local marker_string
    marker_string=$(IFS=" and "; echo "${markers[*]}")
    
    if ! python -m pytest \
        -m "$marker_string" \
        --tb=short \
        --junitxml="$PROJECT_ROOT/test-results-all.xml" \
        "${pytest_args[@]}" \
        tests/; then
        log_error "Some tests failed"
        return 1
    fi
    
    log_success "All tests passed"
    return 0
}

# =============================================================================
# Main Script
# =============================================================================

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit)
                RUN_UNIT=true
                RUN_ALL=false
                shift
                ;;
            --api)
                RUN_API=true
                RUN_ALL=false
                shift
                ;;
            --lambda-local)
                RUN_LAMBDA_LOCAL=true
                RUN_ALL=false
                shift
                ;;
            --integration)
                RUN_INTEGRATION=true
                RUN_ALL=false
                shift
                ;;
            --all)
                RUN_ALL=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-cleanup)
                CLEANUP=false
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
AWS SAM Local Testing Automation Script

Usage: $0 [OPTIONS]

Options:
  --unit              Run only unit tests (fast, no SAM services required)
  --api               Run only API tests (requires SAM local API on port 3000)
  --lambda-local      Run only Lambda local tests (requires SAM local Lambda on port 3001)
  --integration       Run only integration tests
  --all               Run all tests (default)
  --verbose           Enable verbose output
  --no-cleanup        Don't clean up SAM processes after tests (for debugging)
  --help, -h          Show this help message

Examples:
  $0 --unit              # Run unit tests only
  $0 --api               # Run API tests with SAM local API
  $0 --all               # Run all tests (default)
  $0 --verbose --all     # Run all tests with verbose output

Exit Codes:
  0 - Success (all tests passed)
  1 - General error (script failure)
  2 - SAM CLI not found
  3 - SAM service failed to start
  4 - Health check timeout
  5 - Pytest tests failed
EOF
}

# Main function
main() {
    log_info "Starting AWS SAM local testing automation"
    log_debug "Project root: $PROJECT_ROOT"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Configure environment
    configure_environment
    
    # Setup cleanup trap
    setup_trap
    
    # Check SAM CLI
    if ! check_sam_cli; then
        exit 2
    fi
    
    # Determine which services need to be started
    local need_api=false
    local need_lambda=false
    
    if [ "$RUN_API" = true ] || [ "$RUN_ALL" = true ]; then
        need_api=true
    fi
    
    if [ "$RUN_LAMBDA_LOCAL" = true ] || [ "$RUN_ALL" = true ]; then
        need_lambda=true
    fi
    
    # Start required services
    if [ "$need_api" = true ]; then
        if ! start_sam_api; then
            exit 3
        fi
    fi
    
    if [ "$need_lambda" = true ]; then
        if ! start_sam_lambda; then
            exit 3
        fi
    fi
    
    # Run tests based on selection
    local test_result=0
    
    if [ "$RUN_UNIT" = true ]; then
        if ! run_unit_tests; then
            test_result=1
        fi
    elif [ "$RUN_API" = true ]; then
        if ! run_api_tests; then
            test_result=1
        fi
    elif [ "$RUN_LAMBDA_LOCAL" = true ]; then
        if ! run_lambda_local_tests; then
            test_result=1
        fi
    elif [ "$RUN_INTEGRATION" = true ]; then
        if ! run_integration_tests; then
            test_result=1
        fi
    elif [ "$RUN_ALL" = true ]; then
        if ! run_all_tests; then
            test_result=1
        fi
    else
        # Default: run all tests
        if ! run_all_tests; then
            test_result=1
        fi
    fi
    
    # Cleanup (handled by trap, but we can also call it explicitly)
    cleanup
    
    # Exit with appropriate code
    if [ $test_result -eq 0 ]; then
        log_success "All tests completed successfully"
        exit 0
    else
        log_error "Some tests failed"
        exit 5
    fi
}

# Run main function with all arguments
main "$@"