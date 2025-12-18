# =============================================================================
# AWS SAM Testing Makefile
# =============================================================================
#
# This Makefile provides targets for running tests with AWS SAM local services.
# It serves as an alternative to the run_local_tests.sh script with a more
# traditional Unix-style interface.
#
# Usage:
#   make test-unit        # Run unit tests only
#   make test-api         # Run API tests (requires SAM local API)
#   make test-lambda-local # Run Lambda local tests (requires SAM local Lambda)
#   make test-all         # Run all tests with service management
#   make clean            # Clean up test artifacts and processes
#
# =============================================================================

# Configuration
PYTHON = python3
PYTEST = $(PYTHON) -m pytest
SAM = sam
PORT_API = 3000
PORT_LAMBDA = 3001
TIMEOUT = 300
INTERVAL = 2

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Test markers
MARKER_UNIT = unit
MARKER_API = api
MARKER_LAMBDA_LOCAL = lambda_local
MARKER_INTEGRATION = integration
MARKER_SLOW = slow

# File paths
LOG_API = sam_api.log
LOG_LAMBDA = sam_lambda.log
TEST_RESULTS_UNIT = test-results-unit.xml
TEST_RESULTS_API = test-results-api.xml
TEST_RESULTS_LAMBDA = test-results-lambda-local.xml
TEST_RESULTS_INTEGRATION = test-results-integration.xml
TEST_RESULTS_ALL = test-results-all.xml

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Utility Functions
# =============================================================================

define log_info
	@echo -e "$(BLUE)[INFO]$(NC) $(1)"
endef

define log_success
	@echo -e "$(GREEN)[SUCCESS]$(NC) $(1)"
endef

define log_warning
	@echo -e "$(YELLOW)[WARNING]$(NC) $(1)"
endef

define log_error
	@echo -e "$(RED)[ERROR]$(NC) $(1)" >&2
endef

# Check if a command exists
check_command = $(if $(shell command -v $(1)),,$(error "$(1)" command not found. Please install it.))

# Check if port is in use
check_port = $(shell nc -z localhost $(1) 2>/dev/null && echo "in_use" || echo "free")

# Wait for service to be ready
define wait_for_service
	@echo -e "$(BLUE)[INFO]$(NC) Waiting for $(2) on port $(1)..."
	@timeout=0; \
	while [ $$timeout -lt $(TIMEOUT) ]; do \
		if nc -z localhost $(1) 2>/dev/null; then \
			echo -e "$(GREEN)[SUCCESS]$(NC) $(2) is ready on port $(1)"; \
			exit 0; \
		fi; \
		sleep $(INTERVAL); \
		timeout=$$((timeout + $(INTERVAL))); \
		echo -e "$(BLUE)[DEBUG]$(NC) Waited $$timeout seconds for $(2)..."; \
	done; \
	echo -e "$(RED)[ERROR]$(NC) Timeout waiting for $(2) on port $(1) after $(TIMEOUT) seconds" >&2; \
	exit 1
endef

# =============================================================================
# Targets
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo -e "$(BLUE)AWS SAM Testing Makefile$(NC)"
	@echo
	@echo "Usage:"
	@echo "  make <target>"
	@echo
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: check-deps
check-deps: ## Check if required dependencies are installed
	$(call log_info, "Checking dependencies...")
	$(call check_command,$(SAM))
	$(call check_command,$(PYTHON))
	$(call check_command,nc)
	$(call log_success, "All dependencies are installed")

.PHONY: test-unit
test-unit: ## Run unit tests only (fast, no SAM services required)
	$(call log_info, "Running unit tests...")
	@$(PYTEST) \
		--tb=short \
		--junitxml="$(TEST_RESULTS_UNIT)" \
		-v \
		tests/unit/
	$(call log_success, "Unit tests completed")

.PHONY: test-api
test-api: start-api ## Run API tests (requires SAM local API)
	$(call log_info, "Running API tests...")
	@$(PYTEST) \
		-m "$(MARKER_API)" \
		--tb=short \
		--junitxml="$(TEST_RESULTS_API)" \
		-v \
		tests/api/
	$(call log_success, "API tests completed")

.PHONY: test-lambda-local
test-lambda-local: start-lambda ## Run Lambda local tests (requires SAM local Lambda)
	$(call log_info, "Running Lambda local tests...")
	@$(PYTEST) \
		-m "$(MARKER_LAMBDA_LOCAL)" \
		--tb=short \
		--junitxml="$(TEST_RESULTS_LAMBDA)" \
		-v \
		tests/integration/test_lambda_local.py
	$(call log_success, "Lambda local tests completed")

.PHONY: test-integration
test-integration: ## Run integration tests
	$(call log_info, "Running integration tests...")
	@$(PYTEST) \
		-m "$(MARKER_INTEGRATION)" \
		--tb=short \
		--junitxml="$(TEST_RESULTS_INTEGRATION)" \
		-v \
		tests/integration/
	$(call log_success, "Integration tests completed")

.PHONY: test-all
test-all: check-deps start-api start-lambda ## Run all tests with service management
	$(call log_info, "Running all tests...")
	@$(PYTEST) \
		-m "$(MARKER_UNIT) or $(MARKER_API) or $(MARKER_LAMBDA_LOCAL) or $(MARKER_INTEGRATION)" \
		--tb=short \
		--junitxml="$(TEST_RESULTS_ALL)" \
		-v \
		tests/
	$(call log_success, "All tests completed")

.PHONY: test-smoke
test-smoke: ## Run smoke tests (basic functionality)
	$(call log_info, "Running smoke tests...")
	@$(PYTEST) \
		-m "smoke" \
		--tb=short \
		-v \
		tests/
	$(call log_success, "Smoke tests completed")

.PHONY: test-regression
test-regression: ## Run regression tests
	$(call log_info, "Running regression tests...")
	@$(PYTEST) \
		-m "regression" \
		--tb=short \
		-v \
		tests/
	$(call log_success, "Regression tests completed")

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	$(call log_info, "Running tests with coverage...")
	@$(PYTEST) \
		--cov=src \
		--cov-report=term \
		--cov-report=html:coverage_html \
		--cov-report=xml:coverage.xml \
		-v \
		tests/
	$(call log_success, "Coverage report generated: coverage_html/ and coverage.xml")

.PHONY: start-api
start-api: check-deps ## Start SAM local API in background
	$(call log_info, "Starting SAM local API on port $(PORT_API)...")
	@if [ "$(call check_port,$(PORT_API))" = "in_use" ]; then \
		$(call log_warning, "Port $(PORT_API) is already in use"); \
	else \
		$(SAM) local start-api \
			--port $(PORT_API) \
			--template template.yaml \
			--env-vars .env \
			--warm-containers EAGER \
			--debug > "$(LOG_API)" 2>&1 & \
		echo $$! > .sam_api.pid; \
		$(call wait_for_service,$(PORT_API),SAM local API); \
	fi

.PHONY: start-lambda
start-lambda: check-deps ## Start SAM local Lambda in background
	$(call log_info, "Starting SAM local Lambda on port $(PORT_LAMBDA)...")
	@if [ "$(call check_port,$(PORT_LAMBDA))" = "in_use" ]; then \
		$(call log_warning, "Port $(PORT_LAMBDA) is already in use"); \
	else \
		$(SAM) local start-lambda \
			--port $(PORT_LAMBDA) \
			--template template.yaml \
			--env-vars .env \
			--debug > "$(LOG_LAMBDA)" 2>&1 & \
		echo $$! > .sam_lambda.pid; \
		$(call wait_for_service,$(PORT_LAMBDA),SAM local Lambda); \
	fi

.PHONY: stop-api
stop-api: ## Stop SAM local API
	$(call log_info, "Stopping SAM local API...")
	@if [ -f .sam_api.pid ]; then \
		kill $$(cat .sam_api.pid) 2>/dev/null || true; \
		rm -f .sam_api.pid; \
		$(call log_success, "SAM local API stopped"); \
	else \
		$(call log_warning, "No SAM API process found"); \
	fi

.PHONY: stop-lambda
stop-lambda: ## Stop SAM local Lambda
	$(call log_info, "Stopping SAM local Lambda...")
	@if [ -f .sam_lambda.pid ]; then \
		kill $$(cat .sam_lambda.pid) 2>/dev/null || true; \
		rm -f .sam_lambda.pid; \
		$(call log_success, "SAM local Lambda stopped"); \
	else \
		$(call log_warning, "No SAM Lambda process found"); \
	fi

.PHONY: stop-all
stop-all: stop-api stop-lambda ## Stop all SAM services
	$(call log_info, "Stopping all SAM services...")
	@pkill -f "sam local start-api" 2>/dev/null || true
	@pkill -f "sam local start-lambda" 2>/dev/null || true
	$(call log_success, "All SAM services stopped")

.PHONY: status
status: ## Check status of SAM services
	$(call log_info, "Checking SAM services status...")
	@if [ "$(call check_port,$(PORT_API))" = "in_use" ]; then \
		echo -e "$(GREEN)SAM API$(NC): Running on port $(PORT_API)"; \
	else \
		echo -e "$(RED)SAM API$(NC): Not running"; \
	fi
	@if [ "$(call check_port,$(PORT_LAMBDA))" = "in_use" ]; then \
		echo -e "$(GREEN)SAM Lambda$(NC): Running on port $(PORT_LAMBDA)"; \
	else \
		echo -e "$(RED)SAM Lambda$(NC): Not running"; \
	fi

.PHONY: logs-api
logs-api: ## Show SAM API logs
	$(call log_info, "Showing SAM API logs...")
	@if [ -f "$(LOG_API)" ]; then \
		tail -50 "$(LOG_API)"; \
	else \
		$(call log_warning, "No SAM API log file found"); \
	fi

.PHONY: logs-lambda
logs-lambda: ## Show SAM Lambda logs
	$(call log_info, "Showing SAM Lambda logs...")
	@if [ -f "$(LOG_LAMBDA)" ]; then \
		tail -50 "$(LOG_LAMBDA)"; \
	else \
		$(call log_warning, "No SAM Lambda log file found"); \
	fi

.PHONY: logs
logs: logs-api logs-lambda ## Show all SAM logs

.PHONY: clean
clean: stop-all ## Clean up test artifacts and processes
	$(call log_info, "Cleaning up test artifacts...")
	@rm -f .sam_*.pid
	@rm -f $(LOG_API) $(LOG_LAMBDA)
	@rm -f $(TEST_RESULTS_UNIT) $(TEST_RESULTS_API) $(TEST_RESULTS_LAMBDA) $(TEST_RESULTS_INTEGRATION) $(TEST_RESULTS_ALL)
	@rm -f coverage.xml
	@rm -rf coverage_html/
	@rm -rf .pytest_cache/
	@rm -rf __pycache__/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	$(call log_success, "Cleanup completed")

.PHONY: install-deps
install-deps: ## Install Python dependencies
	$(call log_info, "Installing Python dependencies...")
	@pip install -r requirements.txt
	@pip install aws-sam-cli
	$(call log_success, "Dependencies installed")

.PHONY: setup
setup: install-deps ## Setup development environment
	$(call log_info, "Setting up development environment...")
	@pip install -e .
	$(call log_success, "Setup completed")

.PHONY: validate
validate: ## Validate SAM template
	$(call log_info, "Validating SAM template...")
	@$(SAM) validate --template template.yaml
	$(call log_success, "SAM template is valid")

.PHONY: build
build: ## Build SAM application
	$(call log_info, "Building SAM application...")
	@$(SAM) build --use-container
	$(call log_success, "SAM application built")

# =============================================================================
# Test Execution with Cleanup
# =============================================================================

# These targets ensure cleanup happens even if tests fail
.PHONY: safe-test-api
safe-test-api: start-api ## Run API tests with guaranteed cleanup
	@set +e; \
	$(MAKE) test-api; \
	test_result=$$?; \
	$(MAKE) stop-api; \
	exit $$test_result

.PHONY: safe-test-lambda-local
safe-test-lambda-local: start-lambda ## Run Lambda local tests with guaranteed cleanup
	@set +e; \
	$(MAKE) test-lambda-local; \
	test_result=$$?; \
	$(MAKE) stop-lambda; \
	exit $$test_result

.PHONY: safe-test-all
safe-test-all: start-api start-lambda ## Run all tests with guaranteed cleanup
	@set +e; \
	$(MAKE) test-all; \
	test_result=$$?; \
	$(MAKE) stop-all; \
	exit $$test_result

# =============================================================================
# CI/CD Integration
# =============================================================================

.PHONY: ci-test
ci-test: ## Run tests for CI/CD pipeline (with XML output)
	$(call log_info, "Running CI/CD test suite...")
	@$(MAKE) safe-test-all
	$(call log_success, "CI/CD test suite completed")

.PHONY: ci-coverage
ci-coverage: ## Run coverage for CI/CD pipeline
	$(call log_info, "Running CI/CD coverage...")
	@$(PYTEST) \
		--cov=src \
		--cov-report=xml:coverage.xml \
		--junitxml="$(TEST_RESULTS_ALL)" \
		tests/
	$(call log_success, "CI/CD coverage completed")

# =============================================================================
# Development Workflow
# =============================================================================

.PHONY: dev
dev: setup ## Setup development environment
	$(call log_info, "Development environment ready")
	@echo "Available commands:"
	@echo "  make test-unit        # Run unit tests"
	@echo "  make test-api         # Run API tests"
	@echo "  make test-all         # Run all tests"
	@echo "  make clean            # Clean up"
	@echo "  make status           # Check service status"

.PHONY: watch
watch: ## Watch for changes and run tests (requires entr)
	$(call check_command,entr)
	$(call log_info, "Watching for file changes...")
	@find . -name "*.py" -not -path "./.aws-sam/*" -not -path "./.pytest_cache/*" | entr -c make test-unit

# =============================================================================
# Quality Assurance
# =============================================================================

.PHONY: lint
lint: ## Run code linting
	$(call log_info, "Running code linting...")
	@flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	$(call log_success, "Linting completed")

.PHONY: type-check
type-check: ## Run type checking
	$(call log_info, "Running type checking...")
	@mypy src/ --ignore-missing-imports
	$(call log_success, "Type checking completed")

.PHONY: qa
qa: lint type-check test-unit ## Run all quality assurance checks
	$(call log_success, "All quality assurance checks passed")

# =============================================================================
# Documentation
# =============================================================================

.PHONY: docs
docs: ## Generate documentation
	$(call log_info, "Generating documentation...")
	@pydoc-markdown
	$(call log_success, "Documentation generated")

.PHONY: readme
readme: ## Update README with test information
	$(call log