# Testing Framework

This directory contains the testing infrastructure for the 6G OPENRAN project.

## Test Categories

### Unit Tests

Location: `testing/unit-tests/`

- Test individual functions and classes
- Mock external dependencies
- Fast execution

### Integration Tests

Location: `testing/integration-tests/`

- Test component interactions
- End-to-end workflows
- Multiple components running together

### Performance Tests

Location: `testing/performance-tests/`

- Throughput benchmarks
- Latency measurements
- Stress testing
- Load testing

## Running Tests

```bash
# Run all tests
./scripts/testing/run-all-tests.sh

# Run unit tests only
pytest testing/unit-tests/ -v

# Run integration tests
pytest testing/integration-tests/ -v

# Run performance tests
pytest testing/performance-tests/ -v

# Run with coverage
pytest testing/ --cov=ran --cov=core --cov-report=html
```

## Test Results

Test results and coverage reports will be generated in:
- `testing/results/` - Test execution results
- `htmlcov/` - Coverage reports

## Writing Tests

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on writing tests.

## CI/CD Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Release tags

See `.github/workflows/` for CI configuration.
