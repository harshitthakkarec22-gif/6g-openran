# Contributing to 6G OPENRAN

Thank you for your interest in contributing to the 6G OPENRAN project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone. We expect all participants to:

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Publishing others' private information
- Any conduct inappropriate in a professional setting

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Read the [README.md](README.md)
2. Reviewed the [Research Report](docs/RESEARCH_REPORT.md)
3. Set up your [development environment](docs/guides/GETTING_STARTED.md)
4. Familiarized yourself with the architecture

### Setting Up Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/6g-openran.git
cd 6g-openran

# Add upstream remote
git remote add upstream https://github.com/harshitthakkarec22-gif/6g-openran.git

# Create a new branch for your work
git checkout -b feature/your-feature-name
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Code Contributions**
   - Bug fixes
   - New features
   - Performance improvements
   - Refactoring

2. **Documentation**
   - Improving existing documentation
   - Adding new guides or tutorials
   - Fixing typos or clarifications
   - Adding code examples

3. **Testing**
   - Writing unit tests
   - Integration tests
   - Performance tests
   - Identifying and reporting bugs

4. **Research**
   - Algorithm improvements
   - Performance analysis
   - New use case implementations

5. **Community Support**
   - Answering questions in discussions
   - Helping other contributors
   - Improving onboarding materials

### Finding Work

- Check [GitHub Issues](https://github.com/harshitthakkarec22-gif/6g-openran/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Check the project roadmap in README.md
- Propose new features in discussions

## Development Workflow

### Branch Naming Convention

Use descriptive branch names following this pattern:

```
<type>/<short-description>

Examples:
feature/add-cu-implementation
bugfix/fix-amf-registration
docs/improve-getting-started
test/add-smf-unit-tests
refactor/optimize-scheduler
```

Types:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions or modifications
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

### Commit Message Guidelines

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(cu): implement RRC connection establishment

Add RRC connection establishment procedure to CU-CP component.
This includes state machine implementation and message handling.

Closes #123
```

```
fix(smf): correct PDU session establishment timeout

The timeout was set to 5s but should be 30s according to spec.

Fixes #456
```

```
docs(ran): add detailed DU configuration guide

Provide step-by-step instructions for configuring the DU,
including scheduler options and interface settings.
```

## Coding Standards

### General Principles

1. **Write Clean Code**
   - Use meaningful variable and function names
   - Keep functions small and focused
   - Avoid code duplication (DRY principle)
   - Comment complex logic

2. **Follow Language-Specific Conventions**
   - Python: PEP 8
   - C++: Google C++ Style Guide
   - Go: Effective Go guidelines

### Python Style Guide

```python
# Good: descriptive names, proper spacing
def calculate_spectral_efficiency(bandwidth_mhz: float, num_antennas: int) -> float:
    """
    Calculate spectral efficiency for given bandwidth and antenna configuration.
    
    Args:
        bandwidth_mhz: Channel bandwidth in MHz
        num_antennas: Number of antenna elements
    
    Returns:
        Spectral efficiency in bps/Hz
    """
    base_efficiency = 5.0  # bps/Hz for single antenna
    mimo_gain = math.log2(num_antennas)
    return base_efficiency * (1 + mimo_gain * 0.3)

# Bad: unclear names, no documentation
def calc(b, n):
    e = 5.0
    g = math.log2(n)
    return e * (1 + g * 0.3)
```

### C++ Style Guide

```cpp
// Good: clear class structure, proper encapsulation
class MacScheduler {
 public:
  MacScheduler(SchedulerType type, int num_resource_blocks);
  ~MacScheduler();
  
  // Schedule resources for connected UEs
  SchedulingDecision ScheduleDownlink(const std::vector<UeContext>& ues);
  
 private:
  SchedulerType scheduler_type_;
  int num_resource_blocks_;
  
  // Helper function for proportional fair scheduling
  int CalculatePriority(const UeContext& ue) const;
};

// Bad: poor naming, public data members
class sched {
 public:
  int type;
  int rb;
  void schedule(vector<int> u);
};
```

### Configuration Files

Use YAML for configuration:

```yaml
# Good: well-structured, commented
component:
  id: 1
  name: "Component-001"
  
  # Network interfaces
  interfaces:
    control_plane:
      address: "0.0.0.0"
      port: 38412
    user_plane:
      address: "0.0.0.0"
      port: 2152
  
  # QoS parameters
  qos:
    max_sessions: 1000
    default_priority: 8

# Bad: unclear structure
c:
  i: 1
  n: "C1"
  if1: "0.0.0.0:38412"
  if2: "0.0.0.0:2152"
  q: [1000, 8]
```

## Testing Guidelines

### Test Requirements

All code contributions must include tests:

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Verify performance requirements

### Writing Tests

**Python (pytest):**

```python
# tests/test_cu.py
import pytest
from ran.cu import CentralizedUnit

class TestCentralizedUnit:
    def setup_method(self):
        """Set up test fixtures"""
        self.cu = CentralizedUnit(config_file="test_config.yaml")
    
    def test_rrc_connection_establishment(self):
        """Test RRC connection establishment procedure"""
        # Arrange
        ue_id = 1
        request = create_rrc_setup_request(ue_id)
        
        # Act
        response = self.cu.handle_rrc_setup_request(request)
        
        # Assert
        assert response.success is True
        assert response.ue_id == ue_id
        assert self.cu.get_ue_state(ue_id) == "RRC_CONNECTED"
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        with pytest.raises(ValueError):
            CentralizedUnit(config_file="invalid.yaml")
```

**C++ (Google Test):**

```cpp
// tests/test_du.cpp
#include "gtest/gtest.h"
#include "ran/du/distributed_unit.h"

class DistributedUnitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    du_ = std::make_unique<DistributedUnit>("test_config.yaml");
  }
  
  void TearDown() override {
    du_.reset();
  }
  
  std::unique_ptr<DistributedUnit> du_;
};

TEST_F(DistributedUnitTest, MacSchedulerRoundRobin) {
  // Arrange
  std::vector<UeContext> ues = CreateTestUEs(5);
  
  // Act
  auto decision = du_->ScheduleDownlink(ues);
  
  // Assert
  EXPECT_EQ(decision.size(), 5);
  EXPECT_TRUE(VerifyRoundRobinAllocation(decision));
}

TEST_F(DistributedUnitTest, HandleFrontHaulMessage) {
  // Arrange
  auto message = CreateFrontHaulMessage();
  
  // Act
  auto response = du_->ProcessFrontHaulMessage(message);
  
  // Assert
  EXPECT_TRUE(response.success);
  EXPECT_EQ(response.latency_us, 50);
}
```

### Running Tests

```bash
# Run all tests
./scripts/testing/run-all-tests.sh

# Run specific test suite
pytest tests/test_cu.py -v

# Run with coverage
pytest tests/ --cov=ran --cov-report=html

# Run C++ tests
cd build && ctest -V
```

### Test Coverage

- Aim for minimum 80% code coverage
- Critical paths should have 100% coverage
- Add tests for bug fixes to prevent regression

## Documentation

### Documentation Requirements

1. **Code Documentation**
   - Docstrings for all public functions/classes
   - Inline comments for complex logic
   - Type hints (Python) or type declarations (C++)

2. **User Documentation**
   - Update relevant guides when adding features
   - Provide examples for new functionality
   - Include configuration examples

3. **Architecture Documentation**
   - Update architecture docs for structural changes
   - Add sequence diagrams for new flows
   - Document interface changes

### Documentation Format

Use Markdown for documentation:

```markdown
# Component Name

## Overview

Brief description of the component.

## Features

- Feature 1
- Feature 2

## Usage

\`\`\`python
from component import Feature
feature = Feature()
result = feature.process()
\`\`\`

## Configuration

\`\`\`yaml
component:
  option1: value1
  option2: value2
\`\`\`

## API Reference

### Class: Feature

#### Methods

##### process()

Processes the data.

**Parameters:**
- `data` (str): Input data

**Returns:**
- `Result`: Processing result

**Example:**
\`\`\`python
result = feature.process("data")
\`\`\`
```

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   ./scripts/testing/run-all-tests.sh
   ```

3. **Run linters**
   ```bash
   # Python
   pylint ran/
   black ran/
   
   # C++
   clang-format -i ran/**/*.cpp
   ```

4. **Update documentation**
   - Add/update relevant docs
   - Update CHANGELOG.md

### Submitting Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add appropriate labels

3. **PR Template**
   ```markdown
   ## Description
   
   Brief description of changes.
   
   ## Type of Change
   
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   
   Describe testing performed.
   
   ## Checklist
   
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] Commits follow convention
   
   ## Related Issues
   
   Closes #123
   ```

### Review Process

1. Maintainers will review your PR
2. Address feedback and comments
3. Update PR as needed
4. Once approved, PR will be merged

### After Merge

1. Delete your feature branch
2. Pull latest changes
   ```bash
   git checkout main
   git pull upstream main
   ```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code review and collaboration

### Getting Help

- Check existing documentation
- Search closed issues
- Ask in GitHub Discussions
- Tag maintainers if urgent

### Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## Questions?

If you have questions not covered here, please:
1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/harshitthakkarec22-gif/6g-openran/issues)
3. Open a [discussion](https://github.com/harshitthakkarec22-gif/6g-openran/discussions)

Thank you for contributing to 6G OPENRAN!
