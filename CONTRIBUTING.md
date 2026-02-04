# Contributing to Physics Application Suite

Thank you for your interest in contributing to Physics Application Suite! This document provides guidelines and information for contributors to help maintain the quality and consistency of our educational physics software.

## 🎯 Ways to Contribute

### 📝 Code Contributions
- **New Physics Modules**: Add advanced physics simulations with real experimental data
- **Calculator Enhancements**: Expand the 19 calculation types with new physics domains
- **Unit Converter Extensions**: Add new unit categories beyond the current 21
- **Visualization Improvements**: Enhance matplotlib plots and interactive features
- **Security Enhancements**: Strengthen the sanitization and validation systems
- **Performance Optimizations**: Improve calculation speed and memory efficiency

### 📚 Educational Content
- **Real Experimental Data**: Contribute authentic laboratory measurements and datasets
- **Physics Validation**: Verify calculations against published research and textbooks
- **Educational Resources**: Add learning materials, tutorials, and examples
- **Documentation**: Improve user guides, API documentation, and code comments

### 🐛 Bug Reports & Issues
- **Bug Reports**: Identify and report issues with detailed reproduction steps
- **Feature Requests**: Suggest new physics modules or educational enhancements
- **Usability Feedback**: Report user experience issues and interface improvements
- **Educational Accuracy**: Report physics or calculation errors for correction

## 🚀 Getting Started

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/KernFerm/Physics-Application-Suite/.git
   cd physics-application-suite
   ```

2. **Python Environment**
   ```bash
   # Create virtual environment (recommended)
   python -m venv physics-env
   
   # Activate environment
   # Windows:
   physics-env\Scripts\activate
   # Linux/macOS:
   source physics-env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Verify Installation**
   ```bash
   python main.py  # Should launch the application
   ```

### Development Dependencies

For contributors, install additional development tools:
```bash
pip install pytest pytest-qt black flake8 mypy sphinx
```

## 📋 Contribution Guidelines

### Code Style Standards

#### Python Code Style
- **PEP 8 Compliance**: Follow Python PEP 8 style guide
- **Line Length**: Maximum 100 characters (accommodates physics equations)
- **Imports**: Organize imports with standard library, third-party, then local imports
- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Type Hints**: Include type hints for all function parameters and return values

#### Physics-Specific Guidelines
- **Physical Units**: Always include units in variable names and comments
- **Constants**: Use physics_utils.py constants when available
- **Validation**: All physics inputs must use the sanitization system
- **Real Data**: Prefer real experimental data over theoretical calculations
- **Accuracy**: Validate calculations against established physics references

#### PyQt5 GUI Guidelines
- **Styling**: Use consistent dark theme styling across all widgets
- **Layout**: Follow the existing left-panel/right-panel layout pattern
- **Input Fields**: All input fields must use the standardized styling methods
- **Error Handling**: Implement graceful error handling with user-friendly messages
- **Accessibility**: Ensure keyboard navigation and screen reader compatibility

### Code Formatting

Use Black for automatic code formatting:
```bash
black --line-length 100 .
```

Use flake8 for linting:
```bash
flake8 --max-line-length 100 --ignore E203,W503 .
```

### Commit Message Guidelines

Use conventional commit format:
```
type(scope): brief description

Detailed description of changes made.

- List specific changes
- Include physics validation notes
- Reference any educational sources
```

**Types:**
- `feat`: New feature (physics module, calculator function)
- `fix`: Bug fix (calculation error, UI issue)
- `docs`: Documentation changes
- `style`: Code style changes (no functional changes)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `security`: Security improvements
- `data`: Adding or updating experimental data

**Examples:**
```
feat(nuclear): add radioactive decay simulation with real isotope data

- Implements nuclear decay calculations using NIST data
- Adds decay chain visualization with proper half-lives
- Includes uncertainty propagation for realistic measurements
- Validates against published nuclear physics literature

fix(sanitizer): prevent division by zero in physics calculations

- Adds validation for denominator parameters
- Implements graceful error handling for edge cases
- Updates error messages for better user guidance
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run GUI tests
pytest -v tests/test_gui.py
```

### Test Categories

#### Unit Tests
- **Physics Calculations**: Test all calculation functions with known values
- **Input Sanitization**: Verify sanitizer classes handle edge cases
- **Unit Conversions**: Validate conversion accuracy across all categories
- **Error Handling**: Test graceful handling of invalid inputs

#### Integration Tests
- **Module Integration**: Test physics modules with realistic data
- **GUI Integration**: Test PyQt5 widgets and user interactions
- **File Operations**: Test project save/load functionality
- **Security Integration**: Test complete sanitization workflows

#### Educational Tests
- **Physics Accuracy**: Validate against textbook problems and solutions
- **Experimental Data**: Verify real data integration and processing
- **Educational Value**: Test learning outcomes and user comprehension

### Writing New Tests

```python
import pytest
from PyQt5.QtTest import QTest
from physics_utils import PhysicsCalculator, PhysicsSanitizer

class TestPhysicsModule:
    def test_kinetic_energy_calculation(self):
        """Test kinetic energy calculation with real experimental values."""
        calculator = PhysicsCalculator()
        mechanics = calculator.mechanics_calculator()
        
        # Test with basketball data (mass=0.624kg, velocity=8.5m/s)
        result = mechanics["Kinetic Energy"](0.624, 8.5)
        expected = 22.5  # Joules
        
        assert abs(result - expected) < 0.1, f"Expected {expected}, got {result}"
    
    def test_input_sanitization(self):
        """Test input sanitization prevents malicious inputs."""
        sanitizer = PhysicsSanitizer()
        
        # Test dangerous input
        with pytest.raises(ValidationError):
            sanitizer.sanitize_numeric_input("<script>alert('xss')</script>")
```

## 📚 Adding New Physics Modules

### Module Structure

New physics modules should follow this structure:

```python
class NewPhysicsWidget(QWidget):
    """New physics simulation widget with real experimental data."""
    
    def __init__(self):
        super().__init__()
        self.sanitizer = PhysicsSanitizer()  # Always include sanitization
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface following standard layout."""
        # Left panel: Controls and parameters
        # Right panel: Visualization canvas
        # Follow existing styling patterns
        
    def calculate_physics(self):
        """Perform physics calculations with real experimental data."""
        try:
            # 1. Sanitize all inputs
            # 2. Perform calculations using physics_utils
            # 3. Update visualization
            # 4. Display results with proper units
            # 5. Handle errors gracefully
            pass
        except Exception as e:
            self.show_error_message(str(e))
    
    def reset_simulation(self):
        """Reset to initial state with default parameters."""
        pass
```

### Required Components

1. **Input Sanitization**: Use PhysicsSanitizer for all user inputs
2. **Real Data**: Include authentic experimental measurements
3. **Visualization**: Professional matplotlib plots with proper styling
4. **Error Handling**: Graceful handling of edge cases and invalid inputs
5. **Documentation**: Comprehensive docstrings and inline comments
6. **Testing**: Unit tests for all calculation functions
7. **Educational Value**: Clear physics concepts with real-world relevance

### Physics Data Sources

When adding real experimental data:
- **Primary Sources**: Peer-reviewed journal articles
- **Educational Sources**: Established physics textbooks
- **Laboratory Data**: Authentic measurements with proper attribution
- **Institutional Data**: University laboratory measurements
- **Citation**: Always include proper citations in code comments

## 📖 Documentation Standards

### Code Documentation

```python
def calculate_relativistic_energy(mass: float, velocity: float) -> float:
    """Calculate relativistic energy using Einstein's mass-energy equivalence.
    
    Based on experimental validation from particle accelerator measurements.
    Implements proper input validation and handles edge cases near light speed.
    
    Args:
        mass (float): Rest mass in kilograms, must be positive
        velocity (float): Velocity in m/s, must be less than speed of light
        
    Returns:
        float: Relativistic energy in Joules
        
    Raises:
        ValidationError: If inputs are invalid or exceed physical limits
        
    Example:
        >>> calculate_relativistic_energy(9.109e-31, 2.7e8)  # Fast electron
        4.096e-13  # Joules
        
    References:
        Einstein, A. (1905). "Zur Elektrodynamik bewegter Körper"
        NIST Physical Constants Database
    """
```

### Physics Equations

Document physics equations in LaTeX format:
```python
# Relativistic energy equation:
# E = γmc² where γ = 1/√(1 - v²/c²)
# 
# For small velocities (v << c), approximates to:
# E ≈ mc² + ½mv² (rest energy + kinetic energy)
```

## 🔒 Security Considerations

### Input Validation Requirements

All contributions must follow security guidelines:

1. **Sanitize All Inputs**: Use MainApplicationSanitizer or PhysicsSanitizer
2. **Validate Ranges**: Check physics parameters against realistic bounds
3. **Handle Errors**: Implement comprehensive error handling
4. **Prevent Injection**: Protect against code injection and XSS
5. **File Security**: Validate file operations and paths
6. **Memory Safety**: Prevent buffer overflows and memory leaks

### Security Testing

```python
def test_input_security(self):
    """Test that malicious inputs are properly handled."""
    sanitizer = PhysicsSanitizer()
    
    dangerous_inputs = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "eval(malicious_code)",
        "\\x00\\x01\\x02"  # Control characters
    ]
    
    for dangerous_input in dangerous_inputs:
        with pytest.raises(ValidationError):
            sanitizer.sanitize_string_input(dangerous_input)
```

## 📋 Pull Request Process

### Before Submitting

1. **Code Quality**
   - [ ] Code follows style guidelines (Black + flake8)
   - [ ] All tests pass (`pytest`)
   - [ ] Type hints included (`mypy`)
   - [ ] Documentation updated

2. **Physics Validation**
   - [ ] Calculations verified against known solutions
   - [ ] Real experimental data properly cited
   - [ ] Units and constants correctly used
   - [ ] Educational value assessed

3. **Security Review**
   - [ ] All inputs properly sanitized
   - [ ] Error handling implemented
   - [ ] Security tests included
   - [ ] No new vulnerabilities introduced

### Pull Request Template

```markdown
## Description
Brief description of changes and physics domain addressed.

## Type of Change
- [ ] Bug fix (calculation error, UI issue)
- [ ] New feature (physics module, calculator function)
- [ ] Educational enhancement (real data, improved accuracy)
- [ ] Security improvement (sanitization, validation)
- [ ] Documentation update

## Physics Validation
- [ ] Calculations verified against textbook/literature
- [ ] Real experimental data included with citations
- [ ] Units and dimensional analysis confirmed
- [ ] Edge cases and limits properly handled

## Testing
- [ ] Unit tests added for new functions
- [ ] Integration tests updated
- [ ] GUI tests included (if applicable)
- [ ] Security tests added (if applicable)

## Educational Impact
Describe how this change improves the educational value:
- What physics concepts does it teach?
- What real-world applications does it demonstrate?
- How does it enhance student understanding?

## References
List any physics textbooks, papers, or data sources used:
- Source 1: Author, Title, Journal/Publisher, Year
- Source 2: Laboratory data from [Institution], Date
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Physics Review**: Maintainer validates physics accuracy and educational value
3. **Security Review**: Security-focused code review for all changes
4. **Educational Review**: Assessment of learning outcomes and user experience
5. **Final Approval**: Maintainer approval after all checks pass

## 🐛 Reporting Issues

### Bug Report Template

When reporting bugs, please include:

```markdown
**Bug Description**
Clear description of the unexpected behavior.

**Physics Context**
What physics calculation or simulation is affected?

**Steps to Reproduce**
1. Launch application
2. Navigate to [specific module]
3. Enter parameters: [list values]
4. Click [specific button]
5. Observe error/incorrect result

**Expected Behavior**
What should happen according to physics principles?

**Actual Behavior**
What actually happens?

**Environment**
- OS: [Windows 10/macOS/Linux]
- Python version: [3.11.2]
- PyQt5 version: [5.15.7]
- Application version: [2.0.2]

**Screenshots**
If applicable, add screenshots showing the issue.

**Additional Context**
- Are you using real experimental data?
- Does this affect educational outcomes?
- Any relevant physics background or references?
```

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed physics feature.

**Educational Value**
How would this enhance physics education?

**Physics Background**
What physics concepts would this address?

**Proposed Implementation**
Basic idea of how this could be implemented.

**Real-World Applications**
What practical applications would this demonstrate?

**References**
Relevant textbooks, papers, or educational resources.
```

## 🌟 Recognition

Contributors will be recognized in the following ways:

### Code Contributors
- Listed in `CONTRIBUTORS.md` file
- Credited in release notes for major contributions
- Physics modules attributed to original contributors
- Educational enhancements acknowledged in documentation

### Educational Contributors
- Experimental data contributors credited in module documentation
- Physics validation contributors listed in accuracy acknowledgments
- Educational review contributors recognized in teaching resources

### Community Contributors
- Bug reporters acknowledged in issue resolutions
- Feature requesters credited for enhancement ideas
- Documentation contributors listed in help system

## 📞 Community and Support

### Communication Channels
- **GitHub Issues**: Bug reports, feature requests, and discussions
- **GitHub Discussions**: General questions and community support
- **Documentation**: Comprehensive guides and API references
- **Educational Resources**: Physics teaching materials and examples

### Code of Conduct
We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and constructive in all interactions
- Focus on educational value and physics accuracy
- Provide helpful feedback and support to other contributors
- Maintain professionalism in code reviews and discussions
- Respect diverse educational backgrounds and approaches to physics

### Getting Help
- **Development Questions**: Open GitHub issues with "question" label
- **Physics Questions**: Reference established textbooks and peer-reviewed sources
- **Technical Support**: Check existing documentation and issues before asking
- **Educational Guidance**: Discuss pedagogical approaches in GitHub discussions

## 📄 License

By contributing to Physics Application Suite, you agree that your contributions will be licensed under the same GNU General Public License v3.0 that covers the project. This ensures that all educational enhancements remain open source and freely available to the physics education community.

---


- [Discord Invite](https://discord.gg/zQbJJgwbUv)

Thank you for contributing to Physics Application Suite! Your contributions help make physics education more accessible, accurate, and engaging for students and educators worldwide. 🔬⚡🌊

*Together, we're building the future of physics education!*
