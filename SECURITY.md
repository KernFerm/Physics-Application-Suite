# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of Physics Application Suite:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | ✅ Fully Supported |
| 1.x.x   | ❌ No longer supported |

## Reporting a Vulnerability

We take the security of Physics Application Suite seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **Email**: Send detailed information to `security@physics-application-suite.org` (if you have a security contact email)
2. **GitHub Issues**: For non-sensitive security issues, you can create a GitHub issue
3. **Direct Contact**: Contact the maintainers directly through the project repository

### What to Include

When reporting a vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Impact Assessment**: Your assessment of the potential impact
- **Environment Details**: 
  - Operating System (Windows/macOS/Linux)
  - Python version
  - PyQt5 version
  - Physics Application Suite version
- **Proof of Concept**: If possible, provide a minimal example

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Assessment**: Security team will assess within 5 business days
- **Fix Timeline**: Critical issues will be addressed within 14 days
- **Public Disclosure**: Coordinated disclosure after fix is available

## Security Features

### Input Validation & Sanitization

Physics Application Suite includes comprehensive security measures:

- **MainApplicationSanitizer**: Validates and sanitizes all user inputs
- **PhysicsSanitizer**: Specialized validation for physics calculations
- **Range Checking**: All numeric inputs are bounded within safe ranges
- **String Sanitization**: Removes potentially dangerous characters and control sequences
- **Path Validation**: Secure file system operations with path normalization

### File Operations

- **Project Files**: JSON-based project files with input validation
- **Export Functions**: Secure file export with sanitized filenames
- **Import Validation**: Comprehensive validation of imported project data

### Memory Safety

- **Numeric Bounds**: All calculations bounded to prevent overflow/underflow
- **Error Handling**: Comprehensive exception handling for edge cases
- **Resource Management**: Proper cleanup of matplotlib figures and Qt resources

## Security Considerations for Users

### Safe Usage Guidelines

1. **Trusted Sources**: Only install Physics Application Suite from official sources
2. **Project Files**: Only open project files from trusted sources
3. **File Permissions**: Ensure appropriate file permissions for project directories
4. **Python Environment**: Keep your Python installation and dependencies updated

### Data Privacy

- **Local Processing**: All calculations are performed locally on your machine
- **No Network Activity**: The application doesn't transmit data over networks
- **Project Data**: User project data remains on local storage only

### Recommended Environment

- **Python Version**: Use Python 3.11 or newer for latest security patches
- **Dependencies**: Keep PyQt5, matplotlib, and numpy updated to latest stable versions
- **Operating System**: Ensure your OS has latest security updates

## Security Best Practices

### For Developers

If you're contributing to or modifying the codebase:

1. **Input Validation**: Always use the provided sanitizer classes
2. **Error Handling**: Implement comprehensive error handling
3. **Code Review**: Security-focused code reviews for all changes
4. **Testing**: Include security test cases for new features

### For Educational Institutions

- **Network Isolation**: Consider running in isolated environments for classroom use
- **User Permissions**: Run with appropriate user permissions (avoid administrator privileges)
- **File Access**: Monitor file system access in shared environments

## Known Security Considerations

### Physics Calculations

- **Extreme Values**: The application handles extreme physics values safely through input bounds
- **Mathematical Operations**: Protected against division by zero and invalid mathematical operations
- **Memory Usage**: Large dataset visualizations are memory-managed appropriately

### GUI Security

- **Input Fields**: All GUI input fields have validation and sanitization
- **File Dialogs**: File selection dialogs use Qt's secure file handling
- **Window Management**: Proper window lifecycle management prevents resource leaks

## Dependencies Security

We regularly monitor our dependencies for security vulnerabilities:

- **PyQt5**: GUI framework security updates
- **matplotlib**: Plotting library security patches  
- **numpy**: Numerical computing library updates
- **Python Standard Library**: Core Python security updates

## Compliance

This educational software follows standard security practices for:

- **Desktop Applications**: Standard desktop application security guidelines
- **Educational Software**: FERPA and educational privacy considerations
- **Open Source**: Transparent security practices through open source code

## Contact Information

For security-related questions or concerns:

- **Project Maintainers**: Through GitHub repository
- **General Security**: Create an issue with the "security" label
- **Urgent Issues**: Contact maintainers directly

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve the security of Physics Application Suite.

---

**Note**: This is educational software designed for physics learning. While we implement robust security measures, users should follow standard security practices for any desktop application.

*Last Updated: February 2026*
