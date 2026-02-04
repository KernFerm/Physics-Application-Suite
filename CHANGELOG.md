# Changelog

All notable changes to the Physics Application Suite will be documented in this file.

## [2.0.2] - 2026-02-04

### Added - Major Physics Expansion
- **4 New Advanced Physics Modules** added to expand educational scope:
  - **Fluid Dynamics Module**: Advanced flow analysis with Reynolds numbers, Bernoulli's equation, viscosity studies
  - **Relativity Module**: Special and general relativity with time dilation, length contraction, spacetime curvature
  - **Nuclear Physics Module**: Radioactive decay, nuclear reactions, binding energy calculations with real isotope data
  - **Astrophysics Module**: Stellar evolution, cosmological calculations, gravitational physics with astronomical data

### Enhanced - Calculator & Converter Expansion
- **Physics Calculator** expanded from 9 to 19 calculation types:
  - New Fluid Dynamics: Reynolds Number, Bernoulli Pressure, Viscous Flow calculations
  - New Relativity: Lorentz Factor, Time Dilation, Relativistic Energy calculations
  - New Nuclear Physics: Binding Energy, Decay Constant, Half-life calculations
  - New Astrophysics: Schwarzschild Radius, Stellar Luminosity, Hubble Distance calculations
- **Unit Converter** expanded from 9 to 21 unit categories:
  - Added Electromagnetic units: Electric Current, Voltage, Resistance, Magnetic Field
  - Added Thermodynamic units: Heat Capacity, Entropy
  - Added Nuclear units: Activity (Becquerel), Absorbed Dose (Gray), Equivalent Dose (Sievert)
  - Added Astrophysical units: Luminosity (Solar), Distance (Astronomical Units, Light-years, Parsecs)

### Features - Advanced Physics Content
- **Fluid Dynamics**: Laminar/turbulent flow visualization, pressure distribution plots, streamline analysis
- **Relativity**: Spacetime diagrams, relativistic velocity addition, gravitational time dilation
- **Nuclear Physics**: Decay chain visualization, nuclear reaction Q-values, radioactive dating calculations
- **Astrophysics**: Hertzsprung-Russell diagrams, cosmic microwave background analysis, dark matter modeling
- All modules include real experimental data and professional matplotlib visualizations
- Interactive parameter controls with advanced physics-specific validation

### Technical Improvements
- Extended `MainApplicationSanitizer` with specialized validation for advanced physics parameters
- Added physics constants for relativity (c, G), nuclear physics (atomic mass unit, binding energies)
- Enhanced error handling for extreme value calculations (relativistic speeds, nuclear energies)
- Improved memory management for complex visualizations and large datasets

## [2.0.1] - 2026-02-04

### Added
- Comprehensive input sanitization system across all application components
- `MainApplicationSanitizer` class with advanced validation methods
- `SecuritySanitizer` class in installer with path and input validation
- `PhysicsSanitizer` class in physics utilities with range checking
- Functional physics calculator dialog with real-time calculations
- Functional unit converter dialog with comprehensive unit support
- Input validation for all numeric parameters (range, type, bounds checking)
- Protection against malicious input (XSS, injection, invalid characters)
- Error handling for edge cases (division by zero, NaN, infinity values)

### Enhanced
- All physics calculation functions now include input sanitization
- Calculator dialog supports 9 different physics calculation types:
  - Kinetic Energy, Potential Energy, Force, Momentum
  - Power, Frequency, Photon Energy, Electric Field, Magnetic Force
- Unit converter supports 9 unit categories with accurate conversions:
  - Length, Mass, Time, Energy, Power, Force, Pressure, Temperature, Frequency
- Gas law calculations with sanitized experimental data validation
- Wave interference simulations with validated frequency and amplitude inputs
- Circuit analysis with protected resistance and voltage calculations
- Quantum energy levels with bounded parameter validation
- Projectile motion with sanitized trajectory parameters

### Security
- Complete input sanitization preventing malicious data injection
- Numeric input bounds checking with configurable ranges
- String sanitization removing dangerous characters and control sequences
- Path validation in installer preventing directory traversal attacks
- Combo box selection validation against approved options only
- File system operations secured with path normalization

### Fixed
- All placeholder implementations replaced with functional code
- Syntax errors resolved across all modules
- Missing method implementations added to main application class
- Input validation gaps closed in physics simulation modules
- Error handling improved for edge cases and invalid inputs

### Technical Improvements
- Consistent sanitization patterns across all user input handling
- Validation error classes with structured error reporting
- Range checking enums for physics parameter boundaries
- Centralized security validation with reusable sanitizer classes
- Memory-safe input handling preventing buffer overflow scenarios
- Performance-optimized validation with minimal computational overhead

## [2.0.0] - 2026-02-03

### Added
- Complete migration from tkinter to PyQt5 for professional GUI experience
- Real experimental data integration across all physics modules:
  - Mechanics: Basketball, cannonball, and baseball trajectory data with air resistance
  - Thermodynamics: Laboratory gas measurements with Van der Waals corrections
  - Waves: Real interference experiments from tuning forks, lasers, and microwaves
  - Electromagnetism: Authentic circuit measurements and component specifications
  - Quantum: Actual spectroscopy data with experimental uncertainty
- Advanced PyQt5 features:
  - Professional dark theme with Linux-style appearance
  - Complete menu system with keyboard shortcuts
  - Real-time status updates and progress indicators
  - Enhanced data export options (PNG, PDF, SVG, CSV, JSON)
- Improved calculation engine with real-world physics effects
- Comprehensive project management with full state persistence
- Advanced unit converter with specialized physics units
- Interactive formula reference with search functionality
- Automated installer script for easy setup
- Cross-platform launcher scripts (Windows .bat and Unix .sh)

### Enhanced
- All physics simulations now use authentic experimental data
- Matplotlib integration with Qt5Agg backend for better performance
- Error handling and input validation across all modules
- Documentation and help system with detailed usage guides
- Configuration system with customizable themes and behavior

### Fixed
- Menu visibility issues from tkinter version
- Calculation accuracy with real experimental data validation
- Cross-platform compatibility improvements
- Memory usage optimization for large datasets

### Technical Improvements
- Modular architecture for easy extension
- Real-time plotting with zoom and pan capabilities
- Statistical analysis tools for experimental data
- Measurement uncertainty modeling
- Publication-ready plot formatting options

## [1.0.0] - 2025-12-15

### Added
- Initial release with tkinter GUI framework
- Basic physics simulation modules:
  - Mechanics with projectile motion
  - Basic thermodynamics calculations
  - Simple wave simulations
  - Basic electromagnetic field visualization
  - Quantum wave function plotting
- Linux-style dark theme
- Basic calculator and unit converter
- Simple project save/load functionality
- Initial help system and menu structure

### Features
- Tabbed interface for different physics areas
- Real-time parameter updates
- Basic matplotlib plotting integration
- Fundamental physics calculations
- File export capabilities
- Keyboard shortcuts support

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

## Categories

- **Added**: New features
- **Enhanced**: Improvements to existing features
- **Fixed**: Bug fixes
- **Technical Improvements**: Code quality, performance, architecture changes
- **Deprecated**: Features marked for removal in future versions
- **Removed**: Features removed in this version
- **Security**: Security-related changes