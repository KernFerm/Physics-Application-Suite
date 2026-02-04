# Physics Application Suite

A comprehensive physics simulation and calculation tool built with PyQt5, featuring a modern Linux-style GUI with real experimental data integration and advanced visualizations.

## Features

### 🚀 Main Physics Modules
- **Mechanics**: Real projectile motion with actual experimental data from basketball, cannonball, and baseball experiments including air resistance effects
- **Thermodynamics**: Laboratory gas measurements with Van der Waals corrections and real-world thermal analysis
- **Waves & Optics**: Authentic wave interference experiments from tuning forks, lasers, and microwaves with damping and noise effects
- **Electromagnetism**: Real circuit measurements and component specifications with actual laboratory data
- **Quantum Physics**: Actual energy level measurements from spectroscopy with real experimental uncertainty

### 🎨 User Interface
- **PyQt5 Professional GUI**: Modern dark theme with Linux-style professional appearance
- **Advanced UI Elements**: Custom styling with hover effects, smooth animations, and responsive design
- **Complete Menu System**: File, Edit, View, Simulation, Tools, and Help menus with full functionality
- **Tabbed Interface**: Seamless navigation between different physics areas with persistent state
- **Real-time Status Bar**: Live updates, calculation progress, and system time display

### 🔧 Built-in Tools
- **Advanced Physics Calculator**: Comprehensive calculations across all physics areas with real-time validation
- **Smart Unit Converter**: Convert between different measurement systems with precision handling
- **Interactive Formula Reference**: Quick access to physics formulas, equations, and constants with search functionality
- **Project Management**: Save/load simulation configurations with full state persistence
- **Data Export**: Export plots, calculations, and experimental data in multiple formats (PNG, PDF, CSV, JSON)
- **Configuration System**: Customizable themes, fonts, and application behavior

### ⌨️ Keyboard Shortcuts
- `Ctrl+N`: New Project
- `Ctrl+O`: Open Project  
- `Ctrl+S`: Save Project
- `Ctrl+E`: Export Current Results
- `Ctrl+Q`: Exit Application
- `F1`: Help Dialog
- `F11`: Toggle Fullscreen
- `Ctrl+Plus/Minus`: Zoom In/Out plots

## Installation

### Prerequisites
- Python 3.7 or later (Python 3.11+ recommended)
- pip (Python package installer)

### Quick Start

#### Windows
```bash
# Double-click run_pyqt5.bat or open command prompt and run:
.\run_pyqt5.bat
```

#### Linux/macOS
```bash
# Make the script executable
chmod +x run_pyqt5.sh

# Run the application
./run_pyqt5.sh
```

#### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

#### Using the Installer
```bash
# Run the automated installer (handles dependencies and setup)
python installer.py
```

## Dependencies

The application requires the following Python packages:
- `PyQt5 >= 5.15.0` - Modern GUI framework with native look and feel
- `matplotlib >= 3.5.0` - Scientific plotting and visualization with Qt5Agg backend
- `numpy >= 1.21.0` - Numerical calculations and array operations
- `pillow >= 8.3.0` - Image processing and format support

## Usage Guide

### Getting Started
1. Launch the application using one of the methods above
2. The main window opens with a tabbed interface showing all physics modules
3. Select the physics area you want to explore from the tabs
4. Use the left control panel to set experimental parameters
5. Click "Run Simulation" or "Calculate" to execute with real experimental data
6. View results in the integrated matplotlib plots with zoom and pan capabilities

### Physics Modules Overview

#### Mechanics Module
- **Real Projectile Data**: Actual experiments including basketball free throws, military cannonball data, and baseball trajectory analysis
- **Air Resistance Modeling**: Realistic drag coefficients and atmospheric conditions
- **Interactive Visualization**: Real-time trajectory plotting with parameter sensitivity analysis
- **Export Capabilities**: Save trajectory data and plots in professional formats

#### Thermodynamics Module  
- **Laboratory Gas Data**: Real measurements from various gases with Van der Waals corrections
- **Experimental Conditions**: Authentic pressure, temperature, and volume relationships
- **Thermal Analysis**: Heat transfer calculations based on actual material properties
- **Phase Diagrams**: Real substance behavior with experimental phase boundaries

#### Waves & Optics Module
- **Interference Experiments**: Real data from tuning fork, laser, and microwave experiments
- **Damping Effects**: Realistic energy dissipation and environmental noise
- **Optical Phenomena**: Authentic measurements from diffraction and interference setups
- **Signal Processing**: Analysis tools for experimental wave data

#### Electromagnetism Module
- **Circuit Analysis**: Real component specifications and measurement data
- **Field Calculations**: Based on actual experimental setups and measurements
- **Laboratory Equipment**: Simulates real oscilloscopes, multimeters, and signal generators
- **Component Database**: Authentic specifications for resistors, capacitors, and inductors

#### Quantum Physics Module
- **Spectroscopy Data**: Real energy level measurements from various atomic species
- **Experimental Uncertainty**: Authentic measurement precision and statistical analysis
- **Quantum Effects**: Based on actual laboratory observations and published research
- **Wave Function Analysis**: Real experimental validation of quantum mechanical predictions

### Project Management
- **New Project**: Start fresh with default parameters and clean workspace
- **Save Project**: Store complete simulation settings, data, and results as JSON
- **Open Project**: Load previously saved configurations with full state restoration
- **Export Results**: Professional output in PNG, PDF, SVG, and data formats (CSV, JSON)
- **Session Management**: Automatic saving of work sessions and crash recovery

### Tools Menu
- **Physics Calculator**: Advanced scientific calculator with physics-specific functions
- **Unit Converter**: Comprehensive conversion between SI, Imperial, and specialized units
- **Formula Reference**: Searchable database of physics formulas with interactive examples
- **Data Analysis**: Statistical tools for experimental data processing
- **Configuration**: Customize appearance, behavior, and default settings

## Technical Features

### Real Experimental Data Integration
- **Authentic Measurements**: All simulations based on real laboratory data and published research
- **Experimental Uncertainty**: Realistic error bars and statistical analysis
- **Noise Modeling**: Environmental effects and measurement limitations accurately modeled
- **Validation**: Cross-referenced with peer-reviewed physics literature

### Advanced Visualization
- **Professional Plots**: High-quality matplotlib graphs with publication-ready formatting
- **Interactive Features**: Zoom, pan, data cursor, and measurement tools
- **Multi-plot Support**: Compare different experiments and parameter variations
- **Animation Capabilities**: Real-time plotting and dynamic parameter updates
- **Export Options**: Vector and raster formats suitable for presentations and publications

### User Experience Design
- **Responsive Interface**: Adapts to different screen sizes and resolutions
- **Intuitive Controls**: Physics-aware input validation and smart defaults
- **Context Help**: Tooltips, status messages, and integrated documentation
- **Error Handling**: Graceful recovery from invalid inputs with helpful guidance
- **Accessibility**: Keyboard navigation and high contrast options

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **Python**: 3.7 or later
- **RAM**: 4 GB (8 GB recommended for large datasets)
- **Storage**: 100 MB free space
- **Display**: 1024x768 (1920x1080 recommended)

### Recommended Setup
- **Python**: 3.11+ for optimal performance
- **RAM**: 8 GB or more for smooth operation with multiple modules
- **Graphics**: Hardware acceleration for matplotlib rendering
- **Network**: Internet connection for initial dependency installation

## File Structure

```
physics-application/
├── main.py                 # Main PyQt5 application entry point
├── physics_utils.py        # Physics calculation utilities and constants
├── requirements.txt        # Python dependencies
├── config.json            # Application configuration and themes
├── installer.py           # Automated installation script
├── run_pyqt5.bat          # Windows launcher script
├── run_pyqt5.sh           # Linux/macOS launcher script
├── README.md              # This documentation file
└── __pycache__/           # Python compiled bytecode (auto-generated)
```

## Contributing

This comprehensive physics application is designed for educational and research purposes. The modular architecture allows for easy extension:

### Adding New Physics Modules
1. Create a new widget class inheriting from `QWidget`
2. Implement the physics calculations using real experimental data
3. Add visualization using the `PhysicsPlotCanvas` class
4. Register the module in the main application tabs

### Enhancing Existing Features  
1. Add new experimental datasets to existing modules
2. Implement additional visualization options
3. Extend the calculator and unit converter functionality
4. Improve the user interface with new themes or layouts

## License

This project is designed as an educational tool for physics simulation and calculation. It integrates real experimental data from various published sources for educational purposes.

## Support & Troubleshooting

### Common Issues
- **PyQt5 Installation**: Use `pip install PyQt5` or the automated installer
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **Display Issues**: Update graphics drivers for optimal matplotlib performance
- **Permission Errors**: Run installer as administrator on Windows

### Getting Help
1. Check the built-in help system (`F1` or Help menu)
2. Review the interactive formula reference for physics equations
3. Use the keyboard shortcuts reference for efficient navigation
4. Consult the physics utilities documentation for calculation details

## Version History

- **v2.0**: Major PyQt5 upgrade with real experimental data integration
  - Migrated from tkinter to PyQt5 for professional GUI
  - Integrated authentic experimental data across all physics modules  
  - Enhanced visualization with professional matplotlib plots
  - Added comprehensive project management and data export
  - Implemented advanced calculator and unit conversion tools
  - Added full keyboard shortcut support and accessibility features

- **v1.0**: Initial release with basic simulation framework
  - Complete mechanics simulation foundation
  - Linux-style dark theme implementation
  - Basic calculation tools and project management
  - Initial help system and menu structure

## Acknowledgments

- Real experimental data sourced from peer-reviewed physics literature
- PyQt5 framework for modern cross-platform GUI development
- Matplotlib community for scientific visualization tools  
- NumPy project for numerical computation foundation
- Physics education community for validation and feedback

---

**Physics Application Suite v2.0** - Real physics, real data, real results! 🔬⚡🌊📊
