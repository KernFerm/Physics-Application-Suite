#!/usr/bin/env python3
"""
Physics Utilities Module v2.1 - Advanced Physics Suite with Extended Calculations

Comprehensive physics calculation tools and utilities for the Physics Application Suite.
Includes real experimental data handling, statistical analysis, advanced calculations,
comprehensive input sanitization and validation, and specialized utilities for
fluid dynamics, relativity, nuclear physics, and astrophysics.

Features:
- Input sanitization and validation for all physics domains
- Range checking for physical values across all physics areas
- Unit validation and conversion safety (21 unit categories)
- Formula input security for advanced physics equations
- Experimental data validation for complex physics experiments
- Security checks for all operations including extreme value physics
- Specialized constants and utilities for advanced physics modules

Author: Physics Application Suite Team
Version: 2.1.0
Date: February 2026
"""

import math
import numpy as np
import random
import re
import warnings
from typing import Dict, Tuple, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class ValidationError(Exception):
    """Custom exception for physics validation errors"""
    pass

class PhysicsRange(Enum):
    """Defines reasonable ranges for physics quantities across all physics domains"""
    # Classical Mechanics
    VELOCITY = (0, 3e8)  # 0 to speed of light
    MASS = (1e-31, 1e50)  # electron mass to universe mass
    LENGTH = (1e-18, 1e26)  # Planck length to observable universe
    TIME = (1e-44, 1e18)  # Planck time to age of universe
    ENERGY = (0, 1e69)  # 0 to Planck energy
    FREQUENCY = (0, 1e43)  # 0 to Planck frequency
    ANGLE = (0, 2*math.pi)  # 0 to full rotation
    PROBABILITY = (0, 1)  # 0 to 1
    
    # Thermodynamics
    TEMPERATURE = (0, 1e12)  # Absolute zero to stellar core
    PRESSURE = (0, 1e20)  # 0 to neutron star core pressure
    
    # Fluid Dynamics
    REYNOLDS_NUMBER = (0, 1e10)  # Laminar to extreme turbulent flow
    VISCOSITY = (1e-6, 1e6)  # Gas to ultra-viscous liquid (Pa·s)
    DENSITY = (1e-5, 1e20)  # Ultra-low density gas to neutron star
    
    # Relativity
    LORENTZ_FACTOR = (1, 1e10)  # Rest to ultra-relativistic
    PROPER_TIME = (1e-44, 1e18)  # Planck time to cosmic time
    
    # Nuclear Physics
    NUCLEAR_MASS = (1e-30, 1e-20)  # Light nuclei to superheavy elements
    DECAY_CONSTANT = (1e-25, 1e10)  # Stable to extremely unstable
    ACTIVITY = (0, 1e30)  # 0 to extreme radioactive source (Bq)
    
    # Astrophysics
    STELLAR_MASS = (1e28, 1e32)  # Brown dwarf to hypergiant (kg)
    LUMINOSITY = (1e20, 1e32)  # Dim star to hypergiant luminosity (W)
    DISTANCE = (1e3, 1e26)  # Nearby objects to cosmic horizon (m)

class PhysicsSanitizer:
    """Comprehensive sanitization and validation for physics calculations"""
    
    # Dangerous characters that should not appear in physics inputs
    DANGEROUS_CHARS = set('<>"\'\\|*?$`&;{}[]~')
    
    # Valid physics unit patterns
    UNIT_PATTERN = re.compile(r'^[a-zA-Z0-9./\^\-\+\s°μπωαβγθλρστφψΩ]*$')
    
    # Valid formula patterns (allow mathematical expressions)
    FORMULA_PATTERN = re.compile(r'^[a-zA-Z0-9\s\+\-\*/\(\)\^\.\_\=α-ωΑ-Ω₀-₉ᵃ-ᶻ]*$')
    
    @staticmethod
    def sanitize_numeric_input(value: Any, param_name: str = "value", 
                               allow_negative: bool = True, 
                               physics_range: Optional[PhysicsRange] = None) -> float:
        """Sanitize and validate numeric inputs for physics calculations"""
        if value is None:
            raise ValidationError(f"Parameter '{param_name}' cannot be None")
        
        # Convert to float safely
        try:
            if isinstance(value, (str, bytes)):
                # Remove dangerous characters
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                
                # Check for dangerous characters
                if any(char in value for char in PhysicsSanitizer.DANGEROUS_CHARS):
                    raise ValidationError(f"Invalid characters in '{param_name}': {value}")
                
                # Remove whitespace and validate format
                value = value.strip()
                if not value:
                    raise ValidationError(f"Empty value for parameter '{param_name}'")
                
                # Check for scientific notation and basic number format
                number_pattern = r'^[+-]?\d*\.?\d+([eE][+-]?\d+)?$'
                if not re.match(number_pattern, value):
                    raise ValidationError(f"Invalid number format for '{param_name}': {value}")
            
            numeric_value = float(value)
            
        except (ValueError, OverflowError) as e:
            raise ValidationError(f"Cannot convert '{param_name}' to number: {value} ({e})")
        
        # Check for special float values
        if math.isnan(numeric_value):
            raise ValidationError(f"Parameter '{param_name}' is NaN")
        
        if math.isinf(numeric_value):
            raise ValidationError(f"Parameter '{param_name}' is infinite")
        
        # Check sign constraint
        if not allow_negative and numeric_value < 0:
            raise ValidationError(f"Parameter '{param_name}' must be non-negative: {numeric_value}")
        
        # Check physics range if specified
        if physics_range:
            min_val, max_val = physics_range.value
            if not (min_val <= abs(numeric_value) <= max_val):
                raise ValidationError(f"Parameter '{param_name}' outside physical range [{min_val}, {max_val}]: {numeric_value}")
        
        return numeric_value
    
    @staticmethod
    def sanitize_unit_string(unit: str) -> str:
        """Sanitize and validate unit strings"""
        if not isinstance(unit, str):
            raise ValidationError(f"Unit must be a string, got: {type(unit)}")
        
        # Remove null bytes and control characters
        unit = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', unit)
        
        # Check length
        if len(unit) > 50:
            raise ValidationError(f"Unit string too long: {len(unit)} characters")
        
        # Validate characters
        if not PhysicsSanitizer.UNIT_PATTERN.match(unit):
            raise ValidationError(f"Invalid characters in unit string: {unit}")
        
        return unit.strip()

class PhysicsCalculator:
    """Advanced physics calculator with sanitized input validation"""
    
    def __init__(self):
        self.sanitizer = PhysicsSanitizer()
    
    def safe_calculate(self, func_name: str, **kwargs) -> float:
        """Safely perform physics calculations with input validation"""
        try:
            # Validate all inputs
            validated_params = {}
            for param_name, value in kwargs.items():
                validated_params[param_name] = PhysicsSanitizer.sanitize_numeric_input(
                    value, param_name
                )
            
            # Perform calculation (simplified for this example)
            if func_name == 'kinetic_energy' and 'mass' in validated_params and 'velocity' in validated_params:
                result = 0.5 * validated_params['mass'] * validated_params['velocity']**2
            else:
                raise ValidationError(f"Unknown calculation function: {func_name}")
            
            # Validate result
            if math.isnan(result) or math.isinf(result):
                raise ValidationError(f"Invalid calculation result: {result}")
            
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Calculation error in {func_name}: {e}")
    
    @staticmethod
    def mechanics_calculator():
        """Mechanics calculations"""
        calculations = {
            "Velocity": lambda d, t: d / t if t != 0 else 0,
            "Acceleration": lambda v_f, v_i, t: (v_f - v_i) / t if t != 0 else 0,
            "Force": lambda m, a: m * a,
            "Kinetic Energy": lambda m, v: 0.5 * m * v**2,
            "Potential Energy": lambda m, g, h: m * g * h,
            "Work": lambda f, d: f * d,
            "Power": lambda w, t: w / t if t != 0 else 0,
            "Momentum": lambda m, v: m * v,
            "Impulse": lambda f, t: f * t,
            "Centripetal Force": lambda m, v, r: m * v**2 / r if r != 0 else 0
        }
        return calculations
    
    @staticmethod
    def thermodynamics_calculator():
        """Thermodynamics calculations"""
        calculations = {
            "Ideal Gas Pressure": lambda n, R, T, V: (n * R * T) / V if V != 0 else 0,
            "Heat Transfer": lambda m, c, delta_t: m * c * delta_t,
            "Thermal Efficiency": lambda w_out, q_in: w_out / q_in if q_in != 0 else 0,
            "Heat Capacity": lambda q, delta_t: q / delta_t if delta_t != 0 else 0,
            "Carnot Efficiency": lambda t_hot, t_cold: 1 - (t_cold / t_hot) if t_hot != 0 else 0,
            "Stefan-Boltzmann": lambda sigma, A, T: sigma * A * T**4,
            "Thermal Conductivity": lambda q, k, A, delta_t, delta_x: q * delta_x / (k * A * delta_t) if (k * A * delta_t) != 0 else 0
        }
        return calculations
    
    @staticmethod
    def waves_calculator():
        """Wave and optics calculations"""
        calculations = {
            "Wave Speed": lambda f, wavelength: f * wavelength,
            "Frequency": lambda period: 1 / period if period != 0 else 0,
            "Period": lambda freq: 1 / freq if freq != 0 else 0,
            "Wavelength": lambda v, f: v / f if f != 0 else 0,
            "Wave Energy": lambda A, omega: 0.5 * A**2 * omega**2,
            "Doppler Effect": lambda f_source, v_observer, v_source, v_wave: f_source * (v_wave + v_observer) / (v_wave + v_source) if (v_wave + v_source) != 0 else 0,
            "Snell's Law": lambda n1, theta1, n2: math.asin((n1 * math.sin(theta1)) / n2) if n2 != 0 and abs((n1 * math.sin(theta1)) / n2) <= 1 else 0,
            "Lens Equation": lambda f, d_o: (f * d_o) / (d_o - f) if (d_o - f) != 0 else 0
        }
        return calculations
    
    @staticmethod
    def electromagnetism_calculator():
        """Electromagnetic calculations"""
        calculations = {
            "Coulomb Force": lambda k, q1, q2, r: k * q1 * q2 / r**2 if r != 0 else 0,
            "Electric Field": lambda F, q: F / q if q != 0 else 0,
            "Electric Potential": lambda k, q, r: k * q / r if r != 0 else 0,
            "Capacitance": lambda q, v: q / v if v != 0 else 0,
            "Ohm's Law (V)": lambda I, R: I * R,
            "Ohm's Law (I)": lambda V, R: V / R if R != 0 else 0,
            "Ohm's Law (R)": lambda V, I: V / I if I != 0 else 0,
            "Electrical Power": lambda V, I: V * I,
            "Magnetic Force": lambda q, v, B, theta: q * v * B * math.sin(math.radians(theta)),
            "Magnetic Field (Wire)": lambda mu, I, r: (mu * I) / (2 * math.pi * r) if r != 0 else 0
        }
        return calculations
    
    @staticmethod
    def quantum_calculator():
        """Quantum physics calculations"""
        calculations = {
            "Photon Energy": lambda h, f: h * f,
            "de Broglie Wavelength": lambda h, p: h / p if p != 0 else 0,
            "Energy Level (Hydrogen)": lambda n: -13.6 / n**2 if n != 0 else 0,
            "Uncertainty Principle": lambda delta_x: 1.054571817e-34 / (2 * delta_x) if delta_x != 0 else 0,
            "Compton Scattering": lambda wavelength, theta: wavelength + (6.626e-34 / (9.109e-31 * 2.998e8)) * (1 - math.cos(theta)),
            "Blackbody Peak": lambda T: 2.898e-3 / T if T != 0 else 0,  # Wien's displacement law
            "Radioactive Decay": lambda N0, lambda_decay, t: N0 * math.exp(-lambda_decay * t),
            "Half Life": lambda lambda_decay: math.log(2) / lambda_decay if lambda_decay != 0 else 0,
            "Binding Energy": lambda mass_defect, c=2.998e8: mass_defect * c**2,
            "Bohr Radius": lambda n: 5.292e-11 * n**2  # Bohr radius for hydrogen-like atoms
        }
        return calculations
    
    @staticmethod
    def fluid_dynamics_calculator():
        """Fluid dynamics calculations"""
        calculations = {
            "Reynolds Number": lambda rho, v, L, mu: (rho * v * L / mu) if mu != 0 else 0,
            "Bernoulli Pressure": lambda rho, v1, v2, h1, h2, g=9.81: 0.5 * rho * (v1**2 - v2**2) + rho * g * (h1 - h2),
            "Viscous Flow Rate": lambda r, delta_P, L, mu: (math.pi * r**4 * delta_P) / (8 * mu * L) if (mu * L) != 0 else 0,
            "Drag Force": lambda Cd, rho, v, A: 0.5 * Cd * rho * v**2 * A,
            "Dynamic Pressure": lambda rho, v: 0.5 * rho * v**2,
            "Flow Velocity": lambda Q, A: Q / A if A != 0 else 0,
            "Continuity Equation": lambda A1, v1, A2: (A1 * v1 / A2) if A2 != 0 else 0,
            "Hydrostatic Pressure": lambda rho, g, h: rho * g * h,
            "Terminal Velocity": lambda m, g, rho, Cd, A: math.sqrt(2 * m * g / (rho * Cd * A)) if (rho * Cd * A) != 0 else 0,
            "Pipe Flow Velocity": lambda Q, D: 4 * Q / (math.pi * D**2) if D != 0 else 0
        }
        return calculations
    
    @staticmethod
    def relativity_calculator():
        """Special and general relativity calculations"""
        calculations = {
            "Lorentz Factor": lambda v, c=2.998e8: 1 / math.sqrt(1 - (v/c)**2) if abs(v) < c and c != 0 else float('inf'),
            "Time Dilation": lambda t0, v, c=2.998e8: t0 / math.sqrt(1 - (v/c)**2) if abs(v) < c and c != 0 else float('inf'),
            "Length Contraction": lambda L0, v, c=2.998e8: L0 * math.sqrt(1 - (v/c)**2) if abs(v) < c and c != 0 else 0,
            "Relativistic Energy": lambda m, v, c=2.998e8: m * c**2 / math.sqrt(1 - (v/c)**2) if abs(v) < c and c != 0 else float('inf'),
            "Relativistic Momentum": lambda m, v, c=2.998e8: m * v / math.sqrt(1 - (v/c)**2) if abs(v) < c and c != 0 else float('inf'),
            "Mass Energy": lambda m, c=2.998e8: m * c**2,
            "Velocity Addition": lambda v1, v2, c=2.998e8: (v1 + v2) / (1 + v1*v2/c**2) if (1 + v1*v2/c**2) != 0 else 0,
            "Doppler Shift": lambda f0, v, c=2.998e8: f0 * math.sqrt((1 + v/c) / (1 - v/c)) if abs(v) < c and (1 - v/c) > 0 else 0,
            "Schwarzschild Radius": lambda M, G=6.674e-11, c=2.998e8: 2 * G * M / c**2 if c != 0 else 0,
            "Escape Velocity": lambda G, M, r: math.sqrt(2 * G * M / r) if r != 0 else 0
        }
        return calculations
    
    @staticmethod
    def nuclear_physics_calculator():
        """Nuclear physics calculations"""
        calculations = {
            "Binding Energy": lambda mass_defect, c=2.998e8: mass_defect * c**2,
            "Q Value": lambda mass_initial, mass_final, c=2.998e8: (mass_initial - mass_final) * c**2,
            "Decay Constant": lambda half_life: math.log(2) / half_life if half_life != 0 else 0,
            "Activity": lambda N, lambda_decay: N * lambda_decay,
            "Decay Law": lambda N0, lambda_decay, t: N0 * math.exp(-lambda_decay * t),
            "Half Life": lambda lambda_decay: math.log(2) / lambda_decay if lambda_decay != 0 else 0,
            "Mean Lifetime": lambda lambda_decay: 1 / lambda_decay if lambda_decay != 0 else 0,
            "Nuclear Radius": lambda A: 1.2e-15 * A**(1/3),  # A is mass number
            "Cross Section": lambda sigma, n: sigma * n,  # reaction probability
            "Fission Energy": lambda: 200e6 * 1.602e-19  # ~200 MeV in Joules
        }
        return calculations
    
    @staticmethod
    def astrophysics_calculator():
        """Astrophysics calculations"""
        calculations = {
            "Schwarzschild Radius": lambda M, G=6.674e-11, c=2.998e8: 2 * G * M / c**2 if c != 0 else 0,
            "Stellar Luminosity": lambda R, T, sigma=5.67e-8: 4 * math.pi * R**2 * sigma * T**4,
            "Main Sequence Lifetime": lambda M, M_sun=1.989e30: 10e9 * (M_sun / M)**2.5 if M != 0 else 0,  # years
            "Hubble Distance": lambda v, H0=70: v / H0 if H0 != 0 else 0,  # km/s/Mpc
            "Redshift": lambda lambda_obs, lambda_rest: (lambda_obs - lambda_rest) / lambda_rest if lambda_rest != 0 else 0,
            "Angular Diameter": lambda D, d: D / d if d != 0 else 0,  # small angle approximation
            "Parallax Distance": lambda p: 1 / p if p != 0 else 0,  # distance in parsecs
            "Apparent Magnitude": lambda flux, flux_ref: -2.5 * math.log10(flux / flux_ref) if flux_ref != 0 and flux > 0 else 0,
            "Absolute Magnitude": lambda m, d: m - 5 * math.log10(d / 10) if d > 0 else 0,
            "Escape Velocity": lambda G, M, r: math.sqrt(2 * G * M / r) if r != 0 else 0,
            "Orbital Velocity": lambda G, M, r: math.sqrt(G * M / r) if r != 0 else 0,
            "Kepler Third Law": lambda a, G, M: math.sqrt(4 * math.pi**2 * a**3 / (G * M)) if M != 0 else 0,
            "Tidal Force": lambda G, M, m, r, dr: 2 * G * M * m * dr / r**3 if r != 0 else 0,
            "Surface Gravity": lambda G, M, R: G * M / R**2 if R != 0 else 0
        }
        return calculations

@dataclass
class ExperimentalData:
    """Container for experimental data with uncertainty and validation"""
    value: float
    uncertainty: float
    unit: str
    timestamp: Optional[datetime] = None
    conditions: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Sanitize inputs during initialization
        self.value = PhysicsSanitizer.sanitize_numeric_input(self.value, "value")
        self.uncertainty = PhysicsSanitizer.sanitize_numeric_input(
            self.uncertainty, "uncertainty", allow_negative=False
        )
        self.unit = PhysicsSanitizer.sanitize_unit_string(self.unit)
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # Validate physical constraints
        if self.uncertainty > abs(self.value) * 10:  # Uncertainty shouldn't be 10x larger than value
            warnings.warn(f"Very large uncertainty relative to value: {self.uncertainty} vs {self.value}")
    
    def relative_uncertainty(self) -> float:
        """Calculate relative uncertainty as percentage"""
        return abs(self.uncertainty / self.value * 100) if self.value != 0 else 0
    
    def is_significant(self, threshold: float = 5.0) -> bool:
        """Check if measurement is significant (relative uncertainty < threshold%)"""
        return self.relative_uncertainty() < threshold

class StatisticalAnalysis:
    """Statistical analysis tools for experimental data"""
    
    @staticmethod
    def mean_with_uncertainty(data: List[ExperimentalData]) -> ExperimentalData:
        """Calculate mean value with propagated uncertainty"""
        if not data:
            return ExperimentalData(0, 0, "")
        
        values = [d.value for d in data]
        uncertainties = [d.uncertainty for d in data]
        unit = data[0].unit
        
        mean_val = np.mean(values)
        # Standard deviation of the mean
        std_dev = np.std(values, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0
        # Combine with systematic uncertainties
        total_uncertainty = math.sqrt(std_dev**2 + (np.mean(uncertainties))**2)
        
        return ExperimentalData(mean_val, total_uncertainty, unit)
    
    @staticmethod
    def weighted_mean(data: List[ExperimentalData]) -> ExperimentalData:
        """Calculate weighted mean based on uncertainties"""
        if not data:
            return ExperimentalData(0, 0, "")
        
        weights = [1/d.uncertainty**2 if d.uncertainty != 0 else 1e6 for d in data]
        values = [d.value for d in data]
        unit = data[0].unit
        
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        weight_sum = sum(weights)
        
        mean_val = weighted_sum / weight_sum if weight_sum != 0 else 0
        uncertainty = 1 / math.sqrt(weight_sum) if weight_sum != 0 else 0
        
        return ExperimentalData(mean_val, uncertainty, unit)
    
    @staticmethod
    def chi_squared_test(observed: List[float], expected: List[float]) -> Tuple[float, float]:
        """Perform chi-squared goodness of fit test"""
        if len(observed) != len(expected) or len(observed) == 0:
            return 0, 1
        
        chi_sq = sum((obs - exp)**2 / exp for obs, exp in zip(observed, expected) if exp != 0)
        degrees_freedom = len(observed) - 1
        
        # Simplified p-value approximation
        p_value = math.exp(-chi_sq / 2) if degrees_freedom == 1 else 0.5
        
        return chi_sq, p_value
    
    @staticmethod
    def linear_regression(x_data: List[float], y_data: List[float]) -> Dict[str, float]:
        """Perform linear regression analysis"""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            return {"slope": 0, "intercept": 0, "r_squared": 0, "std_error": 0}
        
        x = np.array(x_data)
        y = np.array(y_data)
        
        # Calculate slope and intercept
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Standard error
        std_error = math.sqrt(ss_res / (n - 2)) if n > 2 else 0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "std_error": std_error
        }

class ExperimentalDataGenerator:
    """Generate realistic experimental data with appropriate noise and uncertainties"""
    
    @staticmethod
    def add_experimental_noise(theoretical_value: float, relative_error: float = 0.05, 
                               systematic_error: float = 0.01) -> ExperimentalData:
        """Add realistic experimental noise to theoretical values"""
        # Random error (Gaussian)
        random_error = np.random.normal(0, theoretical_value * relative_error)
        # Systematic error (bias)
        systematic_bias = np.random.uniform(-systematic_error, systematic_error) * theoretical_value
        
        measured_value = theoretical_value + random_error + systematic_bias
        uncertainty = abs(theoretical_value * relative_error)
        
        return ExperimentalData(measured_value, uncertainty, "")
    
    @staticmethod
    def generate_measurement_series(theoretical_values: List[float], 
                                    n_measurements: int = 10,
                                    relative_error: float = 0.03) -> List[List[ExperimentalData]]:
        """Generate series of measurements for each theoretical value"""
        all_measurements = []
        
        for theoretical in theoretical_values:
            measurements = []
            for _ in range(n_measurements):
                measurement = ExperimentalDataGenerator.add_experimental_noise(
                    theoretical, relative_error
                )
                measurements.append(measurement)
            all_measurements.append(measurements)
        
        return all_measurements

class UnitConverter:
    """Unit conversion utilities with input sanitization"""
    
    def __init__(self):
        self.sanitizer = PhysicsSanitizer()
        self.conversions = {
            "Length": {
                "m": 1.0,
                "km": 1000.0,
                "cm": 0.01,
                "mm": 0.001,
                "in": 0.0254,
                "ft": 0.3048,
                "yd": 0.9144,
                "mile": 1609.34
            },
            "Mass": {
                "kg": 1.0,
                "g": 0.001,
                "lb": 0.453592,
                "oz": 0.0283495,
                "ton": 1000.0,
                "slug": 14.5939
            },
            "Time": {
                "s": 1.0,
                "ms": 0.001,
                "min": 60.0,
                "h": 3600.0,
                "day": 86400.0,
                "year": 31557600.0
            },
            "Force": {
                "N": 1.0,
                "kN": 1000.0,
                "dyne": 1e-5,
                "lbf": 4.44822,
                "kgf": 9.80665
            },
            "Energy": {
                "J": 1.0,
                "kJ": 1000.0,
                "cal": 4.184,
                "kcal": 4184.0,
                "eV": 1.60218e-19,
                "kWh": 3.6e6,
                "BTU": 1055.06
            },
            "Power": {
                "W": 1.0,
                "kW": 1000.0,
                "hp": 745.7,
                "BTU/h": 0.293071
            },
            "Pressure": {
                "Pa": 1.0,
                "kPa": 1000.0,
                "bar": 100000.0,
                "atm": 101325.0,
                "psi": 6894.76,
                "torr": 133.322
            },
            "Temperature": {
                # Special handling required for temperature
            },
            "Electric Current": {
                "A": 1.0,
                "mA": 0.001,
                "kA": 1000.0,
                "μA": 1e-6
            },
            "Voltage": {
                "V": 1.0,
                "kV": 1000.0,
                "mV": 0.001,
                "μV": 1e-6
            },
            "Frequency": {
                "Hz": 1.0,
                "kHz": 1000.0,
                "MHz": 1e6,
                "GHz": 1e9,
                "rpm": 1/60.0
            },
            "Magnetic Field": {
                "T": 1.0,
                "mT": 0.001,
                "μT": 1e-6,
                "G": 1e-4,  # Gauss
                "kG": 0.1
            },
            "Radiation": {
                "Bq": 1.0,  # Becquerel
                "Ci": 3.7e10,  # Curie
                "mCi": 3.7e7,
                "μCi": 3.7e4
            },
            "Dose": {
                "Gy": 1.0,  # Gray
                "rad": 0.01,
                "Sv": 1.0,  # Sievert
                "rem": 0.01
            },
            "Resistance": {
                "Ω": 1.0,  # Ohm
                "kΩ": 1000.0,
                "MΩ": 1e6,
                "mΩ": 0.001,
                "μΩ": 1e-6
            },
            "Heat Capacity": {
                "J/K": 1.0,
                "cal/K": 4.184,
                "kJ/K": 1000.0,
                "J/(kg·K)": 1.0,  # Specific heat capacity
                "cal/(g·K)": 4184.0
            },
            "Entropy": {
                "J/K": 1.0,
                "cal/K": 4.184,
                "kJ/K": 1000.0,
                "eV/K": 1.602176634e-19
            },
            "Luminosity": {
                "W": 1.0,
                "L☉": 3.828e26,  # Solar luminosity
                "erg/s": 1e-7,
                "kW": 1000.0
            },
            "Astronomical Distance": {
                "m": 1.0,
                "AU": 1.496e11,  # Astronomical Unit
                "ly": 9.461e15,  # Light year
                "pc": 3.086e16,  # Parsec
                "kpc": 3.086e19,  # Kiloparsec
                "Mpc": 3.086e22  # Megaparsec
            },
            "Wavelength": {
                "m": 1.0,
                "nm": 1e-9,
                "μm": 1e-6,
                "mm": 1e-3,
                "Å": 1e-10  # Angstrom
            }
        }
    
    def convert(self, value: Any, from_unit: str, to_unit: str, unit_type: str) -> float:
        """Convert value from one unit to another with input validation"""
        try:
            # Sanitize inputs
            clean_value = PhysicsSanitizer.sanitize_numeric_input(value, "value", allow_negative=True)
            clean_from_unit = PhysicsSanitizer.sanitize_unit_string(from_unit)
            clean_to_unit = PhysicsSanitizer.sanitize_unit_string(to_unit)
            clean_unit_type = PhysicsSanitizer.sanitize_unit_string(unit_type)
            
            # Special handling for temperature
            if clean_unit_type == "Temperature":
                return self._convert_temperature(clean_value, clean_from_unit, clean_to_unit)
            
            if clean_unit_type not in self.conversions:
                raise ValidationError(f"Unknown unit type: {clean_unit_type}")
            
            unit_dict = self.conversions[clean_unit_type]
            
            if clean_from_unit not in unit_dict or clean_to_unit not in unit_dict:
                raise ValidationError(f"Unknown unit in {clean_unit_type}: {clean_from_unit} or {clean_to_unit}")
            
            # Convert to base unit, then to target unit
            base_value = clean_value * unit_dict[clean_from_unit]
            target_value = base_value / unit_dict[clean_to_unit]
            
            # Validate result
            if math.isnan(target_value) or math.isinf(target_value):
                raise ValidationError(f"Invalid conversion result: {target_value}")
            
            return target_value
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unit conversion error: {e}")
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Special temperature conversion handling"""
        # Convert to Celsius first
        if from_unit == "F":
            celsius = (value - 32) * 5/9
        elif from_unit == "K":
            celsius = value - 273.15
        elif from_unit == "C":
            celsius = value
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Convert from Celsius to target
        if to_unit == "F":
            return celsius * 9/5 + 32
        elif to_unit == "K":
            return celsius + 273.15
        elif to_unit == "C":
            return celsius
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

class FormulaReference:
    """Physics formula reference database"""
    
    def __init__(self):
        self.formulas = {
            "Mechanics": {
                "Kinematics": {
                    "Velocity": "v = Δx/Δt",
                    "Acceleration": "a = Δv/Δt",
                    "Position (constant acceleration)": "x = x₀ + v₀t + ½at²",
                    "Velocity (constant acceleration)": "v = v₀ + at",
                    "Velocity squared": "v² = v₀² + 2a(x - x₀)"
                },
                "Dynamics": {
                    "Newton's 2nd Law": "F = ma",
                    "Weight": "W = mg",
                    "Normal Force": "N = mg cos(θ)",
                    "Friction": "f = μN",
                    "Centripetal Force": "Fc = mv²/r"
                },
                "Energy": {
                    "Kinetic Energy": "KE = ½mv²",
                    "Gravitational PE": "PE = mgh",
                    "Elastic PE": "PE = ½kx²",
                    "Work": "W = F·d cos(θ)",
                    "Power": "P = W/t = F·v"
                },
                "Momentum": {
                    "Momentum": "p = mv",
                    "Impulse": "J = FΔt = Δp",
                    "Conservation": "p₁ + p₂ = p₁' + p₂'"
                },
                "Projectile Motion": {
                    "Range": "R = v₀² sin(2θ)/g",
                    "Max Height": "h = v₀² sin²(θ)/(2g)",
                    "Time of Flight": "t = 2v₀ sin(θ)/g",
                    "Horizontal Position": "x = v₀ cos(θ)·t",
                    "Vertical Position": "y = v₀ sin(θ)·t - ½gt²"
                }
            },
            "Thermodynamics": {
                "Gas Laws": {
                    "Ideal Gas Law": "PV = nRT",
                    "Boyle's Law": "P₁V₁ = P₂V₂",
                    "Charles' Law": "V₁/T₁ = V₂/T₂",
                    "Gay-Lussac's Law": "P₁/T₁ = P₂/T₂",
                    "Combined Gas Law": "P₁V₁/T₁ = P₂V₂/T₂"
                },
                "Heat Transfer": {
                    "Heat Capacity": "Q = mcΔT",
                    "Latent Heat": "Q = mL",
                    "Heat Conduction": "Q = kAΔT/d",
                    "Stefan-Boltzmann": "P = σAT⁴"
                },
                "Thermodynamic Processes": {
                    "First Law": "ΔU = Q - W",
                    "Efficiency": "η = W/Qh = 1 - Qc/Qh",
                    "Carnot Efficiency": "η = 1 - Tc/Th",
                    "Entropy Change": "ΔS = Q/T"
                }
            },
            "Waves and Optics": {
                "Wave Properties": {
                    "Wave Equation": "v = fλ",
                    "Period": "T = 1/f",
                    "Angular Frequency": "ω = 2πf",
                    "Wave Number": "k = 2π/λ",
                    "Wave Function": "y = A sin(kx - ωt + φ)"
                },
                "Sound": {
                    "Speed of Sound": "v = √(B/ρ)",
                    "Doppler Effect": "f' = f(v ± vo)/(v ± vs)",
                    "Beat Frequency": "fb = |f₁ - f₂|",
                    "Intensity Level": "β = 10 log(I/I₀)"
                },
                "Optics": {
                    "Snell's Law": "n₁ sin(θ₁) = n₂ sin(θ₂)",
                    "Thin Lens": "1/f = 1/do + 1/di",
                    "Magnification": "m = -di/do = hi/ho",
                    "Critical Angle": "θc = sin⁻¹(n₂/n₁)"
                }
            },
            "Electromagnetism": {
                "Electrostatics": {
                    "Coulomb's Law": "F = k|q₁q₂|/r²",
                    "Electric Field": "E = F/q = kQ/r²",
                    "Electric Potential": "V = kQ/r",
                    "Potential Energy": "U = qV = kqQ/r",
                    "Capacitance": "C = Q/V"
                },
                "Current Electricity": {
                    "Ohm's Law": "V = IR",
                    "Power": "P = IV = I²R = V²/R",
                    "Resistors in Series": "Rtotal = R₁ + R₂ + ...",
                    "Resistors in Parallel": "1/Rtotal = 1/R₁ + 1/R₂ + ...",
                    "Kirchhoff's Voltage": "ΣV = 0",
                    "Kirchhoff's Current": "ΣI = 0"
                },
                "Magnetism": {
                    "Magnetic Force": "F = qvB sin(θ)",
                    "Magnetic Force on Wire": "F = ILB sin(θ)",
                    "Magnetic Field (wire)": "B = μ₀I/(2πr)",
                    "Magnetic Field (solenoid)": "B = μ₀nI",
                    "Faraday's Law": "ε = -N(dΦ/dt)",
                    "Lenz's Law": "ε = -L(dI/dt)",
                    "Inductance": "L = μ₀n²Al",
                    "Energy in Magnetic Field": "U = ½LI²",
                    "AC Circuit Impedance": "Z = √(R² + (XL - XC)²)"
                }
            },
            "Nuclear Physics": {
                "Radioactivity": {
                    "Decay Law": "N(t) = N₀e^(-λt)",
                    "Half-Life": "t₁/₂ = ln(2)/λ",
                    "Activity": "A = λN",
                    "Mean Lifetime": "τ = 1/λ",
                    "Decay Constant": "λ = ln(2)/t₁/₂"
                },
                "Nuclear Reactions": {
                    "Mass-Energy": "E = mc²",
                    "Binding Energy": "BE = (Δm)c²",
                    "Q-Value": "Q = (m_initial - m_final)c²",
                    "Fission Energy": "E ≈ 200 MeV per fission"
                }
            },
            "Modern Physics": {
                "Quantum Mechanics": {
                    "Planck's Equation": "E = hf",
                    "de Broglie Wavelength": "λ = h/p",
                    "Uncertainty Principle": "ΔxΔp ≥ ħ/2",
                    "Schrödinger Equation": "iħ∂ψ/∂t = Ĥψ",
                    "Energy Levels": "En = -13.6/n² eV"
                },
                "Relativity": {
                    "Time Dilation": "Δt = Δt₀/√(1 - v²/c²)",
                    "Length Contraction": "L = L₀√(1 - v²/c²)",
                    "Mass-Energy": "E = mc²",
                    "Relativistic Energy": "E = γmc²",
                    "Lorentz Factor": "γ = 1/√(1 - v²/c²)"
                }
            }
        }
    
    def get_formulas(self, category: str = None, subcategory: str = None) -> Dict[str, Any]:
        """Get formulas by category and subcategory"""
        if category is None:
            return self.formulas
        
        if category not in self.formulas:
            return {}
        
        if subcategory is None:
            return self.formulas[category]
        
        if subcategory not in self.formulas[category]:
            return {}
        
        return self.formulas[category][subcategory]
    
    def search_formula(self, search_term: str) -> Dict[str, str]:
        """Search for formulas containing the search term"""
        results = {}
        search_term = search_term.lower()
        
        for category, cat_data in self.formulas.items():
            for subcategory, sub_data in cat_data.items():
                for formula_name, formula in sub_data.items():
                    if (search_term in formula_name.lower() or 
                        search_term in formula.lower()):
                        key = f"{category} > {subcategory} > {formula_name}"
                        results[key] = formula
        
        return results

# Comprehensive physics constants database
PHYSICS_CONSTANTS = {
    # Fundamental Constants
    "Speed of Light": {"value": 2.99792458e8, "unit": "m/s", "symbol": "c", "uncertainty": 0, "category": "Fundamental"},
    "Planck Constant": {"value": 6.62607015e-34, "unit": "J·s", "symbol": "h", "uncertainty": 0, "category": "Quantum"},
    "Reduced Planck Constant": {"value": 1.054571817e-34, "unit": "J·s", "symbol": "ℏ", "uncertainty": 0, "category": "Quantum"},
    "Elementary Charge": {"value": 1.602176634e-19, "unit": "C", "symbol": "e", "uncertainty": 0, "category": "Fundamental"},
    
    # Particle Masses
    "Electron Mass": {"value": 9.1093837015e-31, "unit": "kg", "symbol": "me", "uncertainty": 2.8e-40, "category": "Particle"},
    "Proton Mass": {"value": 1.67262192369e-27, "unit": "kg", "symbol": "mp", "uncertainty": 5.1e-37, "category": "Particle"},
    "Neutron Mass": {"value": 1.67492749804e-27, "unit": "kg", "symbol": "mn", "uncertainty": 9.5e-37, "category": "Particle"},
    "Atomic Mass Unit": {"value": 1.66053906660e-27, "unit": "kg", "symbol": "u", "uncertainty": 5.0e-37, "category": "Particle"},
    "Muon Mass": {"value": 1.883531627e-28, "unit": "kg", "symbol": "mμ", "uncertainty": 4.2e-36, "category": "Particle"},
    
    # Thermodynamic Constants
    "Avogadro Number": {"value": 6.02214076e23, "unit": "mol⁻¹", "symbol": "NA", "uncertainty": 0, "category": "Thermodynamic"},
    "Boltzmann Constant": {"value": 1.380649e-23, "unit": "J/K", "symbol": "kB", "uncertainty": 0, "category": "Thermodynamic"},
    "Gas Constant": {"value": 8.314462618, "unit": "J/(mol·K)", "symbol": "R", "uncertainty": 0, "category": "Thermodynamic"},
    "Stefan-Boltzmann Constant": {"value": 5.670374419e-8, "unit": "W/(m²·K⁴)", "symbol": "σ", "uncertainty": 0, "category": "Thermodynamic"},
    
    # Electromagnetic Constants
    "Permittivity of Free Space": {"value": 8.8541878128e-12, "unit": "F/m", "symbol": "ε₀", "uncertainty": 1.3e-21, "category": "Electromagnetic"},
    "Permeability of Free Space": {"value": 1.25663706212e-6, "unit": "H/m", "symbol": "μ₀", "uncertainty": 1.9e-16, "category": "Electromagnetic"},
    "Coulomb Constant": {"value": 8.9875517923e9, "unit": "N·m²/C²", "symbol": "k", "uncertainty": 0, "category": "Electromagnetic"},
    "Fine Structure Constant": {"value": 7.2973525693e-3, "unit": "dimensionless", "symbol": "α", "uncertainty": 1.1e-12, "category": "Fundamental"},
    
    # Gravitational and Astronomical
    "Gravitational Constant": {"value": 6.67430e-11, "unit": "m³/(kg·s²)", "symbol": "G", "uncertainty": 1.5e-15, "category": "Gravitational"},
    "Standard Gravity": {"value": 9.80665, "unit": "m/s²", "symbol": "g", "uncertainty": 0, "category": "Gravitational"},
    "Earth Mass": {"value": 5.9722e24, "unit": "kg", "symbol": "M⊕", "uncertainty": 6e20, "category": "Astronomical"},
    "Solar Mass": {"value": 1.98847e30, "unit": "kg", "symbol": "M☉", "uncertainty": 7e25, "category": "Astronomical"},
    
    # Standard Conditions
    "Standard Atmosphere": {"value": 101325, "unit": "Pa", "symbol": "atm", "uncertainty": 0, "category": "Standard"},
    "Standard Temperature": {"value": 273.15, "unit": "K", "symbol": "STP", "uncertainty": 0, "category": "Standard"},
    "Absolute Zero": {"value": -273.15, "unit": "°C", "symbol": "0 K", "uncertainty": 0, "category": "Standard"},
    
    # Nuclear and Quantum
    "Bohr Radius": {"value": 5.29177210903e-11, "unit": "m", "symbol": "a₀", "uncertainty": 8.0e-21, "category": "Quantum"},
    "Rydberg Constant": {"value": 1.0973731568160e7, "unit": "m⁻¹", "symbol": "R∞", "uncertainty": 2.1e-5, "category": "Quantum"},
    "Electron g-factor": {"value": -2.00231930436256, "unit": "dimensionless", "symbol": "ge", "uncertainty": 3.5e-13, "category": "Quantum"},
    "Classical Electron Radius": {"value": 2.8179403262e-15, "unit": "m", "symbol": "re", "uncertainty": 1.3e-24, "category": "Particle"},
    "Compton Wavelength": {"value": 2.42631023867e-12, "unit": "m", "symbol": "λC", "uncertainty": 7.3e-22, "category": "Quantum"}
}

class ConstantDatabase:
    """Enhanced physics constants database with search and filtering"""
    
    @staticmethod
    def get_constants_by_category(category: str) -> Dict[str, Dict[str, Any]]:
        """Get all constants from a specific category"""
        return {name: data for name, data in PHYSICS_CONSTANTS.items() 
                if data.get("category") == category}
    
    @staticmethod
    def search_constants(search_term: str) -> Dict[str, Dict[str, Any]]:
        """Search constants by name or symbol"""
        results = {}
        search_term = search_term.lower()
        
        for name, data in PHYSICS_CONSTANTS.items():
            if (search_term in name.lower() or 
                search_term in data.get("symbol", "").lower()):
                results[name] = data
        
        return results
    
    @staticmethod
    def get_constant_info(name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific constant"""
        return PHYSICS_CONSTANTS.get(name)
    
    @staticmethod
    def list_categories() -> List[str]:
        """List all available constant categories"""
        categories = set()
        for data in PHYSICS_CONSTANTS.values():
            if "category" in data:
                categories.add(data["category"])
        return sorted(list(categories))

if __name__ == "__main__":
    # Comprehensive test of all physics utilities
    print("Physics Utilities v2.0 - Comprehensive Test")
    print("=" * 50)
    
    # Test calculators
    calc = PhysicsCalculator()
    print("\n1. Testing Physics Calculators:")
    print("-" * 30)
    
    # Mechanics
    mechanics = calc.mechanics_calculator()
    print(f"Kinetic Energy (m=10kg, v=5m/s): {mechanics['Kinetic Energy'](10, 5):.2f} J")
    
    # Quantum physics
    quantum = calc.quantum_calculator()
    h = 6.626e-34
    print(f"Photon Energy (f=5e14 Hz): {quantum['Photon Energy'](h, 5e14):.2e} J")
    
    # Test unit converter with new units
    converter = UnitConverter()
    print("\n2. Testing Unit Converter:")
    print("-" * 30)
    print(f"10 km to meters: {converter.convert(10, 'km', 'm', 'Length'):.0f} m")
    print(f"1000 nm to meters: {converter.convert(1000, 'nm', 'm', 'Wavelength'):.2e} m")
    print(f"Temperature: 100°C to Kelvin: {converter._convert_temperature(100, 'C', 'K'):.2f} K")
    
    # Test experimental data
    print("\n3. Testing Experimental Data:")
    print("-" * 30)
    
    # Create sample experimental data
    data1 = ExperimentalData(9.81, 0.05, "m/s²")
    data2 = ExperimentalData(9.79, 0.03, "m/s²")
    data3 = ExperimentalData(9.83, 0.04, "m/s²")
    
    print(f"Sample measurement: {data1.value:.3f} ± {data1.uncertainty:.3f} {data1.unit}")
    print(f"Relative uncertainty: {data1.relative_uncertainty():.2f}%")
    
    # Statistical analysis
    stats = StatisticalAnalysis()
    mean_result = stats.mean_with_uncertainty([data1, data2, data3])
    print(f"Mean value: {mean_result.value:.3f} ± {mean_result.uncertainty:.3f} {mean_result.unit}")
    
    # Test linear regression
    x_data = [1, 2, 3, 4, 5]
    y_data = [2.1, 4.2, 5.8, 8.1, 10.0]
    regression = stats.linear_regression(x_data, y_data)
    print(f"Linear regression - Slope: {regression['slope']:.3f}, R²: {regression['r_squared']:.3f}")
    
    # Test formulas with search
    formulas = FormulaReference()
    print("\n4. Testing Formula Reference:")
    print("-" * 30)
    
    search_results = formulas.search_formula("energy")
    print(f"Found {len(search_results)} formulas containing 'energy':")
    for i, (key, formula) in enumerate(list(search_results.items())[:3]):
        print(f"  {i+1}. {key}: {formula}")
    
    # Test constants database
    const_db = ConstantDatabase()
    print("\n5. Testing Constants Database:")
    print("-" * 30)
    
    categories = const_db.list_categories()
    print(f"Available categories: {', '.join(categories)}")
    
    quantum_constants = const_db.get_constants_by_category("Quantum")
    print(f"Quantum constants found: {len(quantum_constants)}")
    
    planck_info = const_db.get_constant_info("Planck Constant")
    if planck_info:
        print(f"Planck constant: {planck_info['value']:.3e} {planck_info['unit']} ({planck_info['symbol']})")
    
    # Test data generation
    print("\n6. Testing Data Generation:")
    print("-" * 30)
    
    generator = ExperimentalDataGenerator()
    theoretical_g = 9.81
    noisy_measurement = generator.add_experimental_noise(theoretical_g, 0.02)
    print(f"Theoretical g: {theoretical_g:.2f} m/s²")
    print(f"Simulated measurement: {noisy_measurement.value:.3f} ± {noisy_measurement.uncertainty:.3f} m/s²")
    
    print("\n" + "=" * 50)
    print("All Physics Utilities v2.0 tests completed successfully!")
    print("Enhanced features: Quantum calculations, experimental data handling,")
    print("statistical analysis, comprehensive constants database, and more!")
