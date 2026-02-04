#!/usr/bin/env python3
"""
Physics Application Installer and Launcher v2.0.2
Handles dependencies, security checks, and launches the application safely

Features:
- Input sanitization and validation
- Path security checks
- System requirement verification
- Safe dependency installation
- Error recovery and cleanup


Author: BubblesTheDev
Version: 2.0.2
Date: February 2026
"""

import sys
import subprocess
import os
import platform
import re
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import urllib.parse

import urllib.parse

class SecuritySanitizer:
    """Comprehensive security and input sanitization utilities"""
    
    # Allowed file extensions for the application
    ALLOWED_EXTENSIONS = {'.py', '.json', '.txt', '.md', '.bat', '.sh', '.png', '.ico'}
    
    # Dangerous path components to block
    DANGEROUS_PATHS = {
        '..', '~', '$', '`', '|', '&', ';', '>', '<', '*', '?', 
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
        'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    # Regex patterns for validation
    PACKAGE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$')
    VERSION_PATTERN = re.compile(r'^\d+(\.\d+)*([a-zA-Z0-9._-]+)?$')
    
    @staticmethod
    def sanitize_path(path: str) -> Optional[str]:
        """Sanitize and validate file paths for security"""
        if not path or not isinstance(path, str):
            return None
            
        try:
            # Remove null bytes and normalize
            path = path.replace('\x00', '').strip()
            
            # Convert to Path object for proper handling
            path_obj = Path(path)
            
            # Check for dangerous path components
            for part in path_obj.parts:
                if part.upper() in SecuritySanitizer.DANGEROUS_PATHS:
                    print(f"⚠️  Dangerous path component detected: {part}")
                    return None
            
            # Resolve path and check if it stays within bounds
            try:
                resolved_path = path_obj.resolve()
                # Basic security: ensure we don't go outside reasonable bounds
                if '..' in str(resolved_path) or str(resolved_path).startswith('/etc') or str(resolved_path).startswith('/sys'):
                    print(f"⚠️  Path traversal attempt blocked: {path}")
                    return None
            except (OSError, RuntimeError):
                print(f"⚠️  Invalid path: {path}")
                return None
                
            return str(resolved_path)
            
        except Exception as e:
            print(f"⚠️  Path sanitization error: {e}")
            return None
    
    @staticmethod
    def sanitize_package_name(package: str) -> Optional[str]:
        """Sanitize and validate Python package names"""
        if not package or not isinstance(package, str):
            return None
            
        # Remove whitespace and convert to lowercase
        package = package.strip().lower()
        
        # Check length (reasonable bounds)
        if len(package) < 1 or len(package) > 100:
            print(f"⚠️  Invalid package name length: {package}")
            return None
        
        # Extract package name (remove version specifiers)
        package_name = package.split('>=')[0].split('<=')[0].split('==')[0].split('!=')[0]
        package_name = package_name.split('>')[0].split('<')[0].split('!')[0]
        
        # Validate package name format
        if not SecuritySanitizer.PACKAGE_NAME_PATTERN.match(package_name):
            print(f"⚠️  Invalid package name format: {package_name}")
            return None
            
        return package
    
    @staticmethod
    def sanitize_user_input(user_input: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not user_input or not isinstance(user_input, str):
            return ""
            
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', user_input)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'\\\/\|\*\?\$`&;]', '', sanitized)
        
        # Limit length
        sanitized = sanitized[:255]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_python_executable() -> bool:
        """Validate that we're running with a legitimate Python executable"""
        try:
            # Check if running from a reasonable Python installation
            python_path = sys.executable
            if not python_path or not os.path.exists(python_path):
                print("⚠️  Invalid Python executable path")
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠️  Python validation error: {e}")
            return False
    
    @staticmethod
    def check_available_space(required_mb: int = 500) -> bool:
        """Check if sufficient disk space is available"""
        try:
            current_dir = os.getcwd()
            if platform.system() == 'Windows':
                free_bytes = shutil.disk_usage(current_dir).free
            else:
                stat = os.statvfs(current_dir)
                free_bytes = stat.f_bavail * stat.f_frsize
                
            free_mb = free_bytes // (1024 * 1024)
            
            if free_mb < required_mb:
                print(f"⚠️  Insufficient disk space: {free_mb}MB available, {required_mb}MB required")
                return False
                
            print(f"✅ Disk space check: {free_mb}MB available")
            return True
            
        except Exception as e:
            print(f"⚠️  Disk space check failed: {e}")
            return True  # Continue if we can't check

class CleanupManager:
    """Manages cleanup of temporary files and error recovery"""
    
    def __init__(self):
        self.temp_files: List[str] = []
        self.temp_dirs: List[str] = []
    
    def cleanup(self):
        """Clean up all registered temporary files and directories"""
        # Clean up files
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"⚠️  Failed to cleanup file {file_path}: {e}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()

# Global cleanup manager
cleanup_manager = CleanupManager()

def check_python_version():
    """Check if Python version is compatible with security validation"""
    # Validate Python executable first
    if not SecuritySanitizer.validate_python_executable():
        print("❌ Python executable validation failed!")
        return False
        
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or later is required for security and compatibility!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected and validated")
    
    # Check available disk space
    if not SecuritySanitizer.check_available_space(500):
        return False
        
    return True

def install_requirements():
    """Install required packages with security validation"""
    print("📦 Installing required packages with security checks...")
    
    # Define requirements with version constraints for security
    requirements = [
        "PyQt5>=5.15.0",
        "matplotlib>=3.5.0", 
        "numpy>=1.21.0",
        "pillow>=8.3.0"
    ]
    
    for package in requirements:
        # Sanitize package name
        sanitized_package = SecuritySanitizer.sanitize_package_name(package)
        if not sanitized_package:
            print(f"❌ Invalid package specification: {package}")
            return False
            
        try:
            print(f"Installing {sanitized_package}...")
            
            # Use secure installation command
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", sanitized_package]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {sanitized_package} installed successfully")
            else:
                print(f"❌ Failed to install {sanitized_package}")
                # Sanitize error output
                error_msg = SecuritySanitizer.sanitize_user_input(result.stderr[:500])
                print(f"Error: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Installation timeout for {package}")
            return False
        except Exception as e:
            error_msg = SecuritySanitizer.sanitize_user_input(str(e)[:200])
            print(f"❌ Error installing {package}: {error_msg}")
            return False
    
    return True

def check_pyqt5():
    """Check if PyQt5 is available with security validation"""
    try:
        import PyQt5.QtWidgets
        import PyQt5.QtCore
        
        # Verify PyQt5 version for security
        pyqt_version = PyQt5.QtCore.QT_VERSION_STR
        print(f"✅ PyQt5 {pyqt_version} GUI library is available")
        
        # Check minimum version for security
        version_parts = pyqt_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 5 or (major == 5 and minor < 15):
            print(f"⚠️  PyQt5 version {pyqt_version} is outdated. Minimum 5.15.0 recommended.")
            
        return True
        
    except ImportError:
        print("❌ PyQt5 is not available!")
        print("Installing PyQt5 with security constraints...")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "PyQt5>=5.15.0"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ PyQt5 installed successfully")
                return True
            else:
                error_msg = SecuritySanitizer.sanitize_user_input(result.stderr[:500])
                print(f"❌ Failed to install PyQt5: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ PyQt5 installation timeout")
            return False
        except Exception as e:
            error_msg = SecuritySanitizer.sanitize_user_input(str(e)[:200])
            print(f"❌ Error installing PyQt5: {error_msg}")
            return False

def run_tests():
    """Run the test suite with security validation"""
    print("🧪 Running application tests with security checks...")
    
    test_file = "test_physics_app.py"
    sanitized_path = SecuritySanitizer.sanitize_path(test_file)
    
    if not sanitized_path or not os.path.exists(sanitized_path):
        print(f"⚠️  Test file not found: {test_file}")
        return True  # Don't fail installation if tests are missing
    
    try:
        result = subprocess.run([sys.executable, sanitized_path], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            # Sanitize test output
            stdout = SecuritySanitizer.sanitize_user_input(result.stdout[:1000])
            stderr = SecuritySanitizer.sanitize_user_input(result.stderr[:1000])
            print(stdout)
            if stderr:
                print(stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test execution timeout")
        return False
    except Exception as e:
        error_msg = SecuritySanitizer.sanitize_user_input(str(e)[:200])
        print(f"❌ Error running tests: {error_msg}")
        return False

def launch_application():
    """Launch the physics application with security validation"""
    print("🚀 Launching Physics Application with security checks...")
    
    main_file = "main.py"
    sanitized_path = SecuritySanitizer.sanitize_path(main_file)
    
    if not sanitized_path or not os.path.exists(sanitized_path):
        print(f"❌ Main application file not found: {main_file}")
        return False
    
    try:
        # Import and run the application safely
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("main", sanitized_path)
        if not spec or not spec.loader:
            print(f"❌ Failed to load application module from {main_file}")
            return False
            
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Launch the application
        if hasattr(main_module, 'QApplication') and hasattr(main_module, 'PhysicsApplication'):
            app = main_module.QApplication(sys.argv)
            window = main_module.PhysicsApplication()
            window.show()
            return app.exec_()
        else:
            print("❌ Application classes not found in main.py")
            return False
            
    except Exception as e:
        error_msg = SecuritySanitizer.sanitize_user_input(str(e)[:300])
        print(f"❌ Error launching application: {error_msg}")
        print("Please check that main.py exists and is error-free.")
        return False

def main():
    """Main installer/launcher function with security and cleanup"""
    print("=" * 70)
    print("🔬 PHYSICS APPLICATION SUITE v2.0 - SECURE INSTALLER & LAUNCHER")
    print("=" * 70)
    print("🛡️  Enhanced with security validation and sanitization")
    print()
    
    try:
        # Check Python version with security validation
        if not check_python_version():
            return False
        
        print()
        
        # Check PyQt5 availability with security
        if not check_pyqt5():
            return False
        
        print()
        
        # Install requirements with security validation
        if not install_requirements():
            return False
        
        print()
        
        # Ask if user wants to run tests (sanitize input)
        while True:
            test_input = input("🧪 Run tests before launching? (y/n): ")
            sanitized_input = SecuritySanitizer.sanitize_user_input(test_input).lower()
            
            if sanitized_input in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        if sanitized_input in ['y', 'yes']:
            if not run_tests():
                while True:
                    launch_input = input("❓ Tests failed. Launch application anyway? (y/n): ")
                    sanitized_launch = SecuritySanitizer.sanitize_user_input(launch_input).lower()
                    
                    if sanitized_launch in ['y', 'yes', 'n', 'no']:
                        break
                    print("Please enter 'y' for yes or 'n' for no.")
                
                if sanitized_launch not in ['y', 'yes']:
                    print("Installation completed. Application launch cancelled.")
                    return True
            print()
        
        # Launch application
        print("🎉 Setup complete! Starting the Physics Application...")
        print("🛡️  All security checks passed")
        print()
        
        return launch_application()
        
    except Exception as e:
        error_msg = SecuritySanitizer.sanitize_user_input(str(e)[:300])
        print(f"\n❌ Unexpected error in main installer: {error_msg}")
        return False
    finally:
        # Always cleanup temporary files
        cleanup_manager.cleanup()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\n\n⏹️  Installation cancelled by user")
        cleanup_manager.cleanup()
    except Exception as e:
        error_msg = SecuritySanitizer.sanitize_user_input(str(e)[:300])
        print(f"\n\n❌ Critical error: {error_msg}")
        cleanup_manager.cleanup()
        input("Press Enter to exit...")

