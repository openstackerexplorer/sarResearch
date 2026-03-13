import importlib
import sys
import os

def check_dependencies(requirements_file='requirements.txt'):
    """
    Checks if all libraries listed in requirements.txt are installed.
    """
    print("--- Environment Dependency Check ---\n")
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        return

    # Check Python Version
    print(f"Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("Warning: Python 3.8+ is recommended.")
    else:
        print("Python version is compatible.\n")

    with open(requirements_file, 'r') as f:
        dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    missing = []
    installed_count = 0

    for dep in dependencies:
        # Handle cases where package name differs from import name if necessary
        # Most in requirements.txt match (numpy, torch, etc.)
        package_to_import = dep.split('==')[0].split('>=')[0].strip().replace('-', '_')
        
        # Special mappings
        mapping = {
            "pystac_client": "pystac_client",
            "pyyaml": "yaml",
            "scipy": "scipy",
            "torchvision": "torchvision"
        }
        
        import_name = mapping.get(package_to_import.lower(), package_to_import)
        
        try:
            importlib.import_module(import_name)
            print(f"[OK] {dep}")
            installed_count += 1
        except ImportError:
            print(f"[MISSING] {dep}")
            missing.append(dep)

    print(f"\nSummary: {installed_count}/{len(dependencies)} dependencies installed.")
    
    if missing:
        print("\nTo install missing dependencies, run:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\nAll dependencies are correctly installed. System is ready.")

if __name__ == "__main__":
    check_dependencies()
