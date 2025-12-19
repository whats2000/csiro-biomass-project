#!/usr/bin/env python
"""Verify project setup and dependencies."""

import sys
from pathlib import Path


def check_imports():
    """Check if all required packages can be imported."""
    print("Checking required packages...")
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "TIMM"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("PIL", "Pillow"),
        ("albumentations", "Albumentations"),
        ("cv2", "OpenCV"),
    ]
    
    failed = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name} - {e}")
            failed.append(package_name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: uv sync")
        return False
    
    print("\n✓ All packages installed")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU - training will be slow)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_data():
    """Check if data files exist."""
    print("\nChecking data files...")
    
    data_dir = Path("data")
    required_files = ["train.csv", "test.csv", "sample_submission.csv"]
    
    missing = []
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠ Missing data files: {', '.join(missing)}")
        print("  Ensure all CSV files are in the data/ directory")
        return False
    
    print("\n✓ All data files present")
    return True


def check_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    
    required_dirs = [
        "src/biomass",
        "src/biomass/data",
        "src/biomass/models",
        "src/biomass/training",
        "src/biomass/utils",
        "scripts",
        "data",
    ]
    
    missing = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ not found")
            missing.append(dir_path)
    
    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("\n✓ Project structure correct")
    return True


def check_scripts():
    """Check if scripts are present."""
    print("\nChecking scripts...")
    
    scripts = ["eda.py", "train.py", "inference.py"]
    missing = []
    
    for script in scripts:
        script_path = Path("scripts") / script
        if script_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} not found")
            missing.append(script)
    
    if missing:
        print(f"\n❌ Missing scripts: {', '.join(missing)}")
        return False
    
    print("\n✓ All scripts present")
    return True


def main():
    """Run all checks."""
    print("="*60)
    print("CSIRO Biomass Project - Setup Verification")
    print("="*60)
    
    checks = [
        check_structure,
        check_imports,
        check_cuda,
        check_scripts,
        check_data,
    ]
    
    results = [check() for check in checks]
    
    print("\n" + "="*60)
    if all(results[:4]):  # Structure, imports, CUDA, scripts
        print("✓ Setup complete! Ready to run.")
        print("\nNext steps:")
        print("  1. uv run scripts/eda.py       # Exploratory analysis")
        print("  2. uv run scripts/train.py     # Train with CV")
        print("  3. uv run scripts/inference.py # Generate submission")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    if not results[4]:  # Data check
        print("\n⚠ Note: Data files missing. Add them to continue.")


if __name__ == "__main__":
    main()
