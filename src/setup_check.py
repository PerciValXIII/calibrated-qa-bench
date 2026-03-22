"""
setup_check.py
Run this first to verify your environment is ready.
Usage: python src/setup_check.py
"""

import sys

checks = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [PASS] {label}" + (f" — {result}" if result else ""))
        checks.append(True)
    except Exception as e:
        print(f"  [FAIL] {label} — {e}")
        checks.append(False)

print("\n=== Environment Check: calibrated-qa-bench ===\n")

# Python version
check("Python version", lambda: sys.version)

# Core ML
check("torch", lambda: __import__("torch").__version__)
check("torch CUDA available", lambda: str(__import__("torch").cuda.is_available()))
check("transformers", lambda: __import__("transformers").__version__)
check("datasets", lambda: __import__("datasets").__version__)
check("accelerate", lambda: __import__("accelerate").__version__)

# Calibration & Metrics
check("scikit-learn", lambda: __import__("sklearn").__version__)
check("scipy", lambda: __import__("scipy").__version__)
check("netcal", lambda: __import__("netcal").__version__)

# Data
check("numpy", lambda: __import__("numpy").__version__)
check("pandas", lambda: __import__("pandas").__version__)

# Plotting
check("matplotlib", lambda: __import__("matplotlib").__version__)
check("seaborn", lambda: __import__("seaborn").__version__)

# Utilities
check("tqdm", lambda: __import__("tqdm").__version__)
check("dotenv", lambda: __import__("dotenv") and "installed")
check("wandb", lambda: __import__("wandb").__version__)

# Summary
print(f"\n=== {sum(checks)}/{len(checks)} checks passed ===")
if all(checks):
    print("Environment is ready. Proceed to load_data.py\n")
else:
    print("Fix failing dependencies before proceeding.\n")
    print("Run: pip install -r requirements.txt\n")