#!/usr/bin/env python3
"""
Grep/AST regression sweeps for banned patterns.
"""

import sys
import re
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Banned patterns
BANNED_PATTERNS = [
    (r"Normalize\s*\(\s*mean\s*=\s*\[\s*0\.485", "ImageNet normalization"),
    (r"0\.456.*0\.406|0\.229.*0\.224.*0\.225", "ImageNet stats"),
    (r"pixel_scale_rad\s*=\s*1\.0", "Isotropic fallback pixel_scale_rad=1.0"),
    (r"PhysicsScale\s*\(\s*pixel_scale_arcsec\s*=\s*0\.1\s*\)", "Default PhysicsScale(pixel_scale_arcsec=0.1)"),
    (r"F\.interpolate\([^)]*kappa|F\.interpolate\([^)]*psi|F\.interpolate\([^)]*alpha", "Bilinear interpolation on physics maps"),
]


def check_pattern(pattern, desc, root_dir=Path("src")):
    """Search for banned pattern in Python files."""
    violations = []
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    
    for py_file in root_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            for i, line in enumerate(content.split('\n'), 1):
                if regex.search(line):
                    # Skip comments and docstrings
                    stripped = line.strip()
                    if not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        violations.append((str(py_file.relative_to(project_root)), i, line.strip()[:80]))
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
            continue
    
    return violations


def main():
    """Run all regression checks."""
    print("=" * 60)
    print("Regression Pattern Checks")
    print("=" * 60)
    
    all_violations = {}
    
    for pattern, desc in BANNED_PATTERNS:
        print(f"\nChecking: {desc}")
        violations = check_pattern(pattern, desc)
        
        if violations:
            print(f"  ✗ Found {len(violations)} violation(s):")
            for file, line, snippet in violations[:10]:  # Show first 10
                print(f"    {file}:{line}: {snippet}")
            if len(violations) > 10:
                print(f"    ... and {len(violations) - 10} more")
            all_violations[desc] = violations
        else:
            print(f"  ✓ No violations found")
    
    # Check mlensing for isotropic patterns
    mlensing_dir = Path("mlensing")
    if mlensing_dir.exists():
        print(f"\nChecking mlensing/ for isotropic patterns...")
        for pattern, desc in BANNED_PATTERNS[2:4]:  # Isotropic patterns
            violations = check_pattern(pattern, desc, root_dir=mlensing_dir)
            if violations:
                print(f"  ✗ Found {len(violations)} violation(s) in mlensing/")
                all_violations[f"{desc} (mlensing)"] = violations
            else:
                print(f"  ✓ No violations in mlensing/")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_violations:
        print(f"✗ Found {sum(len(v) for v in all_violations.values())} total violation(s)")
        return 1
    else:
        print("✓ No banned patterns found")
        return 0


if __name__ == "__main__":
    sys.exit(main())

