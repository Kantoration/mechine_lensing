#!/usr/bin/env python3
"""
CI Gates: grep guards and triple-run stability checks.

Fails if:
1. ImageNet normalization detected
2. Isotropic fallbacks detected (pixel_scale_rad=1.0, PhysicsScale default)
3. Tests fail or are flaky (run 3x, fail on inconsistency)
"""

import subprocess
import sys
from pathlib import Path
import re


def check_imagenet_norm():
    """Fail if ImageNet normalization detected in src/."""
    print("🔍 Checking for ImageNet normalization...")
    src_dir = Path("src")
    violations = []

    pattern = re.compile(r"Normalize\s*\(\s*mean\s*=\s*\[\s*0\.485")

    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if pattern.search(content):
                violations.append(str(py_file))
        except Exception:
            continue

    if violations:
        print(f"❌ ImageNet normalization detected in {len(violations)} file(s):")
        for v in violations:
            print(f"   - {v}")
        return False

    print("✅ No ImageNet normalization detected")
    return True


def check_isotropic_fallbacks():
    """Fail if isotropic fallbacks detected in mlensing/."""
    print("🔍 Checking for isotropic fallbacks...")
    mlensing_dir = Path("mlensing")
    violations = []

    patterns = [
        (re.compile(r"pixel_scale_rad\s*=\s*1\.0"), "pixel_scale_rad=1.0"),
        (
            re.compile(r"PhysicsScale\s*\(\s*pixel_scale_arcsec\s*=\s*0\.1\s*\)"),
            "PhysicsScale(pixel_scale_arcsec=0.1)",
        ),
    ]

    for py_file in mlensing_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            for pattern, desc in patterns:
                if pattern.search(content):
                    # Check if it's a default parameter (allow in dataclass defaults)
                    # but not in function bodies
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if pattern.search(line) and "def " not in line[:50]:
                            # Skip dataclass field defaults
                            if ":" in line and (
                                "dataclass" not in lines[max(0, i - 5) : i]
                                or "field" not in line
                            ):
                                violations.append((str(py_file), i + 1, desc))
        except Exception:
            continue

    if violations:
        print(f"❌ Isotropic fallbacks detected in {len(violations)} location(s):")
        for file, line, desc in violations:
            print(f"   - {file}:{line} ({desc})")
        return False

    print("✅ No isotropic fallbacks detected")
    return True


def check_color_jitter_guarded():
    """Warn if unguarded ColorJitter detected (should be opt-in)."""
    print("🔍 Checking for unguarded ColorJitter...")
    src_dir = Path("src")
    mlensing_dir = Path("mlensing")
    violations = []

    # Pattern: ColorJitter not behind a conditional
    pattern = re.compile(r"ColorJitter\s*\(")

    for py_file in list(src_dir.rglob("*.py")) + list(mlensing_dir.rglob("*.py")):
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if pattern.search(line):
                    # Check if it's behind an if/conditional
                    prev_lines = lines[max(0, i - 5) : i]
                    if not any(
                        "if" in pl or "jitter" in pl.lower() or "opt" in pl.lower()
                        for pl in prev_lines
                    ):
                        violations.append((str(py_file), i + 1))
        except Exception:
            continue

    if violations:
        print(
            f"⚠️  Unguarded ColorJitter detected in {len(violations)} location(s) (should be opt-in):"
        )
        for file, line in violations:
            print(f"   - {file}:{line}")
        print("   (This is a warning, not a failure)")

    return True


def run_tests_stability(test_files, max_runs=3):
    """Run tests 3x and fail on inconsistency."""
    print(f"🧪 Running tests {max_runs}x for stability check...")

    results = []
    for run in range(1, max_runs + 1):
        print(f"\nRun {run}/{max_runs}...")
        test_cmd = [sys.executable, "-m", "pytest", "-q", "-x", *test_files]

        result = subprocess.run(test_cmd, capture_output=True, text=True)
        passed = result.returncode == 0

        results.append(passed)
        if not passed:
            print(f"❌ Run {run} failed:")
            print(result.stdout[-500:])  # Last 500 chars
            print(result.stderr[-500:])
        else:
            print(f"✅ Run {run} passed")

    if not all(results):
        print(
            f"\n❌ Tests failed in {sum(1 for r in results if not r)}/{max_runs} run(s)"
        )
        return False

    if len(set(results)) > 1:
        print("\n❌ Tests are flaky (inconsistent results across runs)")
        return False

    print(f"\n✅ All tests passed consistently ({max_runs}/{max_runs} runs)")
    return True


def main():
    """Run all CI gates."""
    print("=" * 60)
    print("CI Gates: P1 Hardening Checks")
    print("=" * 60)

    checks = [
        ("ImageNet normalization check", check_imagenet_norm),
        ("Isotropic fallbacks check", check_isotropic_fallbacks),
        ("ColorJitter guard check", check_color_jitter_guarded),
    ]

    all_passed = True
    for name, check_fn in checks:
        print(f"\n[{name}]")
        if not check_fn():
            all_passed = False

    # Test stability check
    test_files = [
        "tests/test_operators_anisotropic.py",
        "tests/test_fits_loader_meta.py",
        "tests/test_ssl_schedule.py",
        "tests/test_kappa_pooling_area.py",
        "tests/test_tiled_inference_equiv.py",
        "tests/test_sie_smoke.py",
        "tests/test_loader_require_meta.py",
        "tests/test_no_imagenet_norm.py",
        "tests/test_no_isotropic_defaults.py",
        "tests/test_graph_requires_scale.py",
        "tests/test_lensgnn_anisotropic.py",
    ]

    print("\n[Test Stability Check]")
    if not run_tests_stability(test_files):
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All CI gates passed")
        sys.exit(0)
    else:
        print("❌ CI gates failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
