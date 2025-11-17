"""Test package bootstrap.

This module ensures the repository's "src" directory is importable so tests
can import modules like `milp` without installing the package. It also
declares external packages required by these tests.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Add the project's src/ to sys.path so `from milp import ...` works in tests
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# External packages used by tests. Keep this up to date.
EXTERNAL_TEST_REQUIREMENTS = [
    "ortools",  # Solver backend used by milp module
]


def _missing_requirements(packages: list[str]) -> list[str]:
    missing: list[str] = []
    for name in packages:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


_missing = _missing_requirements(EXTERNAL_TEST_REQUIREMENTS)
if _missing:
    # Print a concise hint; do not hard-fail so discovery can
    # still surface the error.
    pkgs = " ".join(_missing)
    print(
        f"[test bootstrap] Missing external packages: {pkgs}.\n"
        f"Install with: pip install {pkgs}",
        file=sys.stderr,
    )
