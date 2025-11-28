"""Unittest entry point for the milp module test suite.

Usage:
    python -m tests
"""

from __future__ import annotations

import unittest


def main() -> None:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=__package__ or ".")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
