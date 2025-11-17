"""Unittest entry point for the orbi test suite.

Usage:
    python -m orbi.test
"""

from __future__ import annotations

import unittest

# Ensure package bootstrap side-effects (sys.path, requirement check) run
from . import EXTERNAL_TEST_REQUIREMENTS  # noqa: F401


def main() -> None:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=__package__ or ".")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
