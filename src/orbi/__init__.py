"""
LibreMIP: A simple, Gurobi-like MILP wrapper backed by OR-Tools.

Public API:
- Model: MILP model wrapper
- GRB: constants mimicking the Gurobi GRB namespace
- Var, LinExpr: basic variable and linear expression helpers

Notes:
- This package is a demonstration and not a drop-in replacement for gurobipy.
"""

from __future__ import annotations

# Runtime version discovery with graceful fallback
# if package metadata is missing
try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except Exception:  # pragma: no cover
    # Very old Python fallback (not expected here)
    version = None
    PackageNotFoundError = Exception


__version__: str
try:
    # Package name should match the installed distribution
    __version__ = version("LibreMIP") if version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"


# Re-export primary symbols from the internal implementation module
from .milp import Model, GRB, Var, LinExpr  # noqa: E402,F401


__all__ = [
    "Model",
    "GRB",
    "Var",
    "LinExpr",
    "__version__",
]
