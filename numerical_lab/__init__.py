# SPDX-License-Identifier: MIT
"""Numerical Error Analysis Studio core package."""

__version__ = "1.2.0"

from .core import (
    ApproximationResult,
    Method,
    ReferenceBackend,
    convergence_scan,
    machine_epsilon,
    scan_sine,
    sine_taylor,
    summarize_scan,
)

__all__ = [
    "ApproximationResult",
    "Method",
    "ReferenceBackend",
    "convergence_scan",
    "machine_epsilon",
    "scan_sine",
    "sine_taylor",
    "summarize_scan",
]
