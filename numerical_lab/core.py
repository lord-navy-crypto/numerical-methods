# SPDX-License-Identifier: MIT
"""Numerically explicit sine approximations used by the interactive lab.

The implementation intentionally exposes intermediate quantities that normal
math libraries hide. This makes range reduction, cancellation, stopping rules,
high-precision references, and floating-point limits visible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Callable

import mpmath as mp
import numpy as np


class Method(str, Enum):
    """Supported Taylor-series evaluation strategies."""

    RAW = "raw"
    RANGE_REDUCED = "range_reduced"


class ReferenceBackend(str, Enum):
    """Reference engines available for error measurement."""

    MPMATH = "mpmath"
    NUMPY = "numpy"


@dataclass(frozen=True)
class ApproximationResult:
    """Result and diagnostics for one scalar sine approximation."""

    x: float
    reduced_x: float
    value: float
    reference: float
    reference_text: str
    numpy_reference: float
    reference_backend: str
    reference_precision_digits: int
    absolute_error: float
    relative_error: float
    ulp_error: float
    allowed_absolute_error: float
    terms_used: int
    last_term: float
    cancellation_ratio: float
    stopping_criterion_met: bool
    accuracy_passed: bool
    numerically_reliable: bool

    @property
    def false_convergence(self) -> bool:
        """Whether the term rule stopped while the reference check failed."""

        return self.stopping_criterion_met and not self.accuracy_passed

    @property
    def normalized_error(self) -> float:
        """Absolute error divided by the scale-aware acceptance bound."""

        if self.allowed_absolute_error == 0.0:
            return 0.0 if self.absolute_error == 0.0 else np.inf
        return self.absolute_error / self.allowed_absolute_error

    @property
    def status(self) -> str:
        """Return a stable, human-readable diagnostic classification."""

        if not np.isfinite(self.value):
            return "non_finite_arithmetic"
        if self.false_convergence:
            return "false_convergence"
        if not self.stopping_criterion_met:
            return "term_limit_reached"
        if not self.accuracy_passed:
            return "accuracy_failure"
        if not self.numerically_reliable:
            return "excessive_cancellation"
        return "reliable"

    @property
    def converged(self) -> bool:
        """Compatibility alias; this means only that the stop rule was met."""

        return self.stopping_criterion_met


def _dtype_type(dtype: str | np.dtype) -> type[np.floating]:
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    return resolved.type


def machine_epsilon(dtype: str | np.dtype = "float64") -> float:
    """Return the spacing between 1 and the next representable value."""

    return float(np.finfo(_dtype_type(dtype)).eps)


def _reduce_sine_argument(x: np.floating, scalar: type[np.floating]) -> np.floating:
    """Reduce an angle to [-pi/2, pi/2] while preserving its sine."""

    pi = scalar(np.pi)
    two_pi = scalar(2.0) * pi
    y = scalar(np.remainder(x + pi, two_pi) - pi)
    half_pi = pi / scalar(2.0)
    if y > half_pi:
        y = pi - y
    elif y < -half_pi:
        y = -pi - y
    return scalar(y)


def _validate_reference_settings(
    backend: ReferenceBackend | str,
    precision_digits: int,
) -> ReferenceBackend:
    selected = ReferenceBackend(backend)
    if not isinstance(precision_digits, Integral):
        raise ValueError("reference_precision_digits must be an integer")
    if not 20 <= precision_digits <= 500:
        raise ValueError("reference_precision_digits must be between 20 and 500")
    return selected


def _reference_diagnostics(
    *,
    original: np.floating,
    value: float,
    dtype: str | np.dtype,
    backend: ReferenceBackend,
    precision_digits: int,
) -> tuple[float, str, float, float, float, float]:
    """Return reference value/text, NumPy value, and three error measures."""

    scalar = _dtype_type(dtype)
    numpy_reference = float(np.sin(original, dtype=scalar))
    spacing = abs(float(np.spacing(scalar(numpy_reference))))
    if spacing == 0.0:
        spacing = float(np.finfo(scalar).smallest_subnormal)

    if backend is ReferenceBackend.NUMPY:
        reference = numpy_reference
        reference_text = format(reference, ".9g" if scalar is np.float32 else ".17g")
        if not np.isfinite(value):
            return reference, reference_text, numpy_reference, np.inf, np.inf, np.inf
        absolute_error = abs(value - reference)
        relative_floor = float(np.finfo(scalar).tiny)
        relative_error = absolute_error / max(abs(reference), relative_floor)
        ulp_error = absolute_error / spacing
        return (
            reference,
            reference_text,
            numpy_reference,
            float(absolute_error),
            float(relative_error),
            float(ulp_error),
        )

    with mp.workdps(precision_digits):
        exact_input = mp.mpf(float(original))
        exact_reference = mp.sin(exact_input)
        reference = float(exact_reference)
        reference_text = mp.nstr(exact_reference, precision_digits)
        if not np.isfinite(value):
            return reference, reference_text, numpy_reference, np.inf, np.inf, np.inf
        high_precision_error = abs(mp.mpf(value) - exact_reference)
        relative_floor = mp.mpf(str(np.finfo(scalar).tiny))
        relative_error = high_precision_error / max(abs(exact_reference), relative_floor)
        ulp_error = high_precision_error / mp.mpf(spacing)
        return (
            reference,
            reference_text,
            numpy_reference,
            float(high_precision_error),
            float(relative_error),
            float(ulp_error),
        )


def sine_taylor(
    x: float,
    *,
    method: Method | str = Method.RANGE_REDUCED,
    dtype: str | np.dtype = "float64",
    tolerance_multiplier: float = 8.0,
    max_terms: int = 120,
    fixed_terms: int | None = None,
    reference_backend: ReferenceBackend | str = ReferenceBackend.MPMATH,
    reference_precision_digits: int = 80,
) -> ApproximationResult:
    """Approximate sin(x) and report numerical diagnostics.

    ``stopping_criterion_met`` reports only whether the final Taylor term was
    small enough. ``accuracy_passed`` separately compares the result with the
    selected reference. This distinction exposes false convergence caused by
    catastrophic cancellation.
    """

    selected = Method(method)
    backend = _validate_reference_settings(reference_backend, reference_precision_digits)
    scalar = _dtype_type(dtype)
    if not np.isfinite(x):
        raise ValueError("x must be finite")
    if tolerance_multiplier <= 0:
        raise ValueError("tolerance_multiplier must be positive")
    if max_terms < 1:
        raise ValueError("max_terms must be at least 1")
    if fixed_terms is not None and not 1 <= fixed_terms <= max_terms:
        raise ValueError("fixed_terms must be between 1 and max_terms")

    original = scalar(x)
    reduced = (
        _reduce_sine_argument(original, scalar)
        if selected is Method.RANGE_REDUCED
        else original
    )
    term = scalar(reduced)
    total = scalar(reduced)
    sum_abs_terms = scalar(abs(term))
    eps = scalar(np.finfo(scalar).eps)
    atol = scalar(tolerance_multiplier) * eps
    stopping_criterion_met = bool(reduced == 0 and fixed_terms is None)
    terms_used = 1
    arithmetic_finite = True

    target_terms = fixed_terms if fixed_terms is not None else max_terms
    if not stopping_criterion_met or fixed_terms is not None:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for term_index in range(1, target_terms):
                denominator = scalar((2 * term_index) * (2 * term_index + 1))
                term = scalar(term * (-reduced * reduced) / denominator)
                total = scalar(total + term)
                sum_abs_terms = scalar(sum_abs_terms + abs(term))
                terms_used = term_index + 1

                if not np.isfinite(term) or not np.isfinite(total) or not np.isfinite(sum_abs_terms):
                    arithmetic_finite = False
                    break

                threshold = scalar(atol + scalar(tolerance_multiplier) * eps * abs(total))
                if fixed_terms is None and abs(term) <= threshold:
                    stopping_criterion_met = True
                    break

    if fixed_terms is not None and arithmetic_finite:
        threshold = scalar(atol + scalar(tolerance_multiplier) * eps * abs(total))
        stopping_criterion_met = bool(abs(term) <= threshold)

    value = float(total) if arithmetic_finite else np.nan
    (
        reference,
        reference_text,
        numpy_reference,
        absolute_error,
        relative_error,
        ulp_error,
    ) = _reference_diagnostics(
        original=original,
        value=value,
        dtype=dtype,
        backend=backend,
        precision_digits=reference_precision_digits,
    )

    cancellation_floor = float(np.finfo(scalar).tiny)
    if arithmetic_finite:
        cancellation_ratio = float(sum_abs_terms) / max(abs(value), cancellation_floor)
    else:
        cancellation_ratio = np.inf
    # Argument reduction uses floating-point pi and remainder operations, so
    # its forward rounding bound grows with the magnitude of the input. The
    # scale-aware threshold avoids pretending that every x has an O(eps)
    # absolute reduction error while still rejecting catastrophic failures.
    allowed_absolute_error = (
        float(tolerance_multiplier)
        * float(eps)
        * max(1.0, abs(float(original)), abs(reference))
    )
    accuracy_passed = bool(np.isfinite(value) and absolute_error <= allowed_absolute_error)
    cancellation_limit = 1.0 / np.sqrt(float(eps))
    numerically_reliable = bool(
        stopping_criterion_met
        and accuracy_passed
        and arithmetic_finite
        and cancellation_ratio <= cancellation_limit
    )

    return ApproximationResult(
        x=float(original),
        reduced_x=float(reduced),
        value=value,
        reference=reference,
        reference_text=reference_text,
        numpy_reference=numpy_reference,
        reference_backend=backend.value,
        reference_precision_digits=reference_precision_digits,
        absolute_error=float(absolute_error),
        relative_error=float(relative_error),
        ulp_error=float(ulp_error),
        allowed_absolute_error=allowed_absolute_error,
        terms_used=terms_used,
        last_term=float(term) if np.isfinite(term) else np.nan,
        cancellation_ratio=float(cancellation_ratio),
        stopping_criterion_met=stopping_criterion_met,
        accuracy_passed=accuracy_passed,
        numerically_reliable=numerically_reliable,
    )


def scan_sine(
    x_values: np.ndarray,
    *,
    method: Method | str,
    dtype: str | np.dtype = "float64",
    tolerance_multiplier: float = 8.0,
    max_terms: int = 120,
    reference_backend: ReferenceBackend | str = ReferenceBackend.MPMATH,
    reference_precision_digits: int = 80,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate a sine method across x; every returned row shares that x-axis."""

    x_array = np.asarray(x_values, dtype=float)
    if x_array.ndim != 1 or x_array.size == 0:
        raise ValueError("x_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x_array)):
        raise ValueError("all x_values must be finite")

    selected_method = Method(method)
    selected_backend = _validate_reference_settings(
        reference_backend, reference_precision_digits
    )
    resolved_dtype = np.dtype(dtype).name
    rows: list[ApproximationResult] = []
    total = int(x_array.size)
    callback_stride = max(1, total // 100)
    for index, x_value in enumerate(x_array, start=1):
        rows.append(
            sine_taylor(
                float(x_value),
                method=selected_method,
                dtype=dtype,
                tolerance_multiplier=tolerance_multiplier,
                max_terms=max_terms,
                reference_backend=selected_backend,
                reference_precision_digits=reference_precision_digits,
            )
        )
        if progress_callback and (index == total or index % callback_stride == 0):
            progress_callback(index, total)

    return {
        "x": np.array([row.x for row in rows]),
        "method": np.full(total, selected_method.value, dtype=object),
        "dtype": np.full(total, resolved_dtype, dtype=object),
        "reference_backend": np.full(total, selected_backend.value, dtype=object),
        "reference_precision_digits": np.full(
            total, reference_precision_digits, dtype=int
        ),
        "reduced_x": np.array([row.reduced_x for row in rows]),
        "approximation": np.array([row.value for row in rows]),
        "reference": np.array([row.reference for row in rows]),
        "reference_text": np.array([row.reference_text for row in rows], dtype=object),
        "numpy_reference": np.array([row.numpy_reference for row in rows]),
        "absolute_error": np.array([row.absolute_error for row in rows]),
        "relative_error": np.array([row.relative_error for row in rows]),
        "ulp_error": np.array([row.ulp_error for row in rows]),
        "allowed_absolute_error": np.array([row.allowed_absolute_error for row in rows]),
        "normalized_error": np.array([row.normalized_error for row in rows]),
        "terms_used": np.array([row.terms_used for row in rows], dtype=int),
        "cancellation_ratio": np.array([row.cancellation_ratio for row in rows]),
        "stopping_criterion_met": np.array(
            [row.stopping_criterion_met for row in rows], dtype=bool
        ),
        "accuracy_passed": np.array([row.accuracy_passed for row in rows], dtype=bool),
        "numerically_reliable": np.array(
            [row.numerically_reliable for row in rows], dtype=bool
        ),
        "false_convergence": np.array(
            [row.false_convergence for row in rows], dtype=bool
        ),
        "status": np.array([row.status for row in rows], dtype=object),
    }


def summarize_scan(result: dict[str, np.ndarray]) -> dict[str, float | int | str]:
    """Summarize a scan without hiding non-finite or failed calculations."""

    required = {
        "x",
        "absolute_error",
        "normalized_error",
        "terms_used",
        "stopping_criterion_met",
        "accuracy_passed",
        "numerically_reliable",
        "false_convergence",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"scan result is missing fields: {', '.join(sorted(missing))}")

    x = np.asarray(result["x"], dtype=float)
    absolute_error = np.asarray(result["absolute_error"], dtype=float)
    normalized_error = np.asarray(result["normalized_error"], dtype=float)
    terms = np.asarray(result["terms_used"], dtype=int)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("scan result must contain a non-empty one-dimensional x array")
    if any(np.asarray(result[key]).shape != x.shape for key in required - {"x"}):
        raise ValueError("all scan result fields must have the same shape as x")

    finite_errors = np.isfinite(absolute_error)
    if finite_errors.any():
        finite_values = absolute_error[finite_errors]
        worst_local = int(np.argmax(finite_values))
        worst_index = int(np.flatnonzero(finite_errors)[worst_local])
        mean_error = float(np.mean(finite_values))
        median_error = float(np.median(finite_values))
        rms_error = float(np.sqrt(np.mean(np.square(finite_values))))
        max_error = float(absolute_error[worst_index])
        worst_x = float(x[worst_index])
    else:
        mean_error = median_error = rms_error = max_error = np.inf
        worst_x = float(x[0])

    return {
        "points": int(x.size),
        "finite_points": int(finite_errors.sum()),
        "mean_absolute_error": mean_error,
        "median_absolute_error": median_error,
        "rms_absolute_error": rms_error,
        "maximum_absolute_error": max_error,
        "worst_x": worst_x,
        "maximum_normalized_error": float(np.nanmax(normalized_error)),
        "maximum_terms_used": int(np.max(terms)),
        "stop_rule_rate": float(np.mean(result["stopping_criterion_met"])),
        "accuracy_pass_rate": float(np.mean(result["accuracy_passed"])),
        "reliability_rate": float(np.mean(result["numerically_reliable"])),
        "false_convergence_count": int(np.sum(result["false_convergence"])),
    }


def convergence_scan(
    x: float,
    *,
    method: Method | str,
    dtype: str | np.dtype = "float64",
    max_terms: int = 60,
    reference_backend: ReferenceBackend | str = ReferenceBackend.MPMATH,
    reference_precision_digits: int = 80,
) -> dict[str, np.ndarray]:
    """Evaluate one x using 1..max_terms; the independent axis is term count."""

    if max_terms < 1:
        raise ValueError("max_terms must be at least 1")
    rows = [
        sine_taylor(
            x,
            method=method,
            dtype=dtype,
            max_terms=max_terms,
            fixed_terms=terms,
            reference_backend=reference_backend,
            reference_precision_digits=reference_precision_digits,
        )
        for terms in range(1, max_terms + 1)
    ]
    return {
        "terms": np.arange(1, max_terms + 1, dtype=int),
        "approximation": np.array([row.value for row in rows]),
        "absolute_error": np.array([row.absolute_error for row in rows]),
        "relative_error": np.array([row.relative_error for row in rows]),
        "ulp_error": np.array([row.ulp_error for row in rows]),
        "last_term": np.array([abs(row.last_term) for row in rows]),
        "cancellation_ratio": np.array([row.cancellation_ratio for row in rows]),
        "accuracy_passed": np.array([row.accuracy_passed for row in rows], dtype=bool),
    }
