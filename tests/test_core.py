import numpy as np
import pytest

from numerical_lab.core import (
    Method,
    ReferenceBackend,
    convergence_scan,
    machine_epsilon,
    scan_sine,
    sine_taylor,
    summarize_scan,
)


def test_machine_epsilon_matches_numpy() -> None:
    assert machine_epsilon("float64") == np.finfo(np.float64).eps
    assert machine_epsilon("float32") == np.finfo(np.float32).eps


@pytest.mark.parametrize("x", [-80.0, -10.5, -np.pi, 0.0, np.pi / 2, 11.0, 80.0])
def test_range_reduction_is_accurate_on_notebook_domain(x: float) -> None:
    result = sine_taylor(
        x,
        method=Method.RANGE_REDUCED,
        dtype="float64",
        reference_backend=ReferenceBackend.MPMATH,
        reference_precision_digits=100,
    )
    assert result.absolute_error < 5e-14
    assert result.terms_used <= 20
    assert result.accuracy_passed
    assert result.numerically_reliable


def test_raw_taylor_exposes_instability_for_large_argument() -> None:
    result = sine_taylor(80.0, method=Method.RAW, dtype="float64")
    assert result.absolute_error > 1.0
    assert result.cancellation_ratio > 1e10
    assert result.stopping_criterion_met
    assert not result.accuracy_passed
    assert not result.numerically_reliable
    assert result.false_convergence
    assert result.status == "false_convergence"
    assert result.normalized_error > 1.0


def test_mpmath_reference_preserves_more_than_float_display_digits() -> None:
    result = sine_taylor(np.pi, reference_precision_digits=80)
    assert result.reference_backend == "mpmath"
    assert result.reference_precision_digits == 80
    assert result.reference_text.startswith("0.000000000000000122464679")
    assert len(result.reference_text) >= 75


def test_numpy_reference_remains_available() -> None:
    result = sine_taylor(1.25, reference_backend=ReferenceBackend.NUMPY)
    assert result.reference_backend == "numpy"
    assert result.reference == result.numpy_reference


def test_zero_is_handled_without_an_unnecessary_second_term() -> None:
    result = sine_taylor(0.0)
    assert result.terms_used == 1
    assert result.value == 0.0
    assert result.numerically_reliable


def test_scan_keeps_scanned_quantity_as_x_axis() -> None:
    x_values = np.linspace(-3.0, 3.0, 17)
    result = scan_sine(x_values, method=Method.RANGE_REDUCED)
    np.testing.assert_array_equal(result["x"], x_values)
    assert "stopping_criterion_met" in result
    assert "accuracy_passed" in result
    assert "numerically_reliable" in result
    assert "normalized_error" in result
    assert "false_convergence" in result
    assert "status" in result
    assert np.all(result["reference_backend"] == "mpmath")
    assert np.all(result["reference_precision_digits"] == 80)
    assert np.all(result["method"] == "range_reduced")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_range_reduced_scan_passes_scale_aware_accuracy_bound(dtype: str) -> None:
    x_values = np.linspace(-80.0, 80.0, 401)
    result = scan_sine(x_values, method=Method.RANGE_REDUCED, dtype=dtype)
    assert np.all(result["accuracy_passed"])
    assert np.all(result["numerically_reliable"])


def test_convergence_axis_is_term_count() -> None:
    result = convergence_scan(np.pi / 2, method=Method.RANGE_REDUCED, max_terms=25)
    np.testing.assert_array_equal(result["terms"], np.arange(1, 26))
    assert result["absolute_error"][-1] <= result["absolute_error"][0]


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        sine_taylor(np.inf)
    with pytest.raises(ValueError):
        sine_taylor(1.0, fixed_terms=0)
    with pytest.raises(ValueError):
        scan_sine(np.array([]), method=Method.RAW)
    with pytest.raises(ValueError):
        sine_taylor(1.0, reference_precision_digits=10)
    with pytest.raises(ValueError):
        sine_taylor(1.0, reference_backend="unknown")
    with pytest.raises(ValueError):
        sine_taylor(1.0, reference_precision_digits=80.5)


def test_scan_summary_identifies_worst_point_and_rates() -> None:
    x_values = np.linspace(-80.0, 80.0, 161)
    result = scan_sine(x_values, method=Method.RAW)
    summary = summarize_scan(result)
    assert summary["points"] == 161
    assert summary["maximum_absolute_error"] >= summary["median_absolute_error"]
    assert summary["maximum_normalized_error"] > 1.0
    assert summary["false_convergence_count"] > 0
    assert 0.0 <= summary["reliability_rate"] <= 1.0


def test_scan_summary_rejects_incomplete_input() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        summarize_scan({"x": np.array([0.0])})
