# SPDX-License-Identifier: MIT
"""Streamlit interface for the Numerical Error Analysis Studio."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from numerical_lab.core import (
    Method,
    ReferenceBackend,
    convergence_scan,
    machine_epsilon,
    scan_sine,
    sine_taylor,
    summarize_scan,
)

APP_VERSION = "1.2.0"
METHOD_LABELS = {
    "Range-reduced Taylor (recommended)": Method.RANGE_REDUCED,
    "Raw Taylor (instability diagnostic)": Method.RAW,
}
REFERENCE_LABELS = {
    "mpmath arbitrary precision (recommended)": ReferenceBackend.MPMATH,
    "NumPy floating-point comparison": ReferenceBackend.NUMPY,
}

st.set_page_config(
    page_title="Numerical Error Analysis Studio",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(145deg, #f6fbff 0%, #ffffff 46%, #f5f7fb 100%);}
    .hero {padding: 1.45rem 1.7rem; border-radius: 20px; color: white;
      background: linear-gradient(120deg, #102a43 0%, #145a73 55%, #168aad 100%);
      box-shadow: 0 14px 34px rgba(16,42,67,.18); margin-bottom: 1rem;}
    .hero h1 {font-size: 2.15rem; margin: 0 0 .35rem 0;}
    .hero p {margin: 0; opacity: .94; font-size: 1rem;}
    .note {padding: .85rem 1rem; border-left: 4px solid #168aad; border-radius: 8px;
      background: #eef8fc; margin: .5rem 0 1rem 0;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #d9e4ec;
      padding: .72rem; border-radius: 14px; box-shadow: 0 5px 16px rgba(16,42,67,.05);}
    </style>
    """,
    unsafe_allow_html=True,
)


def default_config() -> dict[str, object]:
    return {
        "method_label": "Range-reduced Taylor (recommended)",
        "dtype": "float64",
        "reference_label": "mpmath arbitrary precision (recommended)",
        "reference_precision_digits": 80,
        "tolerance_multiplier": 8.0,
        "max_terms": 120,
        "x_min": -80.0,
        "x_max": 80.0,
        "scan_points": 1000,
        "single_x": float(np.pi / 2),
        "convergence_terms": 50,
    }


def initialize_state() -> None:
    for key, value in default_config().items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("scan_result", None)
    st.session_state.setdefault("comparison_result", None)


def _validated_config(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    candidate = default_config()
    for key, fallback in candidate.items():
        if key in raw:
            candidate[key] = type(fallback)(raw[key])
    if candidate["method_label"] not in METHOD_LABELS:
        raise ValueError("unknown calculation method")
    if candidate["reference_label"] not in REFERENCE_LABELS:
        raise ValueError("unknown reference backend")
    if candidate["dtype"] not in {"float32", "float64"}:
        raise ValueError("floating-point precision must be float32 or float64")
    if not 20 <= int(candidate["reference_precision_digits"]) <= 200:
        raise ValueError("reference precision must be between 20 and 200 digits")
    if float(candidate["x_min"]) >= float(candidate["x_max"]):
        raise ValueError("x maximum must be greater than x minimum")
    if not 50 <= int(candidate["scan_points"]) <= 5000:
        raise ValueError("scan points must be between 50 and 5000")
    if not 10 <= int(candidate["max_terms"]) <= 200:
        raise ValueError("maximum Taylor terms must be between 10 and 200")
    if float(candidate["tolerance_multiplier"]) <= 0:
        raise ValueError("tolerance multiplier must be positive")
    return candidate


def load_uploaded_config(uploaded_file) -> None:
    if uploaded_file is None:
        return
    signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("loaded_config_signature") == signature:
        return
    try:
        loaded = json.loads(uploaded_file.getvalue().decode("utf-8"))
        for key, value in _validated_config(loaded).items():
            st.session_state[key] = value
        st.session_state.scan_result = None
        st.session_state.comparison_result = None
        st.session_state.loaded_config_signature = signature
        st.success("Configuration loaded. Review the controls before running.")
    except (json.JSONDecodeError, ValueError, TypeError, UnicodeDecodeError) as exc:
        st.error(f"Could not load configuration: {exc}")


def current_config() -> dict[str, object]:
    return {
        "schema": "numerical-error-studio-config-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        **{key: st.session_state[key] for key in default_config()},
    }


def selected_reference_backend() -> ReferenceBackend:
    return REFERENCE_LABELS[st.session_state.reference_label]


def positive_for_log(values: np.ndarray, dtype: str) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), np.finfo(np.dtype(dtype)).tiny)


def line_figure(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    title: str,
    x_title: str,
    y_title: str,
    log_y: bool = False,
) -> go.Figure:
    colors = ["#146c94", "#ef8354", "#3a7d44", "#7b2cbf", "#bc4749"]
    figure = go.Figure()
    for index, (label, values) in enumerate(series.items()):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                name=label,
                line={"width": 2, "color": colors[index % len(colors)]},
                hovertemplate=f"{x_title}: %{{x:.8g}}<br>{label}: %{{y:.8g}}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_type="log" if log_y else "linear",
        template="plotly_white",
        height=420,
        margin={"l": 55, "r": 25, "t": 60, "b": 55},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    return figure


def run_scan(method: Method) -> dict[str, np.ndarray]:
    x_values = np.linspace(
        float(st.session_state.x_min),
        float(st.session_state.x_max),
        int(st.session_state.scan_points),
    )
    progress = st.progress(0, text="Preparing scan…")

    def update(done: int, total: int) -> None:
        progress.progress(done / total, text=f"Evaluating scan points: {done:,}/{total:,}")

    result = scan_sine(
        x_values,
        method=method,
        dtype=st.session_state.dtype,
        tolerance_multiplier=float(st.session_state.tolerance_multiplier),
        max_terms=int(st.session_state.max_terms),
        reference_backend=selected_reference_backend(),
        reference_precision_digits=int(st.session_state.reference_precision_digits),
        progress_callback=update,
    )
    progress.progress(1.0, text="Scan complete")
    return result


def scan_report(result: dict[str, np.ndarray]) -> bytes:
    report = {
        "schema": "numerical-error-studio-report-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        "configuration": current_config(),
        "summary": summarize_scan(result),
        "column_order": list(result),
    }
    return json.dumps(_json_safe(report), indent=2, allow_nan=False).encode("utf-8")


def _json_safe(value):
    """Convert NumPy and non-finite values into portable JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "nan"
        return "infinity" if value > 0 else "-infinity"
    return value


initialize_state()
st.markdown(
    """
    <section class="hero">
      <h1>Numerical Error Analysis Studio</h1>
      <p>See the behavior first, read the decisive metrics next, and inspect every data row last.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment configuration")
    load_uploaded_config(st.file_uploader("Import configuration", type=["json"]))
    st.selectbox("Method", list(METHOD_LABELS), key="method_label")
    st.selectbox("Floating-point precision", ["float64", "float32"], key="dtype")
    st.selectbox("Reference engine", list(REFERENCE_LABELS), key="reference_label")
    st.slider(
        "mpmath decimal precision",
        20,
        200,
        step=10,
        key="reference_precision_digits",
        disabled=selected_reference_backend() is ReferenceBackend.NUMPY,
    )
    st.caption("The oracle precision does not change the float32/float64 Taylor arithmetic.")
    st.number_input(
        "Tolerance multiplier × machine epsilon",
        min_value=1.0,
        max_value=1000.0,
        step=1.0,
        key="tolerance_multiplier",
    )
    st.slider("Maximum Taylor terms", 10, 200, key="max_terms")
    st.divider()
    st.caption(f"Machine epsilon: {machine_epsilon(st.session_state.dtype):.8e}")
    st.caption(f"Studio version: {APP_VERSION}")
    st.download_button(
        "Download configuration",
        data=json.dumps(current_config(), indent=2).encode("utf-8"),
        file_name="numerical_error_studio_config.json",
        mime="application/json",
        width="stretch",
    )
    if st.button("Reset configuration", width="stretch"):
        for key, value in default_config().items():
            st.session_state[key] = value
        st.session_state.scan_result = None
        st.session_state.comparison_result = None
        st.rerun()

overview_tab, scan_tab, single_tab, comparison_tab, validation_tab = st.tabs(
    ["Overview", "Parameter scan", "Single-point analysis", "Method comparison", "Validation"]
)

with overview_tab:
    st.subheader("What the studio measures")
    st.markdown(
        r"""
        The Taylor series for $\sin(x)$ is mathematically valid for every finite $x$, yet its
        direct floating-point evaluation can fail through cancellation. This studio separates
        a **small final term**, an **accurate answer**, and a **numerically reliable algorithm**.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated domain", "−80 to 80 rad")
    c2.metric("Arithmetic", "float32 / float64")
    c3.metric("Scan axis", "x (radians)")
    c4.metric("Default oracle", "mpmath · 80 dps")
    st.markdown(
        '<div class="note"><b>Reading order:</b> every results page presents plots first, '
        "decision metrics second, and downloadable complete data last.</div>",
        unsafe_allow_html=True,
    )
    st.latex(r"\sin x = \sum_{n=0}^{\infty}(-1)^n\frac{x^{2n+1}}{(2n+1)!}")
    st.markdown(
        r"Range reduction maps the angle into $[-\pi/2,\pi/2]$ before evaluating the recurrence. "
        "Smaller intermediate terms sharply reduce cancellation and make the stopping rule meaningful."
    )

with scan_tab:
    st.subheader("Parameter scan")
    c1, c2, c3 = st.columns(3)
    c1.number_input("x minimum (radians)", key="x_min", step=1.0)
    c2.number_input("x maximum (radians)", key="x_max", step=1.0)
    c3.slider("Number of scan points", 50, 5000, step=50, key="scan_points")
    invalid_bounds = float(st.session_state.x_min) >= float(st.session_state.x_max)
    if invalid_bounds:
        st.error("x maximum must be greater than x minimum.")
    run_col, clear_col = st.columns([3, 1])
    run_clicked = run_col.button(
        "Run parameter scan", type="primary", disabled=invalid_bounds, width="stretch"
    )
    if clear_col.button("Clear results", width="stretch"):
        st.session_state.scan_result = None
        st.rerun()
    if run_clicked:
        st.session_state.scan_result = run_scan(METHOD_LABELS[st.session_state.method_label])

    result = st.session_state.scan_result
    if result is None:
        st.info("Choose a range and run the scan. The result remains available across tabs.")
    else:
        frame = pd.DataFrame(result)
        summary = summarize_scan(result)
        st.caption("Plots — visual evidence")
        st.plotly_chart(
            line_figure(
                result["x"],
                {f"{result['reference_backend'][0]} reference": result["reference"],
                 "Taylor approximation": result["approximation"]},
                title="Reference and approximation across x",
                x_title="x (radians)",
                y_title="sin(x)",
            ),
            width="stretch",
        )
        p1, p2 = st.columns(2)
        p1.plotly_chart(
            line_figure(
                result["x"],
                {"Absolute error": positive_for_log(result["absolute_error"], st.session_state.dtype),
                 "Acceptance bound": positive_for_log(result["allowed_absolute_error"], st.session_state.dtype)},
                title="Error versus the scale-aware acceptance bound",
                x_title="x (radians)",
                y_title="Magnitude",
                log_y=True,
            ),
            width="stretch",
        )
        p2.plotly_chart(
            line_figure(
                result["x"],
                {"Normalized error": positive_for_log(result["normalized_error"], st.session_state.dtype),
                 "Pass/fail threshold": np.ones_like(result["x"])},
                title="Normalized error (values above 1 fail)",
                x_title="x (radians)",
                y_title="Absolute error / allowed error",
                log_y=True,
            ),
            width="stretch",
        )
        st.caption("Key metrics — decision summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Maximum error", f"{summary['maximum_absolute_error']:.3e}")
        m2.metric("Worst x", f"{summary['worst_x']:.7g}")
        m3.metric("RMS error", f"{summary['rms_absolute_error']:.3e}")
        m4.metric("Accuracy pass", f"{summary['accuracy_pass_rate'] * 100:.1f}%")
        m5.metric("Reliable", f"{summary['reliability_rate'] * 100:.1f}%")
        false_count = int(summary["false_convergence_count"])
        if false_count:
            st.warning(
                f"False convergence occurred at {false_count:,} point(s): the term rule stopped, "
                "but the reference-based accuracy test failed."
            )
        else:
            st.success("No false-convergence points were found in this completed scan.")
        st.caption("Complete data — export and inspection")
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download complete scan (CSV)",
            frame.to_csv(index=False).encode("utf-8"),
            "numerical_error_scan.csv",
            "text/csv",
            width="stretch",
        )
        d2.download_button(
            "Download reproducibility report (JSON)",
            scan_report(result),
            "numerical_error_scan_report.json",
            "application/json",
            width="stretch",
        )
        with st.expander("View all numerical rows"):
            st.dataframe(frame, width="stretch", hide_index=True)

with single_tab:
    st.subheader("Single-point convergence")
    c1, c2 = st.columns(2)
    c1.number_input("Single x value (radians)", key="single_x", format="%.10f")
    c2.slider("Terms in convergence study", 5, 150, key="convergence_terms")
    method = METHOD_LABELS[st.session_state.method_label]
    single = sine_taylor(
        float(st.session_state.single_x),
        method=method,
        dtype=st.session_state.dtype,
        tolerance_multiplier=float(st.session_state.tolerance_multiplier),
        max_terms=int(st.session_state.max_terms),
        reference_backend=selected_reference_backend(),
        reference_precision_digits=int(st.session_state.reference_precision_digits),
    )
    convergence = convergence_scan(
        float(st.session_state.single_x),
        method=method,
        dtype=st.session_state.dtype,
        max_terms=int(st.session_state.convergence_terms),
        reference_backend=selected_reference_backend(),
        reference_precision_digits=int(st.session_state.reference_precision_digits),
    )
    st.caption("Plot — convergence behavior")
    st.plotly_chart(
        line_figure(
            convergence["terms"],
            {"Absolute error": positive_for_log(convergence["absolute_error"], st.session_state.dtype),
             "|last Taylor term|": positive_for_log(convergence["last_term"], st.session_state.dtype)},
            title=f"Convergence at x = {float(st.session_state.single_x):.8g}",
            x_title="Number of Taylor terms",
            y_title="Magnitude",
            log_y=True,
        ),
        width="stretch",
    )
    st.caption("Key metrics — decision summary")
    a, b, c, d, e = st.columns(5)
    a.metric("Approximation", f"{single.value:.12g}")
    b.metric("Reference", f"{single.reference:.12g}")
    c.metric("Absolute error", f"{single.absolute_error:.3e}")
    d.metric("Normalized error", f"{single.normalized_error:.3e}")
    e.metric("Status", single.status.replace("_", " ").title())
    if single.false_convergence:
        st.error("False convergence: the stop rule passed while the reference accuracy test failed.")
    elif not single.numerically_reliable:
        st.warning("This result is not classified as reliable; inspect the diagnostics below.")
    else:
        st.success("The stopping, accuracy, and cancellation checks all passed.")
    st.caption("Complete data — internal values and convergence table")
    detail_frame = pd.DataFrame(convergence)
    with st.expander("View reference and all diagnostics"):
        st.code(single.reference_text, language=None)
        st.json({
            "reference_backend": single.reference_backend,
            "reference_precision_digits": single.reference_precision_digits,
            "reduced_x": single.reduced_x,
            "terms_used": single.terms_used,
            "last_term": single.last_term,
            "cancellation_ratio": single.cancellation_ratio,
            "allowed_absolute_error": single.allowed_absolute_error,
            "ulp_error": single.ulp_error,
            "stopping_criterion_met": single.stopping_criterion_met,
            "accuracy_passed": single.accuracy_passed,
            "numerically_reliable": single.numerically_reliable,
        })
        st.dataframe(detail_frame, width="stretch", hide_index=True)
    st.download_button(
        "Download convergence data (CSV)",
        detail_frame.to_csv(index=False).encode("utf-8"),
        "single_point_convergence.csv",
        "text/csv",
    )

with comparison_tab:
    st.subheader("Method comparison")
    st.write("Both methods use the same x values, precision, oracle, and term limit.")
    if st.button("Run method comparison", type="primary"):
        x_values = np.linspace(
            float(st.session_state.x_min), float(st.session_state.x_max), int(st.session_state.scan_points)
        )
        progress = st.progress(0, text="Running range-reduced Taylor…")
        reduced = scan_sine(
            x_values, method=Method.RANGE_REDUCED, dtype=st.session_state.dtype,
            tolerance_multiplier=float(st.session_state.tolerance_multiplier),
            max_terms=int(st.session_state.max_terms), reference_backend=selected_reference_backend(),
            reference_precision_digits=int(st.session_state.reference_precision_digits),
        )
        progress.progress(0.5, text="Running raw Taylor…")
        raw = scan_sine(
            x_values, method=Method.RAW, dtype=st.session_state.dtype,
            tolerance_multiplier=float(st.session_state.tolerance_multiplier),
            max_terms=int(st.session_state.max_terms), reference_backend=selected_reference_backend(),
            reference_precision_digits=int(st.session_state.reference_precision_digits),
        )
        progress.progress(1.0, text="Comparison complete")
        st.session_state.comparison_result = {"reduced": reduced, "raw": raw}
    comparison = st.session_state.comparison_result
    if comparison is not None:
        reduced, raw = comparison["reduced"], comparison["raw"]
        st.caption("Plot — method behavior")
        st.plotly_chart(
            line_figure(
                reduced["x"],
                {"Range-reduced": positive_for_log(reduced["absolute_error"], st.session_state.dtype),
                 "Raw Taylor": positive_for_log(raw["absolute_error"], st.session_state.dtype)},
                title="Absolute error comparison", x_title="x (radians)",
                y_title="Absolute error", log_y=True,
            ),
            width="stretch",
        )
        reduced_summary, raw_summary = summarize_scan(reduced), summarize_scan(raw)
        st.caption("Key metrics — side-by-side summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reduced max error", f"{reduced_summary['maximum_absolute_error']:.3e}")
        m2.metric("Raw max error", f"{raw_summary['maximum_absolute_error']:.3e}")
        m3.metric("Reduced reliable", f"{reduced_summary['reliability_rate'] * 100:.1f}%")
        m4.metric("Raw reliable", f"{raw_summary['reliability_rate'] * 100:.1f}%")
        comparison_frame = pd.DataFrame({
            "x": reduced["x"], "reference": reduced["reference"],
            "range_reduced_value": reduced["approximation"],
            "range_reduced_absolute_error": reduced["absolute_error"],
            "range_reduced_normalized_error": reduced["normalized_error"],
            "range_reduced_status": reduced["status"],
            "raw_value": raw["approximation"], "raw_absolute_error": raw["absolute_error"],
            "raw_normalized_error": raw["normalized_error"], "raw_status": raw["status"],
        })
        st.caption("Complete data — export and inspection")
        st.download_button(
            "Download complete comparison (CSV)",
            comparison_frame.to_csv(index=False).encode("utf-8"),
            "taylor_method_comparison.csv",
            "text/csv",
        )
        with st.expander("View all comparison rows"):
            st.dataframe(comparison_frame, width="stretch", hide_index=True)

with validation_tab:
    st.subheader("Built-in numerical validation")
    checks = []
    for test_x in (-80.0, -10.0, -np.pi, 0.0, np.pi / 2, 10.0, 80.0):
        check = sine_taylor(
            test_x, method=Method.RANGE_REDUCED, dtype="float64",
            reference_backend=ReferenceBackend.MPMATH, reference_precision_digits=100,
        )
        checks.append({
            "x": test_x, "absolute_error": check.absolute_error,
            "normalized_error": check.normalized_error, "terms": check.terms_used,
            "status": check.status, "passed": check.accuracy_passed and check.terms_used <= 20,
        })
    check_frame = pd.DataFrame(checks)
    st.caption("Plot — validation error profile")
    st.plotly_chart(
        line_figure(
            check_frame.x.to_numpy(),
            {"Absolute error": positive_for_log(check_frame.absolute_error.to_numpy(), "float64")},
            title="Validation points across the notebook domain",
            x_title="x (radians)", y_title="Absolute error", log_y=True,
        ),
        width="stretch",
    )
    st.caption("Key metrics — validation result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Checks passed", f"{int(check_frame.passed.sum())}/{len(check_frame)}")
    c2.metric("Largest validation error", f"{check_frame.absolute_error.max():.3e}")
    c3.metric("Maximum terms", f"{int(check_frame.terms.max())}")
    raw_diagnostic = sine_taylor(
        80.0, method=Method.RAW, dtype="float64",
        reference_backend=ReferenceBackend.MPMATH, reference_precision_digits=100,
    )
    if bool(check_frame.passed.all()) and raw_diagnostic.false_convergence:
        st.success("Domain validation and the intentional false-convergence detector check passed.")
    else:
        st.error("One or more built-in checks failed.")
    st.caption("Complete data — validation rows")
    with st.expander("View all validation rows"):
        st.dataframe(check_frame, width="stretch", hide_index=True)
    st.warning(
        "For extremely large arguments, robust range reduction requires specialized constants and algorithms. "
        "This educational implementation is validated on −80 ≤ x ≤ 80."
    )
