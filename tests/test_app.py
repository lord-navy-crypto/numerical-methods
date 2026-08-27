from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _button_by_label(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def test_app_loads_with_expected_sections() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Parameter scan",
        "Single-point analysis",
        "Method comparison",
        "Validation",
    ]
    assert any(button.label == "Run parameter scan" for button in app.button)
    assert any(metric.label == "Scan axis" for metric in app.metric)
    assert any(selectbox.label == "Reference engine" for selectbox in app.selectbox)
    assert any(metric.label == "Default oracle" for metric in app.metric)


def test_parameter_scan_runs_without_ui_exception() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    _button_by_label(app, "Run parameter scan").click()
    app.run(timeout=30)
    assert not app.exception
    metric_labels = [metric.label for metric in app.metric]
    assert "Maximum error" in metric_labels
    assert "Accuracy pass" in metric_labels
    assert "Reliable" in metric_labels
