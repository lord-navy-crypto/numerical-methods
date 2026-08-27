# Numerical Error Analysis Studio

[![Tests](https://github.com/OWNER/numerical-error-analysis-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/numerical-error-analysis-studio/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive, reproducible laboratory for investigating Taylor-series
evaluation, floating-point error, range reduction, cancellation, stopping
rules, and false convergence.

This repository is the polished successor to the original `jason_codes.ipynb`
experiment. It retains the central question—how accurately can a finite Taylor
recurrence evaluate `sin(x)`?—and turns it into a tested application with an
independent numerical core, high-precision oracle, structured exports, and
GitHub automation.

## Result-reading workflow

Every analysis page follows one consistent order:

1. **Plots first** — see the trend, failure region, or convergence behavior.
2. **Key metrics second** — read the worst error, failure rate, and reliability status.
3. **Complete data last** — inspect or download every numerical row.

## Highlights

- Raw and range-reduced Taylor algorithms
- `float32` and `float64` arithmetic
- Independent `mpmath` oracle with 20–200 decimal digits
- Absolute, relative, ULP, and normalized error
- Scale-aware acceptance bounds
- Separate stopping, accuracy, and reliability decisions
- Explicit false-convergence detection
- Cancellation-amplification diagnostics
- Parameter scans with `x` on the horizontal axis
- Single-point studies with Taylor term count on the horizontal axis
- Worst-point, RMS, median, pass-rate, and reliability summaries
- CSV data export plus JSON configuration and reproducibility reports
- Built-in validation on the original `−80 ≤ x ≤ 80` domain
- Cross-platform launchers, automated tests, and GitHub templates

## Quick start

### macOS

1. Download and unzip the release.
2. Double-click `RUN_NUMERICAL_METHODS_LAB.command`.
3. Wait for the local environment to be prepared and the browser to open.

If macOS blocks the launcher, open Terminal in the project folder and run:

```bash
chmod +x RUN_NUMERICAL_METHODS_LAB.command
./RUN_NUMERICAL_METHODS_LAB.command
```

### Windows

Double-click `RUN_NUMERICAL_METHODS_LAB.bat`.

### Manual launch

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## What the diagnostics mean

| Diagnostic | Meaning |
|---|---|
| Absolute error | Distance between the approximation and selected reference |
| Relative error | Absolute error scaled by reference magnitude; unstable near roots |
| ULP error | Error measured in local floating-point spacings |
| Allowed absolute error | Scale-aware acceptance bound based on machine epsilon and input magnitude |
| Normalized error | Absolute error divided by the allowed error; values above `1` fail |
| Cancellation ratio | Sum of absolute term magnitudes divided by the final magnitude |
| Stop rule | Whether the last term became sufficiently small |
| Accuracy pass | Whether the final value passes the reference-based bound |
| Reliable | Whether stopping, accuracy, finite arithmetic, and cancellation checks all pass |

False convergence occurs when the stop rule passes but reference accuracy
fails. This distinction is the central educational feature of the studio.

## Numerical scope

The Taylor identity is valid for every finite argument, but direct
floating-point evaluation can be destroyed by cancellation. The range-reduced
algorithm maps an angle into `[-π/2, π/2]`, keeping intermediate terms small.

The educational reducer is validated on `−80 ≤ x ≤ 80`. Correct reduction of
extremely large arguments is a separate numerical problem that requires
specialized constants and algorithms. The interface states this limit instead
of implying universal accuracy.

`mpmath` is used only as an independent oracle. It does not replace the
float32/float64 Taylor calculation under study. NumPy remains available as a
floating-point comparison, but it is not presented as exact truth.

## Repository layout

```text
.
├── app.py
├── numerical_lab/
│   ├── __init__.py
│   └── core.py
├── presets/
│   └── notebook_reference.json
├── tests/
│   ├── test_app.py
│   └── test_core.py
├── docs/
│   └── NUMERICAL_MODEL.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/tests.yml
│   ├── dependabot.yml
│   └── pull_request_template.md
├── LICENSE
├── NOTICE
├── CONTRIBUTING.md
├── SECURITY.md
├── CITATION.cff
├── CHANGELOG.md
└── requirements.txt
```

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q app.py numerical_lab tests
```

GitHub Actions runs the same suite on supported Python versions.

## Publishing this folder to GitHub

Create an empty GitHub repository, replace `OWNER` in `README.md`,
`pyproject.toml`, and `CITATION.cff`, then run:

```bash
git init
git add .
git commit -m "Release Numerical Error Analysis Studio v1.2.0"
git branch -M main
git remote add origin https://github.com/OWNER/numerical-error-analysis-studio.git
git push -u origin main
```

Do not commit `.venv/`; it is excluded by `.gitignore`.

## License and attribution

Project-authored source code and documentation are available under the
[MIT License](LICENSE), copyright © 2026 Jason and contributors.

Third-party packages retain their own licenses and copyrights. See [NOTICE](NOTICE)
for the dependency boundary. No Project Chrono source code is included in this
repository; the word “numerical” here refers to numerical analysis in general.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Report
security-sensitive problems privately as described in [SECURITY.md](SECURITY.md),
not through a public issue.
