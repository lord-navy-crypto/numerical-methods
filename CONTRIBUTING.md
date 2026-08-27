# Contributing

Thank you for improving the Numerical Error Analysis Studio.

## Before opening a pull request

1. Create a focused branch from `main`.
2. Keep numerical algorithms separate from Streamlit presentation code.
3. Add or update tests for every behavior change.
4. Preserve the results order: plots, key metrics, then complete data.
5. Run `python -m pytest -q` and `python -m compileall -q app.py numerical_lab tests`.
6. Update `CHANGELOG.md` when a user-visible feature changes.

## Numerical changes

Document the reference value, precision, domain, error measure, and acceptance
criterion. Never label a floating-point comparison as exact. If a method is
validated only on a bounded domain, state that limit in code, tests, and UI.

## Issues

Use the bug template for reproducible defects and the feature template for new
ideas. Do not include passwords, tokens, private datasets, or other secrets.

By contributing, you agree that your contribution may be distributed under
the repository's MIT License.
