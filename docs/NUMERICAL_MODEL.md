# Numerical Model and Reliability Decisions

## Evaluated recurrence

The application evaluates the sine Taylor series through a recurrence rather
than recomputing powers and factorials. For term index `n`, the next term is
obtained from the current term using

```text
a[n+1] = a[n] * (-x²) / ((2n + 2)(2n + 3)).
```

The raw method uses the input directly. The range-reduced method first maps the
input into `[-π/2, π/2]` while preserving its sine.

## Stopping rule

Adaptive evaluation stops when the magnitude of the last term is no greater
than a mixed absolute/relative threshold scaled by machine epsilon. A hard term
limit prevents unbounded iteration. Reaching this threshold is not treated as
proof that the accumulated sum is correct.

## Independent reference

The recommended reference is `mpmath.sin` at user-selected precision. The
input is the exact binary floating-point value selected for the experiment.
NumPy can be selected for comparison, but its result is another floating-point
calculation rather than an exact mathematical value.

## Reliability classification

A result is reliable only when:

- arithmetic remains finite;
- the stopping criterion is reached;
- absolute error is within the scale-aware bound; and
- the cancellation ratio does not exceed the precision-dependent threshold.

Normalized error is `absolute_error / allowed_absolute_error`. Values no
greater than one pass the accuracy check. False convergence means the stopping
criterion passed while the accuracy check failed.

## Domain limit

The range reducer uses the selected NumPy floating-point type and NumPy's value
of π. It is validated on `−80 ≤ x ≤ 80`. It is not a replacement for a
production-quality argument reducer for extremely large inputs.
