## Executive summary (read this first)

The experimental daily model changes conditional uncertainty and tail shape.
For example, a recent large shock raises near-term forecast variance, which then
returns towards its historical level. This is an explicit hypothesis, not a proven
forecast improvement. Gaussian remains the CLI default. Monthly CPI is unchanged.

## Run the candidate

```bash
forecast --panels /input/panels --text /input/text --asof 2024-06-28 \
  --out /output/forecast.parquet --text-policy conservative \
  --statistical-model adaptive-t --n-draws 4000
```

Five daily distribution choices permit separate comparisons:

| Choice | Conditional variance | Forecast tail shape |
|---|---|---|
| gaussian | Existing constant historical variance | Existing Gaussian |
| adaptive-gaussian | Fixed GARCH-style recursion | Gaussian |
| student-t | Constant historical variance | Shared Student-t scale |
| adaptive-t | Fixed GARCH-style recursion | Shared Student-t scale |
| historical-t | Constant historical variance | History-dependent Student-t or Gaussian |

## Conditional variance

GARCH describes variance changing in response to earlier squared unexpected changes.
The [official ARCH documentation](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html)
describes the conditional-variance equation and Student-t error models.
This implementation uses that variance recursion with fixed alpha 0.06, beta 0.92
and an intercept of 0.02 times the 120-row historical sample variance.
It does not install ARCH, fit a likelihood, or estimate those coefficients.
Forecast variance reverts towards historical variance at persistence 0.98 per day.
Forecasts integrate the expected variances over the authored business-day horizons.

Adaptive modes move empirical correlation 10% towards the identity matrix.
This shrinkage reduces extreme correlations estimated from short samples.
The Gaussian control retains its existing correlation estimator.

## Heavy-tail hypothesis

A Student-t distribution assigns more probability to extreme standardized changes.
The degrees of freedom are fixed at seven. Each draw receives one positive risk
scale for the whole forecast path, including every asset and horizon. Its squared
scale has mean one, so tail shape changes without mechanically increasing modeled
variance. The resulting forecast is a multivariate variance mixture. It is not a
fitted Student-t GARCH simulation, and daily shocks are not unconditionally independent.

Level forecast centres and cumulative log-return drift remain as in the control.
Monthly cards retain their existing sampler and CPI trend; the adaptive option has
no effect on those draws. All history is truncated at the forecast origin.
Stale history remains stale: this model does not infer current volatility from old
panels or invent a contemporaneous proxy series.

## Validation and limitations

The historical-t refinement estimates excess kurtosis from standardized changes
in the same 120-row pre-cutoff history. Kurtosis measures the fourth moment, or
sensitivity to large standardized changes. Nonconstant assets are pooled using
their median excess kurtosis. Values at or below 0.25 keep Gaussian innovations.
Otherwise, half shrinkage towards Gaussian gives degrees of freedom equal to
4 plus 12 divided by excess kurtosis, clipped to [5, 60]. These are explicit
heuristics, not likelihood estimates or independently selected optimum thresholds.
The refinement was proposed after development diagnostics and needs fresh validation.

Synthetic tests check covariance, tail shape, recent-shock response, shared horizon
paths, future-row exclusion and both formal CLI routes. Historical experiments call
the shared scorer and remain outside this public repository.
Previously inspected windows can diagnose effects but cannot establish independent
accuracy. Increasing tail probability can hurt central and tail scores; conditional
variance can react to a shock that has already passed. These assumptions need testing.
Metadata records the actual model, fixed coefficients, cumulative variance and tail
parameters. Explicit options cannot be combined with the older drift, width, EWMA
or empirical-bootstrap calibrations.
