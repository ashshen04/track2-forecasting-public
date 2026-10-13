## Executive summary (read this first)

Freeze unused historical return-target windows before comparing four fixed policies.
For example, first run with --plan and all prior summaries, then run with --run.
All historical targets, cohort files and scoring reports remain outside this repository.
This experiment evaluates a candidate and does not change the formal forecast policy.

## Run the comparison

```bash
.venv/bin/python scripts/validate_return_calibration.py --plan \
  --prior-summary /private/tmp/prior-numeric/summary.json /private/tmp/prior-text/summary.json \
  --out /private/tmp/new-return-calibration
.venv/bin/python scripts/validate_return_calibration.py --run \
  --out /private/tmp/new-return-calibration
```

Supply every known prior numerical and text summary. The example paths must exist.
The planner also understands event-response intervals embedded in event analog reports.
It records source hashes. Older summaries lacking assets use the authored grid and
adjustment keys conservatively, with the inference and card hash recorded explicitly.
The exclusion ledger is only as complete as the supplied experiment history.

## Frozen candidate and data rules

The four policies are the current reference, half drift, width multiplied by 1.2,
and half drift combined with width multiplied by 1.2. Drift is the mean historical
log-return step projected over the horizon. Width is the scale of random innovations.
All policies share each seed's correlated Gaussian shocks and statistical history.
Only daily cumulative log-return targets are eligible. Authored assets and horizons
are preserved. Neither monthly targets nor level forecasts are calibrated here.

Before any scoring, cohort.json freezes feasible windows, seeds 1/7/19 and 1,000 draws.
The planner prioritizes more assets, then longer maximum horizons, then unit name.
Each card receives at most 12 latest feasible windows with at least 121 history rows.
Prior inspected intervals and new reserved intervals are excluded for every shared asset.
Intervals are open at the origin and closed at the final target. Touching boundaries
are allowed. Windows for disjoint asset sets can still coincide in calendar time.
Missing weekday return observations are refused. No observations are filled.
The scorer checks source hashes against the frozen cohort before computing predictions.
Model inputs are physically truncated at each origin. Target observations are used
only for evaluating predictions. Scoring formulas come from the shared toolkit.

## Acceptance and interpretation

The fixed combination needs at least 12 intervals spanning three origin years.
At each seed it must lower median paired composite error, win more intervals than
it loses, and avoid increases in median joint and tail error. Numerical tolerance
for joint and tail medians is 1e-10 percentage points. No automatic promotion occurs.
An individual policy's results are reported even when the combination fails.

Percentage changes compare each policy against reference on the identical interval.
Raw scores from different task grids are not averaged. Components with a zero
reference score have no meaningful percentage change and are omitted from that median.
Inspect component denominator counts, source warnings and asset/time subgroups before
interpreting an aggregate result. Repeated seeds test sampling robustness only.
Shared training data and coincident market regimes still create dependencies.
The earliest remaining windows may be much older than previously inspected periods.
Fixed snapshots do not reconstruct historical publication vintages or establish
future performance. A candidate needs additional recent-period evidence before deployment.
