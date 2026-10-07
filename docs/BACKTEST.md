## Executive summary (read this first)

Run `python scripts/backtest.py --out /private/tmp/t2-backtest-01` from the
repository root after activating `.venv`. This scans every unit and runs the
same Gaussian random-walk scaffold used by the worked example. It uses no text.
Results must be outside this public repository. Choose a new directory per run.

```bash
source .venv/bin/activate
python scripts/backtest.py --out /private/tmp/t2-backtest-01
python scripts/backtest.py --pattern 't2-F1-*' --folds 5 --out /private/tmp/t2-f1-01
python scripts/backtest.py --pattern 't2-EXAMPLE-*' --out /private/tmp/t2-example-01
```

A holdout is a historical interval withheld from the model and used for scoring.
For example, a forecast made on a Friday at horizon 1 targets the following Monday.
The runner selects up to three recent non-overlapping holdouts per unit. Each needs
at least 63 complete historical dates. Change this with `--min-history`.
The default is 1,000 draws. The adapter fixes its random seed to 1.

The original horizon keys and asset order remain unchanged. Level targets use
the exact weekday endpoint. Cumulative log-return targets sum the shared toolkit
conversion over each future weekday. Missing dates or values are never filled.
This conservative Monday-Friday convention can reject market-holiday intervals;
it is not an exchange-calendar or official historical target-date reconstruction.

Monthly declarations are explicitly skipped. The scaffold does not implement their
observation-period mapping, and these snapshots do not establish historical release
availability or vintages. Sparse data, insufficient history, and missing endpoints
also produce explicit skip reasons. Errors produce `failed` and a nonzero exit code.
A completed unit can have fewer folds than requested; inspect the fold count.

Each model call receives only panel rows at or before its simulated cutoff.
No current unit text or future rows reach the model. This is a retrospective
fixed-snapshot numerical experiment, not a reconstruction of historical information
availability. The original unit's future target is never scored.

`summary.json` contains settings, counts, all unit results and toolkit version.
Per-unit JSON files contain source hashes, dates, warnings and score components.
`report.md` is the readable status index. Temporary output directories may be cleaned
by the operating system; use a durable directory outside this repository to retain runs.

Scores call the existing Track 2 scorer, including its single-cell weight handling,
which delegates metric formulas to `qfbench2-common`. No scoring math is copied.
Scores are raw, unnormalized diagnostics. Compare methods only on identical units
and fold dates. There is no cross-unit average because scales differ. Many units
share historical observations, so their results are not independent evidence.
This runner does not run submission admissibility gates or official evaluations.
Example-panel provenance is unverified; its results demonstrate the workflow only.

## Compare the two shipped implementations

```bash
python scripts/backtest.py --models scaffold reference \
  --include t2-F1-aud-on-hold-2016 t2-F1-measured-pace-2004 \
  t2-F3-brexit-joint-2016 t2-F3-covid-curve-2020 \
  t2-F4-covid-mkt-2020 t2-F4-factor-stress-2008 \
  --out /private/tmp/t2-comparison-01
```

`scaffold` is the example's ThetaARIMABaseline Gaussian random-walk placeholder.
`reference` calls the numerical sampler used by the official forecast CLI.
The latter models asset correlations and uses different drift rules. Comparing
these two complete implementations does not isolate the effect of correlation.

Both implementations receive the same truncated panels, assets, horizon keys,
origins and sample count. Random seed 1 is fixed for each. The report shows paired
raw scores per unit; a negative reference-minus-scaffold difference favours the
reference. Component scores for each fold are in `model_scores`. The legacy
`scores` and `mean_raw_composite` fields refer to the first selected model.
The summary records implementation hashes so that subsequent edits are detectable.

This initial six-unit subset is diagnostic development data. It is not an untouched
validation set. Before tuning new parameters, choose separate chronological
evaluation periods and account for shared histories across units. A unit name
describes its original task; earlier holdouts need not cover that named event.

## Isolate cross-asset dependence

Use `--models reference reference-shuffled` with the same `--include` list.
The shuffled variant first calls the reference sampler unchanged, then permutes
draw indices separately for assets after the first, using permutation seed 9173.
Each asset's vector across horizons moves together. Individual marginal samples,
their CRPS and tail scores, and within-asset horizon relationships are preserved.
Only the pairing across assets changes. Residual sample correlation can remain.
Single-asset tasks are unchanged and act as controls.

The dependence report explicitly uses `reference minus shuffled`: negative favours
the original dependence. This differs from the earlier scaffold comparison's order;
always read the column heading. One permutation and three dates per unit are only
a diagnostic, not evidence of statistical significance or performance on unseen data.
