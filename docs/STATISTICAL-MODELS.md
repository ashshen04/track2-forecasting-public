## Executive summary (read this first)

Compare statistical candidates with LLM calls disabled before changing the forecast default.
For example, compare a random walk against mean reversion on the same historical dates.
All historical targets and scoring reports must remain outside this public repository.
The experimental candidates do not change the submission command or existing package.

## Run a paired experiment

```bash
.venv/bin/python scripts/backtest.py \
  --models reference reference-ewma reference-bootstrap \
  reference-drift-zero reference-drift-half reference-drift-full \
  reference-width-12 reference-mean-reversion-centre reference-mean-reversion \
  --folds 6 --min-history 121 --n-draws 1000 \
  --exclude-intervals-from /private/tmp/prior-run/summary.json \
  --out /private/tmp/new-statistical-run
```

The exclusion input must exist. Include every prior scored interval, including text experiments,
in a compatible numerical summary before claiming a previously unscored experiment.
See [BACKTEST.md](BACKTEST.md) for the interval exclusion format and limitations.
Six complete folds permit an older-three/newer-three comparison. A fold is one historical
forecast origin and its targets. Report incomplete units separately.
Methods receive identical pre-cutoff panel rows and authored horizon keys.
The shared toolkit calculates all scoring formulas. Scores are raw within-unit diagnostics.
Cards can share observations. Their results are not independent trials.

## Mean reversion candidates

For example, a level above a fitted equilibrium moves partway back each day.
Mean reversion means a tendency to return towards that equilibrium.
An AR(1), or first-order autoregressive, model predicts today's level from yesterday's level.
`baselines/mean_reversion.py` fits each asset using at most 120 adjacent business-day pairs.
Pairs spanning missing weekdays or large data gaps are excluded.
The implementation clips fitted persistence to [0, 1], then moves it halfway towards 1.
Persistence is the coefficient on the previous level. A coefficient of 1 is a random walk.
This fixed shrinkage reduces the influence of unstable short-window estimates.
Nonstationary fits and constant series use zero-drift random walks.

`reference-mean-reversion-centre` changes only the predicted centre.
It preserves the reference model's innovation samples and their cross-horizon dependence.
An innovation is an unpredictable change. This comparison isolates the centre assumption.
`reference-mean-reversion` also changes uncertainty propagation.
It fits joint residual covariance and simulates correlated Gaussian shocks each business day.
Covariance describes how asset shocks vary together. All horizons share the same simulated path.
Residuals are deviations from the fitted one-step prediction.
Stationary paths can have much narrower long-horizon intervals than a random walk.
This narrowing needs separate tail validation.

Both candidates retain the reference forecast for cumulative log-return targets.
Daily return observations are increments, so they are not treated as reverting price levels.
Monthly tasks are unsupported in this experiment and are explicitly skipped by the runner.
No calendar-vintage reconstruction or regime-switching model is supplied.

## Interpret the result

Inspect marginal, joint and tail components separately, rather than only the composite.
Marginal scores describe each individual target distribution. Joint scores measure dependence.
Tail scores emphasize extreme outcomes. A lower composite can conceal a worse tail component.
Compare fixed candidates on newer folds. If selecting among candidates, use only older folds,
then lock the choice before reporting newer-fold scores. Do not select using those newer scores.
Per-unit selection does not establish a deployment policy for a previously unseen task.
Any follow-up tuned using these results needs additional previously unused evaluation periods.
