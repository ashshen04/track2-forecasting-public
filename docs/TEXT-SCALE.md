## Executive summary (read this first)

Structured text correction can be multiplied by zero, one quarter, one half or one.
For example, `forecast ... --text-policy structured --text-scale 0.5` halves the
validated correction. `--text-scale 0` produces the numerical forecast without a
model call. The default multiplier remains one. No candidate is promoted by this change.

## Forecast behavior

The multiplier changes the centre, meaning the location of the forecast distribution.
It preserves the numerical model's uncertainty, covariance of simulated residuals,
and shared shocks across horizons. Relative forecast centres can still change, so
the joint score can change too. It applies after dated evidence, event timing and horizon checks.
An unsupported event remains rejected at every multiplier. The structured adjustment
ledger records the multiplier and the applied horizon shifts.

The option is available through both forecast entry points. Non-default multipliers
require structured mode. Values outside zero to one, including non-finite values, are
refused. Existing conservative and default structured predictions retain their behavior.

## Paired research evaluation

`scripts/validate_text_scale.py` first freezes a manifest, then replays saved version-3
replies. Its fixed multipliers are 0, 0.25, 0.5 and 1. Its numerical seeds are 1, 7
and 19. Each comparison uses identical draws. Saved model failures retain their
numerical fallback and remain in the research denominator. Reply reuse is not a new
model observation. The exemplar is replayed but excluded from aggregate loss.

Example, with paths outside this repository:

```bash
python scripts/validate_text_scale.py --plan --source-summary /tmp/v3/summary.json --availability /tmp/availability.json --out /tmp/text-scales
python scripts/validate_text_scale.py --run --out /tmp/text-scales
```

The objective is the mean loss across cards, after averaging repeated windows within
each card and clipping each research loss to the published zero-to-eight range.
Three component errors are normalized before their authored weights are applied by
the existing scorer. The single-cell weight rule comes from that scorer too.

`scripts/m0_normalization_proxy.py` estimates normalization for daily grids from the
[published M0 procedure](M0-BASELINE.md). It uses 300 trailing rows, dated alignment,
gap removal, direct return rows, and a Gaussian path shared across horizons. Two
independent simulations each use 500 forecast draws and 256 baseline-drawn outcomes.
The existing scorer computes every component. No CRPS, variogram or tail formula is
reimplemented. Monte Carlo means averaging over simulated outcomes. Real future
observations cannot enter normalization. Conditional simulation standard errors are
recorded; these do not include forecast-ensemble sampling error.

These are research proxies. They differ from the organizer's exact analytical scales,
particularly through simulation and finite forecast draws. Monthly step conversion is
unsupported here. A daily grid uses its declared horizons. The proxy neither creates
verified reference files nor produces rankable results.

## Evidence and promotion

The availability audit excludes previously scored intervals globally by shared asset.
If no new interval survives, amplitude comparisons remain inspected-window diagnostics.
A temporal split does not make already inspected data independent. The runner cannot
promote a candidate or start the half-drift stage from these diagnostics. Additional
independent evidence is required before changing the submission strategy.

Scored results, saved replies, input hashes and simulated normalization artifacts stay
outside this public repository. The submitted image and ZIP are separate artifacts and
are not rebuilt by either script.
