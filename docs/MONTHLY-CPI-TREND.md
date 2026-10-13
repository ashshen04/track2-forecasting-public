## Executive summary (read this first)

Experimental monthly CPI trends change centres while preserving the existing joint
random shocks and widths. For example, `--monthly-trend damped` uses twelve earlier
monthly changes and reduces each subsequent trend increment by a factor of 0.9.
The option applies only to explicitly declared monthly CPI index level targets.
The formal default is `auto`: damped trend for declared monthly CPI targets, and no
trend for other targets. `--monthly-trend none` retains the original CPI control.
Validation and submission packaging are separate steps.

## Fixed candidate models

All candidates use the same twelve consecutive monthly changes. `recent` uses their
mean as a constant monthly trend. `half` uses half that mean. `damped` uses the full
mean for the first transition and multiplies each following transition by 0.9 again.
`none` reproduces the existing no-trend forecast. Negative trends remain possible.

The target observation period determines how many increments are accumulated.
A horizon key of 140 can correspond to eight monthly transitions. The code retains
140 in the output and accumulates eight trend increments. Publication lag is already
included in those transitions. It never divides horizon keys by an assumed month length.

The same Gaussian shocks are reused for each policy at a fixed seed. Adding a centre
shift preserves covariance, meaning the dependence of the simulated residuals, but
changes relative centres and can therefore change the joint score. Widths and tail
shape are not tuned in this comparison.

## Formal forecast opt-in

```bash
forecast --panels /input/panels --text /input/text --asof 2023-07-12 --card /input/card.toml --out /output/forecast.parquet --text-policy structured --monthly-trend damped
```

Both forecast entry points support the option. An explicit trend requires monthly
level targets and a `value_unit` beginning with `cpi_index_`. Daily targets, unresolved
monthly mappings and other macro targets refuse the opt-in. Structured text remains
daily-only. Monthly CPI trends use the statistical model without an LLM adjustment
under either text policy.
Thirteen consecutive observations are required to fit twelve monthly changes.
Missing months are not filled. The sidecar records the policy and horizon-specific shifts.

## Real-time validation

`scripts/validate_monthly_cpi.py` reads an externally supplied official ALFRED export
in the format “Observations by Real-Time Period”. Each observation has validity dates
for its historical revisions. A training value must be valid at the simulated origin.
The benchmark's 45-day panel lag is applied as an additional observation-date cutoff.
Later revisions cannot enter training. [ALFRED documentation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
describes the real-time dates.

The two research grids copy the published CPI card asset and horizon keys. One uses
explicit eight- and nine-month periods with a fixed later evaluation snapshot. The
other uses an explicit two-month period and the earliest archived release value.
These conventions stay separate. A missing initial release is not replaced with a
later revision. Research periods are generated separately; no authored card is edited
and no organizer target or answer is accessed.

Planning freezes the source hashes, prior inspection ledger, parameters and globally
disjoint CPI forecast intervals. Chronological month-end origins alternate the two
grids. Development targets must resolve before the fixed 2020 boundary. Holdout origins
must be on or after it. Intervals crossing the boundary are purged. Training history
can overlap, and economic regimes remain dependent.

Development compares all four fixed policies. Its equal-card mean loss locks one
candidate before holdout is scored. Holdout compares only that candidate and `none`.
Acceptance requires at least eight development intervals, six holdout intervals across
three origin years, and improved mean loss at all three draw seeds and both independent
normalization simulations. Component regressions are reported, rather than automatically
vetoing an improvement in the composite.

The existing scorer computes every component and its weights, including the single-cell
rule. Input-only normalization follows the published M0 model with explicit monthly
steps. Monte Carlo normalization means averaging scores against simulated baseline
outcomes. Its finite forecast draws and simulated outcomes yield a research proxy,
not the organizer's exact scale or a rankable score.

Example workflow, with all data and artifacts outside this public repository:

```bash
python scripts/validate_monthly_cpi.py --plan --source-zip /tmp/cpi.zip --source-request /tmp/source.json --prior-summary /tmp/prior/summary.json --out /tmp/cpi-check
python scripts/validate_monthly_cpi.py --development --out /tmp/cpi-check
python scripts/validate_monthly_cpi.py --holdout --out /tmp/cpi-check
```

The research scripts make no model request, change no default policy, rebuild no image,
or uploads a submission. A research acceptance result does not guarantee a leaderboard gain.
