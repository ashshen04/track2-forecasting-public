## Executive summary (read this first)

Validate return-centre policies on a fresh external factor-data snapshot.
For example, download official daily French CSV ZIPs outside this repository,
freeze an unscored recent cohort, and compare reference, half and zero drift.
All observations, historical outcomes and scoring reports remain outside this repo.
This research runner does not change the formal forecast or submission package.

## Inputs and commands

Use the daily three-factor and daily momentum CSV ZIPs linked from the
[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
Record the downloaded file hashes. The runner reads ZIP members without extracting them.

```bash
.venv/bin/python scripts/validate_recent_drift.py --plan \
  --three-zip /private/tmp/three-daily.zip \
  --momentum-zip /private/tmp/momentum-daily.zip \
  --prior-summary /private/tmp/prior-run/summary.json \
  --start 2025-01-02 --end 2026-10-11 \
  --out /private/tmp/recent-drift-run
.venv/bin/python scripts/validate_recent_drift.py --run \
  --out /private/tmp/recent-drift-run
```

The ZIP and prior-summary paths must exist. Supply all known prior experiment summaries.
The end date is capped by both data sources' last observation. Defaults specify the
initial study period; explicitly supply dates when conducting a different study.
MKT maps to Mkt-RF, the market excess return. It is not total market return.
SMB, HML and MOM map to the provider's factor returns. Percent values are divided by
100. Missing-value sentinels remain missing, and duplicate dates are refused.

## Freeze and score

The runner preserves the authored asset grids and horizon keys from three source cards:
a multi-factor 21/63-day grid, a multi-factor 7-day grid, and a market-only 21-day grid.
These are research grids applied to external data. They are not evaluations of those
cards' original forecast dates or named economic events.
The longer grid is planned first. Feasible windows are chosen latest first.
All prior inspected and newly reserved intervals are excluded for a shared asset.
Disjoint asset sets may occupy the same period and share economic conditions.

The planner writes cohort.json and the normalized input panel before any scoring.
The scorer checks hashes for the frozen panel and authored grid cards.
Every model input stops at its simulated forecast origin. Only the scorer sees targets.
Policies use 1,000 draws and seeds 1, 7 and 19. Gaussian innovations, uncertainty,
correlations and sampling seeds match. Only historical mean-return drift changes.
All scoring formulas come through the existing shared-toolkit scorer.

## Acceptance and limits

Half drift needs at least 20 intervals across two origin years. At every seed,
median paired composite error must decline, wins must exceed losses, and median
joint and tail errors must not increase, allowing 1e-10 percentage points of
numerical tolerance. This acceptance rule is recorded before scoring.
Passing this rule would still require review of coverage and source limitations.
The runner does not promote a candidate automatically.

Daily horizons use the existing Monday-Friday convention. Missing market-holiday
returns are never filled. Long cumulative-return horizons can consequently have no
eligible windows. Report this explicitly rather than extrapolating short-horizon gains.
The current provider snapshot includes historical revisions and its 2025 underlying
data-format change. It does not reconstruct the data available at each historical origin.
Repeated seeds assess sampling sensitivity and do not create new independent evidence.
Component percentages with zero reference scores are omitted; denominator counts are reported.
