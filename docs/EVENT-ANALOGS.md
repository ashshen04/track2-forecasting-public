## Executive summary (read this first)

Test historical event averages as small adjustments to a statistical forecast centre.
For example, compare future yield changes after an announced rate hike with changes
after earlier hikes. The experiment preserves Gaussian baseline shocks.
It does not change the formal forecast or submission package.
Historical responses and scoring reports must remain outside the public repository.

## Run the experiment

```bash
.venv/bin/python scripts/validate_event_analog.py \
  --unit t2-F3-sep-dots-curve-2023 \
  --evaluation-start 2015-01-01 \
  --out /private/tmp/event-analog-run
```

The authored assets and horizon keys are preserved. A new output directory is required.
The experiment compares the statistical baseline, raw event mean, shrunk event mean,
and half the ordinary historical trend. The latter is a control for general trend,
rather than evidence specifically associated with events.
All formulas for competition scoring come from the shared toolkit.

## Event identification and timing

An analog is a past event used to inform a current forecast.
The initial event class is an announced FOMC increase in the federal funds target.
FOMC means Federal Open Market Committee, the US monetary policy committee.
Classification requires a dated statement that explicitly says the committee decided
to raise its target. Statements anticipating a future hike do not qualify.
Corpus dates and file hashes provide provenance. Duplicate announcement dates count once.
The text corpus is an incomplete retrospective sample, not a complete meeting calendar.
No LLM or external model request is used for this initial event classification.

Forecasts start at the next business day's close. This avoids claiming a prediction
of the already observed announcement-day reaction. Responses are changes from that
origin to the exact business-day endpoint. Missing endpoints are never filled.
Monday-Friday business days are used; an exchange-holiday calendar is not reconstructed.
Panel publication vintages and intraday timing are also not reconstructed.

## Estimate and constrain the centre adjustment

For each horizon, use only analogs whose response has ended by the forecast origin.
Future panel rows are physically removed before analog fitting and baseline fitting.
Select the most recent non-overlapping analog response intervals first.
Use at least three completed analogs. Fewer analogs leave that horizon unchanged.
The raw variant adds the mean historical level change to the baseline centre.
The shrunk variant multiplies that mean by n/(n+10), where n is the peer count.
It then caps the shift at half the baseline horizon standard deviation.
These constants are fixed before this initial evaluation.
Shrinkage means reducing an uncertain estimate towards zero.
The centre adjustment preserves the baseline innovations and the horizon ordering.
It supplies an associative signal, rather than a causal estimate of the event's effect.

## Evaluation limits and follow-up

Evaluation events are selected chronologically with non-overlapping forecast intervals,
without examining their scores. Seeds 1, 7 and 19 check sensitivity to random sampling.
Repeated seeds reuse the same events and do not increase the independent sample size.
These periods may already have been inspected in earlier numerical experiments.
This is development evidence, rather than untouched accuracy validation.

Matching only hike direction mixes action sizes, policy cycles and market expectations.
Expected and unexpected actions can have different implications.
Historical price responses must never be used to label an event as surprising.
A stronger matcher needs expectations and regime features observable before the forecast.
Any matcher designed using this experiment needs additional evaluation periods.
Do not promote event means to the formal forecast without that evidence.
