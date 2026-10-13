## Executive summary (read this first)

The structured forecast validates event dates before changing its centre.
For example, an action effective January 1 cannot be treated as a new action at
a January 2 origin. A separately cited action scheduled for January 6 can qualify.
The code checks cited dates and rejects missing timing. It does not prove that a
policy will cause the model's claimed market response. No scoring formula changes.

## One dated example

Suppose a synthetic document released on January 1, 2020 says:
"Additional purchases will occur on January 6, 2020."
The forecast origin is January 2. Its authored daily horizon keys are `[1, 2, 20]`.
The corresponding end dates are January 3, January 6 and January 30.
January 6 is inside the second and third horizons. The first horizon keeps its
original forecast samples. Horizon keys in the output remain `[1, 2, 20]`.

This revision uses the same Monday-to-Friday business-day calendar as the daily
historical evaluator. It does not add a market-holiday calendar. Monthly horizon
keys cannot use this conversion. Structured monthly mode keeps the statistical
forecast without a model request.

## Required driver fields

An event driver is a supplied future action linked to one requested asset.
The model still reports its effect, mechanism and cited source spans. A source
span is an identified excerpt from a dated document sent to the model.
It must additionally report these fields:

- `announcement_date`: the ISO date of the announcing source's release. It must
  equal the timestamp of at least one source cited for the driver and precede or
  equal the forecast origin. A later report does not establish an earlier original
  announcement date. Missing announcement evidence causes abstention.
- `effective_date`: the action's start date. That complete date must appear in
  the driver's cited text. A document timestamp alone cannot supply it.
- `remaining_future_change`: a short verbatim description of an unfulfilled future
  action, its ISO `date`, and its own source-span `evidence`. The description and
  complete date must appear in the cited spans. The date must follow the origin,
  be at or after the effective date, and fall within a requested horizon.

An ISO date uses `YYYY-MM-DD`. Date presence also recognizes full-year English
dates such as January 6, 2020 and 6 January 2020, including common three-letter
month abbreviations. It does not infer missing years, relative dates or vague
timing such as "soon". Unknown fields use JSON `null` and the driver is skipped.

Already effective policy is eligible only when the document separately dates an
unfulfilled future increment. Maintaining the existing stance or saying its
effects persist does not establish a new dated action. Current stance, past
changes and unsupported conditional scenarios retain their existing exclusions.

## Application and audit

The prompt version is `structured-drivers-v3-event-time`. Old replies without the
required fields cannot trigger adjustments. The new version invalidates reply
cache identities. Old predictions and submission images retain their old code;
changing this source does not update an already built or uploaded image.

The resolver records accepted event dates, eligible horizon keys and the reasons
for discarded drivers in `analysis_audit`. The directional overlay records actual
per-horizon shifts. It changes only eligible horizons and preserves the numerical
forecast's joint shocks. Existing shift bounds and conflict abstention remain.
These shifts are still an approximation; this revision does not learn event
persistence or an effect-size model.

The historical evaluator uses the same mask for samples and direction diagnostics.
Synthetic tests cover completed events mislabeled as scheduled, missing and invalid
dates, uncited dates, future increments after old events, horizon boundaries and
exact parity with the formal forecast entrypoint. They establish mechanics only.

Date presence and a verbatim excerpt cannot prove that the cited date describes
the claimed action, that the action is unfulfilled, or that the causal direction
is correct. General evidence entailment is a later improvement. New live replies
and uninspected held-out histories are required to assess predictive benefit.
