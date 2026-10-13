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

For an expanded paired comparison including experimental mean reversion, see
[STATISTICAL-MODELS.md](STATISTICAL-MODELS.md).

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

## Compare equal-weight and EWMA volatility

```bash
python scripts/backtest.py --models reference reference-ewma \
  --min-history 121 --folds 3 --n-draws 1000 \
  --include t2-F1-aud-on-hold-2016 t2-F1-measured-pace-2004 \
  t2-F3-brexit-joint-2016 t2-F3-covid-curve-2020 \
  t2-F4-covid-mkt-2020 t2-F4-factor-stress-2008 \
  --out /private/tmp/t2-ewma-01
```

For example, a large recent change receives more weight than an equally large old
change. EWMA means exponentially weighted moving average. This candidate uses
exponential weights with a fixed half-life of 20 aligned observations: a change
20 rows older receives half the weight. Pandas computes the bias-corrected weighted
variance, and its square root supplies the standard deviation.

Both variants use at most the latest 120 aligned changes. Only the standard
deviations change. Drift, correlation estimates, seeds, draw counts, and origins
remain identical. Correlations retain equal weights to isolate one modelling choice.
Monthly targets are outside this experiment. The default `forecast` command still
uses equal weights; this candidate is available through the numerical sampler and
backtest runner only.

The report shows each unit's paired scores, within-unit percentage change and fold
win count. No raw score average across units is meaningful. These six units are the
existing development subset, not independent validation. Keep experimental results
outside this public repository. A favourable result needs further chronological
validation; a mixed result does not justify replacing the default.

## Calibrate the forecast centre with chronological validation

```bash
python scripts/backtest.py \
  --models reference reference-drift-zero reference-drift-half reference-drift-full \
  --folds 6 --selection-folds 3 --min-history 121 --n-draws 1000 \
  --include t2-F1-aud-on-hold-2016 t2-F1-measured-pace-2004 \
  t2-F3-brexit-joint-2016 t2-F3-covid-curve-2020 \
  t2-F4-covid-mkt-2020 t2-F4-factor-stress-2008 \
  --out /private/tmp/t2-drift-validation-01
```

For example, a mean daily level change of 0.02 gives a horizon-10 shift of
0, 0.1 or 0.2 under zero, half or full trend. Trend here means the historical
mean change projected over the forecast horizon. Shrinkage means multiplying
that mean by a number between zero and one to reduce its influence.

These three candidates use the same latest 120 aligned changes as `reference`.
Level targets use first differences. Cumulative log-return targets use the shared
toolkit's return conversion. The centre is the anchor plus the trend times the
horizon; return targets have a zero anchor. Standard deviations, correlations,
draw counts, seeds and cumulative innovations match across candidates.
The numerical sampler accepts `drift_scale=0`, `0.5` or `1`; omitting it keeps
the deployed policy (zero level trend and full historical return trend).
The submission CLI and its House-model text path keep that deployed policy.
Monthly trend calibration is unsupported and explicitly refused by the sampler.

Each unit selects the lowest mean raw composite using only its three oldest
holdouts. Ties favour the existing reference. That choice is locked before
reporting performance on the three newer holdouts. No forecast interval overlaps
another within the unit. A run with fewer than six valid holdouts is marked
`insufficient_folds` and does not establish validation. Set `--selection-folds`
to another positive count smaller than `--folds` to change this partition.
The option requires `reference` and all three trend candidates.

The JSON field `drift_validation` records the chosen model, selection and validation
origins, selection means, and newer component scores for every candidate. The
readable report shows the locked choice's percentage changes and fold wins.
Negative changes favour the choice; a zero reference component has no meaningful
percentage and is reported as `n/a`. Results remain outside this public repository.

This is a per-unit diagnostic. It does not learn a policy for unseen cards or
establish independent evidence from shared histories. Historical release vintages
are not reconstructed. The six development units were already inspected in earlier
experiments. Their newer intervals are held out from this run's parameter selection,
but are not an untouched final evaluation set. Text is never used; a text comparison
requires corpora available at each historical origin and must be a separate experiment.
Retain the existing default unless a predeclared policy also improves on additional,
previously unused periods.

## Calibrate interval width on previously unscored intervals

```bash
python scripts/backtest.py \
  --models reference reference-width-08 reference-width-12 \
  --folds 6 --selection-folds 3 --min-history 121 --n-draws 1000 \
  --exclude-intervals-from /private/tmp/earlier-run/summary.json \
  --out /private/tmp/t2-width-validation-01
```

For example, a standard deviation of 0.1 becomes 0.08, 0.1 or 0.12 under the
three candidates. Standard deviation measures the spread around a distribution's
centre. The candidates multiply the estimated standard deviations by 0.8, 1 or
1.2. This changes the scale of the same correlated innovations. It preserves the
theoretical forecast centre, drift, asset correlations and cumulative paths.
It does not change the volatility estimation window or introduce time weighting.
`reference` is the 1.0 control. The sampler's `width_scale` defaults to 1.0;
nonpositive or nonfinite values are refused. Nondefault monthly widths are unsupported.

Supply every earlier numerical backtest summary after `--exclude-intervals-from`.
The runner merges their historical forecast intervals by asset. A new interval is
rejected if it overlaps a previously scored interval for any of its assets, even
if that interval was recorded under another card name. Forecast intervals are
open at the origin and closed at the final target date. Their boundary dates can
touch without sharing a target. Training history may still overlap. Invalid or
empty exclusion summaries fail rather than silently disabling the exclusion.
The output records each supplied summary's hash and every excluded-origin count.

As in the drift experiment, each unit chooses a width on the oldest three valid
holdouts and locks it for the newer three. Ties keep `reference`. Incomplete
six-fold sets do not establish validation. Choose one complete drift or width
candidate set when using `--selection-folds`; combining them is ambiguous and refused.
The JSON field `width_validation` includes all candidate component scores on the
newer folds, the locked choice, percentage changes and wins. Only within-unit
score changes are comparable. No overall raw score is reported.

The excluded periods are protected from direct target reuse, but cards can still
share the new holdouts. This is not independent evidence across units. When recent
periods have already been inspected, the remaining holdouts can be much older;
chronology is enforced within this new experiment, not relative to all earlier runs.
This per-unit diagnostic does not define a deployment policy for unseen cards.
Compare marginal, joint and tail changes before adopting a width. Keep the current
default if composite gains are inconsistent or other components materially worsen.

## Validate the language-model text overlay

For example, this audits text availability without making a model call:

```bash
python scripts/validate_text.py \
  --backtest /private/tmp/t2-drift-validation-all-20261009-v260/summary.json \
  --out /private/tmp/t2-text-audit-01
```

The runner selects one latest eligible historical origin per unit. It requires at
least one indexed document dated within 90 days before that origin. It ranks units
by text freshness within each declared family, then rotates across the four families
for a default budget of 12 calls. Scores never influence case selection. Other older
documents remain in the prompt, as in the current reasoning agent. Change the freshness
threshold with `--max-document-age-days` and the call budget with `--max-calls`.
Use `--include UNIT_ID` to diagnose one eligible unit without changing its origin.
Rejected replies are saved in the external result's `model_reply` for inspection.
For a paired excerpt experiment, run the same cases again with `--excerpt-policy
relevant`. This keeps each document's first 1,500 characters and up to four disjoint
1,000-character windows ranked by fixed policy-keyword counts. Markers identify the
windows' positions in the source. Short documents remain unchanged; documents without
matching later keywords retain their first 6,000 characters. Total length remains at
most 6,000 characters per document. Dates, document IDs, case selection, statistical
samples and adjustment limits match the default `head` policy. The submission's
default remains `head`; this option is an experiment, not a validated improvement.

With configured House access, add `--run`. Missing configuration, failed calls and
invalid replies are explicit failures, never zero-difference accuracy observations.
Partial live runs return nonzero and retain each result. Successful zero adjustments
are recorded separately from forecasts whose samples changed.

For a local NVIDIA API experiment, generate a key through the
[public Nemotron 3 Super endpoint](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b)
and run this in your terminal:

```bash
.venv/bin/python scripts/validate_text.py \
  --backtest /private/tmp/t2-drift-validation-all-20261009-v260/summary.json \
  --out /private/tmp/t2-text-nvidia-01 \
  --nvidia --prompt-api-key --run --max-calls 12
```

Alternatively, put `MODEL_API_KEY=your-key` in the repository's `.env.local` and
omit `--prompt-api-key`. Local validation loads that file automatically; an existing
environment value takes precedence. The file is excluded from Git and Docker.
The key is not saved in the results. `--nvidia`
selects `https://integrate.api.nvidia.com/v1` and
`nvidia/nemotron-3-super-120b-a12b`. To use another experimental interface, configure
`MODEL_ENDPOINT`, `MODEL_NAME`, and optionally `MODEL_API_KEY`, then use `--local-api
--run`. A local experiment refuses any inherited `MODEL_TOKEN`; House credentials
must use the audited House route. Vendor experiments do not change the submission CLI.

Both variants reuse the same statistical samples and conservative text policy. Panels
are checked against the source backtest's hashes. Historical outcomes and baseline
scores are recomputed; no old score is reused. Only rows at or before the simulated
origin reach the sampler, and only indexed text at or before that date reaches the
model. The report includes component changes and the accepted adjustment ledger.

This initial budget is a diagnostic, not sufficient evidence for deployment. The public
vendor's snapshot and serving stack have not been verified against the organizer's
[House snapshot](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md).
The model may remember later events even when a prompt forbids using them. A favourable
historical result therefore needs additional previously unused periods and a House
evaluation before claiming official performance. Keep all experiment output outside
this repository. Temporary directories can be removed by the operating system.

## Empirical daily innovations

Example: add `--models reference reference-bootstrap --folds 6` to a numerical
backtest command. Include every previously inspected summary with
`--exclude-intervals-from` when evaluating new periods.

Bootstrap means drawing historical daily change vectors with replacement.
The experimental sampler centres the last 120 aligned vectors and corrects their
population covariance to match the sample covariance of the Gaussian reference.
It resamples all assets together. Daily shocks accumulate across authored horizons.
For return targets it uses log returns and preserves the reference drift.
The default sampler stays Gaussian. Monthly sampling and EWMA combinations are refused.

This comparison isolates innovation shape, including historical skew and joint tails.
It cannot create extreme events absent from a short history. Repeated daily sampling
also assumes no serial dependence. Evaluate a choice made on older folds against
newer folds; incomplete sets do not establish validation. Cards sharing assets are
not independent evidence. Keep scores and historical outcomes outside this repository.

## Diagnose errors before changing another model

Example:

```bash
.venv/bin/python scripts/diagnose_errors.py \
  --summaries /private/tmp/t2-backtest/summary.json \
  --out /private/tmp/t2-error-diagnosis
```

Pass several already inspected summaries to `--summaries`. The script checks panel
hashes and card grids. It physically truncates training rows, reconstructs holdouts,
and requires the shared-toolkit reference scores to match the recorded scores.
It removes duplicate unit origins across summaries. It never selects a new model.

The report separates families, panels, horizons and unit-level provenance warnings.
A standardized residual is the observed error divided by the forecast standard
deviation. Coverage is the fraction of observed targets inside a forecast interval.
PIT, the probability integral transform, measures the forecast probability below
an observed target. PIT values should be uniform for calibrated continuous forecasts.
These are diagnostics, not replacements for toolkit scores. Neutral forecasts
abstain from predicting direction. A historical-mean direction probe is labelled
separately and is not an adopted forecasting policy.

The report also counts indexed, existing text documents available at each simulated
origin. An origin with no eligible text cannot test reasoning augmentation.
No text contents or model API are used by this audit.

Card-weighted results count repeated asset events. Event-weighted results keep the
first forecast for each asset, origin, horizon and target type in input-summary
order. Forecasts may differ with their asset grid. Neither weighting makes overlapping
histories independent. Missing provenance warnings do not prove data quality.
Monthly tasks and historical release vintages remain outside this numerical audit.
Raw component weights cannot reveal loss attribution under official normalization.
Any diagnosis-driven change needs untouched or prospective validation. All output,
including reconstructed target values in `cells.csv`, must remain outside this repo.

## Text-aligned cohorts and directional evidence

Example: build eligible windows, then audit their development split:

```bash
.venv/bin/python scripts/build_text_cohort.py --out /private/tmp/t2-text-cohort
.venv/bin/python scripts/validate_text.py \
  --cohort /private/tmp/t2-text-cohort/cohort.json \
  --split development --policy both --excerpt-policy relevant \
  --out /private/tmp/t2-text-development --max-calls 12
```

The cohort builder finds up to three non-overlapping windows per daily card.
Every window has an existing document dated at or before its origin and within
90 days. The authored horizons are preserved. Missing exact targets, short history
and unsupported monthly release vintages are reported. It does not rank dates
by scores. It hashes panels and indexed corpus files.

The global chronological split uses 70% of unique origin dates for its boundary.
Development target intervals must end before that boundary. Holdout origins must
start on or after it. Intervals crossing the boundary are purged. The frozen manifest
records all selected windows. Document availability is not proof of relevance.
These are existing public historical periods; earlier experiments may already have
inspected their outcomes. The split protects this experiment's development process,
not an assertion of independent historical evidence or freedom from LLM memory.

`--policy directional` uses a separate experimental prompt. It asks for an
asset-specific causal mechanism, opposing consideration, pricing assessment and
an exact quotation from supplied evidence. Named documents and quoted text are
checked. This checks attribution, not the truth of the causal inference.
A neutral reply is a valid abstention without a fabricated citation.

Weak evidence shifts the longest-horizon centre by 0.10 standard deviations.
Moderate evidence judged unlikely to be priced shifts it by 0.25. Information
judged likely priced produces no shift. These fixed bounds are development choices,
not calibrated confidence probabilities. Shorter-horizon shifts scale linearly
with the authored horizon. Innovations, widths and cross-asset pairing are unchanged.
The submission CLI retains its existing text policy.

`--policy both` evaluates the previous conservative overlay and the directional
version on identical draws. `--max-calls` caps actual API calls, so 12 calls cover
at most six paired windows. `--thinking --model-max-tokens 6000` enables reasoning
with a larger completion budget; the House output cap still applies. Local NVIDIA
calls require the existing `--nvidia --run` flags. Without `--run`, no calls occur.
Use the same frozen settings with `--split holdout` after development. Do not tune
on that split and continue calling it a holdout. Failures and invalid adjustments
remain separate from accuracy comparisons. Directional diagnostics report attempts,
correct calls and abstentions; abstentions are never counted as correct directions.

For a narrower policy cohort, add `--require-policy-text` to the builder. This
requires a recent official policy or macro document, or a speech containing at
least two fixed policy phrases. It is a lexical filter, not proof that the document
has information about the target asset. The manifest records policy-document counts.
Cohorts created with different filters have different temporal boundaries; do not
compare their scores as if they shared identical validation windows.

The directional validator checks each complete quoted excerpt against the supplied
text, allowing whitespace differences. Quotes longer than the requested display
budget are verified in full before the audit ledger truncates them to 500 characters.
A fabricated suffix is rejected even if the first 500 characters are authentic.
Reprocessing a saved reply after an attribution-display fix requires no new API
call. Such replays must retain original statuses and apply the same check to every
reply, without selecting based on scores. They are development evidence.

## Frozen evidence and stability experiments

Example: freeze an experiment before making model calls:

```bash
.venv/bin/python scripts/validate_stability.py \
  --cohort /private/tmp/t2-policy-cohort/cohort.json \
  --prior-assessment /private/tmp/t2-prior-assessment/assessment.json \
  --out /private/tmp/t2-stability --max-calls 100
# Run the same frozen experiment. Repeat this command to resume.
.venv/bin/python scripts/validate_stability.py \
  --cohort /private/tmp/t2-policy-cohort/cohort.json \
  --prior-assessment /private/tmp/t2-prior-assessment/assessment.json \
  --out /private/tmp/t2-stability --max-calls 100 --resume --nvidia --run
```

The runner requires the existing 50-window policy cohort. It plans 98 requests:
50 initial judgments, 24 repeated judgments, 12 quotation comparisons and up to
12 administrative-text controls. The diagnostic cases come from three earlier
improvements, one adverse result, four rejected replies and four abstentions.
Their selection uses earlier results. They test reliability, not independent
forecasting accuracy. All historical windows are labelled previously inspected.

An evidence span is a fixed piece of supplied text. The span variant assigns
stable identifiers to 400-character pieces. The model returns identifiers instead
of copying quotations. The validator resolves each identifier to its source text,
document timestamp and excerpt offsets. Unknown identifiers and conflicting
document identifiers are rejected. Offsets refer to the supplied excerpt.
`validate_text.py --evidence-format span` also exposes this experimental format.
The quotation format and the submission default remain available unchanged.

The manifest freezes inputs, implementation hashes, toolkit version and settings.
Changed files prevent resumption. Each request consumes a persistent budget claim
before network access. Parsed replies are saved before validation. Resumption
reuses saved replies and does not retry a claimed request lacking a saved reply.
A process lock prevents concurrent runs in the same output directory. There are
two workers. The fixed model enables reasoning with a 6000-token completion budget.

The report checks three operational gates. At least 90% of the initial 50 replies
must validate for every requested asset. Repeated decisions must agree on direction
in at least 80% of pairs; failed or missing decisions count as disagreements.
At least eight controls must validate, and at most 10% may shift the forecast.
Controls use existing text dated on or before the origin. A lexical non-policy
filter approximates irrelevant administrative text; it cannot certify irrelevance.
Unavailable controls are reported explicitly.

Each valid saved reply is scored with the shared toolkit at draw seeds 1, 7 and 19.
This checks simulation sensitivity without additional model calls. Reports group
relative changes by unit and never average raw scores across unlike assets.
Passing operational gates does not establish independent forecasting benefit.
Experiment artifacts and observed targets must stay outside this public repository.

The experimental prompt revision `directional-v2-decision-rules` separates asset
relevance, future change, pricing evidence, opposing evidence and the final decision.
For example, a policy rate staying high does not alone establish that a yield will
rise from its current level. Public or routine guidance does not alone establish
that information is already priced. Unsupported pricing comparisons use `unclear`.
Unclear pricing limits strength to `weak`; it does not alone require abstention.
Neutral means no justified correction, not a prediction of a flat market.

Replies also request `pricing_assessment` and `decision_reason`. These diagnostic
fields remain in saved model replies; they are not additional admissibility gates.
The evidence validator and shift bounds are unchanged. Prompt changes invalidate
an earlier experiment's implementation hash. Create a new frozen experiment to
evaluate this revision. Earlier model replies cannot validate a new prompt.

Revision `directional-v3-asset-and-change` names the exact requested asset IDs.
Numeric direction contracts translate authored units into the meaning of up and
down. For example, higher CHF-per-USD means USD strengthens against CHF. Explicit
mixed currency groups are supported. Missing or conflicting FX conventions remain
unknown; observed levels and ticker names are not used to guess them.

The prompt requests `residual_change` and `change_basis` in saved replies. These
describe the future increment relative to the statistical forecast. Maintained
policy alone and unsupported conditional scenarios call for abstention. A final
self-check reconciles asset keys, numeric signs, pricing evidence and opposing
channels. These are prompt instructions, not proof that the reasoning is true.
Attribution validation, shift bounds and operational thresholds remain unchanged.
Existing lexical text controls retain their frozen contents for comparability.
Some may contain substantive policy passages; their trigger rate must not be
interpreted as a certified false-positive rate without semantic review.

Three consecutive connection errors now stop new requests. The stop is persisted
in `network_circuit.json` and remains latched across resumption. Already in-flight
requests may finish. Jobs prevented from calling the model report
`not_attempted_network`; their budget is not consumed. A slow model response alone
does not trip this connection safeguard. Saved replies still replay offline.
Do not silently reset a stopped batch or erase its failures. After connectivity
recovers, freeze a separately budgeted experiment and retain the failed batch.

## Structured evidence analysis

Example: freeze a small diagnostic experiment using an existing stability manifest:

```bash
.venv/bin/python scripts/validate_analysis.py \
  --source-manifest /private/tmp/t2-stability/manifest.json \
  --out /private/tmp/t2-analysis --max-calls 12
.venv/bin/python scripts/validate_analysis.py \
  --source-manifest /private/tmp/t2-stability/manifest.json \
  --out /private/tmp/t2-analysis --max-calls 12 --resume --nvidia --run
```

`baselines/evidence_analysis.py` asks the model for cited causal drivers rather than
a final adjustment. A driver describes a future increment, declared timing, an
economic effect and its source spans. Supporting and opposing drivers are retained.
For example, an effect that strengthens JPY converts to a lower JPY-per-USD quote.
Code performs that conversion using authored units. Missing conventions are not
guessed. Unknown sources and mismatched asset keys are refused.

Only scheduled actions or explicit future paths declared within the horizon can
trigger a correction. Current stance and unsupported conditional scenarios are
recorded but do not trigger one. Conflicting actionable signs produce abstention;
there is no learned driver-weighting formula. Unsupported pricing claims are
downgraded to unclear. The existing bounded directional overlay and shared scoring
toolkit are reused. Citation checks and typed effects cannot prove the underlying
economic inference or the model's interpretation of timing.

The runner selects the same 12 earlier diagnostic cases without using new scores.
It saves raw analysis before validation and replays it at three draw seeds without
new calls. Input and implementation hashes, persistent call budgets, process locks
and connection-error stopping are preserved. This is a functional smoke test on
selected inspected history, not independent accuracy or repeatability validation.

An optional `--cache-dir /private/tmp/t2-analysis-cache` reuses replies only when
the full prompt, model settings and endpoint identity match. Model keys are not
stored. Cached replies still require validation. Failed model calls are not cached.
Without that flag caching is disabled. Cache hits consume no new model calls and
must never be counted as independent repeated decisions. Source documents are
parsed once for all requested assets per case. Cache efficiency does not establish
predictive accuracy. The submission default remains unchanged.

The return-target revisions `directional-v4-return-target` and
`structured-drivers-v2-return-target` recognize the card's `target_type = "log_return"`.
For example, a supported upward MOM return driver now describes a more positive
cumulative log return. Previously, these experiments recognized only
`cumulative_log_return`, which is a value-unit description rather than the card's
target-type spelling. This could reject a correctly typed return effect or accept
a generic numeric effect instead. The earlier experimental spelling remains
supported for offline callers. Return targets require an asset-specific `return`
effect; Treasury yield and generic numeric effects are refused.

Create a new frozen experiment for these revisions. Existing reply caches and
earlier results do not validate the revised prompt or resolver. The regression
checks establish target interpretation and bounded shifts, not accuracy uplift.

## Formal structured-policy comparison

For example, audit a frozen cohort without making model requests:

```bash
.venv/bin/python scripts/validate_text.py \
  --cohort /private/tmp/t2-text-cohort-policy-20261009-01/cohort.json \
  --split holdout --policy structured --excerpt-policy relevant \
  --out /private/tmp/t2-formal-structured-audit \
  --max-calls 12 --model-max-tokens 4000
```

The corresponding submission command is `forecast ... --text-policy structured`.
The historical evaluator and submission use the same unit contracts, prompt,
resolver and horizon-scaled shift. A synthetic parity test checks sample equality
with the formal entrypoint. Keep `--excerpt-policy relevant` for parity. Structured
evidence always uses source spans, regardless of the quote-format CLI default.
The conservative submission default is unchanged. Structured monthly forecasts
retain the statistical distribution without a model call.

For live local experiments, add `--nvidia --run` within the chosen call budget.
Use `--cache-dir /private/tmp/t2-formal-analysis-cache` for exact-input reuse.
The report records actual model calls and cache hits separately. The corresponding
submission flag is `--text-cache`. Reusing a reply at three draw seeds tests
simulation sensitivity; it does not measure independent model repeatability.

Inspect prior experiment intervals before treating any cohort as unseen. Exclude
overlapping origin-to-target intervals sharing assets, not only identical unit
names. A previously inspected holdout is now development evidence. If exclusions
leave no windows, obtain new held-out histories or retain the current default.
Fresh model calls on old windows do not restore independence. Historical model
memory and revised source vintages also remain limitations. Do not promote this
policy based only on citation validity or simulated replies.

The structured event-time revision now validates cited calendar dates and applies
shifts only to horizons containing the remaining future increment. Direction
diagnostics use the same per-horizon shifts as the formal forecast samples.
Freeze a new experiment for `structured-drivers-v3-event-time`; older model replies
without the new fields are valid only for rejection diagnostics. See
[EVENT-TIMING.md](EVENT-TIMING.md). Historical evidence and scores stay outside this
public repository. Existing packaged images and submission ZIPs are not rebuilt
by source changes.
