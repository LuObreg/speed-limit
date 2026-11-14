# Requirements Specification

This specification makes the supplied case-study requirements measurable for
the course prototype, not a fully compliant or safety-approved implementation.
It records both the original intent and the project's chosen prototype policy.
The case study (Case_Study_Speed_Limit_V220.pdf) is the source for REQ1-REQ3
and SPEC1-SPEC4. See [project decisions](./project_decisions.md) for rationale
and implementation gaps, and [stakeholder notes](./stakeholder_meeting_notes.md)
for open questions.

## Functional requirements

**SPEC1 — Darkness.** A valid illuminance reading is classified as dark when it
is strictly below 500 millilux.

**SPEC2 — Black-ice danger.** Valid readings indicate black-ice danger when the
water level is strictly above 1000 µm and the temperature is strictly below
0°C. Both conditions must be true.

**SPEC3 — Learned relationship.** The neural network shall estimate the
relationship between weather/light conditions, the speed limit in force, and
the number of near-accidents. The speed limit in force is represented by
sensor ID `-1`. The model's prediction target is near-accidents per hour,
conditional on weather/light features and a candidate speed.

**SPEC4 — Near-accident definition.** A near-accident is one recorded event
where a car skids, comes too close to another car, or comes too close to the
highway guardrail.

**REQ1 — Weather/light speed decision.** When SPEC1 or SPEC2 is true, the
prototype estimates the mean near-accident count for each candidate speed.
Candidates are the speed limits observed in the Year N data (40, 50, 60, 70
and 80 km/h); only those below the configured normal speed are eligible, so
the result is always a reduction. Among eligible candidates, the highest one
estimated at or below one near-accident per hour is selected. If none reaches
the target, the eligible candidate with the lowest estimated rate is selected
(ties go to the lower speed) and the decision is flagged as "target not met".
If no eligible candidate exists, the operator decides.

The requirement's intended safety outcome is a realized average of at most
one near-accident per triggered hour over an agreed operating period. The
model's estimated rate is only a decision proxy. Because the historical data
is observational, retrospective evaluation cannot prove the counterfactual
outcome of a speed that was not applied. Stakeholders must define how that
safety outcome will be independently validated before any real-world claim.

**REQ2 — Air-quality decision.** When current air quality is classified as
high, the system shall propose a speed reduction. For this MVP, European AQI
of 60 or higher is the configurable trigger, and the LLM proposal is bounded
to a configurable reduction of 10–20 km/h. These values are project
assumptions, not legal or professor-prescribed thresholds, and require
stakeholder validation.

**Original/prototype gap:** the original course requirement asks the LLM to
decide whether and how much to reduce speed. The prototype instead decides
whether deterministically with the AQI gate; the LLM supplies only a bounded
amount. Preserve this policy pending stakeholder resolution, not as an
approved reinterpretation of the original requirement.

**REQ3 — Normal default.** When required inputs and services are valid and
neither REQ1 nor REQ2 requires a reduction, the speed limit shall be 80 km/h.
This default must not be used to disguise missing data or a failed decision
component.

## Operational assumptions selected by the project owner

- The formal trigger in this specification (darkness OR black-ice danger) is
  authoritative for the prototype, rather than the less precise phrase “bad
  weather” in the original project description. **Stakeholder confirmation
  remains open.**
- REQ1's primary evaluation population is the hours when SPEC1 or SPEC2
  triggers the neural-network path. Report the all-hours mean as a secondary
  measure.
- Evaluate chronologically with separate training, validation, and test
  periods; do not use test results for model selection.
- Evaluate the Year N model on Year N+1 before using Year N+1 in annual
  retraining. Join features and event labels separately for each period,
  then concatenate the resulting training sets; do not pool repeated calendar
  labels across periods.
- Interpolate only whole internal sensor gaps no longer than six hours.
  Longer gaps remain wholly unfilled; do not fill their first six hours.
- At runtime, when a required input or decision component fails, the system
  does not choose a speed on its own: it asks the operator to choose one,
  showing the current metrics or the last available ones. The operator's
  speed applies until the system can decide on its own again. **Implemented
  for the simulation; requires safety-owner approval.**
- Exclude sensor 19 from the temperature feature until its type is confirmed.
- Assume sensor 16 is traffic for this project; do not include it in the current
  weather/light feature set unless a separate model-input decision is made.
- Exclude licence plates from application processing, derived data, logs, and
  model inputs.
- Treat the application as a simulation/course prototype, not as certified
  highway infrastructure.

## Verification

The router test cases and the AQ/LLM test cases are documented in exercises
10 and 11. The code's [golden harness fixtures](../source/air_quality_llm.py)
are engineering example ranges, not expert-validated answers. Other
verification slices include physical input/output bounds and invalid
classes; chronological NN splits, baseline comparisons and candidate support;
headerless ingestion, unit normalization, privacy, period-local labels and
whole-gap interpolation; simulator lifecycle and consumer failures; and
configuration consistency. Commands and coverage policy live in
[README](./README.md#testing), while model evaluation lives in
[project decisions](./project_decisions.md#6-verification-evidence).
