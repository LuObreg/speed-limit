# Stakeholder Meeting Notes

## Meeting details

- **Purpose:** Clarify open requirements, data assumptions, and approvals for
  the SpeedLimit course prototype.

## 1. Safety goal and weather trigger

- Confirm whether the formal trigger—darkness OR black-ice danger—is intended
  to define the assignment's broader phrase “bad weather.” What other weather
  conditions, if any, should trigger a reduction, and how are they measured?
- The Year N+1 model MAE (0.874) is slightly worse than the mean-baseline MAE
  (0.873). Is this prediction quality sufficient for the prototype's intended
  use, and what measurable improvement or acceptance criterion is required?
- Do stakeholders want a separate analysis of the association between recorded
  speed and near-accident counts? Such an association would not establish that
  changing the speed limit causes a change in near-accidents.
- What should the system do if no historically supported candidate speed is
  estimated to meet the target?
- Which locations or jurisdictions are in scope? The system cannot be assumed
  compliant worldwide because traffic laws and permitted speed ranges differ.
- What certification or approval process is required before any operational
  use?

## 2. Speed policy and failure behavior

- What legal speed range and minimum speed apply in the selected jurisdiction?
  Current configured bounds are prototype values, not regulatory approval.
- When the system cannot decide, it asks the operator to choose the speed.
  What should the physical sign show while waiting, and how quickly must the
  operator respond?
- What alerts, escalation path, and physical sign behavior are required when
  sensor input, API, model, or another required component is unavailable?
- Who may override, pause, or resume the system? Should an override require
  authentication, a reason, an expiry time, and an audit record?

## 3. Data and model

- Confirm sensor meanings and units: the project assumes sensors 15 and 18 are
  temperature and sensor 16 is traffic; sensor 19 has readings but no sensor-map
  entry. Confirm whether sensor ID `-1` always records the active speed limit.
- Confirm authoritative, versioned schemas, units, timestamp semantics,
  provenance, missing-data and duplicate-record rules, interpolation policy,
  and whether the headerless Year N+1 file has the same column order as Year N.
- Confirm that licence plates can be excluded from application processing and
  all derived data, and whether there is any legitimate need to retain them.
- Are the January-only periods representative? Can additional seasons, roads,
  and a future holdout period be provided?
- Should crash damage/injuries be part of the model objective, or should the
  target remain the specified near-accident count?
- Besides the current annual schedule, what evidence or threshold should
  trigger retraining?
- Is the current model prediction quality acceptable for the prototype, given
  that its Year N+1 MAE is slightly worse than the mean baseline? What
  measurable acceptance criterion should apply?

## 4. Air quality and external services

- Which AQI scale and authoritative source define poor air quality for the
  highway? Are the current AQI threshold and 10–20 km/h LLM proposal acceptable?
- Should the pollution response be a fixed/deterministic rule, an authorized
  operator decision, or an LLM proposal? What rule maps pollution levels to a
  reduction, and who approves and validates it?
- Which location and timestamp should be used for historical replay?
- What data freshness, fallback, alert, and escalation rules apply when AQI is
  stale or unavailable?

## 5. Operations and quality requirements

- What measurable availability, update latency, recovery, reliability,
  cybersecurity, privacy, retention, access-control, and incident-reporting
  requirements apply?
- What does “safe,” “reliable,” and “secure” mean in measurable, testable
  acceptance terms?
- Who is accountable for monitoring, and what training and authority do
  operators need?
- For the electric-vehicle exemption, is there a trusted vehicle-specific input?
  How should its privacy and spoofing risks be handled? A global sign cannot
  selectively exempt an individual vehicle.

## 6. Evidence: counter-intuitive model behavior

Dashboard scenarios with the runtime (annual) model produce these estimated
near-accidents per hour per candidate speed (target: ≤ 1.0):

| Scenario | 40 | 50 | 60 | 70 | 80 | Published sign |
| --- | --- | --- | --- | --- | --- | --- |
| Black ice in daylight (-3 °C, 1500 µm, daylight) | 0.65 | 0.48 | 0.69 | 0.93 | 1.13 | 70 km/h (target met) |
| Black ice at night (-5 °C, 2000 µm, dark) | 1.79 | 1.92 | 2.22 | 2.49 | 2.58 | 40 km/h (lowest risk, target not met) |
| Night, dry road (10 °C, 0 µm, dark) | 2.71 | 2.54 | 2.39 | 2.31 | 2.29 | 70 km/h (lowest risk, target not met) |

Observations:

- At night, no candidate speed reaches the target, so the system selects the
  reduced speed with the lowest estimated risk and flags the unmet target.
- On a dry road at night, the estimated risk **decreases** as speed increases.
  This contradicts the domain expectation and reflects historical association
  (for example, lower speeds recorded during worse conditions), not the causal
  effect of choosing a speed. As a result, the system picks 70 km/h rather
  than a lower speed at night on a dry road.
- Estimated risk is not monotonic in speed even in daylight (50 km/h is
  estimated safer than 40 km/h).

This is evidence that the current training approach is not fit for choosing a
speed limit. Questions and proposals for stakeholders:

- Should the model be constrained so that estimated risk never decreases when
  speed increases under the same conditions (monotonic constraint), or should
  the speed effect come from an approved domain rule instead of being learned?
- Can data be provided where the speed limit varied independently of weather
  (for example, controlled trials or other road sections), so the speed effect
  can be estimated rather than confounded with conditions?
- Should training use only triggered hours (darkness or black ice), weight
  them more heavily, or model the interaction between darkness and black ice
  explicitly?
- Is the 1.0 near-accidents/hour target achievable in the worst conditions? Is
  selecting the lowest-risk reduced speed when the target is not met
  acceptable, or should the minimum configured speed be used instead?
- Which acceptance tests (for example, "black ice at night never yields a
  higher speed than black ice in daylight") must a retrained model pass before
  deployment?
