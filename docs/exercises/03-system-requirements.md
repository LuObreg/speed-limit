# Exercise 03: System Requirements

## Assignment

Complete the system requirements and requirements specification; record
assumptions and a glossary; review requirements for common natural-language
errors.

## Existing Evidence

### Functional requirements

- **Darkness:** a valid illuminance reading below 500 millilux is classified
  as dark.
- **Black-ice danger:** both water level above 1000 micrometers and temperature
  below 0 degrees Celsius indicate black-ice danger.
- **Weather/light decision:** darkness OR black-ice danger activates the
  neural-network path. The model estimates hourly near-accident counts for
  historically supported candidate speeds of 40, 50, 60, 70, and 80 km/h.
  The prototype aims to choose the highest candidate below the normal speed
  whose estimated mean is at most one near-accident per triggered hour. If no
  lower candidate meets that estimate, it flags the unmet target and selects
  the lower candidate with the lowest estimated risk.
- **Near-accident label:** one recorded skid, close approach to another car,
  or close approach to the guardrail counts as one near-accident.
- **Air quality:** the prototype uses European AQI 60 or higher as a
  configurable trigger and asks the LLM for a reduction from 10 to 20 km/h.
  These are project assumptions. The original course requirement gives the
  LLM both the decision whether to reduce and the amount; the prototype's
  deterministic gate is a known difference.
- **Normal operation:** when all required inputs and services are valid and no
  reduction applies, the speed is 80 km/h.
- **Failure:** a required input or component failure must not be reported as a
  valid no-reduction decision. The simulation flags the failure and asks the
  operator to choose the speed based on the current or last available metrics.

### Assumptions

The project assumes that the formal darkness/black-ice trigger represents the
case study's phrase “bad weather”; that the European AQI threshold and LLM
range are suitable for the prototype; that sensor 16 is traffic; that sensor
19 is unclassified and excluded; that only historically observed candidate
speeds are supported; and that Year N+1 has the same column order as Year N
despite having no header. The app excludes licence plates. None of these
assumptions constitutes stakeholder approval.

### Glossary

- **Candidate speed:** a historically observed speed tested by the model.
- **Triggered hour:** an hour when darkness or black-ice danger is true.
- **Near-accident rate:** the hourly count of events in the defined
  near-accident categories, averaged over a specified period.
- **Operator decision:** the speed chosen by the operator when the system
  cannot decide on its own; it applies until the system can decide again.
- **Software 1.0 / 2.0 / 3.0:** deterministic code / trained machine-learning
  model / prompted LLM.
- **A/T software:** application-domain components (A) versus technical
  infrastructure components (T).

### Natural-language quality review

The original terms “bad weather,” “reduce speed,” “high air quality,” and
“regular basis” are ambiguous without thresholds, units, timing, and decision
rules. The prototype makes some of them measurable, but the selected
interpretations remain assumptions. The one-near-accident target needs a
defined population and evaluation period. The claim that a model estimate
means a speed reduction will improve safety is not supported by observational
data. Legal jurisdiction, failure response, and operator authority also need
explicit acceptance criteria before any real-world use.
