# Exercise 04: Non-Functional Requirements

## Assignment

Define non-functional requirements for the SpeedLimit system.

Source: SE4AI_Requirements_Structuring_V720.pdf, slide 10, exercise
SpeedLimit07.

## Existing Evidence

- **Safety and bounded output:** every speed sent to the simulated sign must
 be within configured operating bounds. The bounds require approval for the
 applicable jurisdiction before real deployment.
- **Failure transparency:** invalid required input or component failure must
 be visible to the operator, not reported as a successful “no reduction
 needed” result. The system then asks the operator to choose the speed.
- **Traceability:** retain enough non-sensitive telemetry to explain input
 validity, active decision path, AI proposal, guardrail result, published
 speed, and degraded or override status.
- **Privacy:** licence plates must not be used, logged, or retained in the
 application, derived data, or model inputs.
- **Changeability:** business thresholds and policy values must be centralized
 in validated configuration. A policy change must be reflected in
 requirements, tests, and documentation.
- **Verifiability:** each component needs at least five meaningful test cases
 for equivalence classes, boundaries, and failure behavior. Coverage is
 supporting evidence, not proof of model quality or safety.
- **Usability and human oversight:** the operator interface should display
 the current decision and degraded state and permit pausing and manual
 override in the simulation.

The current material does not set measurable targets for availability, update
latency, recovery time, reliability, cybersecurity, retention periods, access
control, incident reporting, or operator response time. Those targets need a
defined deployment context and stakeholder agreement.
