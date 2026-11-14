# Exercise 12: Human-in-the-Loop

## Assignment

Would you place one or more humans in the loop? Where exactly, and what skills
would they need?

## Existing Evidence

I would use a human-on-the-loop model as an operator: the software
processes readings and proposes a speed, while an operator watches the
live behaviour and has permissions to override the value proposed by the service. The operator should be
shown the measured values, active decision path, proposed and published speed,
service/model status, and whether the operator must decide the speed.

An operator should intervene when sensor values are implausible or missing,
the API/model/LLM is unavailable, the system reports that the risk target is
not met, the result is degraded, or a real incident requires a response.
The operator should be able to pause the simulation, enter or correct a
reading for demonstration, override the proposed speed, and record the reason.

Operators need training in the traffic rules applicable to their role, the
system's thresholds and limitations, sensor and service failure indicators,
degraded versus valid decisions, override procedures, incident escalation,
privacy, and cybersecurity. Model or data specialists need additional skills
in sensor quality, model evaluation, uncertainty, data drift, and the
difference between observational predictions and causal safety evidence.
Traffic-safety, legal, and environmental specialists should approve the
relevant policy and thresholds.

The prototype implements part of this: the operator sees every decision with
its explanation, can run predefined test scenarios (which pause the live
replay), chooses the speed when the system cannot, and can override it.
Free entry of readings and recording an override reason are not implemented.
Staffing, operator authority, authentication, required approvals, escalation
contacts and response-time targets have not been established. The simulated
hourly cadence is not a justified real-world response-time requirement.
