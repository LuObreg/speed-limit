# Exercise 11: Air-Quality LLM Test Cases

## Assignment

Test the air-quality part of the system. It asks an LLM how much to lower the
speed limit when the air is polluted. Explain what information it receives,
whether we know the correct answer and how, then write at least five tests.

## Existing Evidence

The system receives an air-quality score called AQI. It may also receive extra
information, such as which pollutants were measured, the location, and the
time. If AQI is 60 or higher, the system asks the LLM to suggest lowering the
speed by 10 to 20 km/h. These numbers were chosen for this project; they have
not been checked by air-quality or traffic-safety experts.

The system checks that the LLM's reply contains a usable number within the
allowed range. If the reply is not usable, it asks once more. If that also
fails, the controller marks the decision as failed and asks the operator to
choose the speed. It does not treat a failure as proof that no reduction is
needed.

| Test | Input or situation | Expected result |
| --- | --- | --- |
| AQ-01 | AQI is 59 and weather does not require a reduction. | Do not ask the LLM. Use 80 km/h if all other required information is valid. |
| AQ-02 | Run once with AQI 60 and once with AQI 61. The test LLM replies with a reduction of 14 km/h. | Ask the LLM in both runs and accept the 14 km/h reduction. |
| AQ-03 | The LLM replies with JSON containing `speed_reduction_kmh: 14`. | Read the number and accept a 14 km/h reduction. |
| AQ-04 | In separate runs, the LLM suggests reductions of 50 km/h and 1 km/h. | Limit the suggestions to the allowed maximum of 20 and minimum of 10 km/h. |
| AQ-05 | The first reply cannot be read; the second gives a reduction of 12 km/h. | Ask once more, then accept the 12 km/h reduction. |
| AQ-06 | Both replies cannot be used; one contains `true` instead of a number. | Mark the decision as failed and ask the operator to choose the speed. |
| AQ-07 | The LLM service cannot be reached. | Mark the decision as failed. Do not silently choose a made-up reduction. |
| AQ-08 | Some extra information, such as location or pollutant readings, is missing. | Build the request using the information that is available. |
| AQ-09 | The air-quality service fails or gives an invalid score, such as a negative or non-number value. | Mark the decision as failed; do not report that no speed reduction is needed. |

We do not have a verified answer for the correct speed reduction in real
traffic. The example ranges used by the tests (AQI 62 -> 10-13, AQI 72 ->
11-16, AQI 95 -> 16-20, and AQI 130 -> 18-20 km/h) are guesses chosen for the
project, not rules confirmed by experts. The LLM cannot prove that its own
answer is correct. These tests check how the software handles replies; they do
not prove that a real LLM gives correct or safe advice.
