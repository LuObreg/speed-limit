# Exercise 14: Full MVP Implementation

## Assignment

After completing the other SpeedLimit home exercises, implement an MVP with
both a neural-network training stage (clean and transform supplied data) and
an operational stage that sets the speed limit regularly. Sensors may be
simulated through a GUI; high-performance deployment is not required.

## Existing Evidence

The MVP includes a training workflow and an operational simulation. The
training workflow reads the supplied CSV files, excludes licence plates,
cleans and aggregates sensor readings, aligns near-accident labels by hour,
builds model features, trains a neural network, evaluates chronologically,
and saves model artifacts and reports. The Year N model can be evaluated on
Year N+1 before a separate annual model is trained.

The operational workflow replays sensor readings with a simulator, validates
inputs, routes weather/light conditions to the neural network, obtains air
quality from an API, invokes an LLM through a bounded evaluator when the
configured trigger is met, and combines proposals in the speed controller.
Guardrails bound inputs and outputs. A GUI displays telemetry and provides
simulation controls and operator override. A command-line interface supports
training, evaluation, retraining, and running the application.

The implementation uses configuration for business policy and technical
settings. Automated tests use controlled clients and fixtures; they do not
prove live service reliability or safety. This is a minimum viable course
prototype and is not deployed to control public-road signs.
