# SpeedLimit Highway Prototype

An SE4AI course simulation combining weather sensors, a neural-network risk
model, an air-quality LLM evaluator, and deterministic guardrails under
human-on-the-loop supervision. Observational predictions do not establish
that changing a sign reduces accidents. This is not approved road-control
software.

## Documentation Map

- [Requirements specification](./requirements_specification.md): the case-study
  requirements (REQ1-REQ3, SPEC1-SPEC4) made measurable, plus assumptions.
- [Project decisions](./project_decisions.md): decision records, traceability,
  model limitations, and verification evidence.
- [Data notes](./data.md): supplied files, data-quality findings, and how the
  code handles them.
- [Stakeholder questions](./stakeholder_meeting_notes.md): open questions
  (what I would have done first if it was real software that I had to develop!).
- [Model training report](../models/training_report.md): plain-language reading
  of the training curves and the Year N+1 check.
- [Exercises](./exercises/): answers to the individual course exercises.
- [Diagrams](./diagrams/): rendered UML views (listed below).

The original case study (Case_Study_Speed_Limit_V220.pdf) and the lecture
slides are course material and are not part of this repository.

## Repository Layout

| Path | Content |
| --- | --- |
| `source/` | Application code (one module per component, see the table below). |
| `tests/` | Automated tests (pytest). |
| `settings.py`, `configuration.toml` | Configuration records/loader and the configured values. |
| `data/` | Supplied case-study CSV files (unchanged). |
| `models/year_n/`, `models/annual/` | Trained model of Year N and of the annual retraining; both are used by the workflow, they are not backups. |
| `docs/` | This documentation. |

Logs, bytecode, pytest/coverage caches and virtual environments are generated
locally and excluded by `.gitignore`.

## Diagrams

- [use_case_diagram.svg](./diagrams/use_case_diagram.svg): actors and use cases.
- [class_diagram_data.svg](./diagrams/class_diagram_data.svg): supplied data and
  the model-ready hourly data.
- [class_diagram_learning.svg](./diagrams/class_diagram_learning.svg) and
  [class_diagram_decision.svg](./diagrams/class_diagram_decision.svg): classes of
  the code.
- [component_diagram_full.svg](./diagrams/component_diagram_full.svg),
  [component_diagram_weather_light.svg](./diagrams/component_diagram_weather_light.svg)
  and [data_flow_diagram.svg](./diagrams/data_flow_diagram.svg): training and
  production phase.
- [sequence_diagram.svg](./diagrams/sequence_diagram.svg): one hourly update.


## Setup and Operation

Requirements: Python 3.11 or newer (for `tomllib`). Tkinter and a display are
needed only for the GUI (on Debian/Ubuntu: `sudo apt install python3-tk`).
Run every command from the repository root.


```bash
PYTHONPATH=.:source python3 -m app
```

| Command | Effect |
| --- | --- |
| `PYTHONPATH=.:source python3 -m app train` | Train on Year N and save the model in `models/year_n/`. |
| `PYTHONPATH=.:source python3 -m app evaluate` | Evaluate the Year N model on Year N+1; write `models/year_plus_one_evaluation.json`. |
| `PYTHONPATH=.:source python3 -m app retrain` | Run `evaluate` first, then train on Year N + Year N+1 and save in `models/annual/`. |
| `PYTHONPATH=.:source python3 -m app run` | Start the simulation with the operator dashboard (`--ui headless` logs decisions only). |

`run` needs internet access for the [Open-Meteo](https://open-meteo.com)
air-quality API and a local [Ollama](https://ollama.com) service with the
configured model (`ollama pull llama3.2:3b`). Without them the system does not
guess: the decision is marked as degraded and the dashboard asks the operator
to choose the speed. The air-quality location is Vienna and the data is
current, so it does not match the historical hour being replayed.

### Using the GUI

The GUI combines the operator's view of the current speed-limit decision with
controls for simulating different situations. The live replay displays
historical sensor readings alongside the proposed speed, decision explanation,
and history. Choose a scenario from the dropdown to pause the replay and test a
specific flow, such as night driving, black ice, high air pollution, or an
invalid sensor reading. The dashboard shows how the system responds, and lets
the operator set or clear a manual speed override.

## Canonical Component Table

A means application-domain logic; T means technical infrastructure.
Software 1.0 is deterministic code, 2.0 a learned model, and 3.0 a prompted
LLM. SRP/SOC and DRY guide boundaries; deterministic harnesses bound AI effects
without certifying their correctness. The operator can observe, run test
scenarios, and override the simulation rather than approve every update.

| Module / component | Type | Generation | Responsibility |
| --- | --- | --- | --- |
| [app](../source/app.py): ApplicationFactory, Application, CLI | T | 1.0 | Compose collaborators, logging, training/evaluation and runtime workflows. |
| [settings](../settings.py): settings records and loader | A values / T loader and technical values | 1.0 | Validate centralized policy and infrastructure configuration. |
| [data_io](../source/data_io.py): CSVDataReader | T | 1.0 | Read CSVs, including headerless Year N+1; omit licence plates. |
| [data_pipeline](../source/data_pipeline.py): DataCleaner, FeatureEngineer | A | 1.0 | Clean and normalize readings, align hourly labels, build the features shared by training and inference. |
| [nn_model](../source/nn_model.py): SpeedLimitNN, reports | A | 2.0 | Estimate observational hourly risk; evaluate and score supported speeds. |
| [nn_model](../source/nn_model.py): ModelArtifactStore protocol | T interface | 1.0 | Define artifact persistence boundary. |
| [model_artifacts](../source/model_artifacts.py): FilesystemModelArtifactStore | T | 1.0 | Persist model, history, reports and plots. |
| [controller_state](../source/controller_state.py): ThreadSafeControllerState | T | 1.0 | Hold the operator's override and fallback speed safely across threads. |
| [weather_router](../source/weather_router.py): WeatherRouter | A | 1.0 | Trigger NN on darkness OR black-ice danger. |
| [air_quality_api](../source/air_quality_api.py): AirQualityAPIClient | T | 1.0 | Fetch/cache AQ metrics and handle transport/response errors. |
| [air_quality_llm](../source/air_quality_llm.py): evaluator and GoldenAnswerHarness | A | 3.0 / 1.0 harness | Prompt for an amount; parse, retry, clamp, and check engineering fixtures. |
| [llm_clients](../source/llm_clients.py): LLMClient, OllamaClient | T | 1.0 | Isolate LLM HTTP transport. |
| [guardrails](../source/guardrails.py): InputGuardrail, OutputGuardrail | A | 1.0 | Check physical input bounds and configured output range. |
| [speed_controller](../source/speed_controller.py): SpeedLimitController, Decision | A | 1.0 | Orchestrate 2.0/3.0 proposals, lowest-valid selection, defaults and degraded policy. |
| [sensor_simulator](../source/sensor_simulator.py): SensorSimulator | T | 1.0 | Replay hourly readings and manage lifecycle. |
| [gui_app](../source/gui_app.py): OperatorDashboard | T | 1.0 | Show decisions and explanations; run test scenarios; accept operator speed and override. |

## Testing

```powershell
python -m pytest                                  # full suite, 100% line and branch coverage required
python -m pytest --cov-report=html                # plus an HTML coverage report in htmlcov/
python -m pytest tests/test_weather_router.py --no-cov   # one file, without the coverage gate
```

[pytest.ini](../pytest.ini) sets the paths and the coverage gate, so no
`PYTHONPATH` is needed for tests. The tests use scripted LLM replies, a local
fake HTTP server and GUI doubles; they do not check the live Ollama model, the
live API, or road safety. Test design is described in exercises 08, 10 and 11;
the latest recorded run is in
[project decisions](./project_decisions.md#6-verification-evidence).

## Attribution

Air-quality data: [Open-Meteo](https://open-meteo.com), CC BY 4.0, based on
CAMS ENSEMBLE (Copernicus Atmosphere Monitoring Service). Preserve attribution
when using its data.
