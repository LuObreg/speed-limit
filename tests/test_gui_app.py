"""Headless tests for the dashboard presentation and controls."""

import queue
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gui_app


class Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self, *_args, **kwargs):
        self.options = dict(kwargs)
        self.children = []
        self.lines = 0
        self.items = {}
        self.calls = []

    def __getattr__(self, name):
        if name in {"columnconfigure", "rowconfigure", "title", "geometry", "protocol",
                    "mainloop", "destroy", "grid", "configure", "heading", "column",
                    "tag_configure", "see", "set", "yview", "after", "itemconfig", "create_oval",
                    "add", "bind"}:
            return lambda *args, **kwargs: self.calls.append((name, args, kwargs))
        raise AttributeError(name)

    def configure(self, **kwargs):
        self.options.update(kwargs)
        self.calls.append(("configure", (), kwargs))

    def create_text(self, *_args, **_kwargs):
        return "sign-text"

    def insert(self, *_args, **kwargs):
        self.lines += 1
        if "columns" in self.options:
            item_id = f"item-{self.lines}"
            self.items[item_id] = kwargs
            self.children.insert(0, item_id)
            return item_id

    def get_children(self):
        return tuple(self.children)

    def delete(self, item, *_args):
        if item in self.items:
            del self.items[item]
            self.children.remove(item)
        else:
            self.calls.append(("delete", (item,), {}))

    def index(self, _index):
        return f"{self.lines + 1}.0"


class Simulator:
    seconds_per_sim_hour = 2.0
    callback_error = None
    is_paused = False
    is_running = True
    position = 1
    total_rows = 5
    cycle = 0

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop_with_error(self, error):
        self.callback_error = error


class Controller:
    def __init__(self, *, error=None):
        self.error = error
        self.overrides = []
        self.operator_speeds = []
        self.processed = []
        self.process_metadata = []

    def set_manual_override(self, value):
        self.overrides.append(value)
        return value

    def set_operator_speed(self, value):
        self.operator_speeds.append(value)
        return value

    def process(self, reading, origin, air_quality=None, scenario_id=None, commit_state=True):
        if self.error:
            raise self.error
        self.processed.append((reading, origin, air_quality))
        self.process_metadata.append((scenario_id, commit_state))


@pytest.fixture()
def dashboard(monkeypatch):
    for module, names in (
        (gui_app.tk, ("Tk", "Label", "Canvas", "Text", "Toplevel")),
        (gui_app.ttk, ("Frame", "Label", "LabelFrame", "Treeview", "Scrollbar",
                       "Button", "Spinbox", "Combobox", "Notebook")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, Widget)
    monkeypatch.setattr(gui_app.tk, "StringVar", Variable)
    sim, controller = Simulator(), Controller()
    view = gui_app.OperatorDashboard(controller, sim, queue.Queue(), (20, 80), 80)
    return view, sim, controller


def test_presentation_formatters_cover_valid_missing_and_malformed_values():
    assert gui_app._fmt(None) == "invalid / n.a."
    assert gui_app._fmt(2.5, ".1f", "°") == "2.5°"
    assert gui_app._fmt("bad") == "bad"
    assert gui_app.format_time(None) == "-"
    assert gui_app.format_time("2026-01-02T03:04:05") == "2026-01-02T03:04"

    decision = SimpleNamespace(override_active=True, degraded=True, origin="manual_test",
                               timestamp="now", speed_kmh=40, decided_by="MANUAL",
                               automatic_speed_kmh=70)
    assert gui_app.format_flags(decision) == "OVERRIDE OPERATOR TEST"
    assert gui_app.format_history_row(decision) == ("now", "40", "MANUAL", "70",
                                                    "OVERRIDE OPERATOR TEST")
    decision.speed_kmh = decision.automatic_speed_kmh = None
    assert gui_app.format_history_row(decision)[1::2] == ("-", "-")


def test_problem_reason_picks_the_failing_step():
    reason = gui_app.OperatorDashboard.problem_reason
    assert reason(("REQ1: NN not needed", "REQ2: FAILED (cannot reach Ollama)")) == \
        "REQ2: FAILED (cannot reach Ollama)"
    assert reason(("REQ1: NN has no candidate speed below 80 km/h",)).startswith("REQ1: NN has no candidate")
    assert reason(("REQ2: FAILED " + "x" * 100,)).endswith("...")
    assert reason(()) == "a check failed (see log)"


def test_telemetry_and_log_tag_variants():
    decision = SimpleNamespace(
        reading={"temperature_c": None, "water_level_um": 10, "illuminance_millilux": None},
        aq_metrics={},
    )
    rows = dict(gui_app.format_telemetry(decision))
    assert rows["Temperature"] == "invalid / n.a."
    assert rows["Light"] == "invalid / n.a."
    assert rows["European AQI"] == "unavailable"
    assert rows["PM10 / PM2.5"] == "-"

    decision.reading["illuminance_millilux"] = 5000
    decision.aq_metrics = {"european_aqi": 60, "pm10": 2, "pm2_5": 1}
    rows = dict(gui_app.format_telemetry(decision))
    assert rows["Light"] == "5.0 lux"
    assert rows["European AQI"] == "60"

    for line, tag in [
        ("request FAILED", "warn"), ("AQ unavailable", "warn"), ("REQ1 skipped", "warn"),
        ("manual OVERRIDE set", "override"), ("OPERATOR DECISION: 45 km/h", "override"),
        ("output GUARDRAIL", "guard"),
        ("call LLM", "llm"), ("call NN", "nn"), ("REQ3 fallback", "default"),
        ("ordinary message", "plain"),
    ]:
        assert gui_app.log_tag(line) == tag


def test_explain_covers_every_decision_path():
    base = dict(weather_triggered=False, nn_speed_kmh=None, nn_risk_estimates={}, nn_target_met=None,
                aq_metrics={"european_aqi": 20}, aq_high=False, llm_reduction_kmh=None,
                override_active=False, degraded=False,
                decided_by="DEFAULT", speed_kmh=80, automatic_speed_kmh=80)
    explain = gui_app.OperatorDashboard.explain

    def why(**changes):
        return explain(SimpleNamespace(**{**base, **changes}), 80)

    normal = why()
    assert normal["Weather (REQ1)"].startswith("Daylight")
    assert normal["Air quality (REQ2)"] == "AQI 20 is below the pollution limit -> no reduction needed."
    assert normal["Result"] == "Nothing requires a reduction -> normal speed 80 km/h."

    assert why(weather_triggered=None)["Weather (REQ1)"].startswith("Cannot check")
    assert "proposes 50 km/h" in why(weather_triggered=True, nn_speed_kmh=50, nn_target_met=True)["Weather (REQ1)"]
    lowest = why(weather_triggered=True, nn_speed_kmh=40, nn_target_met=False,
                 nn_risk_estimates={40: 1.79, 70: 2.49})["Weather (REQ1)"]
    assert "lowest estimated risk: 40 km/h (1.79 near-accidents/hour)" in lowest
    no_candidate = why(weather_triggered=True, nn_risk_estimates={80: 1.5})["Weather (REQ1)"]
    assert "no candidate speed below 80 km/h" in no_candidate
    assert "failed" in why(weather_triggered=True)["Weather (REQ1)"]

    assert why(aq_high=None)["Air quality (REQ2)"].startswith("Check failed")
    assert why(aq_high=True, aq_metrics={"european_aqi": 95}, llm_reduction_kmh=15)[
        "Air quality (REQ2)"] == "AQI 95 is high -> the LLM proposes -15 km/h = 65 km/h."

    assert "override sets 40 km/h" in why(override_active=True, speed_kmh=40)["Result"]
    assert "operator must choose" in why(degraded=True, speed_kmh=None)["Result"]
    assert "operator chose 45 km/h" in why(degraded=True, speed_kmh=45)["Result"]
    assert why(decided_by="NN", speed_kmh=50)["Result"] == "The lowest proposal wins -> 50 km/h (from NN)."


def test_scenario_inputs_build_reading_and_air_quality():
    base = datetime(2026, 2, 3, 4, 5, 6)
    for name, (temp, water, illum, hour, (aqi, _, _), expected) in gui_app.OperatorDashboard.SCENARIOS.items():
        reading, air = gui_app.OperatorDashboard.scenario_inputs(name, base)
        assert reading == {"timestamp": datetime(2026, 2, 3, hour), "temperature_c": temp,
                           "water_level_um": water, "illuminance_millilux": illum}
        assert air["european_aqi"] == aqi and air["source"] == "scenario"
        assert expected


def test_dashboard_build_run_close_and_operator_controls(dashboard):
    view, sim, controller = dashboard
    view._append_log("within history limit", "plain")
    view.run()
    assert view.root.calls

    view._ovr_var.set("bad")
    view._set_override()
    assert "finite number" in view._ovr_label.options["text"]
    view._ovr_var.set("55")
    view._set_override()
    assert controller.overrides == [55]
    view._ovr_var.set("55.5")
    view._set_override()
    assert controller.overrides[-1] == 56
    view._ovr_var.set("55.9")
    view._set_override()
    assert controller.overrides[-1] == 56
    view._ovr_var.set("55.4")
    view._set_override()
    assert controller.overrides[-1] == 55
    view._ovr_var.set("inf")
    view._set_override()
    assert "finite" in view._ovr_label.options["text"]
    view._ovr_var.set("1e308")
    view._set_override()
    assert controller.overrides[-1] == 80
    view._clear_override()
    assert controller.overrides[-1] is None

    view._on_close()
    assert view._closing
    view._poll()


def test_dashboard_poll_show_and_log_trimming(dashboard, monkeypatch):
    view, sim, _ = dashboard
    decision = SimpleNamespace(
        timestamp=datetime(2026, 1, 2, 3), speed_kmh=50, decided_by="NN_LLM",
        automatic_speed_kmh=60, override_active=False, degraded=False, origin="simulator",
        reading={"temperature_c": 1, "water_level_um": 2, "illuminance_millilux": 3},
        aq_metrics={"european_aqi": 70, "pm10": 10, "pm2_5": 5}, weather_triggered=False,
        nn_speed_kmh=50, aq_high=True, llm_reduction_kmh=10, llm_source="llm",
        log=("REQ3 default", "INPUT GUARDRAIL invalid"),
    )
    view._queue.put(decision)
    monkeypatch.setattr(gui_app, "MAX_HISTORY_ROWS", 1)
    monkeypatch.setattr(gui_app, "MAX_LOG_LINES", 1)
    view._poll()
    for _ in range(50):
        view._queue.put(decision)
    view._poll()
    assert view._banner.options["text"].startswith("LIVE REPLAY")
    assert len(view._tree.get_children()) == 1
    assert any(call[0] == "delete" for call in view._log.calls)
    view._show(SimpleNamespace(**{**vars(decision), "override_active": True, "degraded": False}))
    assert view._banner.options["text"].startswith("MANUAL OVERRIDE")
    view._show(SimpleNamespace(**{**vars(decision), "override_active": False, "degraded": True}))
    assert view._banner.options["text"] == (
        "OPERATOR DECISION ACTIVE - INPUT GUARDRAIL invalid; sign set to 50 km/h by the operator")
    awaiting = SimpleNamespace(**{**vars(decision), "degraded": True, "speed_kmh": None,
                                  "automatic_speed_kmh": None, "reading": {}, "aq_metrics": None,
                                  "aq_high": None})
    view._show(awaiting)
    assert view._banner.options["text"] == "OPERATOR DECISION REQUIRED - INPUT GUARDRAIL invalid"
    assert view._delta.options["text"] == "awaiting operator decision"
    assert view._prompt is not None
    assert "Temperature: 1.0 °C (last available, 2026-01-02 03:00)" in view._prompt_metrics.get()
    view._show(awaiting)                       # refreshes the open dialog instead of opening another
    view._show(decision)
    assert view._prompt is None
    view._scenario = "Night, dry road"
    view._show(SimpleNamespace(**{**vars(decision), "origin": "scenario", "speed_kmh": 80}))
    assert view._banner.options["text"].startswith("SCENARIO - Night, dry road")
    assert view._delta.options["text"].startswith("no reduction")

    shown_lines = view._log.lines
    view._queue.put(decision)
    view._poll()
    assert view._log.lines == shown_lines      # replay tick dropped while a scenario is shown

    sim.callback_error = RuntimeError("failed")
    view._refresh_status()
    assert "STOPPED" in view._status.options["text"]
    sim.callback_error = None
    sim.is_paused = True
    view._refresh_status()
    assert "PAUSED" in view._status.options["text"]
    sim.is_paused = False
    view._refresh_status()
    assert "running" in view._status.options["text"]
    sim.is_running = False
    view._refresh_status()
    assert "stopped" in view._status.options["text"]

    view._append_log("short message", "plain")


def test_dashboard_scenarios_and_processing_failures(dashboard, monkeypatch):
    view, sim, controller = dashboard

    class ImmediateThread:
        def __init__(self, target, args, **_):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(gui_app.threading, "Thread", ImmediateThread)
    view._scenario_var.set("Black ice at night")
    view._run_scenario()
    assert sim.is_paused and view._scenario == "Black ice at night"
    reading, origin, air = controller.processed[0]
    assert origin == "scenario" and reading["temperature_c"] == -5.0 and air["european_aqi"] == 20.0
    assert view._scenario_msg.options["text"].startswith("Live replay paused. Expected:")
    view._show(SimpleNamespace(
        timestamp=None, speed_kmh=50, decided_by="NN", automatic_speed_kmh=50, override_active=False,
        degraded=False, origin="scenario", reading={}, aq_metrics={"european_aqi": 20},
        weather_triggered=True, nn_speed_kmh=50, nn_target_met=True, aq_high=False, llm_reduction_kmh=None,
        llm_source=None,
        log=()))
    assert view._delta.options["text"] == "normal 80 km/h -> 50 km/h (-30 km/h)"
    assert "Black ice at night" in view._banner.options["text"]
    assert view._why_vars["Result"].get() == "The lowest proposal wins -> 50 km/h (from NN)."

    view._scenario_var.set(view.LIVE_REPLAY)
    view._run_scenario()
    assert not sim.is_paused and view._scenario is None
    assert view._scenario_msg.options["text"] == view.LIVE_MSG
    assert controller.operator_speeds == [None, None]     # every scenario change clears it

    controller.error = RuntimeError("API error")
    view._process_scenario({"timestamp": datetime.now()}, {}, view._scenario_id)
    assert isinstance(sim.callback_error, RuntimeError)


def test_operator_prompt_applies_speed_and_reruns_scenario(dashboard, monkeypatch):
    view, _, controller = dashboard
    started = []
    monkeypatch.setattr(view, "_start_scenario", started.append)
    view._ask_operator(SimpleNamespace(log=("REQ2: FAILED (no LLM)",)))
    assert view._prompt_reason.get() == "Problem: REQ2: FAILED (no LLM)"

    view._prompt_speed.set("abc")
    view._apply_operator_speed()
    assert view._prompt_msg.get() == "enter a finite number" and view._prompt is not None

    view._prompt_speed.set("44")
    view._apply_operator_speed()
    assert controller.operator_speeds == [44] and view._prompt is None and started == []
    assert view._delta.options["text"] == "operator chose 44 km/h (applies at the next update)"

    view._scenario = "Faulty temperature sensor"
    view._ask_operator(SimpleNamespace(log=()))
    view._prompt_speed.set("60")
    view._apply_operator_speed()
    assert started == ["Faulty temperature sensor"]


def test_scenario_results_are_correlated_and_running_scenarios_cannot_overlap(dashboard, monkeypatch):
    view, sim, controller = dashboard
    workers = []

    class DeferredThread:
        def __init__(self, target, args, **_):
            self.target, self.args = target, args
            workers.append(self)

        def start(self):
            pass

    monkeypatch.setattr(gui_app.threading, "Thread", DeferredThread)
    view._scenario_var.set("Night, dry road")
    view._run_scenario()
    first_id = view._scenario_id
    view._run_scenario()
    assert len(workers) == 1

    view._scenario_var.set(view.LIVE_REPLAY)
    view._run_scenario()
    assert view._scenario is None and not sim.is_paused

    stale = SimpleNamespace(origin="scenario", scenario_id=first_id)
    current_scenario = SimpleNamespace(origin="scenario", scenario_id=first_id + 1)
    current_live = SimpleNamespace(origin="simulator", scenario_id=None)
    assert not view._is_current_decision(stale)
    assert not view._is_current_decision(current_scenario)
    assert view._is_current_decision(current_live)

    view._scenario = "Night, dry road"
    view._scenario_id = first_id
    assert view._is_current_decision(stale)
    view._scenario = None
    workers[0].target(*workers[0].args)
    assert view._scenario_done.is_set()
    assert controller.process_metadata == [(first_id, False)]
    view._poll()
    assert not view._scenario_running
