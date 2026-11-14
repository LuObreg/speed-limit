"""Type T / Software 1.0 Tkinter dashboard for human-on-the-loop supervision.

All widget updates run on Tk's thread; potentially slow scenario evaluations run
in a worker thread.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Any

LOG = logging.getLogger("speedlimit.gui")

POLL_MS = 100
MAX_HISTORY_ROWS = 300
MAX_LOG_LINES = 1500


def _fmt(value: Any, spec: str = ".1f", unit: str = "") -> str:
    if value is None:
        return "invalid / n.a."
    try:
        return f"{value:{spec}}{unit}"
    except (TypeError, ValueError):
        return str(value)


def format_time(timestamp: Any) -> str:
    return str(timestamp)[:16] if timestamp is not None else "-"


def format_flags(decision: Any) -> str:
    flags = []
    if decision.override_active:
        flags.append("OVERRIDE")
    if decision.degraded:
        flags.append("OPERATOR")
    if decision.origin != "simulator":
        flags.append("TEST")
    return " ".join(flags)


def format_history_row(decision: Any) -> tuple[str, str, str, str, str]:
    return (format_time(decision.timestamp), _fmt(decision.speed_kmh, "d").replace("invalid / n.a.", "-"),
            decision.decided_by, _fmt(decision.automatic_speed_kmh, "d").replace("invalid / n.a.", "-"),
            format_flags(decision))


def format_telemetry(decision: Any) -> list[tuple[str, str]]:
    r = decision.reading
    illum = r.get("illuminance_millilux")
    aq = decision.aq_metrics or {}
    return [
        ("Temperature", _fmt(r.get("temperature_c"), ".1f", " °C")),
        ("Water on road", _fmt(r.get("water_level_um"), ".0f", " µm")),
        ("Light", "invalid / n.a." if illum is None else f"{illum / 1000:,.1f} lux"),
        ("European AQI", _fmt(aq.get("european_aqi"), ".0f") if aq else "unavailable"),
        ("PM10 / PM2.5", f"{_fmt(aq.get('pm10'), '.0f')} / {_fmt(aq.get('pm2_5'), '.0f')} µg/m³" if aq else "-"),
    ]


def log_tag(line: str) -> str:
    """Colour class of one controller log line."""
    up = line.upper()
    if "FAILED" in up or "UNAVAILABLE" in up or "SKIPPED" in up or "INPUT GUARDRAIL" in up:
        return "warn"
    if "OVERRIDE" in up or up.startswith("OPERATOR DECISION"):
        return "override"
    if "GUARDRAIL" in up:
        return "guard"
    if "LLM" in up:
        return "llm"
    if "NN" in up:
        return "nn"
    if up.startswith("REQ3"):
        return "default"
    return "plain"


class OperatorDashboard:
    """Type T / Software 1.0 - operator panel for the speed-limit system."""

    TAG_COLORS = {"nn": "#1f5fbf", "llm": "#7b3fb0", "default": "#555555",
                  "guard": "#b36b00", "override": "#d35400", "warn": "#c0392b", "plain": "#222222"}

    LIVE_REPLAY = "Live replay - historical Year N sensors + live air quality"
    LIVE_MSG = ("Live replay: one hour of recorded sensor data is replayed every few seconds. "
                "Pick a scenario above to freeze the replay and test one situation.")
    EXPLANATION_STEPS = ("Weather (REQ1)", "Air quality (REQ2)", "Result")
    # name -> (temperature C, water um, illuminance millilux, hour, (AQI, PM10, PM2.5), expected behaviour)
    SCENARIOS: dict[str, tuple[float, float, float, int, tuple[float, float, float], str]] = {
        "Clear day, dry road, clean air": (
            15.0, 0.0, 20_000_000.0, 12, (20.0, 12.0, 6.0),
            "Nothing dangerous: REQ1 and REQ2 are not triggered, so REQ3 keeps the normal speed."),
        "Night, dry road": (
            10.0, 0.0, 100.0, 23, (20.0, 12.0, 6.0),
            "Darkness (REQ1) -> the NN picks the highest speed whose estimated risk meets the target."),
        "Black ice in daylight": (
            -3.0, 1_500.0, 20_000_000.0, 10, (20.0, 12.0, 6.0),
            "Water on the road below 0 °C (REQ1) -> the NN reduces the speed."),
        "Black ice at night": (
            -5.0, 2_000.0, 50.0, 3, (20.0, 12.0, 6.0),
            "Darkness AND black ice (REQ1) -> the NN usually proposes its strongest reduction."),
        "High air pollution (needs Ollama)": (
            15.0, 0.0, 20_000_000.0, 12, (95.0, 95.0, 62.0),
            "AQI above the threshold (REQ2) -> the LLM reduces the normal speed by 10-20 km/h. "
            "Without a running LLM the system asks the operator to choose the speed."),
        "Night + high air pollution (needs Ollama)": (
            5.0, 0.0, 100.0, 22, (95.0, 95.0, 62.0),
            "REQ1 and REQ2 both fire -> the lower of the NN and LLM proposals is published."),
        "Faulty temperature sensor": (
            999.0, 0.0, 20_000_000.0, 12, (20.0, 12.0, 6.0),
            "The input guardrail rejects 999 °C -> REQ1 cannot be assessed, so the system asks "
            "the operator to choose the speed."),
    }
    MISSING_VALUES = ("invalid / n.a.", "unavailable", "-")

    def __init__(self, controller: Any, simulator: Any, telemetry_queue: "queue.Queue",
                 speed_range: tuple[int, int], default_kmh: int) -> None:
        self._controller, self._sim, self._queue = controller, simulator, telemetry_queue
        self._lo, self._hi = speed_range
        self._default = default_kmh
        self._scenario: str | None = None       # None = live replay
        self._scenario_id = 0
        self._scenario_running = False
        self._scenario_done = threading.Event()
        self._last_known: dict[str, tuple[str, Any]] = {}   # telemetry row -> (last valid text, time)
        self._prompt: Any = None
        self._closing = False

        self.root = tk.Tk()
        self.root.title("SpeedLimit Highway - Operator Dashboard (human on the loop)")
        self.root.geometry("1200x850")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    def run(self) -> None:
        """Blocks until the window is closed."""
        self.root.after(POLL_MS, self._poll)
        self.root.mainloop()

    def _build_ui(self) -> None:
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        bold = ("TkDefaultFont", 10, "bold")

        self._banner = tk.Label(root, text="Starting live replay ...", bg="#dff0d8", fg="#2d5a1e",
                                font=("TkDefaultFont", 11, "bold"), pady=6)
        self._banner.grid(row=0, column=0, columnspan=2, sticky="ew")

        left = ttk.Frame(root, padding=10)
        left.grid(row=1, column=0, sticky="ns")
        right = ttk.Frame(root, padding=(0, 10, 10, 10))
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self._sign = tk.Canvas(left, width=190, height=190, highlightthickness=0)
        self._sign.grid(row=0, column=0, pady=(0, 4))
        self._sign.create_oval(8, 8, 182, 182, fill="white", outline="#c0392b", width=16)
        # Explicit fill: the canvas default follows the OS theme and can be white on white.
        self._sign_text = self._sign.create_text(95, 95, text="--", fill="black",
                                                 font=("Arial", -72, "bold"))
        self._delta = ttk.Label(left, text="waiting for first reading ...", justify="center",
                                font=("TkDefaultFont", 11, "bold"))
        self._delta.grid(row=1, column=0)
        self._sign_caption = ttk.Label(left, text="", justify="center")
        self._sign_caption.grid(row=2, column=0, pady=(0, 8))

        tele = ttk.LabelFrame(left, text="Sensor readings", padding=8)
        tele.grid(row=3, column=0, sticky="ew")
        self._tele_vars: dict[str, tk.StringVar] = {}
        for n, name in enumerate(["Temperature", "Water on road", "Light", "European AQI", "PM10 / PM2.5"]):
            ttk.Label(tele, text=name + ":").grid(row=n, column=0, sticky="w")
            var = tk.StringVar(value="-")
            ttk.Label(tele, textvariable=var).grid(row=n, column=1, sticky="w", padx=(8, 0))
            self._tele_vars[name] = var

        ovr = ttk.LabelFrame(left, text="Manual override (operator)", padding=8)
        ovr.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(ovr, text="km/h:").grid(row=0, column=0, sticky="w")
        self._ovr_var = tk.StringVar(value=str(self._default))
        ttk.Spinbox(ovr, from_=self._lo, to=self._hi, increment=5, width=5,
                    textvariable=self._ovr_var).grid(row=0, column=1, padx=4)
        ttk.Button(ovr, text="Set", command=self._set_override, width=5).grid(row=0, column=2)
        ttk.Button(ovr, text="Clear", command=self._clear_override, width=6).grid(row=0, column=3, padx=(4, 0))
        self._ovr_label = ttk.Label(ovr, text="no override active", wraplength=220)
        self._ovr_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        self._status = ttk.Label(left, text="", justify="left", foreground="#666666", wraplength=240)
        self._status.grid(row=5, column=0, sticky="w", pady=(8, 0))

        scen = ttk.LabelFrame(right, text="1. Choose what to simulate", padding=8)
        scen.grid(row=0, column=0, sticky="ew")
        scen.columnconfigure(0, weight=1)
        self._scenario_var = tk.StringVar(value=self.LIVE_REPLAY)
        combo = ttk.Combobox(scen, textvariable=self._scenario_var, state="readonly",
                             values=[self.LIVE_REPLAY, *self.SCENARIOS])
        combo.grid(row=0, column=0, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._run_scenario())
        self._scenario_button = ttk.Button(scen, text="Run again", command=self._run_scenario, width=10)
        self._scenario_button.grid(row=0, column=1, padx=(6, 0))
        self._scenario_msg = ttk.Label(scen, text=self.LIVE_MSG, wraplength=800, justify="left")
        self._scenario_msg.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        why = ttk.LabelFrame(right, text="2. Why does the sign show this speed?", padding=8)
        why.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        why.columnconfigure(1, weight=1)
        self._why_vars: dict[str, tk.StringVar] = {}
        for n, step in enumerate(self.EXPLANATION_STEPS):
            ttk.Label(why, text=step + ":", font=bold).grid(row=n, column=0, sticky="nw", pady=2)
            var = tk.StringVar(value="-")
            ttk.Label(why, textvariable=var, wraplength=750, justify="left").grid(
                row=n, column=1, sticky="w", padx=(8, 0), pady=2)
            self._why_vars[step] = var

        tabs = ttk.Notebook(right)
        tabs.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        hist = ttk.Frame(tabs, padding=4)
        tabs.add(hist, text="History")
        hist.columnconfigure(0, weight=1)
        hist.rowconfigure(0, weight=1)
        cols = ("time", "speed", "by", "auto", "flags")
        self._tree = ttk.Treeview(hist, columns=cols, show="headings", height=8)
        for c, w, t in zip(cols, (140, 70, 140, 110, 200), ("Time", "Sign km/h", "Decided by",
                                                           "System km/h", "Flags")):
            self._tree.heading(c, text=t)
            self._tree.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(hist, command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        logf = ttk.Frame(tabs, padding=4)
        tabs.add(logf, text="Technical log")
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self._log = tk.Text(logf, height=10, wrap="word", state="disabled")
        lsb = ttk.Scrollbar(logf, command=self._log.yview)
        self._log.configure(yscrollcommand=lsb.set)
        self._log.grid(row=0, column=0, sticky="nsew")
        lsb.grid(row=0, column=1, sticky="ns")
        for tag, colour in self.TAG_COLORS.items():
            self._log.tag_configure(tag, foreground=colour)
        self._log.tag_configure("head", font=("TkDefaultFont", 9, "bold"))

        self._prompt_reason = tk.StringVar(value="")
        self._prompt_metrics = tk.StringVar(value="")
        self._prompt_speed = tk.StringVar(value="")
        self._prompt_msg = tk.StringVar(value="")

    def _ask_operator(self, d: Any) -> None:
        """Open (or refresh) the dialog asking the operator to choose the speed."""
        if self._prompt is None:
            self._prompt = tk.Toplevel(self.root)
            self._prompt.title("Operator decision required")
            self._prompt.protocol("WM_DELETE_WINDOW", self._close_prompt)
            frame = ttk.Frame(self._prompt, padding=12)
            frame.grid(row=0, column=0, sticky="nsew")
            ttk.Label(frame, text="The system cannot decide the speed limit safely on its own. "
                                  "Choose the speed based on the metrics below.",
                      wraplength=440, justify="left", font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=0, columnspan=3, sticky="w")
            ttk.Label(frame, textvariable=self._prompt_reason, wraplength=440, justify="left",
                      foreground="#c0392b").grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 6))
            ttk.Label(frame, textvariable=self._prompt_metrics, justify="left").grid(
                row=2, column=0, columnspan=3, sticky="w")
            ttk.Label(frame, text="Speed limit km/h:").grid(row=3, column=0, sticky="w", pady=(10, 0))
            ttk.Spinbox(frame, from_=self._lo, to=self._hi, increment=5, width=5,
                        textvariable=self._prompt_speed).grid(row=3, column=1, padx=4, pady=(10, 0))
            ttk.Button(frame, text="Apply", command=self._apply_operator_speed).grid(
                row=3, column=2, pady=(10, 0))
            ttk.Label(frame, textvariable=self._prompt_msg, foreground="#c0392b").grid(
                row=4, column=0, columnspan=3, sticky="w")
        self._prompt_reason.set("Problem: " + self.problem_reason(d.log))
        self._prompt_metrics.set("\n".join(self._metric_lines()))

    def _metric_lines(self) -> list[str]:
        """Current metrics; invalid ones are replaced by the last available value."""
        lines = []
        for name, var in self._tele_vars.items():
            text = var.get()
            if text in self.MISSING_VALUES and name in self._last_known:
                known, timestamp = self._last_known[name]
                text = f"{known} (last available, {format_time(timestamp)})"
            lines.append(f"{name}: {text}")
        return lines

    def _apply_operator_speed(self) -> None:
        value = self._parse_speed(self._prompt_speed.get())
        if value is None:
            self._prompt_msg.set("enter a finite number")
            return
        effective = self._controller.set_operator_speed(value)
        self._close_prompt()
        self._delta.configure(text=f"operator chose {effective} km/h (applies at the next update)")
        if self._scenario is not None:
            self._start_scenario(self._scenario)

    def _close_prompt(self) -> None:
        if self._prompt is not None:
            self._prompt.destroy()
            self._prompt = None

    def _parse_speed(self, text: str) -> int | None:
        try:
            requested = float(text.replace(",", "."))
        except ValueError:
            return None
        if not math.isfinite(requested):
            return None
        return min(self._hi, max(self._lo, math.floor(requested + 0.5)))

    @classmethod
    def scenario_inputs(cls, name: str, base: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
        """Scenario name -> (sensor reading, air-quality metrics) for the controller."""
        temp, water, illum, hour, (aqi, pm10, pm25), _ = cls.SCENARIOS[name]
        timestamp = base.replace(hour=hour, minute=0, second=0, microsecond=0)
        reading = {"timestamp": timestamp, "temperature_c": temp,
                   "water_level_um": water, "illuminance_millilux": illum}
        air = {"source": "scenario", "location": "simulated scenario", "time": timestamp.isoformat(),
               "units": {"pm10": "µg/m³", "pm2_5": "µg/m³"},
               "european_aqi": aqi, "pm10": pm10, "pm2_5": pm25}
        return reading, air

    @staticmethod
    def problem_reason(log: tuple[str, ...]) -> str:
        """First log line explaining why the system could not decide on its own."""
        for line in log:
            if log_tag(line) == "warn" or "no candidate speed" in line:
                return line if len(line) <= 90 else line[:87] + "..."
        return "a check failed (see log)"

    @staticmethod
    def explain(d: Any, default_kmh: int) -> dict[str, str]:
        """Plain-language reason for each decision step."""
        if d.weather_triggered is None:
            weather = "Cannot check: a weather sensor value is invalid."
        elif not d.weather_triggered:
            weather = "Daylight and no black ice -> no reduction needed."
        elif d.nn_speed_kmh is not None and d.nn_target_met is False:
            weather = (f"Dark or black ice, and no speed reaches the risk target -> the neural network "
                       f"picks the speed with the lowest estimated risk: {d.nn_speed_kmh} km/h "
                       f"({d.nn_risk_estimates[d.nn_speed_kmh]:.2f} near-accidents/hour).")
        elif d.nn_speed_kmh is not None:
            weather = f"Dark or black ice -> the neural network proposes {d.nn_speed_kmh} km/h."
        elif d.nn_risk_estimates:
            weather = (f"Dark or black ice, but the neural network has no candidate speed below "
                       f"{default_kmh} km/h.")
        else:
            weather = "Dark or black ice, but the neural network failed (see technical log)."

        aqi = (d.aq_metrics or {}).get("european_aqi")
        if d.aq_high is None:
            air = "Check failed: air-quality data or the LLM is unavailable (see technical log)."
        elif not d.aq_high:
            air = f"AQI {aqi:.0f} is below the pollution limit -> no reduction needed."
        else:
            air = (f"AQI {aqi:.0f} is high -> the LLM proposes -{d.llm_reduction_kmh} km/h "
                   f"= {default_kmh - d.llm_reduction_kmh} km/h.")

        if d.override_active:
            result = (f"The operator override sets {d.speed_kmh} km/h "
                      f"(the system alone would show {d.automatic_speed_kmh} km/h).")
        elif d.degraded and d.speed_kmh is None:
            result = "Not every check succeeded, so the system does not guess: the operator must choose the speed."
        elif d.degraded:
            result = f"Not every check succeeded, so the operator chose {d.speed_kmh} km/h."
        elif d.decided_by == "DEFAULT":
            result = f"Nothing requires a reduction -> normal speed {default_kmh} km/h."
        else:
            result = f"The lowest proposal wins -> {d.speed_kmh} km/h (from {d.decided_by})."
        return dict(zip(OperatorDashboard.EXPLANATION_STEPS, (weather, air, result)))

    def _run_scenario(self) -> None:
        name = self._scenario_var.get()
        if name not in self.SCENARIOS:
            self._scenario_id += 1
            self._scenario = None
            self._controller.set_operator_speed(None)
            self._sim.resume()
            self._scenario_msg.configure(text=self.LIVE_MSG)
            return
        if self._scenario_running:
            self._scenario_msg.configure(text="A scenario is still running; wait for it to finish.")
            return
        self._controller.set_operator_speed(None)
        self._start_scenario(name)

    def _start_scenario(self, name: str) -> None:
        self._scenario_id += 1
        scenario_id = self._scenario_id
        self._scenario = name
        self._scenario_running = True
        self._scenario_done.clear()
        self._sim.pause()
        reading, air = self.scenario_inputs(name, datetime.now())
        self._scenario_msg.configure(text=f"Live replay paused. Expected: {self.SCENARIOS[name][-1]}")
        threading.Thread(target=self._process_scenario, args=(reading, air, scenario_id), daemon=True,
                         name="scenario").start()      # may call the LLM: keep UI responsive

    def _process_scenario(self, reading: dict[str, Any], air: dict[str, Any], scenario_id: int) -> None:
        try:
            self._controller.process(
                reading, origin="scenario", air_quality=air,
                scenario_id=scenario_id, commit_state=False,
            )  # result arrives via the queue
        except Exception as exc:  # noqa: BLE001
            self._sim.stop_with_error(exc)
            LOG.exception("scenario evaluation failed")
        finally:
            self._scenario_done.set()

    def _set_override(self) -> None:
        value = self._parse_speed(self._ovr_var.get())
        if value is None:
            self._ovr_label.configure(text="enter a finite number")
            return
        effective = self._controller.set_manual_override(value)
        self._ovr_var.set(str(effective))
        self._ovr_label.configure(text=f"override ARMED: {effective} km/h (applies at the next update)")

    def _clear_override(self) -> None:
        self._controller.set_manual_override(None)
        self._ovr_label.configure(text="override cleared (applies at the next update)")

    def _on_close(self) -> None:
        self._closing = True
        self.root.destroy()

    def _poll(self) -> None:
        if self._closing:
            return
        try:
            for _ in range(50):                       # bounded work per tick
                decision = self._queue.get_nowait()
                if self._is_current_decision(decision):
                    self._show(decision)
        except queue.Empty:
            pass
        if self._scenario_running and self._scenario_done.is_set():
            self._scenario_running = False
        self._refresh_status()
        self.root.after(POLL_MS, self._poll)

    def _is_current_decision(self, decision: Any) -> bool:
        """Live ticks only while replaying; scenario results only for the scenario on screen."""
        if decision.origin == "simulator":
            return self._scenario is None
        return self._scenario is not None and decision.scenario_id == self._scenario_id

    def _show(self, d: Any) -> None:
        self._sign.itemconfig(self._sign_text, text="?" if d.speed_kmh is None else str(d.speed_kmh))
        by = d.decided_by.replace("_", " ")
        self._sign_caption.configure(text=f"decided by {by}")
        if d.speed_kmh is None:
            self._delta.configure(text="awaiting operator decision")
        else:
            reduction = self._default - d.speed_kmh
            self._delta.configure(
                text=f"normal {self._default} km/h -> {d.speed_kmh} km/h (-{reduction} km/h)" if reduction > 0
                else f"no reduction (normal {self._default} km/h)")
        if d.override_active:
            self._banner.configure(text=f"MANUAL OVERRIDE ACTIVE - sign held at {d.speed_kmh} km/h by the operator",
                                   bg="#fdebd0", fg="#a04000")
        elif d.degraded and d.speed_kmh is None:
            self._banner.configure(text=f"OPERATOR DECISION REQUIRED - {self.problem_reason(d.log)}",
                                   bg="#f5b7b1", fg="#78281f")
        elif d.degraded:
            self._banner.configure(text=f"OPERATOR DECISION ACTIVE - {self.problem_reason(d.log)}; "
                                        f"sign set to {d.speed_kmh} km/h by the operator",
                                   bg="#fcf3cf", fg="#7d6608")
        elif d.origin != "simulator":
            self._banner.configure(text=f"SCENARIO - {self._scenario} (live replay paused)",
                                   bg="#d6eaf8", fg="#1b4f72")
        else:
            self._banner.configure(text="LIVE REPLAY - the system decides the speed for each recorded hour",
                                   bg="#dff0d8", fg="#2d5a1e")
        for name, text in format_telemetry(d):
            self._tele_vars[name].set(text)
            if text not in self.MISSING_VALUES:
                self._last_known[name] = (text, d.timestamp)
        for step, text in self.explain(d, self._default).items():
            self._why_vars[step].set(text)
        if d.speed_kmh is None:
            self._ask_operator(d)
        else:
            self._close_prompt()

        self._tree.insert("", 0, values=format_history_row(d))
        for old in self._tree.get_children()[MAX_HISTORY_ROWS:]:
            self._tree.delete(old)

        shown = "awaiting operator" if d.speed_kmh is None else f"{d.speed_kmh} km/h"
        self._append_log(f"{format_time(d.timestamp)}  ->  {shown}  ({by})  {format_flags(d)}".rstrip(),
                         "head")
        for line in d.log:
            self._append_log("    " + line, log_tag(line))

    def _append_log(self, text: str, tag: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n", tag)
        extra = int(self._log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if extra > 0:
            self._log.delete("1.0", f"{extra + 1}.0")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _refresh_status(self) -> None:
        s = self._sim
        if s.callback_error is not None:
            state = f"STOPPED: {s.callback_error}"
        else:
            state = "PAUSED" if s.is_paused else ("running" if s.is_running else "stopped")
        self._status.configure(text=f"Live replay: {state} - hour {s.position} of {s.total_rows} "
                                    f"(cycle {s.cycle + 1}), {s.seconds_per_sim_hour:g} s per hour")
