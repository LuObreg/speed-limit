"""Tests for speed controller."""

import pytest
import threading

from tests._support import DAY, NIGHT, ScriptedClient, make_controller
from air_quality_api import AirQualityAPIError
from llm_clients import LLMClientError

class TestController:
    def test_process_calls_are_serialized(self, make_controller, monkeypatch):
        controller = make_controller()
        active = 0
        max_active = 0
        state_lock = threading.Lock()
        both_started = threading.Barrier(3)

        def fake_process(*_args):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            threading.Event().wait(0.02)
            with state_lock:
                active -= 1

        monkeypatch.setattr(controller, "_process", fake_process)

        def call_process():
            both_started.wait()
            controller.process(DAY)

        workers = [threading.Thread(target=call_process) for _ in range(2)]
        for worker in workers:
            worker.start()
        both_started.wait()
        for worker in workers:
            worker.join()

        assert max_active == 1

    def test_req3_default_when_nothing_applies(self, make_controller):
        d = make_controller().process(DAY)
        assert (d.speed_kmh, d.decided_by, d.degraded) == (80, "DEFAULT", False)

    def test_req1_darkness_uses_nn(self, make_controller):
        c = make_controller(nn=55)
        d = c.process(NIGHT)
        assert (d.speed_kmh, d.decided_by, c.nn_fake.calls) == (50, "NN", 1)

    def test_nn_is_not_called_when_router_says_no(self, make_controller):
        c = make_controller()
        c.process(DAY)
        assert c.nn_fake.calls == 0

    def test_black_ice_uses_nn(self, make_controller):
        d = make_controller(nn=40).process({**DAY, "water_level_um": 1500.0, "temperature_c": -3.0})
        assert (d.speed_kmh, d.decided_by) == (40, "NN")

    def test_req2_high_pollution_uses_llm(self, make_controller):
        d = make_controller(aqi=95).process(DAY)
        assert d.decided_by == "LLM" and 60 <= d.speed_kmh <= 70 and d.llm_source == "llm"

    def test_pollution_below_threshold_does_not_call_llm(self, make_controller):
        d = make_controller(aqi=59.9).process(DAY)
        assert d.aq_high is False and d.llm_reduction_kmh is None and d.speed_kmh == 80

    def test_supplied_air_quality_replaces_api_fetch(self, make_controller):
        c = make_controller(aqi=AirQualityAPIError("must not be called"))
        d = c.process(DAY, origin="scenario", air_quality={"european_aqi": 95.0})
        assert d.decided_by == "LLM" and d.aq_metrics == {"european_aqi": 95.0} and not d.degraded

    def test_scenario_decision_does_not_replace_operational_state(self, make_controller):
        controller = make_controller(aqi=20)
        live_decision = controller.process(DAY)
        controller.set_operator_speed(50)

        scenario_decision = controller.process(
            DAY, origin="scenario", air_quality={"european_aqi": 95.0},
            scenario_id=4, commit_state=False,
        )

        assert scenario_decision.scenario_id == 4
        assert scenario_decision.speed_kmh < live_decision.speed_kmh
        assert controller._state.get_operator_speed() == 50

    # Check the AQI gate immediately below, exactly at, and immediately above its threshold.
    @pytest.mark.parametrize("offset,expected_high", [
        pytest.param(-0.001, False, id="below-threshold"),
        pytest.param(0.0, True, id="at-threshold"),
        pytest.param(0.001, True, id="above-threshold"),
    ])
    def test_air_quality_threshold_boundary(self, S, make_controller, offset, expected_high):
        threshold = S.business.air_quality.high_pollution_threshold
        decision = make_controller(aqi=threshold + offset).process(DAY)
        assert decision.aq_high is expected_high
        assert decision.decided_by == ("LLM" if expected_high else "DEFAULT")

    # The controller publishes the stricter proposal when NN or LLM recommends the lower limit.
    @pytest.mark.parametrize("nn,aqi,speed,by", [
        pytest.param(55, 72, 50, "NN", id="nn-proposal-is-lower"),
        pytest.param(70, 72, 67, "LLM", id="llm-proposal-is-lower"),
        pytest.param(67, 72, 60, "NN", id="nn-wins-by-larger-reduction"),
    ])
    def test_lower_of_nn_and_llm_wins(self, make_controller, nn, aqi, speed, by):
        d = make_controller(nn=nn, aqi=aqi).process(NIGHT)
        assert (d.speed_kmh, d.decided_by) == (speed, by)

    def test_air_quality_failure_asks_the_operator(self, make_controller):
        decision = make_controller(aqi=AirQualityAPIError("down")).process(DAY)
        assert (decision.speed_kmh, decision.automatic_speed_kmh) == (None, None)
        assert decision.degraded and decision.decided_by == "AWAITING_OPERATOR"
        assert "awaiting operator" in str(decision)
        assert any("REQ2: FAILED" in line for line in decision.log)

    def test_llm_failure_asks_the_operator(self, make_controller):
        controller = make_controller(
            aqi=95, llm_client=ScriptedClient(LLMClientError("provider unavailable")))
        decision = controller.process(DAY)
        assert decision.speed_kmh is None and decision.degraded

    def test_nn_failure_asks_the_operator(self, make_controller):
        d = make_controller(nn=RuntimeError("boom")).process(NIGHT)
        assert (d.speed_kmh, d.degraded, d.decided_by) == (None, True, "AWAITING_OPERATOR")

    def test_operator_decision_is_used_until_the_system_recovers(self, make_controller):
        controller = make_controller(nn=RuntimeError("boom"))
        assert controller.set_operator_speed(45) == 45
        chosen = controller.process(NIGHT)
        assert (chosen.speed_kmh, chosen.decided_by, chosen.degraded) == (45, "OPERATOR", True)
        assert controller.process(DAY).speed_kmh == 80
        assert controller._state.get_operator_speed() is None
        assert controller.process(NIGHT).speed_kmh is None

    @pytest.mark.parametrize("requested,effective", [(5, 20), (120, 80), (None, None)])
    def test_operator_speed_is_clamped_by_output_guardrail(self, make_controller, requested, effective):
        assert make_controller().set_operator_speed(requested) == effective

    def test_no_candidate_meets_target_selects_lowest_estimated_risk(self, make_controller):
        controller = make_controller()
        controller.nn_fake.predict_risk = lambda _features: [2.71, 2.54, 2.39, 2.31, 2.29]
        d = controller.process(NIGHT)
        assert (d.speed_kmh, d.decided_by, d.degraded, d.nn_target_met) == (70, "NN", False, False)
        assert any("TARGET NOT MET" in line and "70 km/h (2.31)" in line for line in d.log)

    def test_equal_risks_above_target_select_the_lowest_speed(self, make_controller):
        controller = make_controller()
        controller._nn.value = "none"
        d = controller.process(NIGHT)
        assert (d.speed_kmh, d.degraded, d.nn_target_met) == (40, False, False)

    def test_req1_does_not_count_normal_speed_as_a_reduction(self, make_controller):
        # If the only candidate is the normal speed, there is no reduction to select: ask the operator.
        controller = make_controller(nn=500)
        assert controller.process(DAY).speed_kmh == 80
        controller.nn_fake.candidate_speeds = (80,)

        decision = controller.process(NIGHT)

        assert (decision.speed_kmh, decision.degraded, decision.decided_by) == (
            None, True, "AWAITING_OPERATOR"
        )
        assert "REQ1: NN has no candidate speed below 80 km/h" in decision.log

    def test_req1_selects_lower_candidate_when_default_also_meets_target(self, make_controller):
        # When both 70 and 80 meet the target, only 70 is an actual reduction from normal speed.
        controller = make_controller(nn=500)
        controller.nn_fake.candidate_speeds = (70, 80)

        decision = controller.process(NIGHT)

        assert (decision.speed_kmh, decision.degraded, decision.nn_target_met) == (70, False, True)
        assert "REQ1: NN selects 70 km/h (estimated risk 0.50 <= target 1 near-accidents/hour)" in decision.log

    def test_candidate_range_limits_nn_speed(self, make_controller):
        # An estimate above target still reduces, a valid reduction is accepted, and an out-of-range value is clamped.
        unsafe = make_controller(nn=5).process(NIGHT)
        assert unsafe.speed_kmh == 40 and not unsafe.degraded and unsafe.nn_target_met is False
        assert make_controller(nn=500).process(NIGHT).speed_kmh == 70
        controller = make_controller(nn=500)
        controller.nn_fake.candidate_speeds = (10,)
        guarded = controller.process(NIGHT)
        assert guarded.speed_kmh == 20
        assert any("OUTPUT GUARDRAIL" in line for line in guarded.log)

    def test_candidate_above_normal_speed_is_not_selected(self, make_controller):
        # A speed above the normal limit cannot be selected as a weather-triggered reduction.
        controller = make_controller(nn=500)
        controller.nn_fake.candidate_speeds = (100,)
        decision = controller.process(NIGHT)
        assert decision.speed_kmh is None and decision.degraded
        assert "REQ1: NN has no candidate speed below 80 km/h" in decision.log

    def test_invalid_candidate_estimates_are_treated_as_nn_failure(self, make_controller):
        controller = make_controller()
        controller.nn_fake.candidate_speeds = (40, 50)
        controller.nn_fake.predict_risk = lambda _features: [0.5]
        decision = controller.process(NIGHT)
        assert decision.degraded and decision.speed_kmh is None
        assert any("invalid candidate estimates" in line for line in decision.log)

    def test_invalid_sensor_is_flagged_and_req1_skipped(self, make_controller):
        d = make_controller().process({**DAY, "temperature_c": 999.0})
        assert d.degraded and d.speed_kmh is None
        assert d.weather_triggered is None and d.input_violations

    def test_manual_override_publishes_operator_value(self, make_controller):
        c = make_controller(nn=55)
        assert c.set_manual_override(30) == 30
        d = c.process(NIGHT, origin="manual_test")
        assert (d.speed_kmh, d.automatic_speed_kmh, d.decided_by, d.override_active, d.origin) == \
               (30, 50, "MANUAL_OVERRIDE", True, "manual_test")

    @pytest.mark.parametrize("requested,effective", [(5, 20), (120, 80), (45, 45)])
    def test_override_is_clamped_by_output_guardrail(self, make_controller, requested, effective):
        assert make_controller().set_manual_override(requested) == effective

    def test_override_can_be_cleared(self, make_controller):
        c = make_controller()
        c.set_manual_override(30)
        c.set_manual_override(None)
        assert c.process(DAY).speed_kmh == 80 and c._state.get_override() is None

    def test_listeners_receive_decisions_and_failures_are_isolated(self, make_controller):
        c, seen = make_controller(), []
        c.add_listener(lambda d: 1 / 0)
        c.add_listener(seen.append)
        d = c.process(DAY)
        assert seen == [d]

    def test_decision_string_includes_both_status_flags(self, make_controller):
        controller = make_controller()
        automatic = controller.process(DAY)
        assert "[OVERRIDE]" not in str(automatic)
        controller.set_manual_override(35)
        manual = controller.process({**DAY, "temperature_c": 999}, origin="manual_test")
        rendered = str(manual)
        assert "[OVERRIDE]" in rendered and "[OPERATOR NEEDED]" in rendered

    # Invalid AQI types and non-finite numbers must degrade instead of appearing safe.
    @pytest.mark.parametrize("aqi", [
        pytest.param(True, id="boolean"),
        pytest.param("60", id="numeric-string"),
        pytest.param(float("nan"), id="nan"),
    ])
    def test_invalid_aqi_is_reported_and_asks_the_operator(self, make_controller, aqi):
        decision = make_controller(aqi=aqi).process(DAY)
        assert decision.degraded and decision.speed_kmh is None
