"""Tests for thread-safe controller state."""

from controller_state import ThreadSafeControllerState


def test_state_starts_without_an_override():
    assert ThreadSafeControllerState().get_override() is None


def test_state_starts_without_an_operator_speed():
    assert ThreadSafeControllerState().get_operator_speed() is None


def test_state_stores_and_clears_an_override():
    state = ThreadSafeControllerState()
    state.set_override(55)
    assert state.get_override() == 55
    state.set_override(None)
    assert state.get_override() is None


def test_state_stores_and_clears_an_operator_speed():
    state = ThreadSafeControllerState()
    state.set_operator_speed(45)
    assert state.get_operator_speed() == 45
    state.set_operator_speed(None)
    assert state.get_operator_speed() is None
