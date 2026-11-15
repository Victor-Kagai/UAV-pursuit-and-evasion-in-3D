import numpy as np
import pytest

from vehicle import Vehicle
from guidance.pronav import proportional_navigation


def test_no_previous_los_gives_zero_rate_and_unchanged_velocity():
    pursuer = Vehicle(0, 0, 0, 100, 0, 0)
    target = Vehicle(1000, 500, 300, 0, 0, 0)
    v, a, rate, los = proportional_navigation(pursuer, target, 0.0)
    assert np.allclose(v, [100, 0, 0])
    assert a == 0.0
    assert np.allclose(rate, 0)
    assert np.isclose(np.linalg.norm(los), 1.0)


def test_speed_is_maintained():
    pursuer = Vehicle(0, 0, 0, 100, 0, 0)
    target = Vehicle(1000, 500, 300, -10, 20, 5)
    _, _, _, los0 = proportional_navigation(pursuer, target, 0.0)
    target.update_position(0.5)
    v, _, _, _ = proportional_navigation(pursuer, target, 0.0, previous_LOS=los0, dt=0.5)
    assert np.isclose(np.linalg.norm(v), 100.0)


def test_acceleration_is_perpendicular_to_los_and_turns_toward_target_motion():
    pursuer = Vehicle(0, 0, 0, 100, 0, 0)
    target = Vehicle(1000, 0, 0, 0, 0, 50)
    _, _, _, los0 = proportional_navigation(pursuer, target, 0.0)
    target.update_position(1.0)
    v, a, rate, los = proportional_navigation(
        pursuer, target, 0.0, previous_LOS=los0, dt=1.0, maintain_speed=False)
    assert a > 0
    assert abs(np.dot(rate, los)) < 1e-9
    assert (v - np.array([100, 0, 0]))[2] > 0


def test_3d_intercept():
    dt = 0.02
    pursuer = Vehicle(0, 0, 0, 300, 0, 0)
    target = Vehicle(3000, 1500, 800, -40, 30, 10)
    prev = None
    min_dist = np.inf
    for _ in range(3000):
        v, _, _, prev = proportional_navigation(
            pursuer, target, 0.0, previous_LOS=prev, dt=dt, navigation_constant=4.0)
        pursuer.vx, pursuer.vy, pursuer.vz = v
        pursuer.update_position(dt)
        target.update_position(dt)
        dist = np.linalg.norm(target.get_position() - pursuer.get_position())
        min_dist = min(min_dist, dist)
        if dist < 5.0:
            break
    assert min_dist < 5.0


def test_zero_velocity_raises():
    with pytest.raises(ValueError):
        proportional_navigation(Vehicle(0, 0, 0, 0, 0, 0), Vehicle(10, 0, 0), 0.0)


def test_coincident_raises():
    with pytest.raises(ValueError):
        proportional_navigation(Vehicle(0, 0, 0, 1, 0, 0), Vehicle(0, 0, 0), 0.0)
