import numpy as np

from vehicle import Vehicle


def test_heading_and_speed_from_velocity():
    v = Vehicle(0, 0, 0, 0, 10, 0)
    assert np.isclose(v.psi, 90.0)
    assert np.isclose(v.theta, 0.0)
    assert np.isclose(v.V, 10.0)


def test_elevation_from_velocity():
    v = Vehicle(0, 0, 0, 0, 0, 5)
    assert np.isclose(v.theta, 90.0)


def test_setting_heading_updates_velocity():
    v = Vehicle(0, 0, 0, 10, 0, 0)
    v.psi = 90.0
    assert np.allclose(v.get_velocity(), [0, 10, 0], atol=1e-9)


def test_update_position_3d():
    v = Vehicle(0, 0, 0, 1, 2, 3)
    v.update_position(2.0)
    assert np.allclose(v.get_position(), [2, 4, 6])
