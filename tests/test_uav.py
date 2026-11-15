import numpy as np

from uav import UAV


def test_escape_vector_has_vertical_component():
    uav = UAV(0, 0, 0, 10, 0, 0)
    escape = uav.compute_escape_vector([(0, 0, 100)])
    assert np.allclose(escape, [0, 0, -1])


def test_escape_vector_is_unit_or_zero():
    uav = UAV(0, 0, 0, 0, 0, 0)
    assert np.isclose(np.linalg.norm(uav.compute_escape_vector([(10, 5, 3), (-4, 2, 8)])), 1.0)
    assert np.allclose(uav.compute_escape_vector([]), 0)


def test_evasion_changes_vz_and_records_3d_history():
    uav = UAV(0, 0, 0, 10, 0, 0, evasion_strength=5.0)
    uav.update_with_evasion(1.0, [(0, 0, 100)])
    assert uav.vz < 0
    assert len(uav.pos_history) == 1
    assert len(uav.pos_history[0]) == 3


def test_max_speed_clamped_in_3d():
    uav = UAV(0, 0, 0, 60, 0, 80, evasion_strength=50.0, max_speed=100.0)
    uav.update_with_evasion(1.0, [(0, 0, 100)])
    assert np.linalg.norm(uav.get_velocity()) <= 100.0 + 1e-9
