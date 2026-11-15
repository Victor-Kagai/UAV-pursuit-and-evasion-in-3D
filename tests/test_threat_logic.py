import numpy as np
import pytest

from threat_logic import pure_pursuit_step


def test_step_moves_toward_target_in_3d():
    new_p, dist, captured = pure_pursuit_step([0, 0, 0], [0, 0, 100], 10.0, 1.0, 1.0)
    assert np.allclose(new_p, [0, 0, 10])
    assert np.isclose(dist, 100.0)
    assert not captured


def test_capture_within_radius():
    new_p, _, captured = pure_pursuit_step([0, 0, 0], [1, 1, 1], 10.0, 1.0, 5.0)
    assert captured
    assert np.allclose(new_p, [1, 1, 1])


def test_does_not_overshoot_capture_radius():
    new_p, _, captured = pure_pursuit_step([0, 0, 0], [0, 0, 10], 100.0, 1.0, 2.0)
    assert not captured
    assert np.allclose(new_p, [0, 0, 8])


def test_rejects_2d_input():
    with pytest.raises(ValueError):
        pure_pursuit_step([0, 0], [1, 1], 1.0, 1.0, 0.1)
