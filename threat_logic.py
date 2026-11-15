import numpy as np


def pure_pursuit_step(p_pos, t_pos, speed, dt, capture_radius):
    """
    One step of 3D pure pursuit.
    p_pos, t_pos: array-like [x, y, z]
    returns (new_p_pos, distance, captured: bool)
    """
    p_pos = np.asarray(p_pos, dtype=float)
    t_pos = np.asarray(t_pos, dtype=float)
    if p_pos.shape != (3,) or t_pos.shape != (3,):
        raise ValueError("p_pos and t_pos must be 3D vectors [x, y, z].")

    d = t_pos - p_pos
    dist = np.linalg.norm(d)

    if dist <= capture_radius:
        return t_pos.copy(), dist, True

    u = d / dist if dist >= 1e-9 else np.zeros(3)

    step_len = min(speed * dt, max(0.0, dist - capture_radius))
    return p_pos + u * step_len, dist, False
