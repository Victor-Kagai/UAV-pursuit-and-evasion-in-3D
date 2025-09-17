import numpy as np

def compute_escape_vector(uav_pos, threat_positions):
    escape = np.array([0.0, 0.0, 0.0])
    for threat in threat_positions:
        diff = np.array(uav_pos) - np.array(threat)
        dist = np.linalg.norm(diff)
        if dist == 0:
            continue
        escape += diff / (dist**2)  # Inverse-square repulsion
    escape = escape / np.linalg.norm(escape)  # Normalize to unit vector
    return tuple(escape)
