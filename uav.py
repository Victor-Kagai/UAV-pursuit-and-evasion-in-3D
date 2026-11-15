import numpy as np
from vehicle import Vehicle


class UAV(Vehicle):
    """UAV class with 3D evasive maneuvering logic."""
    def __init__(self, x, y, z, vx, vy, vz, evasion_strength=5.0, max_speed=100.0):
        super().__init__(x, y, z, vx, vy, vz)
        self.evasion_strength = evasion_strength
        self.max_speed = max_speed
        self.pos_history = []

    def compute_escape_vector(self, threat_positions):
        escape = np.zeros(3)
        uav_pos = self.get_position()

        for threat_pos in threat_positions:
            diff = uav_pos - np.asarray(threat_pos, dtype=float)
            dist = np.linalg.norm(diff)
            if dist == 0:
                continue
            escape += diff / (dist ** 2)

        norm = np.linalg.norm(escape)
        if norm > 0:
            escape = escape / norm
        return escape

    def update_with_evasion(self, dt, threat_positions):
        """Update position with 3D evasion maneuvering."""
        evasion_accel = self.compute_escape_vector(threat_positions) * self.evasion_strength
        velocity = self.get_velocity() + evasion_accel * dt

        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            velocity = velocity / speed * self.max_speed

        self.vx, self.vy, self.vz = velocity

        self.update_position(dt)
        self.pos_history.append((self.x, self.y, self.z))
