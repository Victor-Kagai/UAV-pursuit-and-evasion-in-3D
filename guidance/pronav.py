import numpy as np
from vehicle import Vehicle


def proportional_navigation(pursuer: Vehicle, target: Vehicle, nav_time: float,
                            previous_LOS: np.ndarray = None, dt: float = 0.1,
                            navigation_constant: float = 3.0, maintain_speed: bool = True):
    """
    3D proportional navigation: a = N * Vc * (LOS_rate x LOS_unit).

    Parameters:
        pursuer (Vehicle): The pursuing vehicle
        target (Vehicle): The target vehicle
        nav_time (float): Lookahead time used to predict the target position
        previous_LOS (np.ndarray): LOS unit vector from the previous call
        dt (float): Time step
        navigation_constant (float): PN gain (typically 3-5)
        maintain_speed (bool): Keep the pursuer speed constant

    Returns:
        new_velocity (np.ndarray): New 3D velocity of the pursuer
        acceleration_magnitude (float): Magnitude of the commanded acceleration
        LOS_rate (np.ndarray): 3D LOS angular rate vector (rad/s)
        LOS_unit (np.ndarray): Normalized LOS vector
    """
    target_pos = np.asarray(target.get_position(), dtype=float)
    target_vel = np.asarray(target.get_velocity(), dtype=float)
    pursuer_pos = np.asarray(pursuer.get_position(), dtype=float)
    pursuer_vel = np.asarray(pursuer.get_velocity(), dtype=float)

    speed = np.linalg.norm(pursuer_vel)
    if speed == 0:
        raise ValueError("Pursuer has zero velocity; heading undefined.")

    LOS = (target_pos + target_vel * nav_time) - pursuer_pos
    LOS_norm = np.linalg.norm(LOS)
    if LOS_norm < 1e-10:
        raise ValueError("Pursuer and target are too close; LOS undefined.")
    LOS_unit = LOS / LOS_norm

    LOS_rate = np.zeros(3)
    if previous_LOS is not None:
        prev = np.asarray(previous_LOS, dtype=float)
        prev = prev / np.linalg.norm(prev)
        axis = np.cross(prev, LOS_unit)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            angle = np.arctan2(axis_norm, np.dot(prev, LOS_unit))
            LOS_rate = (axis / axis_norm) * angle / dt

    closing_velocity = -np.dot(target_vel - pursuer_vel, LOS_unit)

    acceleration = navigation_constant * closing_velocity * np.cross(LOS_rate, LOS_unit)
    acceleration_magnitude = float(np.linalg.norm(acceleration))

    new_velocity = pursuer_vel + acceleration * dt

    if maintain_speed:
        new_norm = np.linalg.norm(new_velocity)
        if new_norm > 0:
            new_velocity = new_velocity / new_norm * speed

    return new_velocity, acceleration_magnitude, LOS_rate, LOS_unit
