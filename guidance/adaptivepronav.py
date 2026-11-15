import numpy as np
from typing import Callable, Optional, Tuple, Dict
from vehicle import Vehicle


class TargetEstimator:
    """
    Alpha-Beta-Gamma filter for 3D target state estimation.
    Tracks and smooths position, velocity, and acceleration.
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.01, dt: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.pos_est: Optional[np.ndarray] = None
        self.vel_est = np.zeros(3)
        self.acc_est = np.zeros(3)

    def update(self, measured_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pos_est is None:
            self.pos_est = np.copy(measured_pos)
            return self.pos_est, self.vel_est, self.acc_est

        # Predict next state
        dt_sq_half = 0.5 * self.dt**2
        pos_pred = self.pos_est + self.vel_est * self.dt + self.acc_est * dt_sq_half
        vel_pred = self.vel_est + self.acc_est * self.dt
        acc_pred = self.acc_est

        # Calculate residual (innovation)
        residual = measured_pos - pos_pred

        # Update estimates
        self.pos_est = pos_pred + self.alpha * residual
        self.vel_est = vel_pred + (self.beta / self.dt) * residual
        self.acc_est = acc_pred + (self.gamma / dt_sq_half) * residual

        return self.pos_est, self.vel_est, self.acc_est


def proportional_navigation(pursuer: Vehicle, target: Vehicle, nav_time: float,
                            estimator: Optional[TargetEstimator] = None, dt: float = 0.1,
                            navigation_constant: float = 3.0, maintain_speed: bool = True,
                            gain_schedule: Optional[Callable[[float, float, float, float], float]] = None,
                            max_accel: Optional[float] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Augmented 3D proportional navigation using analytical LOS rate and target estimation.
    a = N * Vc * (LOS_rate x LOS_unit) + (N/2) * a_t_perp
    """
    # 1. State extraction and estimation
    raw_target_pos = np.asarray(target.get_position(), dtype=float)
    pursuer_pos = np.asarray(pursuer.get_position(), dtype=float)
    pursuer_vel = np.asarray(pursuer.get_velocity(), dtype=float)

    if estimator is not None:
        # Filter smooths the noisy raw measurement and derives velocity/acceleration
        t_pos, t_vel, t_acc = estimator.update(raw_target_pos)
    else:
        # Fallback to perfect knowledge without acceleration feed-forward
        t_pos = raw_target_pos
        t_vel = np.asarray(target.get_velocity(), dtype=float)
        t_acc = np.zeros(3)

    speed = np.linalg.norm(pursuer_vel)
    if speed == 0:
        raise ValueError("Pursuer has zero velocity; heading undefined.")

    # 2. LOS and relative kinematics
    r_vec = (t_pos + t_vel * nav_time) - pursuer_pos
    r_sq = np.dot(r_vec, r_vec)
    r_norm = np.sqrt(r_sq)
    
    if r_norm < 1e-10:
        raise ValueError("Pursuer and target are too close; LOS undefined.")
    
    LOS_unit = r_vec / r_norm
    v_rel = t_vel - pursuer_vel
    closing_velocity = -np.dot(v_rel, LOS_unit)

    # Exact analytical LOS rate: Ω = (r × v_rel) / |r|²
    LOS_rate = np.cross(r_vec, v_rel) / r_sq

    # 3. Compute dynamic gain schedule inputs
    if gain_schedule is not None:
        los_rate_mag = float(np.linalg.norm(LOS_rate))
        time_to_go = r_norm / closing_velocity if closing_velocity > 1e-6 else float('inf')
        raw_N = gain_schedule(r_norm, closing_velocity, los_rate_mag, time_to_go)
        N = float(np.clip(raw_N, 2.0, 6.0))
    else:
        N = navigation_constant

    # 4. Augmented PN Guidance Law calculation
    # Base PN term
    pn_accel = N * closing_velocity * np.cross(LOS_rate, LOS_unit)
    
    # Target acceleration feed-forward term (perpendicular to LOS)
    a_t_perp = t_acc - np.dot(t_acc, LOS_unit) * LOS_unit
    aug_accel = (N / 2.0) * a_t_perp
    
    acceleration = pn_accel + aug_accel
    acceleration_magnitude = float(np.linalg.norm(acceleration))

    # 5. Saturate acceleration
    if max_accel is not None and acceleration_magnitude > max_accel:
        acceleration = (acceleration / acceleration_magnitude) * max_accel
        acceleration_magnitude = float(max_accel)

    # 6. Integrate velocity
    new_velocity = pursuer_vel + acceleration * dt
    if maintain_speed:
        new_norm = np.linalg.norm(new_velocity)
        if new_norm > 0:
            new_velocity = new_velocity / new_norm * speed

    info = {"N_used": N, "target_accel_est_mag": float(np.linalg.norm(t_acc))}

    return new_velocity, acceleration_magnitude, LOS_rate, LOS_unit, info