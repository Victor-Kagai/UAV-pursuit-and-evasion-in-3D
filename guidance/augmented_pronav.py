import numpy as np

def augmented_proportional_navigation(p_pos, p_vel, t_pos, t_vel, dt, 
                                      navigation_constant=3.0, 
                                      target_accel=None, 
                                      maintain_speed=True):
    """
    Augmented 3D Proportional Navigation (APN).
    
    a_cmd = N * Vc * (Omega x LOS_unit) + (N/2) * a_t_perp
    
    Parameters:
        p_pos (array-like): Pursuer 3D position [x, y, z]
        p_vel (array-like): Pursuer 3D velocity [vx, vy, vz]
        t_pos (array-like): Target 3D position [x, y, z]
        t_vel (array-like): Target 3D velocity [vx, vy, vz]
        dt (float): Time step
        navigation_constant (float): Fixed PN gain (N)
        target_accel (array-like, optional): Target 3D acceleration. If None, reduces to plain PN.
        maintain_speed (bool): If True, scales the new velocity to match the original speed.
        
    Returns:
        new_p_vel (np.ndarray): The pursuer's commanded velocity for the next step
        a_cmd (np.ndarray): The commanded acceleration vector
    """
    p_pos = np.asarray(p_pos, dtype=float)
    p_vel = np.asarray(p_vel, dtype=float)
    t_pos = np.asarray(t_pos, dtype=float)
    t_vel = np.asarray(t_vel, dtype=float)

    speed = np.linalg.norm(p_vel)
    if speed < 1e-9:
        raise ValueError("Pursuer velocity is near zero; cannot compute guidance.")

    # 1. Compute LOS and Relative Kinematics
    los = t_pos - p_pos
    los_sq = np.dot(los, los)
    los_norm = np.sqrt(los_sq)

    if los_norm < 1e-9:
        raise ValueError("Pursuer and target are co-located; LOS undefined.")

    los_unit = los / los_norm
    v_rel = t_vel - p_vel
    
    closing_velocity = -np.dot(v_rel, los_unit)

    # Analytical LOS rate (Omega) = (r x v_rel) / |r|^2
    omega = np.cross(los, v_rel) / los_sq

    # 2. Base Proportional Navigation Term
    pn_accel = navigation_constant * closing_velocity * np.cross(omega, los_unit)

    # 3. Augmented Feed-Forward Term
    if target_accel is not None:
        a_t = np.asarray(target_accel, dtype=float)
        # Extract the component of target acceleration perpendicular to the LOS
        a_t_perp = a_t - np.dot(a_t, los_unit) * los_unit
        aug_accel = (navigation_constant / 2.0) * a_t_perp
    else:
        aug_accel = np.zeros(3)

    # 4. Total Commanded Acceleration
    a_cmd = pn_accel + aug_accel

    # 5. Integrate to get new velocity
    new_p_vel = p_vel + a_cmd * dt

    # Enforce speed constraint if requested
    if maintain_speed:
        new_speed = np.linalg.norm(new_p_vel)
        if new_speed > 1e-9:
            new_p_vel = (new_p_vel / new_speed) * speed

    return new_p_vel, a_cmd