import numpy as np
from uav import Vehicle  # Import Vehicle class
"""For convenience and reliability across environments;
 I chose to execute the module via python -m guidance.pronav rather than relying on direct imports."""
def proportional_navigation(pursuer: Vehicle, target: Vehicle, nav_time: float,
                            previous_LOS: np.ndarray = None, dt: float = 0.1, 
                            navigation_constant: float = 3.0, maintain_speed: bool = True):
    """
    Complete Proportional Navigation guidance with constant speed maintenance.
    
    Parameters:
        pursuer (Vehicle): The pursuing UAV
        target (Vehicle): The target UAV or object  
        nav_time (float): The navigation time step or lookahead time
        previous_LOS (np.ndarray): LOS vector from previous call (for LOS rate calculation)
        dt (float): Time step for rate calculations
        navigation_constant (float): PN guidance constant (typically 3-5)
        maintain_speed (bool): Whether to maintain constant speed (True for missiles)
    
    Returns:
        new_velocity: The new velocity vector for the pursuer
        acceleration_magnitude: The computed acceleration magnitude
        LOS_rate: Line of sight rate
        LOS_unit: Normalized LOS vector
    """
    # Convert to numpy arrays.I’ve been exploring NumPy arrays to enhance the efficiency and structure of this project incase of any suggestions on improving ,feel free to bring it forth.
    target_pos = np.array(target.get_position(), dtype=float)
    target_vel = np.array(target.get_velocity(), dtype=float)
    pursuer_pos = np.array(pursuer.get_position(), dtype=float)
    pursuer_vel = np.array(pursuer.get_velocity(), dtype=float)
    
    # Get current speed for maintenance
    current_speed = np.linalg.norm(pursuer_vel)

    # Line of sight (LOS) vector after nav_time.Trying now to predict future target position.
    LOS = (target_pos + target_vel * nav_time) - pursuer_pos

    # Normalized LOS vector for guidance commands
    LOS_norm = np.linalg.norm(LOS)
    if LOS_norm < 1e-10: #The error handling prevents division by zero when objects are too close
        raise ValueError("Pursuer and target are too close; LOS undefined.") 
    #On top of this the next couple of lines of code will be on calculating how misaligned the pursuer is with the target direction using vector mathematics
    LOS_unit = LOS / LOS_norm
    
    # Compute LOS angle relative to pursuer velocity.
    dot_product = np.dot(pursuer_vel, LOS)
    pursuer_speed = np.linalg.norm(pursuer_vel)
    if pursuer_speed == 0:
        raise ValueError("Pursuer has zero velocity; heading undefined.")
    
    cos_angle = np.clip(dot_product / (pursuer_speed * LOS_norm), -1.0, 1.0)
    angular_error = np.arccos(cos_angle)

    """
    Calculating LOS rate ;
               How fast the LOS rate is rotating
               Including error handling for edge cases"""
    LOS_rate = 0.0
    if previous_LOS is not None:
        try:
            prev_dot = np.clip(np.dot(previous_LOS, LOS_unit), -1.0, 1.0)
            LOS_angle_change = np.arccos(prev_dot)
            
            cross_prod = np.cross(previous_LOS, LOS_unit)
            if np.isscalar(cross_prod):  # 2D case
                rotation_dir = np.sign(cross_prod)
            else:  # 3D case
                rotation_dir = np.sign(cross_prod[2]) if len(cross_prod) > 2 else 1.0
            
            LOS_rate = rotation_dir * LOS_angle_change / dt
        except:
            LOS_rate = 0.0
    
    # Calculate closing velocity
    relative_vel = target_vel - pursuer_vel
    closing_velocity = -np.dot(relative_vel, LOS_unit)
    
    # Proportional Navigation acceleration command.This is the heart of proportional navigation - applies the classic PN formula and ensures speed maintenance.
    acceleration_magnitude = navigation_constant * closing_velocity * LOS_rate
    
    # Convert acceleration to velocity adjustment
    adjustment = acceleration_magnitude * LOS_unit * dt
    
    # Apply adjustment to velocity
    new_velocity = pursuer_vel + adjustment
    
    # Maintain constant speed 
    if maintain_speed and current_speed > 0:
        new_velocity = (new_velocity / np.linalg.norm(new_velocity)) * current_speed
    
    return new_velocity, acceleration_magnitude, LOS_rate, LOS_unit


"""
             **EMPHASIS**
For convenience and reliability across environments;
 I chose to execute the module via python -m guidance.pronav rather than relying on direct imports."""