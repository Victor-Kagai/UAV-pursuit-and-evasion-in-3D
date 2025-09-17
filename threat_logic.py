import numpy as np

def pure_pursuit_step(p_pos, t_pos, speed, dt, capture_radius):
    """
    One step of pure pursuit.
    p_pos, t_pos: np.array([x,y])
    returns (new_p_pos, distance, captured: bool)
    """
    d = t_pos - p_pos                     
    dist = np.linalg.norm(d)          

   
    if dist <= capture_radius:
        return t_pos.copy(), dist, True

   
    if dist < 1e-9:
        u = np.zeros_like(d)
    else:
        u = d / dist  

    
    step_len = min(speed*dt, max(0.0, dist - capture_radius))
    new_p = p_pos + u * step_len
    return new_p, dist, False
