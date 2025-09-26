import numpy as np
import matplotlib.pyplot as plt

# First define the Vehicle class
class Vehicle:
    
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
    
    def update_position(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def get_position(self):
        return np.array([self.x, self.y])
    
    def get_velocity(self):
        return np.array([self.vx, self.vy])
class UAV(Vehicle):
    """UAV class with evasive maneuvering logic."""
    def __init__(self, x, y, vx, vy, evasion_strength=5.0, max_speed=100.0):
        super().__init__(x, y, vx, vy)
        self.evasion_strength = evasion_strength
        self.max_speed = max_speed
        self.pos_history = []
    
    def compute_escape_vector(self, threat_positions):
        
        escape = np.array([0.0, 0.0])
        uav_pos = np.array([self.x, self.y])
        
        for threat_pos in threat_positions:
            diff = uav_pos - np.array(threat_pos)
            dist = np.linalg.norm(diff)
            if dist == 0:
                continue
            escape += diff / (dist**2)  
        
        if np.linalg.norm(escape) > 0:
            escape = escape / np.linalg.norm(escape)  # Normalize
        return escape
    
    def update_with_evasion(self, dt, threat_positions):
        """Update position with evasion maneuvering."""
        # Calculate evasion vector
        escape_vector = self.compute_escape_vector(threat_positions)
        
        # Apply evasion acceleration
        evasion_accel = escape_vector * self.evasion_strength
        self.vx += evasion_accel[0] * dt
        self.vy += evasion_accel[1] * dt
        
        # Limit maximum speed
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            self.vx = self.vx / speed * self.max_speed
            self.vy = self.vy / speed * self.max_speed
        
        # Update position
        self.update_position(dt)
        self.pos_history.append((self.x, self.y))

if __name__ == "__main__":
    uav = UAV(100, 200, 10, 5)
    print("UAV created successfully!")