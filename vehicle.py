import numpy as np

class Vehicle:
    """Base class for both UAV and Threat - 3D VERSION.Been working on this to ensure the code is realistic."""
    #Learnt a bit on encapsulation for automatic consistency maintenance so I can simulate real world engagements!
    def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0,):
        self._x=x
        self._y=y
        self._z=z
        self._vx=vx
        self._vy=vy
        self._vz=vz
#List for storing derived properties .Sets _psi, _V correctly
        self._update_heading_and_speed()

    def _update_heading_and_speed(self):
        """Update heading angles and speed from velocity components"""
        self._V = float(np.sqrt(self._vx**2 + self._vy**2 + self._vz**2))#Total speed magnitude of the vehicle
        
        if self._V > 0.0:
            # Horizontal heading (azimuth) - in degrees.
            """I settled for arctan2 given it solved an issue that was disturbing
            previously whare arctan by itself failed when vx=0, missing the right quadrant at timees,
            and also the undefined state for (0,0,0)"""
            self._psi = float(np.rad2deg(np.arctan2(self._vy, self._vx)))
            horiz = float(np.sqrt(self._vx**2 + self._vy**2))
            # Vertical heading (elevation) - in degrees  
            self._theta = float(np.rad2deg(np.arcsin(self._vz / self._V)))
        else:
            """I added bits in the code so that I can tone down some of the errors that may be encountered
            in real world scenarios basically in this part normalized psi to [-180, 180)"""
            self._psi = 0.0 #Compass heading
            self._theta = 0.0 #Elevation angle
        
    def _update_velocity_from_heading(self):
        """Update velocity components from heading angles and speed"""
        psi_rad = np.deg2rad(self._psi)
        theta_rad = np.deg2rad(self._theta)
        
        self._vx = self._V * np.cos(theta_rad) * np.cos(psi_rad) #This one will be handling horizontal projection
        self._vy = self._V * np.cos(theta_rad) * np.sin(psi_rad) #For vertical projection
        self._vz = self._V * np.sin(theta_rad)

    # Position properties.Ensures automatic consistency maintenance.
    @property
    def x(self): return self._x
    @x.setter
    def x(self, value): self._x = value
    
    @property
    def y(self): return self._y
    @y.setter
    def y(self, value): self._y = value
    
    @property
    def z(self): return self._z
    @z.setter
    def z(self, value): self._z = value

    # Velocity properties
    @property
    def vx(self): return self._vx
    @vx.setter
    def vx(self, value):
        self._vx = value
        self._update_heading_and_speed()
    
    @property
    def vy(self): return self._vy
    @vy.setter
    def vy(self, value):
        self._vy = value
        self._update_heading_and_speed()
    
    @property
    def vz(self): return self._vz
    @vz.setter
    def vz(self, value):
        self._vz = value
        self._update_heading_and_speed()

    # Heading and speed properties
    @property
    def psi(self): return self._psi
    @psi.setter
    def psi(self, value):
        self._psi = value
        self._update_velocity_from_heading()
    
    @property
    def theta(self): return self._theta
    @theta.setter
    def theta(self, value):
        self._theta = value
        self._update_velocity_from_heading()

    @property
    def V(self): return self._V
    @V.setter
    def V(self, value):
        self._V = value
        self._update_velocity_from_heading()

    #Simple euler integration to update position by just multiplying current velocity with given time step
    def update_position(self, dt):
        """Integrate position over time step dt"""
        self._x += self._vx * dt
        self._y += self._vy * dt
        self._z += self._vz * dt
    
    def get_position(self):
        """Return position as numpy array"""
        return np.array([self.x, self.y, self.z])
    
    def get_velocity(self):
        """Return velocity as numpy array"""
        return np.array([self.vx, self.vy, self.vz])

