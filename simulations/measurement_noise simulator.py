def run_simulation(pursuer: Vehicle, target: Vehicle, steps: int = 500, dt: float = 0.1):
    estimator = TargetEstimator(alpha=0.6, beta=0.15, gamma=0.02, dt=dt)
    
    for step in range(steps):
        # 1. Capture true target position
        true_target_pos = np.asarray(target.get_position())
        
        # 2. Inject Gaussian noise (e.g., standard deviation of 5 meters)
        noise = np.random.normal(loc=0.0, scale=5.0, size=3)
        measured_pos = true_target_pos + noise
        
        # 3. Create a temporary observation object for the guidance law
        class NoisyTarget(Vehicle):
            def get_position(self): return measured_pos
            def get_velocity(self): return np.zeros(3) # Hidden from pursuer; estimator figures this out
        
        noisy_target = NoisyTarget()

        # 4. Execute Augmented Proportional Navigation
        new_vel, acc_mag, los_rate, los_unit, info = proportional_navigation(
            pursuer=pursuer,
            target=noisy_target,
            nav_time=0.0, 
            estimator=estimator,
            dt=dt,
            max_accel=100.0
        )
        
        pursuer.set_velocity(new_vel)
        
        # Advance physics step for both vehicles here...