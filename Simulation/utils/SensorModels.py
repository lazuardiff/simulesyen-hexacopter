import numpy as np
from numpy import sin, cos, pi
import config

class IMUSensor:
    def __init__(self):
        # Noise parameters (standard deviation)
        self.acc_noise_std = 0.05  # m/s^2
        self.gyro_noise_std = 0.01  # rad/s
        
        # Bias parameters
        self.acc_bias = np.array([0.1, -0.05, 0.03])  # Initial bias (m/s^2)
        self.gyro_bias = np.array([0.01, -0.01, 0.005])  # Initial bias (rad/s)
        
        # Parameters for random walk bias model
        self.acc_bias_walk = 0.0001  # m/s^2 per √s
        self.gyro_bias_walk = 0.0001  # rad/s per √s
        
        # Sampling frequency
        self.freq = 100  # Hz
        self.dt = 1.0/self.freq
        
    def measure(self, quad, t):
        # Get true values from quadcopter model
        acc_true = quad.acc  # True acceleration in body frame
        omega_true = quad.omega  # True angular velocity in body frame
        
        # Add noise and bias
        acc_noise = np.random.normal(0, self.acc_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)
        
        # Update bias with random walk model
        self.acc_bias += np.random.normal(0, self.acc_bias_walk * np.sqrt(self.dt), 3)
        self.gyro_bias += np.random.normal(0, self.gyro_bias_walk * np.sqrt(self.dt), 3)
        
        # Final sensor readings
        acc_measured = acc_true + self.acc_bias + acc_noise
        gyro_measured = omega_true + self.gyro_bias + gyro_noise
        
        return acc_measured, gyro_measured

class GPSSensor:
    def __init__(self):
        # Noise parameters
        self.pos_noise_std = 1.5  # m
        self.vel_noise_std = 0.1  # m/s
        
        # GPS update rate (typically slower than IMU)
        self.freq = 5  # Hz
        self.dt = 1.0/self.freq
        self.last_update = 0
        
    def measure(self, quad, t):
        # Only provide measurement at specific frequency
        if t - self.last_update < self.dt:
            return None, None
        
        self.last_update = t
        
        # Get true position and velocity in NED frame
        pos_true = quad.pos
        vel_true = quad.vel
        
        # Add noise
        pos_noise = np.random.normal(0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0, self.vel_noise_std, 3)
        
        # Final GPS readings
        pos_measured = pos_true + pos_noise
        vel_measured = vel_true + vel_noise
        
        return pos_measured, vel_measured

class AltitudeSensor:
    def __init__(self, sensor_type="baro"):
        self.sensor_type = sensor_type
        
        if sensor_type == "baro":
            self.noise_std = 0.2  # m
            self.freq = 20  # Hz
        elif sensor_type == "lidar":
            self.noise_std = 0.05  # m
            self.freq = 40  # Hz
            
        self.dt = 1.0/self.freq
        self.last_update = 0
        
    def measure(self, quad, t):
        # Only provide measurement at specific frequency
        if t - self.last_update < self.dt:
            return None
            
        self.last_update = t
        
        # Get true altitude (negative z in NED, positive z in ENU)
        if (config.orient == "NED"):
            alt_true = -quad.pos[2]
        else:
            alt_true = quad.pos[2]
        
        # Add noise
        alt_noise = np.random.normal(0, self.noise_std)
        
        # Final altitude reading
        alt_measured = alt_true + alt_noise
        
        return alt_measured

class MagnetometerSensor:
    def __init__(self):
        # Noise parameters
        self.noise_std = 0.05  # Gauss
        
        # Magnetic field reference (north direction)
        self.mag_reference = np.array([1.0, 0.0, 0.0])
        
        # Hard and soft iron effects
        self.hard_iron = np.array([0.1, -0.05, 0.03])
        self.soft_iron = np.eye(3) + np.random.normal(0, 0.05, (3, 3))
        
        # Sampling frequency
        self.freq = 50  # Hz
        self.dt = 1.0/self.freq
        self.last_update = 0
        
    def measure(self, quad, t):
        # Only provide measurement at specific frequency
        if t - self.last_update < self.dt:
            return None
            
        self.last_update = t
        
        # Get true attitude
        R = quad.dcm  # Rotation matrix from body to inertial frame
        
        # Calculate magnetic field in body frame
        mag_body_true = R.T @ self.mag_reference
        
        # Apply soft and hard iron effects
        mag_body_distorted = self.soft_iron @ mag_body_true + self.hard_iron
        
        # Add noise
        mag_noise = np.random.normal(0, self.noise_std, 3)
        
        # Final magnetometer reading
        mag_measured = mag_body_distorted + mag_noise
        
        return mag_measured