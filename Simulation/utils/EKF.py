import numpy as np
from numpy import sin, cos
import utils
import config

class EKF:
    def __init__(self, dt=0.01):
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
        # x, y, z: position
        # vx, vy, vz: velocity
        # qw, qx, qy, qz: quaternion attitude
        # bax, bay, baz: accelerometer bias
        # bgx, bgy, bgz: gyroscope bias
        
        self.dt = dt
        self.state_dim = 16
        self.g = 9.81  # gravity constant
        
        # Initial state
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initial quaternion is [1, 0, 0, 0]
        
        # Initial covariance matrix
        self.P = np.eye(self.state_dim)
        self.P[0:3, 0:3] *= 10.0  # Position uncertainty
        self.P[3:6, 3:6] *= 1.0   # Velocity uncertainty
        self.P[6:10, 6:10] *= 0.1  # Attitude uncertainty
        self.P[10:13, 10:13] *= 0.01  # Acc bias uncertainty
        self.P[13:16, 13:16] *= 0.01  # Gyro bias uncertainty
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim)
        self.Q[0:3, 0:3] *= 0.01   # Position process noise
        self.Q[3:6, 3:6] *= 0.1    # Velocity process noise
        self.Q[6:10, 6:10] *= 0.01  # Attitude process noise
        self.Q[10:13, 10:13] *= 0.001  # Acc bias process noise
        self.Q[13:16, 13:16] *= 0.001  # Gyro bias process noise
        
        # Measurement noise covariance
        self.R_imu = np.eye(6)
        self.R_imu[0:3, 0:3] *= 0.1  # Accelerometer measurement noise
        self.R_imu[3:6, 3:6] *= 0.01  # Gyroscope measurement noise
        
        self.R_gps = np.eye(6)
        self.R_gps[0:3, 0:3] *= 2.25  # GPS position noise (1.5^2)
        self.R_gps[3:6, 3:6] *= 0.01  # GPS velocity noise
        
        self.R_alt = np.array([[0.04]])  # Altitude sensor noise (0.2^2)
        self.R_mag = np.eye(3) * 0.0025  # Magnetometer noise (0.05^2)
        
        # Gravity vector in inertial frame
        if (config.orient == "NED"):
            self.g_vec = np.array([0, 0, self.g])
        else:  # ENU
            self.g_vec = np.array([0, 0, -self.g])
    
    def predict(self, acc_m, gyro_m):
        # Extract current state
        p = self.x[0:3]   # Position
        v = self.x[3:6]   # Velocity
        q = self.x[6:10]  # Quaternion [qw, qx, qy, qz]
        ba = self.x[10:13]  # Accelerometer bias
        bg = self.x[13:16]  # Gyroscope bias
        
        # Correct for bias
        acc_corrected = acc_m - ba
        gyro_corrected = gyro_m - bg
        
        # Get rotation matrix from body to inertial frame
        R = utils.quat2Dcm(q)
        
        # Compute acceleration in inertial frame (removing gravity)
        acc_inertial = R @ acc_corrected + self.g_vec
        
        # State prediction
        p_new = p + v * self.dt + 0.5 * acc_inertial * self.dt**2
        v_new = v + acc_inertial * self.dt
        
        # Quaternion update using angular velocity
        q_dot = 0.5 * utils.quatMultiply(q, np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]]))
        q_new = q + q_dot * self.dt
        q_new = q_new / np.linalg.norm(q_new)  # Normalize quaternion
        
        # Bias random walk model
        ba_new = ba  # Assume biases stay constant in prediction step
        bg_new = bg
        
        # Update state
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new
        self.x[10:13] = ba_new
        self.x[13:16] = bg_new
        
        # Compute Jacobian of state transition function
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt  # dpos/dvel = dt*I
        
        # Acceleration Jacobian wrt quaternion is complex - would involve quaternion derivatives
        # This is simplified for computational efficiency
        F[3:6, 6:10] = np.zeros((3, 4))  # Simplified
        
        # Acceleration affected by accelerometer bias
        F[3:6, 10:13] = -R * self.dt
        
        # Quaternion affected by gyroscope bias
        # Simplified for computational efficiency
        F[6:10, 13:16] = np.zeros((4, 3))  # Simplified
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_imu(self, acc_m, gyro_m):
        # IMU measurement: we expect to see gravity in accelerometer when not moving
        # and we expect gyroscope to match our state's angular velocity
        
        # Extract attitude and biases from state
        q = self.x[6:10]
        ba = self.x[10:13]
        bg = self.x[13:16]
        
        # Measurement model: expected accelerometer and gyroscope readings
        R = utils.quat2Dcm(q)
        gravity_body = R.T @ self.g_vec  # Gravity vector in body frame
        
        h_acc = -gravity_body + ba  # Expected accelerometer measurement in body frame
        h_gyro = np.zeros(3) + bg  # Expected gyroscope measurement when not rotating
        
        h = np.hstack((h_acc, h_gyro))
        z = np.hstack((acc_m, gyro_m))
        
        # Measurement residual
        y = z - h
        
        # Measurement Jacobian (H)
        H = np.zeros((6, self.state_dim))
        
        # Sensitivity of accelerometer to attitude and bias
        # Simplified: full implementation would involve quaternion derivatives
        H[0:3, 6:10] = np.zeros((3, 4))  # Simplified
        H[0:3, 10:13] = np.eye(3)  # Accelerometer affected by acc bias
        
        # Sensitivity of gyroscope to bias
        H[3:6, 13:16] = np.eye(3)  # Gyroscope affected by gyro bias
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_imu @ K.T  # Joseph form for numerical stability
    
    def update_gps(self, pos_m, vel_m):
        if pos_m is None or vel_m is None:
            return
        
        # GPS provides direct measurements of position and velocity
        z = np.hstack((pos_m, vel_m))
        h = np.hstack((self.x[0:3], self.x[3:6]))
        
        # Measurement residual
        y = z - h
        
        # Measurement Jacobian
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 3:6] = np.eye(3)  # Velocity
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_gps @ K.T
    
    def update_altitude(self, alt_m):
        if alt_m is None:
            return
        
        # Extract z position from state
        if (config.orient == "NED"):
            h = -self.x[2]  # In NED, altitude is -z
        else:
            h = self.x[2]   # In ENU, altitude is z
        
        # Measurement residual
        y = alt_m - h
        
        # Measurement Jacobian
        H = np.zeros((1, self.state_dim))
        if (config.orient == "NED"):
            H[0, 2] = -1  # Sensitivity to z position
        else:
            H[0, 2] = 1   # Sensitivity to z position
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_alt
        K = self.P @ H.T / S  # Simplified inversion for scalar
        
        # Update state
        self.x = self.x + K.flatten() * y
        
        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
    
    def update_magnetometer(self, mag_m):
        if mag_m is None:
            return
        
        # Extract quaternion from state
        q = self.x[6:10]
        
        # Expected magnetic field measurement in body frame
        mag_reference = np.array([1.0, 0.0, 0.0])  # Magnetic field in inertial frame (North)
        R = utils.quat2Dcm(q)
        h = R.T @ mag_reference  # Expected measurement in body frame
        
        # Measurement residual
        y = mag_m - h
        
        # Measurement Jacobian
        # Simplified - full implementation would involve quaternion derivatives
        H = np.zeros((3, self.state_dim))
        H[0:3, 6:10] = np.zeros((3, 4))  # Sensitivity to attitude - simplified
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_mag @ K.T
    
    def get_state(self):
        # Return position, velocity, quaternion, and euler angles
        pos = self.x[0:3]
        vel = self.x[3:6]
        quat = self.x[6:10]
        
        # Convert quaternion to euler angles
        euler = utils.quatToYPR_ZYX(quat)[::-1]  # [roll, pitch, yaw]
        
        return pos, vel, quat, euler