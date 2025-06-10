"""
EKF Common Classes and Functions
===============================

This file contains the base EKF class and common functions shared between
control input and non-control input variants.

Author: EKF Implementation Team
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


class HexacopterDynamicsFromSimulation:
    """
    Model dinamika hexacopter yang menggunakan data dari simulasi
    """

    def __init__(self):
        # Parameters dari simulasi hexacopter
        self.mass = 1.2  # kg
        self.g = 9.81    # m/s²

        # Inertia matrix
        self.inertia = np.array([[0.0123, 0,      0],
                                [0,      0.0123, 0],
                                [0,      0,      0.0224]])  # kg⋅m²

        # Motor parameters dari simulasi
        self.kTh = 1.076e-5  # thrust coefficient (N/(rad/s)²)
        self.kTo = 1.632e-7  # torque coefficient (Nm/(rad/s)²)

        # Hexacopter geometry
        self.L = 0.225  # arm length (m)

        # Gravity vector in NED frame
        self.g_ned = np.array([0, 0, self.g])

    def predict_acceleration_from_motor_thrusts(self, motor_thrusts, attitude_matrix):
        """
        Predict acceleration dari individual motor thrusts

        Args:
            motor_thrusts: Array of 6 motor thrusts [T1, T2, T3, T4, T5, T6] (N)
            attitude_matrix: 3x3 rotation matrix body-to-NED

        Returns:
            acceleration_ned: 3D acceleration in NED frame [m/s²]
        """
        # Total thrust dalam body frame (semua motor thrust ke atas = -Z direction)
        total_thrust = np.sum(motor_thrusts)
        # Up = -Z dalam body frame
        thrust_body = np.array([0, 0, -total_thrust])

        # Transform thrust ke NED frame
        thrust_ned = attitude_matrix @ thrust_body

        # Total acceleration = thrust/mass + gravity
        acceleration_ned = thrust_ned / self.mass + self.g_ned

        return acceleration_ned

    def predict_acceleration_from_thrust_vector(self, thrust_vector_body, attitude_matrix):
        """
        Predict acceleration dari control thrust vector

        Args:
            thrust_vector_body: Control thrust vector [Fx, Fy, Fz] (N) in body frame
            attitude_matrix: 3x3 rotation matrix body-to-NED

        Returns:
            acceleration_ned: 3D acceleration in NED frame [m/s²]
        """
        # Transform thrust vector ke NED frame
        thrust_ned = attitude_matrix @ thrust_vector_body

        # Total acceleration = thrust/mass + gravity
        acceleration_ned = thrust_ned / self.mass + self.g_ned

        return acceleration_ned

    def predict_angular_acceleration_from_torques(self, control_torques):
        """
        Predict angular acceleration dari control torques

        Args:
            control_torques: Control torques [Mx, My, Mz] (N⋅m)

        Returns:
            angular_acceleration: 3D angular acceleration [rad/s²]
        """
        # τ = I⋅α  =>  α = I⁻¹⋅τ
        angular_acceleration = np.linalg.inv(self.inertia) @ control_torques

        return angular_acceleration


class BaseEKF:
    """
    Base Extended Kalman Filter class for attitude and position estimation

    State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bias_ax, bias_ay, bias_az, bias_gx, bias_gy, bias_gz]
    """

    def __init__(self, dt=0.01):
        self.dt = dt

        # Initialize hexacopter dynamics model
        self.dynamics = HexacopterDynamicsFromSimulation()

        # State dimension: 16 (pos:3, vel:3, quat:4, acc_bias:3, gyro_bias:3)
        self.state_dim = 16

        # Initialize state vector
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initialize quaternion as identity [w, x, y, z]

        # Initialize covariance matrix
        # Error state dimension: 15 (pos:3, vel:3, att:3, acc_bias:3, gyro_bias:3)
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 25.0    # Reduced from 100.0
        self.P[3:6, 3:6] *= 4.0     # Reduced from 10.0
        self.P[6:9, 6:9] = np.diag([0.04, 0.04, 1.0])
        self.P[8, 8] *= 4.0         # REDUCED from 10.0 (yaw uncertainty)
        self.P[9:12, 9:12] *= 0.005  # Reduced from 0.01
        self.P[12:15, 12:15] *= 0.005  # Reduced from 0.01

        # Process noise matrix Q (untuk error states)
        self.Q = np.zeros((15, 15))
        # Position process noise
        self.Q[0:3, 0:3] = np.eye(3) * (0.01 * self.dt**2)**2
        # Velocity process noise
        self.Q[3:6, 3:6] = np.eye(3) * (0.1 * self.dt)**2
        # Attitude process noise
        self.Q[6:9, 6:9] = np.diag(
            [0.005 * self.dt, 0.005 * self.dt, 0.003 * self.dt])**2
        # Acceleration bias random walk
        self.Q[9:12, 9:12] = np.eye(3) * (1e-4 * self.dt)**2
        # Gyroscope bias random walk
        gyro_bias_noise = np.array([1e-5, 1e-5, 5e-5])
        self.Q[12:15, 12:15] = np.diag(gyro_bias_noise * self.dt)**2

        # Measurement noise matrices
        self.R_gps_pos = np.eye(3) * 1.0**2   # GPS position noise
        self.R_gps_vel = np.eye(3) * 0.1**2   # GPS velocity noise
        self.R_baro = np.array([[0.5**2]])    # Barometer noise
        # Increased magnetometer noise
        self.R_mag = np.eye(3) * 0.02**2

        # YAW-SPECIFIC TUNING PARAMETERS
        self.yaw_innovation_gate = 8.0        # Larger gate for yaw innovations
        self.mag_declination = 0.0            # Magnetic declination for your location
        self.enable_mag_bias_learning = True  # Enable magnetometer bias learning
        self.mag_bias = np.zeros(3)           # Magnetometer bias estimate
        self.mag_bias_P = np.eye(3) * 0.01    # Magnetometer bias covariance

        # GPS HEADING ASSISTANCE
        self.use_gps_heading = True           # Use GPS velocity for heading when available
        # Minimum GPS speed (m/s) for heading calculation
        self.gps_heading_threshold = 1.0
        self.gps_heading_weight_max = 0.6  # maximum correction weight

        # Constants
        self.g_ned = np.array([0, 0, 9.81])  # Gravity in NED frame

        # Magnetic field reference in NED
        self.mag_ref_ned = np.array([1.0, 0.0, 0.0])

        self.initialized = False

        # Debug info
        self.prediction_mode = "IMU_ONLY"  # Track which prediction mode is being used

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [w,x,y,z] to rotation matrix"""
        # Ensure quaternion is normalized and valid
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            # Return identity rotation if quaternion is invalid
            return np.eye(3)

        q_normalized = q / q_norm
        r = Rotation.from_quat(
            # [x,y,z,w]
            [q_normalized[1], q_normalized[2], q_normalized[3], q_normalized[0]])
        return r.as_matrix()

    def normalize_quaternion(self, q):
        """Normalize quaternion"""
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.array([1, 0, 0, 0])
        return q / norm

    def skew_symmetric(self, v):
        """Skew symmetric matrix"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def initialize_state(self, gps_pos, gps_vel, initial_acc, initial_gyro, initial_mag=None, true_yaw=None):
        """Initialize state dengan GPS dan IMU data - ENHANCED YAW INITIALIZATION"""

        # Initialize position dan velocity dari GPS
        self.x[0:3] = gps_pos  # Position
        self.x[3:6] = gps_vel  # Velocity

        # Initialize attitude dari accelerometer (asumsi static)
        acc_magnitude = np.linalg.norm(initial_acc)

        if acc_magnitude < 1e-6:
            # If accelerometer data is invalid, use default attitude (level)
            print(
                "Warning: Invalid accelerometer data for initialization. Using level attitude.")
            roll = 0.0
            pitch = 0.0
        else:
            acc_norm = initial_acc / acc_magnitude

            # Calculate initial roll dan pitch with bounds checking
            roll = np.arctan2(-acc_norm[1], -acc_norm[2])
            pitch = np.arctan2(acc_norm[0], np.sqrt(
                acc_norm[1]**2 + acc_norm[2]**2))

            # Limit roll and pitch to reasonable values
            roll = np.clip(roll, -np.pi/3, np.pi/3)  # ±60 degrees
            pitch = np.clip(pitch, -np.pi/3, np.pi/3)  # ±60 degrees

        # ENHANCED YAW INITIALIZATION
        yaw = 0.0
        yaw_source = "default"

        if true_yaw is not None:
            yaw = true_yaw
            yaw_source = "ground_truth"
        elif initial_mag is not None and np.linalg.norm(initial_mag) > 0.1:
            # Try to estimate yaw from magnetometer
            try:
                # Create initial DCM from roll/pitch
                R_initial = Rotation.from_euler(
                    'xy', [roll, pitch]).as_matrix()

                # Expected magnetic field in body frame
                mag_ref_corrected = np.array([
                    np.cos(self.mag_declination),
                    np.sin(self.mag_declination),
                    0.0
                ])

                # Project magnetometer to horizontal plane
                mag_horizontal = initial_mag[:2]
                if np.linalg.norm(mag_horizontal) > 0.1:
                    # Calculate yaw from horizontal magnetometer components
                    yaw = np.arctan2(-mag_horizontal[1],
                                     mag_horizontal[0]) + self.mag_declination
                    yaw_source = "magnetometer"
            except:
                pass
        elif np.linalg.norm(gps_vel[:2]) > 1.0:  # GPS velocity initialization
            # If moving fast enough, use GPS velocity for initial heading
            yaw = np.arctan2(gps_vel[1], gps_vel[0])
            yaw_source = "gps_velocity"

        # Convert to quaternion with validation
        try:
            r = Rotation.from_euler('xyz', [roll, pitch, yaw])
            q_scipy = r.as_quat()  # [x,y,z,w]
            self.x[6:10] = np.array(
                [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # [w,x,y,z]
        except:
            # Fallback to identity quaternion if conversion fails
            print("Warning: Quaternion conversion failed. Using identity quaternion.")
            self.x[6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]

        # Ensure quaternion is normalized
        self.x[6:10] = self.normalize_quaternion(self.x[6:10])

        # ENHANCED BIAS INITIALIZATION
        # Initialize biases with better estimates if available
        if np.linalg.norm(initial_gyro) < 0.01:  # Likely static
            # If gyro readings are very small, they might be mostly bias
            self.x[13:16] = initial_gyro.copy()  # Use as initial bias estimate
        else:
            self.x[13:16] = np.zeros(3)  # Gyro bias

        self.x[10:13] = np.zeros(3)  # Accel bias

        self.initialized = True

        print(f"EKF Initialization completed:")
        print(
            f"  Initial attitude: roll={np.rad2deg(roll):.1f}°, pitch={np.rad2deg(pitch):.1f}°, yaw={np.rad2deg(yaw):.1f}° (from {yaw_source})")
        if initial_mag is not None:
            print(f"  Magnetometer norm: {np.linalg.norm(initial_mag):.3f}")
        print(f"  GPS velocity norm: {np.linalg.norm(gps_vel):.3f} m/s")

    def update_gps_position(self, gps_pos):
        """Update dengan GPS position measurement"""
        if not self.initialized or gps_pos is None:
            return

        z = gps_pos
        h = self.x[0:3]  # Predicted position
        y = z - h  # Innovation

        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        S = H @ self.P @ H.T + self.R_gps_pos
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        # Apply corrections to nominal state
        self.x[0:3] += dx[0:3]  # Position
        self.x[3:6] += dx[3:6]  # Velocity

        # Attitude correction
        dtheta = dx[6:9]
        dq = np.array([1, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq = self.normalize_quaternion(dq)

        q = self.x[6:10]
        q_corrected = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
        ])
        self.x[6:10] = self.normalize_quaternion(q_corrected)

        self.x[10:13] += dx[9:12]   # Accel bias
        self.x[13:16] += dx[12:15]  # Gyro bias

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_pos @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_gps_velocity(self, gps_vel):
        """Update dengan GPS velocity measurement - ENHANCED with heading assistance"""
        if not self.initialized or gps_vel is None:
            return

        z = gps_vel
        h = self.x[3:6]  # Predicted velocity
        y = z - h

        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)

        S = H @ self.P @ H.T + self.R_gps_vel
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        # Apply corrections (same pattern as GPS position)
        self.x[0:3] += dx[0:3]
        self.x[3:6] += dx[3:6]

        dtheta = dx[6:9]
        dq = np.array([1, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq = self.normalize_quaternion(dq)

        q = self.x[6:10]
        q_corrected = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
        ])
        self.x[6:10] = self.normalize_quaternion(q_corrected)

        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_vel @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # GPS HEADING ASSISTANCE for YAW CORRECTION
        self.update_gps_heading(gps_vel)

    def update_gps_heading(self, gps_vel):
        """GPS heading assistance for yaw correction"""
        if not self.use_gps_heading:
            return

        gps_speed = np.linalg.norm(gps_vel[:2])
        if gps_speed > self.gps_heading_threshold:
            # Calculate GPS heading
            gps_heading = np.arctan2(gps_vel[1], gps_vel[0])

            # Get current yaw from quaternion
            q = self.x[6:10]
            R = self.quaternion_to_rotation_matrix(q)
            current_yaw = np.arctan2(R[1, 0], R[0, 0])

            # Calculate innovation
            heading_innovation = gps_heading - current_yaw
            heading_innovation = np.arctan2(
                np.sin(heading_innovation), np.cos(heading_innovation))

            # Apply correction if reasonable
            if abs(heading_innovation) < np.deg2rad(45):  # 20 degree gate
                weight = min(gps_speed / 5.0, 1.0) * \
                    self.gps_heading_weight_max
                yaw_correction = heading_innovation * weight * 0.5

                # Apply as quaternion correction
                dq_yaw = np.array(
                    [np.cos(yaw_correction/2), 0, 0, np.sin(yaw_correction/2)])
                q_corrected = np.array([
                    q[0]*dq_yaw[0] - q[3]*dq_yaw[3],
                    q[1]*dq_yaw[0] + q[2]*dq_yaw[3],
                    q[2]*dq_yaw[0] - q[1]*dq_yaw[3],
                    q[3]*dq_yaw[0] + q[0]*dq_yaw[3]
                ])
                self.x[6:10] = self.normalize_quaternion(q_corrected)

    def update_barometer(self, baro_alt):
        """Update dengan barometer altitude measurement"""
        if not self.initialized or baro_alt is None:
            return

        z = np.array([baro_alt])
        h = np.array([-self.x[2]])  # altitude = -z_ned
        y = z - h

        H = np.zeros((1, 15))
        H[0, 2] = -1

        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        # Apply corrections (same pattern)
        self.x[0:3] += dx[0:3]
        self.x[3:6] += dx[3:6]

        dtheta = dx[6:9]
        dq = np.array([1, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq = self.normalize_quaternion(dq)

        q = self.x[6:10]
        q_corrected = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
        ])
        self.x[6:10] = self.normalize_quaternion(q_corrected)

        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_baro @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_magnetometer(self, mag_body):
        """ENHANCED magnetometer update dengan bias learning dan adaptive noise"""
        if not self.initialized or mag_body is None:
            return

        # Check if measurement is valid
        mag_norm = np.linalg.norm(mag_body)
        if mag_norm < 0.1 or mag_norm > 2.0:
            return

        # MAGNETOMETER BIAS LEARNING
        if self.enable_mag_bias_learning:
            # Simple moving average bias learning
            alpha_bias = 0.001  # Learning rate
            expected_norm = 1.0  # Expected earth magnetic field strength
            bias_correction = (mag_norm - expected_norm) * \
                mag_body / mag_norm * alpha_bias
            self.mag_bias += bias_correction

            # Apply learned bias
            mag_body_corrected = mag_body - self.mag_bias
        else:
            mag_body_corrected = mag_body.copy()

        # Get current rotation matrix
        R_bn = self.quaternion_to_rotation_matrix(self.x[6:10])

        # Expected magnetic field in body frame (with declination)
        mag_ref_ned_corrected = np.array([
            np.cos(self.mag_declination),
            np.sin(self.mag_declination),
            0.0
        ])
        mag_expected_body = R_bn.T @ mag_ref_ned_corrected

        # ADAPTIVE NOISE based on magnetic field consistency
        # Calculate consistency metric
        mag_consistency = abs(
            mag_norm - np.linalg.norm(mag_expected_body)) / np.linalg.norm(mag_expected_body)

        # Increase noise if inconsistent (likely interference)
        noise_scale = 1.0 + 5.0 * mag_consistency  # Scale factor 1-6
        R_mag_adaptive = self.R_mag * noise_scale

        # Use horizontal components primarily for yaw (less affected by pitch/roll errors)
        mag_h = np.array([mag_body_corrected[0], mag_body_corrected[1], 0.0])
        exp_h = np.array([mag_expected_body[0], mag_expected_body[1], 0.0])

        if np.linalg.norm(mag_h) < 0.05 or np.linalg.norm(exp_h) < 0.05:
            return

        # Normalize horizontal components
        mag_h_norm = mag_h / np.linalg.norm(mag_h)
        exp_h_norm = exp_h / np.linalg.norm(exp_h)

        # Innovation for horizontal components only (focus on yaw)
        z = mag_h_norm[:2]
        h = exp_h_norm[:2]
        y = z - h

        # ENHANCED Measurement Jacobian for better yaw sensitivity
        H = np.zeros((2, 15))

        # Primary yaw sensitivity
        H[0, 8] = exp_h_norm[1]   # dx/dψ ≈ my
        H[1, 8] = -exp_h_norm[0]  # dy/dψ ≈ -mx

        # Secondary roll/pitch sensitivity (reduced weighting)
        H[0, 6] = -exp_h_norm[2] * 0.05   # Reduced coupling
        H[1, 7] = -exp_h_norm[2] * 0.05   # Reduced coupling

        # Use adaptive noise for horizontal components
        R_mag_h = R_mag_adaptive[:2, :2]

        # Innovation gating - more permissive for yaw
        S = H @ self.P @ H.T + R_mag_h
        if np.min(np.linalg.eigvals(S)) < 1e-6:
            return

        # Mahalanobis distance test
        y_normalized = y.T @ np.linalg.inv(S) @ y
        if y_normalized > self.yaw_innovation_gate**2:  # Use larger gate
            return

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # Apply innovation with saturation
        y_limited = np.clip(y, -0.3, 0.3)  # Less aggressive limiting
        dx = K @ y_limited

        # Apply corrections with more permissive limits for yaw
        self.x[0:3] += np.clip(dx[0:3], -0.1, 0.1)
        self.x[3:6] += np.clip(dx[3:6], -0.5, 0.5)

        # Attitude correction - focus on yaw
        dtheta = dx[6:9]
        # Allow larger yaw corrections but limit roll/pitch
        dtheta[0] = np.clip(dtheta[0], -0.05, 0.05)  # Roll limit
        dtheta[1] = np.clip(dtheta[1], -0.05, 0.05)  # Pitch limit
        dtheta[2] = np.clip(dtheta[2], -0.35, 0.35)    # Yaw limit (larger)

        dq = np.array([1, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq = self.normalize_quaternion(dq)

        q = self.x[6:10]
        q_corrected = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
        ])
        self.x[6:10] = self.normalize_quaternion(q_corrected)

        # Bias corrections
        self.x[10:13] += np.clip(dx[9:12], -0.01, 0.01)
        self.x[13:16] += np.clip(dx[12:15], -0.001, 0.001)

        # Covariance update
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_mag_h @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(15) * 1e-9

    def get_state(self):
        """Return current state estimates"""
        # Convert quaternion to Euler angles
        q = self.x[6:10]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        euler = r.as_euler('xyz')

        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy(),
            'attitude_euler': euler,
            'quaternion': self.x[6:10].copy(),
            'acc_bias': self.x[10:13].copy(),
            'gyro_bias': self.x[13:16].copy(),
            'position_std': np.sqrt(np.diag(self.P[0:3, 0:3])),
            'velocity_std': np.sqrt(np.diag(self.P[3:6, 3:6])),
            'attitude_std': np.sqrt(np.diag(self.P[6:9, 6:9])),
            'prediction_mode': self.prediction_mode
        }


class EKFWithControl(BaseEKF):
    """
    Extended Kalman Filter with Control Input Integration

    This class enhances the base EKF by incorporating control input data
    from the simulation to improve prediction accuracy, especially for
    position and velocity estimation.
    """

    def __init__(self, dt=0.01):
        super().__init__(dt)

        # Control input specific parameters
        # How much to trust control vs IMU (0-1)
        self.control_trust_factor = 0.7
        self.min_thrust_threshold = 0.1  # Minimum thrust to consider valid (N)
        self.max_thrust_threshold = 100  # Maximum reasonable thrust (N)

        # Control input availability tracking
        self.control_available = False
        self.control_quality = 0.0  # Quality metric (0-1)

    def predict_with_control_input(self, accel_body, gyro_body, control_data=None):
        """
        Enhanced prediction step with control input data

        Workflow:
        1. Validate IMU and control input data
        2. Apply IMU-based prediction (fallback method)
        3. Calculate physics-based prediction from control inputs
        4. Fuse IMU and control predictions based on quality
        5. Propagate state and covariance

        Args:
            accel_body: IMU accelerometer reading [ax, ay, az] (m/s²)
            gyro_body: IMU gyroscope reading [gx, gy, gz] (rad/s)
            control_data: Dictionary containing:
                - 'motor_thrusts': [T1, T2, T3, T4, T5, T6] (N)
                - 'thrust_sp': [Fx, Fy, Fz] (N) 
                - 'total_thrust': scalar total thrust (N)
                - 'control_torques': [Mx, My, Mz] (N⋅m)
        """
        if not self.initialized:
            return

        # === STEP 1: EXTRACT CURRENT STATE ===
        pos = self.x[0:3]
        vel = self.x[3:6]
        q = self.x[6:10]
        acc_bias = self.x[10:13]
        gyro_bias = self.x[13:16]

        # === STEP 2: CORRECT SENSOR MEASUREMENTS ===
        accel_corrected = accel_body - acc_bias
        gyro_corrected = gyro_body - gyro_bias

        # Get current rotation matrix (body to NED)
        R_bn = self.quaternion_to_rotation_matrix(q)

        # === STEP 3: PHYSICS-BASED ACCELERATION PREDICTION ===
        accel_physics = None
        control_quality = 0.0

        if control_data is not None:
            # Priority 1: Individual motor thrusts (most accurate)
            if 'motor_thrusts' in control_data and control_data['motor_thrusts'] is not None:
                motor_thrusts = control_data['motor_thrusts']
                total_thrust = np.sum(motor_thrusts)

                if len(motor_thrusts) == 6 and self.min_thrust_threshold < total_thrust < self.max_thrust_threshold:
                    # Check thrust distribution quality
                    thrust_variance = np.var(motor_thrusts)
                    thrust_mean = np.mean(motor_thrusts)
                    thrust_cv = thrust_variance / \
                        (thrust_mean + 1e-6)  # Coefficient of variation

                    # Quality metric based on thrust consistency and magnitude
                    control_quality = min(
                        1.0, total_thrust / 20.0) * max(0.1, 1.0 - thrust_cv)

                    accel_physics = self.dynamics.predict_acceleration_from_motor_thrusts(
                        motor_thrusts, R_bn)
                    self.prediction_mode = "MOTOR_THRUSTS"

            # Priority 2: Control thrust vector
            elif 'thrust_sp' in control_data and control_data['thrust_sp'] is not None:
                thrust_sp = control_data['thrust_sp']
                thrust_magnitude = np.linalg.norm(thrust_sp)

                if thrust_magnitude > self.min_thrust_threshold:
                    # Slightly lower quality
                    control_quality = min(1.0, thrust_magnitude / 20.0) * 0.8

                    accel_physics = self.dynamics.predict_acceleration_from_thrust_vector(
                        thrust_sp, R_bn)
                    self.prediction_mode = "THRUST_VECTOR"

            # Priority 3: Total thrust (assume vertical)
            elif 'total_thrust' in control_data and control_data['total_thrust'] is not None:
                total_thrust = control_data['total_thrust']

                if self.min_thrust_threshold < total_thrust < self.max_thrust_threshold:
                    control_quality = min(
                        1.0, total_thrust / 20.0) * 0.6  # Lower quality

                    # Vertical thrust assumption
                    thrust_body = np.array([0, 0, -total_thrust])
                    accel_physics = self.dynamics.predict_acceleration_from_thrust_vector(
                        thrust_body, R_bn)
                    self.prediction_mode = "TOTAL_THRUST"

        # === STEP 4: IMU-BASED ACCELERATION (BASELINE) ===
        accel_imu = R_bn @ accel_corrected + self.g_ned

        # === STEP 5: SENSOR FUSION ===
        if accel_physics is not None and control_quality > 0.1:
            # Adaptive fusion based on control quality
            alpha = self.control_trust_factor * control_quality
            accel_fused = alpha * accel_physics + (1 - alpha) * accel_imu
            self.control_available = True
            self.control_quality = control_quality
        else:
            # Fallback to IMU-only prediction
            accel_fused = accel_imu
            self.prediction_mode = "IMU_ONLY"
            self.control_available = False
            self.control_quality = 0.0

        # === STEP 6: KINEMATIC INTEGRATION ===
        # Position and velocity integration
        vel_mid = vel + 0.5 * accel_fused * self.dt  # Midpoint integration
        pos_new = pos + vel_mid * self.dt
        vel_new = vel + accel_fused * self.dt

        # === STEP 7: ATTITUDE INTEGRATION ===
        omega_norm = np.linalg.norm(gyro_corrected)
        if omega_norm > 1e-8:
            # Rodrigues rotation formula for quaternion integration
            axis = gyro_corrected / omega_norm
            angle = omega_norm * self.dt

            # Quaternion increment
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])

            # Quaternion multiplication: q_new = q * dq
            q_new = np.array([
                q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
                q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
                q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
                q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
            ])
        else:
            q_new = q.copy()

        # === STEP 8: UPDATE NOMINAL STATE ===
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:10] = self.normalize_quaternion(q_new)
        # Biases evolve via random walk (no direct update)

        # === STEP 9: ERROR STATE JACOBIAN ===
        F = np.eye(15)

        # Position error propagation
        F[0:3, 3:6] = np.eye(3) * self.dt

        # Velocity error propagation (includes control input effects)
        F[3:6, 6:9] = -R_bn @ self.skew_symmetric(accel_corrected) * self.dt
        F[3:6, 9:12] = -R_bn * self.dt  # Accelerometer bias coupling

        # Attitude error propagation
        F[6:9, 6:9] = np.eye(3) - self.skew_symmetric(gyro_corrected) * self.dt
        F[6:9, 12:15] = -np.eye(3) * self.dt  # Gyroscope bias coupling

        # === STEP 10: COVARIANCE PROPAGATION ===
        self.P = F @ self.P @ F.T + self.Q

        # Ensure positive definite and symmetric
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(15) * 1e-12  # Numerical stability

    def get_state(self):
        """Enhanced state getter with control input information"""
        base_state = super().get_state()

        # Add control-specific information
        base_state.update({
            'control_available': self.control_available,
            'control_quality': self.control_quality,
            'control_trust_factor': self.control_trust_factor
        })

        return base_state
