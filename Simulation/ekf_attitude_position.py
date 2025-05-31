import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


class ImprovedPositionVelocityEKFWithSimulationData:
    """
    Extended Kalman Filter yang menggunakan data control dari simulasi

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
        self.P[0:3, 0:3] *= 100.0   # Position uncertainty
        self.P[3:6, 3:6] *= 10.0    # Velocity uncertainty
        self.P[6:9, 6:9] *= 1.0     # Attitude uncertainty
        self.P[8, 8] *= 10.0        # Higher initial yaw uncertainty
        self.P[9:12, 9:12] *= 0.01  # Accel bias uncertainty
        self.P[12:15, 12:15] *= 0.01  # Gyro bias uncertainty

        # Process noise matrix Q (untuk error states)
        self.Q = np.zeros((15, 15))
        # Position process noise
        self.Q[0:3, 0:3] = np.eye(3) * (0.01 * self.dt**2)**2
        # Velocity process noise
        self.Q[3:6, 3:6] = np.eye(3) * (0.1 * self.dt)**2
        # Attitude process noise
        self.Q[6:9, 6:9] = np.eye(3) * (0.01 * self.dt)**2
        # Acceleration bias random walk
        self.Q[9:12, 9:12] = np.eye(3) * (1e-4 * self.dt)**2
        # Gyroscope bias random walk
        self.Q[12:15, 12:15] = np.eye(3) * (1e-5 * self.dt)**2

        # Measurement noise matrices
        self.R_gps_pos = np.eye(3) * 1.0**2   # GPS position noise
        self.R_gps_vel = np.eye(3) * 0.1**2   # GPS velocity noise
        self.R_baro = np.array([[0.5**2]])    # Barometer noise
        self.R_mag = np.eye(3) * 0.05**2      # Magnetometer noise

        # Constants
        self.g_ned = np.array([0, 0, 9.81])  # Gravity in NED frame

        # Magnetic field reference in NED
        self.mag_ref_ned = np.array([1.0, 0.0, 0.0])

        # Control input mixing parameter
        self.control_trust_factor = 0.7  # How much to trust control vs IMU

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
        """Initialize state dengan GPS dan IMU data"""

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

        # Initialize yaw
        yaw = 0.0
        if true_yaw is not None:
            yaw = true_yaw
        elif initial_mag is not None and np.linalg.norm(initial_mag) > 0.1:
            yaw = 0.0  # Will rely on magnetometer updates

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

        # Initialize biases
        self.x[10:13] = np.zeros(3)  # Accel bias
        self.x[13:16] = np.zeros(3)  # Gyro bias

        self.initialized = True

    def predict_with_simulation_control(self, accel_body, gyro_body, control_data=None):
        """
        Prediction step dengan data control dari simulasi

        Args:
            accel_body: accelerometer reading [ax, ay, az] (m/s²)
            gyro_body: gyroscope reading [gx, gy, gz] (rad/s)  
            control_data: Dictionary dengan control data dari simulasi
        """
        if not self.initialized:
            return

        # Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        q = self.x[6:10]
        acc_bias = self.x[10:13]
        gyro_bias = self.x[13:16]

        # Correct sensor measurements dengan bias
        accel_corrected = accel_body - acc_bias
        gyro_corrected = gyro_body - gyro_bias

        # Get rotation matrix
        R_bn = self.quaternion_to_rotation_matrix(q)  # Body to NED

        # === PHYSICS-BASED ACCELERATION PREDICTION ===
        accel_physics = None

        if control_data is not None:
            # Try different control input sources (prioritized)
            if 'motor_thrusts' in control_data and control_data['motor_thrusts'] is not None:
                # Method 1: Use individual motor thrusts (most accurate)
                motor_thrusts = control_data['motor_thrusts']
                if len(motor_thrusts) == 6 and np.any(motor_thrusts > 0):
                    accel_physics = self.dynamics.predict_acceleration_from_motor_thrusts(
                        motor_thrusts, R_bn)
                    self.prediction_mode = "MOTOR_THRUSTS"

            elif 'thrust_sp' in control_data and control_data['thrust_sp'] is not None:
                # Method 2: Use control thrust vector
                thrust_sp = control_data['thrust_sp']
                if np.linalg.norm(thrust_sp) > 0.1:
                    accel_physics = self.dynamics.predict_acceleration_from_thrust_vector(
                        thrust_sp, R_bn)
                    self.prediction_mode = "THRUST_VECTOR"

            elif 'total_thrust' in control_data and control_data['total_thrust'] is not None:
                # Method 3: Use total thrust (assume vertical)
                total_thrust = control_data['total_thrust']
                if total_thrust > 0.1:
                    thrust_body = np.array(
                        [0, 0, -total_thrust])  # Vertical thrust
                    accel_physics = self.dynamics.predict_acceleration_from_thrust_vector(
                        thrust_body, R_bn)
                    self.prediction_mode = "TOTAL_THRUST"

        # IMU-based acceleration (for comparison/fusion)
        accel_imu = R_bn @ accel_corrected + self.g_ned

        if accel_physics is not None:
            # Fusion: Combine predicted dan measured acceleration
            alpha = self.control_trust_factor
            accel_fused = alpha * accel_physics + (1 - alpha) * accel_imu
        else:
            # Fallback to IMU-only prediction
            accel_fused = accel_imu
            self.prediction_mode = "IMU_ONLY"

        # === KINEMATIC INTEGRATION ===
        # Position dan velocity integration dengan improved acceleration
        vel_mid = vel + 0.5 * accel_fused * self.dt
        pos_new = pos + vel_mid * self.dt
        vel_new = vel + accel_fused * self.dt

        # === ATTITUDE INTEGRATION ===
        # Quaternion integration
        omega_norm = np.linalg.norm(gyro_corrected)
        if omega_norm > 1e-8:
            # Axis-angle representation
            axis = gyro_corrected / omega_norm
            angle = omega_norm * self.dt

            # Quaternion increment
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])

            # Quaternion multiplication
            q_new = np.array([
                q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
                q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
                q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
                q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
            ])
        else:
            q_new = q.copy()

        # Update nominal state
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:10] = self.normalize_quaternion(q_new)
        # Biases remain the same (random walk)

        # === ERROR STATE JACOBIAN ===
        F = np.eye(15)

        # Position error propagation
        F[0:3, 3:6] = np.eye(3) * self.dt

        # Velocity error propagation
        F[3:6, 6:9] = -R_bn @ self.skew_symmetric(accel_corrected) * self.dt
        F[3:6, 9:12] = -R_bn * self.dt

        # Attitude error propagation
        F[6:9, 6:9] = np.eye(3) - self.skew_symmetric(gyro_corrected) * self.dt
        F[6:9, 12:15] = -np.eye(3) * self.dt

        # Propagate error covariance
        self.P = F @ self.P @ F.T + self.Q

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

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
        """Update dengan GPS velocity measurement"""
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
        """Update dengan magnetometer measurement untuk yaw correction"""
        if not self.initialized or mag_body is None:
            return

        # Check if measurement is valid
        mag_norm = np.linalg.norm(mag_body)
        if mag_norm < 0.1 or mag_norm > 2.0:
            return

        # Get current rotation matrix
        R_bn = self.quaternion_to_rotation_matrix(self.x[6:10])

        # Expected magnetic field in body frame
        mag_expected_body = R_bn.T @ self.mag_ref_ned

        # Use only horizontal components for yaw correction
        mag_h = np.array([mag_body[0], mag_body[1], 0.0])
        exp_h = np.array([mag_expected_body[0], mag_expected_body[1], 0.0])

        if np.linalg.norm(mag_h) < 0.1 or np.linalg.norm(exp_h) < 0.1:
            return

        # Normalize horizontal components
        mag_h_norm = mag_h / np.linalg.norm(mag_h)
        exp_h_norm = exp_h / np.linalg.norm(exp_h)

        # Innovation
        z = mag_h_norm[:2]
        h = exp_h_norm[:2]
        y = z - h

        # Measurement Jacobian H (2x15)
        H = np.zeros((2, 15))
        mag_body_normalized = mag_expected_body / \
            np.linalg.norm(mag_expected_body)

        # Focus mainly on yaw
        H[0, 8] = mag_body_normalized[1]   # dx/dψ ≈ my
        H[1, 8] = -mag_body_normalized[0]  # dy/dψ ≈ -mx
        H[0, 6] = -mag_body_normalized[2] * 0.1
        H[1, 7] = -mag_body_normalized[2] * 0.1

        R_mag_h = self.R_mag[:2, :2] * 4.0

        S = H @ self.P @ H.T + R_mag_h
        if np.min(np.linalg.eigvals(S)) < 1e-6:
            return

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        y_limited = np.clip(y, -0.5, 0.5)
        dx = K @ y_limited

        # Apply corrections with limits
        self.x[0:3] += np.clip(dx[0:3], -0.1, 0.1)
        self.x[3:6] += np.clip(dx[3:6], -0.5, 0.5)

        dtheta = dx[6:9]
        dtheta = np.clip(dtheta, -0.1, 0.1)

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

        self.x[10:13] += np.clip(dx[9:12], -0.01, 0.01)
        self.x[13:16] += np.clip(dx[12:15], -0.001, 0.001)

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


def run_ekf_with_simulation_data(csv_file_path, use_control_input=True, use_magnetometer=True):
    """Main function untuk run EKF dengan data dari simulasi"""

    # Load data
    data = pd.read_csv(csv_file_path)
    print(f"Simulation data loaded: {len(data)} samples")

    # Check if control data is available
    control_columns = ['motor_thrust_1', 'motor_thrust_2', 'motor_thrust_3',
                       'motor_thrust_4', 'motor_thrust_5', 'motor_thrust_6']
    has_motor_data = all(col in data.columns for col in control_columns)

    thrust_columns = ['thrust_sp_x', 'thrust_sp_y', 'thrust_sp_z']
    has_thrust_data = all(col in data.columns for col in thrust_columns)

    torque_columns = ['control_torque_x',
                      'control_torque_y', 'control_torque_z']
    has_torque_data = all(col in data.columns for col in torque_columns)

    print(f"Control data availability:")
    print(f"  Motor thrusts: {has_motor_data}")
    print(f"  Thrust vector: {has_thrust_data}")
    print(f"  Control torques: {has_torque_data}")

    # Calculate dt
    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"Sampling time: {dt_mean:.4f} s")

    # Initialize EKF
    ekf = ImprovedPositionVelocityEKFWithSimulationData(dt=dt_mean)

    # Find first timestamp with complete and valid sensor data for initialization
    init_idx = None
    min_start_time = 0.1  # Skip very early timestamps (first 0.1 seconds)

    for i in range(len(data)):
        row = data.iloc[i]

        # Skip very early timestamps where systems might not be active
        if row['timestamp'] < min_start_time:
            continue

        # Check for complete sensor data
        if row['gps_available'] == 1:
            # GPS data validation
            gps_pos_test = np.array(
                [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
            gps_vel_test = np.array(
                [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

            # IMU data validation
            acc_test = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
            gyro_test = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

            # Check if all sensor data is valid (not NaN and not all zeros)
            gps_valid = not np.any(np.isnan(gps_pos_test)) and not np.any(
                np.isnan(gps_vel_test))
            imu_valid = (not np.any(np.isnan(acc_test)) and not np.any(np.isnan(gyro_test)) and
                         np.linalg.norm(acc_test) > 0.1 and np.linalg.norm(gyro_test) >= 0)

            # Optional: Check for motor activity (system is actually flying)
            motor_active = False
            if has_motor_data:
                motor_thrusts = np.array([row['motor_thrust_1'], row['motor_thrust_2'], row['motor_thrust_3'],
                                          row['motor_thrust_4'], row['motor_thrust_5'], row['motor_thrust_6']])
                motor_active = np.sum(motor_thrusts) > 0.1
            else:
                motor_active = True  # Assume active if no motor data

            if gps_valid and imu_valid and motor_active:
                init_idx = i
                print(
                    f"Found valid initialization data at index {i}, time = {row['timestamp']:.3f}s")
                print(f"  GPS pos norm: {np.linalg.norm(gps_pos_test):.3f}m")
                print(f"  IMU acc norm: {np.linalg.norm(acc_test):.3f}m/s²")
                print(f"  IMU gyro norm: {np.linalg.norm(gyro_test):.4f}rad/s")
                if has_motor_data:
                    print(
                        f"  Total motor thrust: {np.sum(motor_thrusts):.3f}N")
                break

    if init_idx is None:
        print("No valid complete sensor data found for initialization!")
        print("Trying with relaxed criteria (GPS + IMU only)...")

        # Fallback: try with just GPS and IMU
        for i in range(len(data)):
            row = data.iloc[i]
            if row['timestamp'] < min_start_time:
                continue

            if row['gps_available'] == 1:
                gps_pos_test = np.array(
                    [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
                gps_vel_test = np.array(
                    [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])
                acc_test = np.array([row['acc_x'], row['acc_y'], row['acc_z']])

                gps_valid = not np.any(np.isnan(gps_pos_test)) and not np.any(
                    np.isnan(gps_vel_test))
                acc_valid = not np.any(
                    np.isnan(acc_test)) and np.linalg.norm(acc_test) > 0.1

                if gps_valid and acc_valid:
                    init_idx = i
                    print(
                        f"Fallback initialization at index {i}, time = {row['timestamp']:.3f}s")
                    break

    if init_idx is None:
        print("No suitable initialization data found!")
        return None

    print(
        f"EKF will be initialized at index {init_idx}, time = {data.iloc[init_idx]['timestamp']:.3f}s")

    # Initialize EKF
    row = data.iloc[init_idx]
    # Updated column names to match simulation output
    gps_pos = np.array(
        [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
    gps_vel = np.array(
        [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])
    initial_acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
    initial_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

    print(f"Initial sensor data:")
    print(
        f"  GPS pos: [{gps_pos[0]:.3f}, {gps_pos[1]:.3f}, {gps_pos[2]:.3f}] m")
    print(
        f"  GPS vel: [{gps_vel[0]:.3f}, {gps_vel[1]:.3f}, {gps_vel[2]:.3f}] m/s")
    print(
        f"  IMU acc: [{initial_acc[0]:.3f}, {initial_acc[1]:.3f}, {initial_acc[2]:.3f}] m/s²")
    print(
        f"  IMU gyro: [{initial_gyro[0]:.3f}, {initial_gyro[1]:.3f}, {initial_gyro[2]:.3f}] rad/s")

    # Get initial magnetometer if available
    initial_mag = None
    if 'mag_available' in data.columns and row['mag_available'] == 1:
        initial_mag = np.array([row['mag_x'], row['mag_y'], row['mag_z']])

    # Initialize with ground truth yaw for testing
    true_yaw = row['true_yaw'] if use_magnetometer else None

    try:
        ekf.initialize_state(gps_pos, gps_vel, initial_acc,
                             initial_gyro, initial_mag, true_yaw)
        print("EKF initialization successful")
    except Exception as e:
        print(f"EKF initialization failed: {str(e)}")
        return None

    # Process all data
    n_samples = len(data)
    results = {
        'timestamp': [],
        'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [],
        'pos_std': [], 'vel_std': [], 'att_std': [],
        'prediction_modes': []
    }

    # Count sensor updates
    prediction_count = 0
    gps_update_count = 0
    baro_update_count = 0
    mag_update_count = 0
    control_used_count = 0

    for i in range(n_samples):
        row = data.iloc[i]

        # IMU data
        accel_body = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro_body = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        # Prepare control data
        control_data = None
        if use_control_input:
            control_data = {}

            # Motor thrusts (most preferred)
            if has_motor_data:
                motor_thrusts = np.array([
                    row['motor_thrust_1'], row['motor_thrust_2'], row['motor_thrust_3'],
                    row['motor_thrust_4'], row['motor_thrust_5'], row['motor_thrust_6']
                ])
                if np.sum(motor_thrusts) > 0.1:  # Valid motor data
                    control_data['motor_thrusts'] = motor_thrusts
                    control_used_count += 1

            # Thrust vector (second preference)
            if has_thrust_data and 'motor_thrusts' not in control_data:
                thrust_sp = np.array(
                    [row['thrust_sp_x'], row['thrust_sp_y'], row['thrust_sp_z']])
                if np.linalg.norm(thrust_sp) > 0.1:
                    control_data['thrust_sp'] = thrust_sp
                    control_used_count += 1

            # Total thrust (fallback)
            if 'total_thrust_sp' in data.columns and len(control_data) == 0:
                total_thrust = row['total_thrust_sp']
                if total_thrust > 0.1:
                    control_data['total_thrust'] = total_thrust
                    control_used_count += 1

            # Control torques (additional info)
            if has_torque_data:
                control_torques = np.array(
                    [row['control_torque_x'], row['control_torque_y'], row['control_torque_z']])
                control_data['control_torques'] = control_torques

        # Prediction step dengan control input
        try:
            ekf.predict_with_simulation_control(
                accel_body, gyro_body, control_data)
            prediction_count += 1
        except Exception as e:
            print(f"Warning: Prediction step failed at sample {i}: {str(e)}")
            continue

        # GPS updates - Updated column names with validation
        if row['gps_available'] == 1:
            gps_pos = np.array(
                [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
            gps_vel = np.array(
                [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

            # Validate GPS data before using it
            if not np.any(np.isnan(gps_pos)) and not np.any(np.isnan(gps_vel)):
                # Reasonable GPS bounds check
                # 10km position, 100m/s velocity
                if np.linalg.norm(gps_pos) < 10000 and np.linalg.norm(gps_vel) < 100:
                    ekf.update_gps_position(gps_pos)
                    ekf.update_gps_velocity(gps_vel)
                    gps_update_count += 1

        # Barometer update with validation
        if row['baro_available'] == 1:
            baro_alt = row['baro_altitude']
            # Reasonable altitude range
            if not np.isnan(baro_alt) and -1000 < baro_alt < 10000:
                ekf.update_barometer(baro_alt)
                baro_update_count += 1

        # Magnetometer update with validation
        if use_magnetometer and 'mag_available' in data.columns and row['mag_available'] == 1:
            mag_body = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
            if not np.any(np.isnan(mag_body)) and np.linalg.norm(mag_body) > 0.1:
                ekf.update_magnetometer(mag_body)
                mag_update_count += 1

        # Store results
        try:
            state = ekf.get_state()
            results['timestamp'].append(row['timestamp'])
            results['position'].append(state['position'])
            results['velocity'].append(state['velocity'])
            results['attitude'].append(state['attitude_euler'])
            results['acc_bias'].append(state['acc_bias'])
            results['gyro_bias'].append(state['gyro_bias'])
            results['pos_std'].append(state['position_std'])
            results['vel_std'].append(state['velocity_std'])
            results['att_std'].append(state['attitude_std'])
            results['prediction_modes'].append(state['prediction_mode'])
        except Exception as e:
            print(f"Warning: Failed to get state at sample {i}: {str(e)}")
            continue

        # Progress indicator
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i}/{n_samples} ({100*i/n_samples:.1f}%)")

    print(
        f"Processing completed: {n_samples} samples, {len(results['timestamp'])} valid results")

    # Check if we have enough valid results
    if len(results['timestamp']) < 100:
        print(
            f"Error: Only {len(results['timestamp'])} valid results. Need at least 100 for analysis.")
        return None

    # Convert to numpy arrays
    for key in ['position', 'velocity', 'attitude', 'acc_bias', 'gyro_bias', 'pos_std', 'vel_std', 'att_std']:
        results[key] = np.array(results[key])

    # Calculate errors - using only valid result indices
    valid_indices = [i for i, t in enumerate(
        data['timestamp']) if t in results['timestamp']]

    true_pos = np.column_stack(
        [data.iloc[valid_indices]['true_pos_x'],
         data.iloc[valid_indices]['true_pos_y'],
         data.iloc[valid_indices]['true_pos_z']])
    true_vel = np.column_stack(
        [data.iloc[valid_indices]['true_vel_x'],
         data.iloc[valid_indices]['true_vel_y'],
         data.iloc[valid_indices]['true_vel_z']])
    true_att = np.column_stack(
        [data.iloc[valid_indices]['true_roll'],
         data.iloc[valid_indices]['true_pitch'],
         data.iloc[valid_indices]['true_yaw']])

    pos_error = results['position'] - true_pos
    vel_error = results['velocity'] - true_vel
    att_error = results['attitude'] - true_att

    # Handle angle wrapping
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # Statistics
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    # Count prediction modes
    prediction_modes = results['prediction_modes']
    mode_counts = {}
    for mode in prediction_modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    print("\n" + "="*80)
    print(f"EKF WITH SIMULATION CONTROL DATA - RESULTS")
    print("="*80)

    print(f"\nControl Input Statistics:")
    print(
        f"  Control data used: {control_used_count}/{n_samples} ({100*control_used_count/n_samples:.1f}%)")
    print(f"  Prediction mode distribution:")
    for mode, count in mode_counts.items():
        print(f"    {mode}: {count} ({100*count/len(prediction_modes):.1f}%)")

    print(f"\nSensor Update Statistics:")
    print(f"  Total predictions: {prediction_count}")
    print(f"  GPS updates: {gps_update_count}")
    print(f"  Barometer updates: {baro_update_count}")
    print(f"  Magnetometer updates: {mag_update_count}")

    print(f"\n--- POSITION ESTIMATION RESULTS ---")
    print(
        f"  RMSE [X,Y,Z]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(f"  Total RMSE: {np.linalg.norm(pos_rmse):.4f} m")

    print(f"\n--- VELOCITY ESTIMATION RESULTS ---")
    print(
        f"  RMSE [X,Y,Z]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(f"  Total RMSE: {np.linalg.norm(vel_rmse):.4f} m/s")

    print(f"\n--- ATTITUDE ESTIMATION RESULTS ---")
    print(
        f"  RMSE [Roll,Pitch,Yaw]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] deg")

    print("="*80)

    return ekf, results, data


def plot_ekf_simulation_results(ekf_results, data):
    """Plot hasil EKF vs ground truth dari simulasi"""

    time = np.array(ekf_results['timestamp'])

    # Ground truth data
    true_pos = np.column_stack(
        [data['true_pos_x'], data['true_pos_y'], data['true_pos_z']])
    true_vel = np.column_stack(
        [data['true_vel_x'], data['true_vel_y'], data['true_vel_z']])
    true_att = np.column_stack(
        [data['true_roll'], data['true_pitch'], data['true_yaw']])

    # EKF estimates
    est_pos = ekf_results['position']
    est_vel = ekf_results['velocity']
    est_att = ekf_results['attitude']

    # Plot Position Comparison
    plt.figure(figsize=(15, 10))
    plt.suptitle('EKF Position Estimation vs Ground Truth', fontsize=16)

    for i, label in enumerate(['X (North)', 'Y (East)', 'Z (Down)']):
        plt.subplot(3, 1, i+1)
        plt.plot(time, true_pos[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_pos[:, i], 'r--', linewidth=2, label='EKF Estimate')
        plt.fill_between(time,
                         est_pos[:, i] - 2*ekf_results['pos_std'][:, i],
                         est_pos[:, i] + 2*ekf_results['pos_std'][:, i],
                         alpha=0.3, color='red', label='2σ Uncertainty')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Position {label} (m)')
        plt.title(f'Position {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot Velocity Comparison
    plt.figure(figsize=(15, 10))
    plt.suptitle('EKF Velocity Estimation vs Ground Truth', fontsize=16)

    for i, label in enumerate(['X (North)', 'Y (East)', 'Z (Down)']):
        plt.subplot(3, 1, i+1)
        plt.plot(time, true_vel[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_vel[:, i], 'r--', linewidth=2, label='EKF Estimate')
        plt.fill_between(time,
                         est_vel[:, i] - 2*ekf_results['vel_std'][:, i],
                         est_vel[:, i] + 2*ekf_results['vel_std'][:, i],
                         alpha=0.3, color='red', label='2σ Uncertainty')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Velocity {label} (m/s)')
        plt.title(f'Velocity {label}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot Attitude Comparison
    plt.figure(figsize=(15, 10))
    plt.suptitle('EKF Attitude Estimation vs Ground Truth', fontsize=16)

    for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
        plt.subplot(3, 1, i+1)
        plt.plot(time, np.rad2deg(true_att[:, i]),
                 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, np.rad2deg(est_att[:, i]),
                 'r--', linewidth=2, label='EKF Estimate')
        plt.fill_between(time,
                         np.rad2deg(est_att[:, i] - 2 *
                                    ekf_results['att_std'][:, i]),
                         np.rad2deg(est_att[:, i] + 2 *
                                    ekf_results['att_std'][:, i]),
                         alpha=0.3, color='red', label='2σ Uncertainty')
        plt.xlabel('Time (s)')
        plt.ylabel(f'{label} (deg)')
        plt.title(f'{label} Angle')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot Control Usage Analysis
    prediction_modes = ekf_results['prediction_modes']

    plt.figure(figsize=(12, 6))
    plt.suptitle('EKF Control Input Usage Analysis', fontsize=16)

    # Count modes over time windows
    window_size = 1000  # samples
    n_windows = len(prediction_modes) // window_size

    mode_timeline = []
    time_windows = []

    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window_modes = prediction_modes[start:end]

        # Count most common mode in this window
        mode_count = {}
        for mode in window_modes:
            mode_count[mode] = mode_count.get(mode, 0) + 1

        most_common = max(mode_count, key=mode_count.get)
        mode_timeline.append(most_common)
        time_windows.append(time[start + window_size//2])

    # Plot control mode usage
    plt.subplot(1, 2, 1)
    modes = list(set(prediction_modes))
    mode_colors = {'IMU_ONLY': 'red', 'MOTOR_THRUSTS': 'green',
                   'THRUST_VECTOR': 'blue', 'TOTAL_THRUST': 'orange'}

    for i, mode in enumerate(mode_timeline):
        color = mode_colors.get(mode, 'gray')
        plt.scatter(time_windows[i], i, c=color, s=20)

    plt.yticks(range(len(modes)), modes)
    plt.xlabel('Time (s)')
    plt.ylabel('Prediction Mode')
    plt.title('Control Input Usage Over Time')
    plt.grid(True, alpha=0.3)

    # Plot mode distribution
    plt.subplot(1, 2, 2)
    mode_counts = {}
    for mode in prediction_modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    modes = list(mode_counts.keys())
    counts = list(mode_counts.values())
    colors = [mode_colors.get(mode, 'gray') for mode in modes]

    plt.pie(counts, labels=modes, colors=colors, autopct='%1.1f%%')
    plt.title('Prediction Mode Distribution')

    plt.tight_layout()


# Test the implementation
if __name__ == "__main__":
    # Update with your simulation data file
    csv_file_path = "logs/complete_flight_data_with_geodetic_20250530_183803.csv"

    print("="*80)
    print("EKF WITH SIMULATION CONTROL DATA INTEGRATION")
    print("="*80)

    # Test with control input
    print("\n--- Testing EKF with Simulation Control Data ---")
    try:
        results = run_ekf_with_simulation_data(
            csv_file_path, use_control_input=True, use_magnetometer=True)

        if results is not None:
            ekf, results_data, data = results

            # Plot comprehensive results
            plot_ekf_simulation_results(results_data, data)

            plt.show()
        else:
            print("EKF processing failed!")

    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file_path}")
        print("Please run the simulation first to generate the data file.")
        print("Expected format: complete_flight_data_with_geodetic_YYYYMMDD_HHMMSS.csv")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    # Compare with IMU-only prediction
    print("\n\n--- Comparison: Testing WITHOUT Control Input ---")
    try:
        results_no_control = run_ekf_with_simulation_data(
            csv_file_path, use_control_input=False, use_magnetometer=True)
        if results_no_control is not None:
            print("\nComparison completed. Check the improvement with control input.")
    except Exception as e:
        print(f"Comparison test failed: {str(e)}")
