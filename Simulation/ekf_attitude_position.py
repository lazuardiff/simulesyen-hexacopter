import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class ImprovedPositionVelocityEKF:
    """
    Extended Kalman Filter dengan attitude estimation untuk posisi dan velocity
    INCLUDING MAGNETOMETER untuk yaw estimation yang lebih baik

    State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bias_ax, bias_ay, bias_az, bias_gx, bias_gy, bias_gz]
    - Position: [px, py, pz] dalam frame NED (m)
    - Velocity: [vx, vy, vz] dalam frame NED (m/s)
    - Attitude: [qw, qx, qy, qz] quaternion (NED frame)
    - Accel bias: [bias_ax, bias_ay, bias_az] (m/s²)
    - Gyro bias: [bias_gx, bias_gy, bias_gz] (rad/s)
    """

    def __init__(self, dt=0.01):
        self.dt = dt

        # State dimension: 16 (pos:3, vel:3, quat:4, acc_bias:3, gyro_bias:3)
        self.state_dim = 16

        # Initialize state vector
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initialize quaternion as identity [w, x, y, z]

        # Initialize covariance matrix
        # Error state dimension: 15 (pos:3, vel:3, att:3, acc_bias:3, gyro_bias:3)
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 100.0  # Position uncertainty
        self.P[3:6, 3:6] *= 10.0   # Velocity uncertainty
        self.P[6:9, 6:9] *= 1.0    # Attitude uncertainty (increased for yaw)
        self.P[8, 8] *= 10.0       # Higher initial yaw uncertainty
        self.P[9:12, 9:12] *= 0.01  # Accel bias uncertainty
        self.P[12:15, 12:15] *= 0.01  # Gyro bias uncertainty

        # Process noise matrix Q (untuk error states)
        self.Q = np.zeros((15, 15))
        # Position process noise (dari velocity integration)
        self.Q[0:3, 0:3] = np.eye(3) * (0.01 * self.dt**2)**2
        # Velocity process noise (dari acceleration integration)
        self.Q[3:6, 3:6] = np.eye(3) * (0.1 * self.dt)**2
        # Attitude process noise (dari gyro integration)
        self.Q[6:9, 6:9] = np.eye(3) * (0.01 * self.dt)**2
        # Acceleration bias random walk
        self.Q[9:12, 9:12] = np.eye(3) * (1e-4 * self.dt)**2
        # Gyroscope bias random walk
        self.Q[12:15, 12:15] = np.eye(3) * (1e-5 * self.dt)**2

        # Measurement noise matrices
        self.R_gps_pos = np.eye(3) * 1.0**2  # GPS position noise
        self.R_gps_vel = np.eye(3) * 0.1**2  # GPS velocity noise
        self.R_baro = np.array([[0.5**2]])   # Barometer noise
        # Magnetometer noise (dari SensorModels.py)
        self.R_mag = np.eye(3) * 0.05**2

        # Constants
        self.g_ned = np.array([0, 0, 9.81])  # Gravity in NED frame

        # Magnetic field reference in NED
        # Match the sensor model: mag_reference = [1.0, 0.0, 0.0]
        self.mag_ref_ned = np.array([1.0, 0.0, 0.0])

        # Alternative: use realistic magnetic field with inclination
        # mag_inclination = 60 * np.pi/180  # degrees
        # self.mag_ref_ned = np.array([
        #     np.cos(mag_inclination),
        #     0.0,
        #     np.sin(mag_inclination)
        # ])

        self.initialized = False

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [w,x,y,z] to rotation matrix"""
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # [x,y,z,w]
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
        # Dalam kondisi static: acc_body ≈ -g_body
        acc_norm = initial_acc / np.linalg.norm(initial_acc)

        # Calculate initial roll dan pitch (asumsi yaw = 0)
        roll = np.arctan2(-acc_norm[1], -acc_norm[2])
        pitch = np.arctan2(acc_norm[0], np.sqrt(
            acc_norm[1]**2 + acc_norm[2]**2))

        # Initialize yaw dari magnetometer jika tersedia
        yaw = 0.0
        if true_yaw is not None:
            # Use ground truth for initial yaw if available (for debugging)
            yaw = true_yaw
            print(f"Using ground truth initial yaw: {np.rad2deg(yaw):.1f} deg")
        elif initial_mag is not None and np.linalg.norm(initial_mag) > 0.1:
            # For initial yaw, assume drone is level or use simple approach
            # Just use the horizontal components of magnetometer
            # In many cases, initial yaw can be set to 0 if unknown

            # Simple approach: use atan2 of horizontal components
            # This assumes the magnetometer is calibrated and aligned
            # yaw = np.arctan2(-initial_mag[1], initial_mag[0])

            # For now, use ground truth if available or set to 0
            yaw = 0.0  # Will rely on magnetometer updates to correct this
            print(
                f"Initial yaw set to: {np.rad2deg(yaw):.1f} deg (will be corrected by magnetometer)")

        # Convert to quaternion
        r = Rotation.from_euler('xyz', [roll, pitch, yaw])
        q_scipy = r.as_quat()  # [x,y,z,w]
        self.x[6:10] = np.array(
            [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # [w,x,y,z]

        # Initialize biases
        self.x[10:13] = np.zeros(3)  # Accel bias
        self.x[13:16] = np.zeros(3)  # Gyro bias

        self.initialized = True
        print(f"EKF initialized:")
        print(f"  Position: {self.x[0:3]}")
        print(f"  Velocity: {self.x[3:6]}")
        print(
            f"  Initial attitude (deg): roll={np.rad2deg(roll):.1f}, pitch={np.rad2deg(pitch):.1f}, yaw={np.rad2deg(yaw):.1f}")

    def predict(self, accel_body, gyro_body):
        """
        Prediction step menggunakan IMU data

        Args:
            accel_body: accelerometer reading dalam body frame [ax, ay, az] (m/s²)
            gyro_body: gyroscope reading dalam body frame [gx, gy, gz] (rad/s)
        """
        if not self.initialized:
            return

        # Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        q = self.x[6:10]
        acc_bias = self.x[10:13]
        gyro_bias = self.x[13:16]

        # PERBAIKAN 1: Correct sensor measurements dengan bias
        accel_corrected = accel_body - acc_bias
        gyro_corrected = gyro_body - gyro_bias

        # PERBAIKAN 2: Get rotation matrix dari quaternion
        R_bn = self.quaternion_to_rotation_matrix(q)  # Body to NED

        # PERBAIKAN 3: Transform acceleration ke NED frame dan kompensasi gravitasi
        # Specific force equation: f = a - g
        # Sehingga: a = f + g = R @ (acc_body - bias) + g_ned
        accel_ned = R_bn @ accel_corrected + self.g_ned

        # PERBAIKAN 4: Kinematic integration dengan RK2
        # Position dan velocity integration
        vel_mid = vel + 0.5 * accel_ned * self.dt
        pos_new = pos + vel_mid * self.dt
        vel_new = vel + accel_ned * self.dt

        # PERBAIKAN 5: Quaternion integration
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

        # PERBAIKAN 6: Error state Jacobian F matrix (15x15)
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

        # Measurement model: z = h(x) + v
        z = gps_pos
        h = self.x[0:3]  # Predicted position
        y = z - h  # Innovation

        # Measurement Jacobian H (3x15)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)  # GPS measures position directly

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps_pos

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update error state
        dx = K @ y

        # Apply corrections to nominal state (ESQUA method)
        self.x[0:3] += dx[0:3]  # Position
        self.x[3:6] += dx[3:6]  # Velocity

        # Attitude correction (small angle approximation)
        dtheta = dx[6:9]
        dq = np.array([1, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq = self.normalize_quaternion(dq)

        # Apply quaternion correction
        q = self.x[6:10]
        q_corrected = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
        ])
        self.x[6:10] = self.normalize_quaternion(q_corrected)

        # Bias corrections
        self.x[10:13] += dx[9:12]   # Accel bias
        self.x[13:16] += dx[12:15]  # Gyro bias

        # Update covariance (Joseph form)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_pos @ K.T

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update_gps_velocity(self, gps_vel):
        """Update dengan GPS velocity measurement"""
        if not self.initialized or gps_vel is None:
            return

        # Measurement model
        z = gps_vel
        h = self.x[3:6]  # Predicted velocity
        y = z - h  # Innovation

        # Measurement Jacobian H (3x15)
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)  # GPS measures velocity directly

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps_vel

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update (similar to position update)
        dx = K @ y

        # Apply corrections (same as position update method)
        self.x[0:3] += dx[0:3]
        self.x[3:6] += dx[3:6]

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

        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        # Update covariance
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_vel @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_barometer(self, baro_alt):
        """Update dengan barometer altitude measurement"""
        if not self.initialized or baro_alt is None:
            return

        # Measurement model: altitude = -position_z (NED frame)
        z = np.array([baro_alt])
        h = np.array([-self.x[2]])  # altitude = -z_ned
        y = z - h

        # Measurement Jacobian H (1x15)
        H = np.zeros((1, 15))
        H[0, 2] = -1  # d(altitude)/d(z_ned) = -1

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_baro

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update (same pattern)
        dx = K @ y

        self.x[0:3] += dx[0:3]
        self.x[3:6] += dx[3:6]

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

        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        # Update covariance
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_baro @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_magnetometer(self, mag_body):
        """
        Update dengan magnetometer measurement untuk yaw correction

        Args:
            mag_body: magnetometer reading dalam body frame [mx, my, mz]
        """
        if not self.initialized or mag_body is None:
            return

        # Check if measurement is valid
        mag_norm = np.linalg.norm(mag_body)
        if mag_norm < 0.1 or mag_norm > 2.0:  # Typical earth field is ~0.25-0.65 Gauss
            return

        # Get current rotation matrix
        R_bn = self.quaternion_to_rotation_matrix(self.x[6:10])

        # Expected magnetic field in body frame
        mag_expected_body = R_bn.T @ self.mag_ref_ned

        # Use only horizontal components for yaw correction
        # This is more robust as vertical component is affected by inclination uncertainty

        # Project to horizontal plane in body frame
        # Remove z component (down in body frame)
        mag_h = np.array([mag_body[0], mag_body[1], 0.0])
        exp_h = np.array([mag_expected_body[0], mag_expected_body[1], 0.0])

        # Check if horizontal components are significant
        if np.linalg.norm(mag_h) < 0.1 or np.linalg.norm(exp_h) < 0.1:
            return

        # Normalize horizontal components
        mag_h_norm = mag_h / np.linalg.norm(mag_h)
        exp_h_norm = exp_h / np.linalg.norm(exp_h)

        # Innovation (use only x and y components)
        z = mag_h_norm[:2]
        h = exp_h_norm[:2]
        y = z - h

        # Measurement Jacobian H (2x15) - only x,y components
        # For small angle approximation: δ(R*m) ≈ -[R*m]× * δθ
        H = np.zeros((2, 15))

        # Only yaw affects horizontal magnetic field significantly
        # Simplified Jacobian focusing on yaw
        mag_body_normalized = mag_expected_body / \
            np.linalg.norm(mag_expected_body)

        # Derivative of horizontal mag field w.r.t attitude errors
        # Focus mainly on yaw (z-axis rotation)
        H[0, 8] = mag_body_normalized[1]   # dx/dψ ≈ my
        H[1, 8] = -mag_body_normalized[0]  # dy/dψ ≈ -mx

        # Smaller influence from roll and pitch
        H[0, 6] = -mag_body_normalized[2] * 0.1  # Reduced influence
        H[1, 7] = -mag_body_normalized[2] * 0.1

        # Measurement noise for horizontal components only
        R_mag_h = self.R_mag[:2, :2] * 4.0  # Increase noise for robustness

        # Innovation covariance
        S = H @ self.P @ H.T + R_mag_h

        # Check for positive definite S
        if np.min(np.linalg.eigvals(S)) < 1e-6:
            return

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        # Limit the correction magnitude
        y_limited = np.clip(y, -0.5, 0.5)

        # Update error state
        dx = K @ y_limited

        # Apply corrections with limits
        self.x[0:3] += np.clip(dx[0:3], -0.1, 0.1)     # Position
        self.x[3:6] += np.clip(dx[3:6], -0.5, 0.5)     # Velocity

        # Attitude correction with special care for yaw
        dtheta = dx[6:9]
        # Limit attitude corrections
        dtheta = np.clip(dtheta, -0.1, 0.1)  # ~5.7 degrees max

        # Create error quaternion
        dq = np.array([1, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2])
        dq = self.normalize_quaternion(dq)

        # Apply quaternion correction
        q = self.x[6:10]
        q_corrected = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
        ])
        self.x[6:10] = self.normalize_quaternion(q_corrected)

        # Bias corrections with limits
        self.x[10:13] += np.clip(dx[9:12], -0.01, 0.01)    # Accel bias
        self.x[13:16] += np.clip(dx[12:15], -0.001, 0.001)  # Gyro bias

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_mag_h @ K.T

        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)

        # Add small diagonal to maintain positive definiteness
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
            'attitude_std': np.sqrt(np.diag(self.P[6:9, 6:9]))
        }


def run_improved_position_velocity_ekf_with_mag(csv_file_path, use_magnetometer=True):
    """Main function dengan improved EKF including magnetometer"""

    # Load data
    data = pd.read_csv(csv_file_path)
    print(f"Data loaded: {len(data)} samples")

    # Calculate dt
    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"Sampling time: {dt_mean:.4f} s")

    # Initialize improved EKF
    ekf = ImprovedPositionVelocityEKF(dt=dt_mean)

    # Find first valid GPS data untuk initialization
    init_idx = None
    for i in range(len(data)):
        if data.iloc[i]['gps_available'] == 1:
            init_idx = i
            break

    if init_idx is None:
        print("No GPS data found!")
        return None

    # Initialize EKF
    row = data.iloc[init_idx]
    gps_pos = np.array([row['gps_pos_x'], row['gps_pos_y'], row['gps_pos_z']])
    gps_vel = np.array([row['gps_vel_x'], row['gps_vel_y'], row['gps_vel_z']])
    initial_acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
    initial_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

    # Check initial ground truth attitude
    print(f"\nInitial ground truth attitude:")
    print(f"  Roll: {np.rad2deg(row['true_roll']):.1f} deg")
    print(f"  Pitch: {np.rad2deg(row['true_pitch']):.1f} deg")
    print(f"  Yaw: {np.rad2deg(row['true_yaw']):.1f} deg")

    # Get initial magnetometer if available
    initial_mag = None
    if 'mag_available' in data.columns and row['mag_available'] == 1:
        initial_mag = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
        print(f"\nInitial magnetometer reading: {initial_mag}")
        print(f"Magnetometer magnitude: {np.linalg.norm(initial_mag):.3f}")

    # Pass true yaw for debugging
    true_yaw = row['true_yaw'] if use_magnetometer else None
    ekf.initialize_state(gps_pos, gps_vel, initial_acc,
                         initial_gyro, initial_mag, true_yaw)

    # Process all data
    n_samples = len(data)
    results = {
        'timestamp': [],
        'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [],
        'pos_std': [], 'vel_std': [], 'att_std': []
    }

    # Count sensor updates
    mag_update_count = 0
    gps_update_count = 0
    baro_update_count = 0

    print("Processing data...")
    for i in range(n_samples):
        row = data.iloc[i]

        # IMU data
        accel_body = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro_body = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        # Prediction step
        ekf.predict(accel_body, gyro_body)

        # GPS updates
        if row['gps_available'] == 1:
            gps_pos = np.array(
                [row['gps_pos_x'], row['gps_pos_y'], row['gps_pos_z']])
            gps_vel = np.array(
                [row['gps_vel_x'], row['gps_vel_y'], row['gps_vel_z']])
            ekf.update_gps_position(gps_pos)
            ekf.update_gps_velocity(gps_vel)
            gps_update_count += 1

        # Barometer update
        if row['baro_available'] == 1:
            ekf.update_barometer(row['baro_altitude'])
            baro_update_count += 1

        # Magnetometer update
        if use_magnetometer and 'mag_available' in data.columns and row['mag_available'] == 1:
            mag_body = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
            # Debug: print first few magnetometer updates
            if mag_update_count < 5:
                state = ekf.get_state()
                print(
                    f"\nMag update {mag_update_count}: mag={mag_body}, yaw_est={np.rad2deg(state['attitude_euler'][2]):.1f}°")
            ekf.update_magnetometer(mag_body)
            mag_update_count += 1

        # Store results
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

        if i % 1000 == 0:
            print(f"Processed {i}/{n_samples} ({100*i/n_samples:.1f}%)")

    # Convert to numpy arrays
    for key in ['position', 'velocity', 'attitude', 'acc_bias', 'gyro_bias', 'pos_std', 'vel_std', 'att_std']:
        results[key] = np.array(results[key])

    # Calculate errors
    true_pos = np.column_stack(
        [data['true_pos_x'], data['true_pos_y'], data['true_pos_z']])
    true_vel = np.column_stack(
        [data['true_vel_x'], data['true_vel_y'], data['true_vel_z']])
    true_att = np.column_stack(
        [data['true_roll'], data['true_pitch'], data['true_yaw']])

    pos_error = results['position'] - true_pos
    vel_error = results['velocity'] - true_vel
    att_error = results['attitude'] - true_att

    # Handle angle wrapping for attitude errors
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # Statistics
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    pos_mae = np.mean(np.abs(pos_error), axis=0)
    vel_mae = np.mean(np.abs(vel_error), axis=0)
    att_mae = np.mean(np.abs(att_error), axis=0)

    pos_max_error = np.max(np.abs(pos_error), axis=0)
    vel_max_error = np.max(np.abs(vel_error), axis=0)
    att_max_error = np.max(np.abs(att_error), axis=0)

    print("\n" + "="*80)
    print(
        f"IMPROVED EKF {'WITH' if use_magnetometer else 'WITHOUT'} MAGNETOMETER - COMPLETE RESULTS")
    print("="*80)

    print(f"\nSensor Update Statistics:")
    print(f"  Total samples: {n_samples}")
    print(
        f"  GPS updates: {gps_update_count} ({100*gps_update_count/n_samples:.1f}%)")
    print(
        f"  Barometer updates: {baro_update_count} ({100*baro_update_count/n_samples:.1f}%)")
    print(
        f"  Magnetometer updates: {mag_update_count} ({100*mag_update_count/n_samples:.1f}%)")

    print(f"\n--- POSITION ESTIMATION RESULTS ---")
    print(
        f"  RMSE [X,Y,Z]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(
        f"  MAE  [X,Y,Z]: [{pos_mae[0]:.4f}, {pos_mae[1]:.4f}, {pos_mae[2]:.4f}] m")
    print(
        f"  Max Error [X,Y,Z]: [{pos_max_error[0]:.4f}, {pos_max_error[1]:.4f}, {pos_max_error[2]:.4f}] m")
    print(f"  Total RMSE: {np.linalg.norm(pos_rmse):.4f} m")

    print(f"\n--- VELOCITY ESTIMATION RESULTS ---")
    print(
        f"  RMSE [X,Y,Z]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(
        f"  MAE  [X,Y,Z]: [{vel_mae[0]:.4f}, {vel_mae[1]:.4f}, {vel_mae[2]:.4f}] m/s")
    print(
        f"  Max Error [X,Y,Z]: [{vel_max_error[0]:.4f}, {vel_max_error[1]:.4f}, {vel_max_error[2]:.4f}] m/s")
    print(f"  Total RMSE: {np.linalg.norm(vel_rmse):.4f} m/s")

    print(f"\n--- ATTITUDE ESTIMATION RESULTS ---")
    print(
        f"  RMSE [Roll,Pitch,Yaw]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] deg")
    print(
        f"  MAE  [Roll,Pitch,Yaw]: [{np.rad2deg(att_mae[0]):.3f}, {np.rad2deg(att_mae[1]):.3f}, {np.rad2deg(att_mae[2]):.3f}] deg")
    print(
        f"  Max Error [Roll,Pitch,Yaw]: [{np.rad2deg(att_max_error[0]):.3f}, {np.rad2deg(att_max_error[1]):.3f}, {np.rad2deg(att_max_error[2]):.3f}] deg")

    print(f"\n--- BIAS ESTIMATION RESULTS ---")
    print(
        f"  Final Accelerometer Bias: [{results['acc_bias'][-1][0]:.5f}, {results['acc_bias'][-1][1]:.5f}, {results['acc_bias'][-1][2]:.5f}] m/s²")
    print(
        f"  Final Gyroscope Bias: [{np.rad2deg(results['gyro_bias'][-1][0]):.4f}, {np.rad2deg(results['gyro_bias'][-1][1]):.4f}, {np.rad2deg(results['gyro_bias'][-1][2]):.4f}] deg/s")

    print(f"\n--- FINAL STATE UNCERTAINTIES (2σ) ---")
    print(
        f"  Position: [{2*results['pos_std'][-1][0]:.4f}, {2*results['pos_std'][-1][1]:.4f}, {2*results['pos_std'][-1][2]:.4f}] m")
    print(
        f"  Velocity: [{2*results['vel_std'][-1][0]:.4f}, {2*results['vel_std'][-1][1]:.4f}, {2*results['vel_std'][-1][2]:.4f}] m/s")
    print(f"  Attitude: [{2*np.rad2deg(results['att_std'][-1][0]):.3f}, {2*np.rad2deg(results['att_std'][-1][1]):.3f}, {2*np.rad2deg(results['att_std'][-1][2]):.3f}] deg")

    print("="*80)

    # Add update counts to results
    results['gps_updates'] = gps_update_count
    results['baro_updates'] = baro_update_count
    results['mag_updates'] = mag_update_count

    return ekf, results, data


def plot_ekf_results_with_mag(ekf, results, data):
    """
    Enhanced plotting function including magnetometer effects
    """

    # Extract data
    time = np.array(results['timestamp'])

    # Ground truth data
    true_pos = np.column_stack(
        [data['true_pos_x'], data['true_pos_y'], data['true_pos_z']])
    true_vel = np.column_stack(
        [data['true_vel_x'], data['true_vel_y'], data['true_vel_z']])
    true_attitude = np.column_stack(
        [data['true_roll'], data['true_pitch'], data['true_yaw']])

    # EKF estimates
    est_pos = results['position']
    est_vel = results['velocity']
    est_att = results['attitude']

    # Calculate errors
    pos_error = est_pos - true_pos
    vel_error = est_vel - true_vel
    att_error = est_att - true_attitude

    # Handle angle wrapping for attitude errors
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # =============================================================================
    # PLOT 1: Attitude Estimation with Magnetometer Focus
    # =============================================================================
    fig1 = plt.figure(figsize=(20, 15))
    fig1.suptitle('Attitude Estimation with Magnetometer',
                  fontsize=16, fontweight='bold')

    # Find magnetometer update times
    mag_available = data['mag_available'].values == 1 if 'mag_available' in data.columns else np.zeros_like(
        time, dtype=bool)
    mag_times = time[mag_available]

    att_labels = ['Roll', 'Pitch', 'Yaw']

    for i in range(3):
        # Attitude comparison
        plt.subplot(3, 3, i+1)
        plt.plot(time, np.rad2deg(
            true_attitude[:, i]), 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, np.rad2deg(est_att[:, i]),
                 'r-', linewidth=2, label='EKF Estimate')

        # Mark magnetometer updates (especially important for yaw)
        if len(mag_times) > 0 and i == 2:  # Yaw
            plt.scatter(mag_times[::100], np.rad2deg(est_att[mag_available, i][::100]),
                        c='purple', s=5, alpha=0.5, label='Mag Updates')

        # Plot uncertainty bounds
        att_std = results['att_std']
        plt.fill_between(time,
                         np.rad2deg(est_att[:, i] - 2*att_std[:, i]),
                         np.rad2deg(est_att[:, i] + 2*att_std[:, i]),
                         alpha=0.3, color='red', label='2σ Uncertainty')

        plt.xlabel('Time (s)')
        plt.ylabel(f'{att_labels[i]} (deg)')
        plt.title(f'{att_labels[i]} Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Attitude error
        plt.subplot(3, 3, i+4)
        plt.plot(time, np.rad2deg(att_error[:, i]), 'r-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.fill_between(time,
                         np.rad2deg(-2*att_std[:, i]),
                         np.rad2deg(2*att_std[:, i]),
                         alpha=0.3, color='gray', label='2σ Bounds')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Error (deg)')
        plt.title(f'{att_labels[i]} Error')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error statistics
        plt.subplot(3, 3, i+7)
        plt.hist(np.rad2deg(att_error[:, i]), bins=50,
                 alpha=0.7, color='red', density=True)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)

        mean_error = np.rad2deg(np.mean(att_error[:, i]))
        std_error = np.rad2deg(np.std(att_error[:, i]))
        rmse_error = np.rad2deg(np.sqrt(np.mean(att_error[:, i]**2)))

        plt.axvline(x=mean_error, color='r', linestyle='-',
                    label=f'Mean: {mean_error:.2f}°')
        plt.axvline(x=std_error, color='orange', linestyle='-',
                    label=f'Std: {std_error:.2f}°')
        plt.axvline(x=-std_error, color='orange', linestyle='-')

        plt.xlabel(f'Error (deg)')
        plt.ylabel('Density')
        plt.title(
            f'{att_labels[i]} Error Distribution\nRMSE: {rmse_error:.2f}°')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ekf_attitude_with_magnetometer.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # PLOT 2: Magnetometer Impact Analysis
    # =============================================================================
    if 'mag_available' in data.columns:
        fig2 = plt.figure(figsize=(16, 10))
        fig2.suptitle('Magnetometer Impact on Yaw Estimation',
                      fontsize=16, fontweight='bold')

        # Yaw error over time with magnetometer availability
        ax1 = plt.subplot(211)
        plt.plot(time, np.rad2deg(
            att_error[:, 2]), 'r-', linewidth=1.5, label='Yaw Error')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Highlight magnetometer update periods
        if len(mag_times) > 0:
            for i in range(len(mag_times)-1):
                if mag_times[i+1] - mag_times[i] < 0.05:  # Consecutive updates
                    plt.axvspan(mag_times[i], mag_times[i+1],
                                alpha=0.1, color='purple')

        plt.xlabel('Time (s)')
        plt.ylabel('Yaw Error (deg)')
        plt.title('Yaw Error with Magnetometer Update Periods (purple shading)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Magnetometer measurements
        ax2 = plt.subplot(212)
        mag_data = np.column_stack(
            [data['mag_x'], data['mag_y'], data['mag_z']])
        mag_magnitude = np.linalg.norm(mag_data[mag_available], axis=1)

        if len(mag_times) > 0:
            plt.plot(mag_times, mag_magnitude, 'p-',
                     markersize=2, label='Mag Magnitude')
            plt.xlabel('Time (s)')
            plt.ylabel('Magnetic Field (Gauss)')
            plt.title('Magnetometer Magnitude')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('magnetometer_impact_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    # =============================================================================
    # PLOT 3: Position and Velocity Results
    # =============================================================================
    fig3 = plt.figure(figsize=(20, 10))
    fig3.suptitle('Position and Velocity Estimation Results',
                  fontsize=16, fontweight='bold')

    axes_labels = ['North (X)', 'East (Y)', 'Down (Z)']

    for i in range(3):
        # Position
        plt.subplot(2, 3, i+1)
        plt.plot(time, true_pos[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_pos[:, i], 'r-', linewidth=2, label='EKF Estimate')
        # Removed uncertainty visualization
        plt.xlabel('Time (s)')
        plt.ylabel(f'Position {axes_labels[i]} (m)')
        plt.title(f'Position {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Velocity
        plt.subplot(2, 3, i+4)
        plt.plot(time, true_vel[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_vel[:, i], 'r-', linewidth=2, label='EKF Estimate')
        # Removed uncertainty visualization
        plt.xlabel('Time (s)')
        plt.ylabel(f'Velocity {axes_labels[i]} (m/s)')
        plt.title(f'Velocity {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ekf_position_velocity_results.png',
                dpi=300, bbox_inches='tight')
    plt.show()


# Main function
if __name__ == "__main__":
    # Update with your actual file
    csv_file_path = "logs/all_sensor_data_20250527_072440.csv"

    print("="*80)
    print("IMPROVED POSITION & VELOCITY EKF WITH MAGNETOMETER FOR HEXACOPTER")
    print("="*80)

    # Test with magnetometer
    print("\n--- Testing WITH Magnetometer ---")
    try:
        results = run_improved_position_velocity_ekf_with_mag(
            csv_file_path, use_magnetometer=True)

        if results is not None:
            ekf, results_data, data = results
            print("\nImproved EKF with Magnetometer processing completed successfully!")

            # Plot comprehensive results
            print("\nGenerating comprehensive plots...")
            plot_ekf_results_with_mag(ekf, results_data, data)

            print("\nAll plots have been saved as PNG files:")
            print("- ekf_attitude_with_magnetometer.png")
            print("- magnetometer_impact_analysis.png")
            print("- ekf_position_velocity_results.png")

        else:
            print("EKF processing failed!")

    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file_path}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    # Optionally test without magnetometer for comparison
    print("\n\n--- Testing WITHOUT Magnetometer for comparison ---")
    try:
        results_no_mag = run_improved_position_velocity_ekf_with_mag(
            csv_file_path, use_magnetometer=False)
        if results_no_mag is not None:
            print("\nComparison completed. Check the difference in yaw estimation.")
    except Exception as e:
        print(f"Comparison test failed: {str(e)}")
