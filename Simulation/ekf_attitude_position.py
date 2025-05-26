import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class EnhancedPositionVelocityEKF:
    """
    Enhanced Extended Kalman Filter dengan Safety Checks dan Adaptive Capabilities
    """

    def __init__(self, dt=0.01):
        self.dt = dt
        self.state_dim = 16

        # Initialize state vector
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initialize quaternion as identity [w, x, y, z]

        # Initialize covariance matrix
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 100.0  # Position uncertainty
        self.P[3:6, 3:6] *= 10.0   # Velocity uncertainty
        self.P[6:9, 6:9] *= 0.1    # Attitude uncertainty
        self.P[9:12, 9:12] *= 0.01  # Accel bias uncertainty
        self.P[12:15, 12:15] *= 0.01  # Gyro bias uncertainty

        # Base process noise matrix Q
        self.Q_base = np.zeros((15, 15))
        self.Q_base[0:3, 0:3] = np.eye(3) * (0.01 * self.dt**2)**2
        self.Q_base[3:6, 3:6] = np.eye(3) * (0.1 * self.dt)**2
        self.Q_base[6:9, 6:9] = np.eye(3) * (0.01 * self.dt)**2
        self.Q_base[9:12, 9:12] = np.eye(3) * (1e-4 * self.dt)**2
        self.Q_base[12:15, 12:15] = np.eye(3) * (1e-5 * self.dt)**2

        # Current process noise (adaptive)
        self.Q = self.Q_base.copy()

        # Measurement noise matrices
        self.R_gps_pos = np.eye(3) * 1.0**2
        self.R_gps_vel = np.eye(3) * 0.1**2
        self.R_baro = np.array([[0.5**2]])

        # Constants
        self.g_ned = np.array([0, 0, 9.81])
        self.initialized = False

        # === SAFETY CHECKS PARAMETERS ===
        self.innovation_gate_chi2 = 9.21  # Chi-square 95% confidence for 3 DOF
        self.max_position_innovation = 10.0  # meters
        self.max_velocity_innovation = 5.0   # m/s
        self.max_altitude_innovation = 5.0   # meters
        self.min_gps_accuracy = 20.0         # meters
        self.max_imu_accel = 50.0           # m/s² (5G limit)
        self.max_imu_gyro = 10.0            # rad/s

        # === ADAPTIVE PARAMETERS ===
        self.motion_intensity_window = 10
        self.recent_accels = []
        self.recent_gyros = []
        self.adaptive_factor_min = 0.5
        self.adaptive_factor_max = 3.0

        # === STATISTICS TRACKING ===
        self.gps_rejections = 0
        self.baro_rejections = 0
        self.innovation_stats = []
        self.covariance_resets = 0

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

    # === SAFETY CHECK FUNCTIONS ===

    def check_imu_saturation(self, accel_body, gyro_body):
        """Check for IMU saturation or unrealistic values"""
        accel_mag = np.linalg.norm(accel_body)
        gyro_mag = np.linalg.norm(gyro_body)

        if accel_mag > self.max_imu_accel:
            print(f"WARNING: High acceleration detected: {accel_mag:.2f} m/s²")
            return False

        if gyro_mag > self.max_imu_gyro:
            print(
                f"WARNING: High angular rate detected: {np.rad2deg(gyro_mag):.2f} deg/s")
            return False

        return True

    def check_covariance_health(self):
        """Check covariance matrix health and fix if needed"""
        try:
            # Check positive definiteness
            eigenvals = np.linalg.eigvals(self.P)
            min_eigenval = np.min(eigenvals)

            if min_eigenval <= 0:
                print(
                    f"WARNING: Covariance not positive definite (min eigenval: {min_eigenval:.2e})")
                self.fix_covariance()
                self.covariance_resets += 1
                return False

            # Check for numerical instability (very large values)
            max_diagonal = np.max(np.diag(self.P))
            if max_diagonal > 1e6:
                print(
                    f"WARNING: Large covariance values detected (max: {max_diagonal:.2e})")
                self.fix_covariance()
                self.covariance_resets += 1
                return False

        except np.linalg.LinAlgError:
            print("ERROR: Covariance matrix computation failed")
            self.fix_covariance()
            self.covariance_resets += 1
            return False

        return True

    def fix_covariance(self):
        """Fix corrupted covariance matrix"""
        # Reset to reasonable values
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 10.0   # Position uncertainty
        self.P[3:6, 3:6] *= 1.0    # Velocity uncertainty
        self.P[6:9, 6:9] *= 0.1    # Attitude uncertainty
        self.P[9:12, 9:12] *= 0.01  # Accel bias uncertainty
        self.P[12:15, 12:15] *= 0.01  # Gyro bias uncertainty
        print("Covariance matrix reset to safe values")

    def innovation_gate_check(self, innovation, S, gate_threshold=None):
        """Chi-square innovation gating test"""
        if gate_threshold is None:
            gate_threshold = self.innovation_gate_chi2

        try:
            # Check if S is positive definite
            eigenvals = np.linalg.eigvals(S)
            if np.any(eigenvals <= 0):
                print("WARNING: Innovation covariance matrix not positive definite")
                return False

            # Mahalanobis distance
            S_inv = np.linalg.inv(S)
            mahal_dist = innovation.T @ S_inv @ innovation

            # For scalar case, extract the value
            if np.isscalar(mahal_dist) or mahal_dist.size == 1:
                mahal_dist = float(mahal_dist)
            else:
                mahal_dist = float(mahal_dist[0, 0])

            if mahal_dist > gate_threshold:
                print(
                    f"Innovation gate FAILED: {mahal_dist:.2f} > {gate_threshold:.2f}")
                return False

            # Store statistics
            self.innovation_stats.append(mahal_dist)
            return True

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"ERROR: Innovation covariance computation failed: {e}")
            return False

    def check_gps_quality(self, gps_pos, gps_vel, gps_accuracy=None):
        """Check GPS measurement quality"""
        # Check GPS accuracy if available
        if gps_accuracy is not None and gps_accuracy > self.min_gps_accuracy:
            print(
                f"GPS accuracy too low: {gps_accuracy:.1f}m > {self.min_gps_accuracy:.1f}m")
            return False

        # Check for reasonable position values (only if gps_pos is provided)
        if gps_pos is not None:
            pos_mag = np.linalg.norm(gps_pos)
            if pos_mag > 1e6:  # More than 1000km from origin
                print(f"GPS position unreasonable: {pos_mag:.0f}m")
                return False

        # Check for reasonable velocity values (only if gps_vel is provided)
        if gps_vel is not None:
            vel_mag = np.linalg.norm(gps_vel)
            if vel_mag > 100:  # More than 100 m/s (360 km/h)
                print(f"GPS velocity unreasonable: {vel_mag:.1f}m/s")
                return False

        return True

    # === ADAPTIVE CAPABILITIES ===

    def update_motion_intensity(self, accel_body, gyro_body):
        """Update motion intensity for adaptive process noise"""
        # Store recent measurements
        self.recent_accels.append(np.linalg.norm(accel_body))
        self.recent_gyros.append(np.linalg.norm(gyro_body))

        # Keep only recent history
        if len(self.recent_accels) > self.motion_intensity_window:
            self.recent_accels.pop(0)
            self.recent_gyros.pop(0)

    def compute_adaptive_process_noise(self):
        """Compute adaptive process noise based on motion intensity"""
        if len(self.recent_accels) < 3:
            return self.Q_base.copy()

        # Compute motion intensity indicators
        accel_std = np.std(self.recent_accels)
        gyro_std = np.std(self.recent_gyros)

        # Normalized motion indicators (0 = stationary, 1+ = high motion)
        # Scale based on 2 m/s² reference
        accel_factor = min(accel_std / 2.0, 3.0)
        # Scale based on 0.5 rad/s reference
        gyro_factor = min(gyro_std / 0.5, 3.0)

        # Combined adaptive factor
        motion_factor = max(accel_factor, gyro_factor)
        adaptive_factor = np.clip(1.0 + motion_factor,
                                  self.adaptive_factor_min,
                                  self.adaptive_factor_max)

        # Scale process noise based on motion intensity
        Q_adaptive = self.Q_base.copy()

        # Increase velocity and attitude process noise during high motion
        Q_adaptive[3:6, 3:6] *= adaptive_factor  # Velocity
        Q_adaptive[6:9, 6:9] *= adaptive_factor  # Attitude

        return Q_adaptive

    def initialize_state(self, gps_pos, gps_vel, initial_acc, initial_gyro):
        """Enhanced initialization with safety checks"""

        # Check GPS quality first
        if not self.check_gps_quality(gps_pos, gps_vel):
            print("ERROR: Initial GPS data quality check failed")
            return False

        # Check IMU saturation
        if not self.check_imu_saturation(initial_acc, initial_gyro):
            print("WARNING: Initial IMU data shows saturation, proceeding with caution")

        # Initialize position and velocity from GPS
        self.x[0:3] = gps_pos
        self.x[3:6] = gps_vel

        # Initialize attitude from accelerometer (static assumption)
        acc_norm = initial_acc / np.linalg.norm(initial_acc)

        # Calculate initial roll and pitch (assume yaw = 0)
        roll = np.arctan2(-acc_norm[1], -acc_norm[2])
        pitch = np.arctan2(acc_norm[0], np.sqrt(
            acc_norm[1]**2 + acc_norm[2]**2))
        yaw = 0.0

        # Convert to quaternion
        r = Rotation.from_euler('xyz', [roll, pitch, yaw])
        q_scipy = r.as_quat()  # [x,y,z,w]
        self.x[6:10] = np.array(
            [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # [w,x,y,z]

        # Initialize biases
        self.x[10:13] = np.zeros(3)  # Accel bias
        self.x[13:16] = np.zeros(3)  # Gyro bias

        # Reduce initial uncertainty after good initialization
        self.P[0:3, 0:3] *= 0.1  # Reduce position uncertainty
        self.P[3:6, 3:6] *= 0.1  # Reduce velocity uncertainty

        self.initialized = True
        print(f"Enhanced EKF initialized successfully:")
        print(f"  Position: {self.x[0:3]}")
        print(f"  Velocity: {self.x[3:6]}")
        print(
            f"  Initial attitude (deg): roll={np.rad2deg(roll):.1f}, pitch={np.rad2deg(pitch):.1f}, yaw={np.rad2deg(yaw):.1f}")

        return True

    def predict(self, accel_body, gyro_body):
        """Enhanced prediction step with adaptive process noise"""
        if not self.initialized:
            return

        # Safety check for IMU data
        if not self.check_imu_saturation(accel_body, gyro_body):
            print("Skipping prediction due to IMU saturation")
            return

        # Update motion intensity for adaptive capabilities
        self.update_motion_intensity(accel_body, gyro_body)

        # Get adaptive process noise
        self.Q = self.compute_adaptive_process_noise()

        # Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        q = self.x[6:10]
        acc_bias = self.x[10:13]
        gyro_bias = self.x[13:16]

        # Correct sensor measurements with bias
        accel_corrected = accel_body - acc_bias
        gyro_corrected = gyro_body - gyro_bias

        # Get rotation matrix from quaternion
        R_bn = self.quaternion_to_rotation_matrix(q)

        # Transform acceleration to NED frame and compensate gravity
        accel_ned = R_bn @ accel_corrected + self.g_ned

        # Kinematic integration with RK2
        vel_mid = vel + 0.5 * accel_ned * self.dt
        pos_new = pos + vel_mid * self.dt
        vel_new = vel + accel_ned * self.dt

        # Quaternion integration
        omega_norm = np.linalg.norm(gyro_corrected)
        if omega_norm > 1e-8:
            axis = gyro_corrected / omega_norm
            angle = omega_norm * self.dt

            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])

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

        # Error state Jacobian F matrix (15x15)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = -R_bn @ self.skew_symmetric(accel_corrected) * self.dt
        F[3:6, 9:12] = -R_bn * self.dt
        F[6:9, 6:9] = np.eye(3) - self.skew_symmetric(gyro_corrected) * self.dt
        F[6:9, 12:15] = -np.eye(3) * self.dt

        # Propagate error covariance with adaptive Q
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry

        # Check covariance health
        self.check_covariance_health()

    def update_gps_position(self, gps_pos, gps_accuracy=None):
        """Enhanced GPS position update with safety checks"""
        if not self.initialized or gps_pos is None:
            return False

        # Quality check
        if not self.check_gps_quality(gps_pos, None, gps_accuracy):
            self.gps_rejections += 1
            return False

        # Measurement model
        z = gps_pos
        h = self.x[0:3]
        y = z - h  # Innovation

        # Innovation magnitude check
        innovation_mag = np.linalg.norm(y)
        if innovation_mag > self.max_position_innovation:
            print(f"GPS position innovation too large: {innovation_mag:.2f}m")
            self.gps_rejections += 1
            return False

        # Measurement Jacobian H (3x15)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps_pos

        # Innovation gating
        if not self.innovation_gate_check(y, S):
            self.gps_rejections += 1
            return False

        # Kalman gain with improved numerical stability
        try:
            # Add small regularization to avoid numerical issues
            S_reg = S + np.eye(S.shape[0]) * 1e-9
            K = self.P @ H.T @ np.linalg.inv(S_reg)
        except np.linalg.LinAlgError:
            print("ERROR: Failed to compute Kalman gain for GPS position")
            self.gps_rejections += 1
            return False

        # Update error state
        dx = K @ y

        # Apply corrections to nominal state (ESQUA method)
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

        self.x[10:13] += dx[9:12]   # Accel bias
        self.x[13:16] += dx[12:15]  # Gyro bias

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_pos @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_gps_velocity(self, gps_vel, gps_accuracy=None):
        """Enhanced GPS velocity update with safety checks"""
        if not self.initialized or gps_vel is None:
            return False

        # Quality check
        if not self.check_gps_quality(None, gps_vel, gps_accuracy):
            self.gps_rejections += 1
            return False

        # Measurement model
        z = gps_vel
        h = self.x[3:6]
        y = z - h

        # Innovation magnitude check
        innovation_mag = np.linalg.norm(y)
        if innovation_mag > self.max_velocity_innovation:
            print(
                f"GPS velocity innovation too large: {innovation_mag:.2f}m/s")
            self.gps_rejections += 1
            return False

        # Measurement Jacobian H (3x15)
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps_vel

        # Innovation gating
        if not self.innovation_gate_check(y, S):
            self.gps_rejections += 1
            return False

        try:
            # Add small regularization to avoid numerical issues
            S_reg = S + np.eye(S.shape[0]) * 1e-9
            K = self.P @ H.T @ np.linalg.inv(S_reg)
        except np.linalg.LinAlgError:
            print("ERROR: Failed to compute Kalman gain for GPS velocity")
            self.gps_rejections += 1
            return False

        # Update (similar to position update)
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
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_vel @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_barometer(self, baro_alt):
        """Enhanced barometer update with safety checks"""
        if not self.initialized or baro_alt is None:
            return False

        # Measurement model
        z = np.array([baro_alt])
        h = np.array([-self.x[2]])  # altitude = -z_ned
        y = z - h

        # Innovation magnitude check
        innovation_mag = abs(y[0])
        if innovation_mag > self.max_altitude_innovation:
            print(f"Barometer innovation too large: {innovation_mag:.2f}m")
            self.baro_rejections += 1
            return False

        # Measurement Jacobian H (1x15)
        H = np.zeros((1, 15))
        H[0, 2] = -1

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_baro

        # Single DOF innovation gating (Chi-square 95% = 3.84)
        if not self.innovation_gate_check(y, S, gate_threshold=3.84):
            self.baro_rejections += 1
            return False

        try:
            # Add small regularization to avoid numerical issues
            S_reg = S + np.eye(S.shape[0]) * 1e-9
            K = self.P @ H.T @ np.linalg.inv(S_reg)
        except np.linalg.LinAlgError:
            print("ERROR: Failed to compute Kalman gain for barometer")
            self.baro_rejections += 1
            return False

        # Update
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

        return True

    def get_state(self):
        """Return current state estimates with enhanced diagnostics"""
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
            'gps_rejections': self.gps_rejections,
            'baro_rejections': self.baro_rejections,
            'covariance_resets': self.covariance_resets
        }

    def print_diagnostics(self):
        """Print diagnostic information"""
        print("\n" + "="*50)
        print("ENHANCED EKF DIAGNOSTICS")
        print("="*50)
        print(f"GPS position rejections: {self.gps_rejections}")
        print(f"Barometer rejections: {self.baro_rejections}")
        print(f"Covariance resets: {self.covariance_resets}")

        if len(self.innovation_stats) > 0:
            print(f"Innovation statistics:")
            print(f"  Mean: {np.mean(self.innovation_stats):.2f}")
            print(f"  Std: {np.std(self.innovation_stats):.2f}")
            print(f"  Max: {np.max(self.innovation_stats):.2f}")

        if len(self.recent_accels) > 0:
            print(f"Motion intensity:")
            print(f"  Accel std: {np.std(self.recent_accels):.3f} m/s²")
            print(
                f"  Gyro std: {np.rad2deg(np.std(self.recent_gyros)):.3f} deg/s")


def run_enhanced_position_velocity_ekf(csv_file_path):
    """Main function to run Enhanced EKF"""

    # Load data
    data = pd.read_csv(csv_file_path)
    print(f"Data loaded: {len(data)} samples")

    # Calculate dt
    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"Sampling time: {dt_mean:.4f} s")

    # Initialize Enhanced EKF
    ekf = EnhancedPositionVelocityEKF(dt=dt_mean)

    # Find first valid GPS data for initialization
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

    # Check for NaN values in initialization data
    if (np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel)) or
            np.any(np.isnan(initial_acc)) or np.any(np.isnan(initial_gyro))):
        print("ERROR: Invalid initialization data contains NaN values!")
        return None

    if not ekf.initialize_state(gps_pos, gps_vel, initial_acc, initial_gyro):
        print("Failed to initialize Enhanced EKF!")
        return None

    # Process all data
    n_samples = len(data)
    results = {
        'timestamp': [],
        'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [],
        'pos_std': [], 'vel_std': [], 'att_std': [],
        'gps_updates': [], 'baro_updates': []
    }

    print("Processing data with Enhanced EKF...")
    gps_update_count = 0
    baro_update_count = 0

    for i in range(n_samples):
        row = data.iloc[i]

        # IMU data
        accel_body = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro_body = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        # Check for NaN or invalid values in IMU data
        if np.any(np.isnan(accel_body)) or np.any(np.isnan(gyro_body)):
            print(
                f"WARNING: Invalid IMU data at time {row['timestamp']:.3f}s, skipping...")
            continue

        # Prediction step
        ekf.predict(accel_body, gyro_body)

        # GPS updates with enhanced safety checks
        gps_updated = False
        if row['gps_available'] == 1:
            gps_pos = np.array(
                [row['gps_pos_x'], row['gps_pos_y'], row['gps_pos_z']])
            gps_vel = np.array(
                [row['gps_vel_x'], row['gps_vel_y'], row['gps_vel_z']])

            # Check for NaN or invalid values
            if not (np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel))):
                # Try GPS position update
                if ekf.update_gps_position(gps_pos):
                    gps_update_count += 1
                    gps_updated = True

                # Try GPS velocity update
                if ekf.update_gps_velocity(gps_vel):
                    gps_updated = True
            else:
                print(
                    f"WARNING: Invalid GPS data at time {row['timestamp']:.3f}s")

        # Barometer update with enhanced safety checks
        baro_updated = False
        if row['baro_available'] == 1:
            baro_alt = row['baro_altitude']
            # Check for NaN or invalid values
            if not np.isnan(baro_alt) and np.isfinite(baro_alt):
                if ekf.update_barometer(baro_alt):
                    baro_update_count += 1
                    baro_updated = True
            else:
                print(
                    f"WARNING: Invalid barometer data at time {row['timestamp']:.3f}s")

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
        results['gps_updates'].append(gps_updated)
        results['baro_updates'].append(baro_updated)

        if i % 100 == 0:
            print(f"Processed {i}/{n_samples} ({100*i/n_samples:.1f}%)")

    # Convert to numpy arrays
    for key in ['position', 'velocity', 'attitude', 'acc_bias', 'gyro_bias', 'pos_std', 'vel_std', 'att_std']:
        results[key] = np.array(results[key])

    # Calculate errors
    true_pos = np.column_stack(
        [data['true_pos_x'], data['true_pos_y'], data['true_pos_z']])
    true_vel = np.column_stack(
        [data['true_vel_x'], data['true_vel_y'], data['true_vel_z']])
    true_attitude = np.column_stack(
        [data['true_roll'], data['true_pitch'], data['true_yaw']])

    pos_error = results['position'] - true_pos
    vel_error = results['velocity'] - true_vel
    att_error = results['attitude'] - true_attitude

    # Handle angle wrapping for attitude errors
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # Calculate RMSE
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    # Print comprehensive results
    print("\n" + "="*80)
    print("ENHANCED EKF RESULTS")
    print("="*80)
    print(
        f"Total GPS updates successful: {gps_update_count}/{np.sum(data['gps_available'])} ({100*gps_update_count/np.sum(data['gps_available']):.1f}%)")
    print(
        f"Total Barometer updates successful: {baro_update_count}/{np.sum(data['baro_available'])} ({100*baro_update_count/np.sum(data['baro_available']):.1f}%)")
    print()
    print("POSITION RMSE:")
    print(f"  North (X): {pos_rmse[0]:.4f} m")
    print(f"  East (Y):  {pos_rmse[1]:.4f} m")
    print(f"  Down (Z):  {pos_rmse[2]:.4f} m")
    print(f"  Total:     {np.linalg.norm(pos_rmse):.4f} m")
    print()
    print("VELOCITY RMSE:")
    print(f"  North (X): {vel_rmse[0]:.4f} m/s")
    print(f"  East (Y):  {vel_rmse[1]:.4f} m/s")
    print(f"  Down (Z):  {vel_rmse[2]:.4f} m/s")
    print(f"  Total:     {np.linalg.norm(vel_rmse):.4f} m/s")
    print()
    print("ATTITUDE RMSE:")
    print(f"  Roll:      {np.rad2deg(att_rmse[0]):.3f} deg")
    print(f"  Pitch:     {np.rad2deg(att_rmse[1]):.3f} deg")
    print(f"  Yaw:       {np.rad2deg(att_rmse[2]):.3f} deg")
    print()
    print("FINAL BIAS ESTIMATES:")
    print(
        f"  Accelerometer: [{results['acc_bias'][-1][0]:.4f}, {results['acc_bias'][-1][1]:.4f}, {results['acc_bias'][-1][2]:.4f}] m/s²")
    print(f"  Gyroscope:     [{np.rad2deg(results['gyro_bias'][-1][0]):.4f}, {np.rad2deg(results['gyro_bias'][-1][1]):.4f}, {np.rad2deg(results['gyro_bias'][-1][2]):.4f}] deg/s")

    # Print Enhanced EKF diagnostics
    ekf.print_diagnostics()

    return ekf, results, data, pos_error, vel_error, att_error


def plot_enhanced_ekf_results(ekf, results, data, pos_error, vel_error, att_error):
    """
    Plot comprehensive results from Enhanced EKF estimation vs ground truth
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

    # GPS measurements for comparison
    gps_available = data['gps_available'].values == 1
    gps_times = time[gps_available]
    gps_pos = np.column_stack(
        [data['gps_pos_x'], data['gps_pos_y'], data['gps_pos_z']])[gps_available]
    gps_vel = np.column_stack(
        [data['gps_vel_x'], data['gps_vel_y'], data['gps_vel_z']])[gps_available]

    # Barometer data
    baro_available = data['baro_available'].values == 1
    baro_times = time[baro_available]
    baro_alt = data['baro_altitude'].values[baro_available]

    print("Creating Enhanced EKF plots...")

    # =============================================================================
    # PLOT 1: Position Estimation Results
    # =============================================================================
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle('Enhanced EKF: Position Estimation Results',
                  fontsize=16, fontweight='bold')

    axes_labels = ['North (X)', 'East (Y)', 'Down (Z)']

    for i in range(3):
        # Position comparison
        plt.subplot(3, 3, i+1)
        plt.plot(time, true_pos[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_pos[:, i], 'r-', linewidth=2, label='Enhanced EKF')

        # Plot uncertainty bounds
        pos_std = results['pos_std']
        plt.fill_between(time,
                         est_pos[:, i] - 2*pos_std[:, i],
                         est_pos[:, i] + 2*pos_std[:, i],
                         alpha=0.3, color='red', label='2σ Uncertainty')

        # GPS measurements
        if len(gps_times) > 0:
            plt.scatter(gps_times, gps_pos[:, i], c='blue', s=20, alpha=0.7,
                        marker='o', label='GPS Measurements')

        plt.xlabel('Time (s)')
        plt.ylabel(f'Position {axes_labels[i]} (m)')
        plt.title(f'Position {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Position error
        plt.subplot(3, 3, i+4)
        plt.plot(time, pos_error[:, i], 'r-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.fill_between(time, -2*pos_std[:, i], 2*pos_std[:, i],
                         alpha=0.3, color='gray', label='2σ Bounds')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Error {axes_labels[i]} (m)')
        plt.title(f'Position Error {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error histogram
        plt.subplot(3, 3, i+7)
        plt.hist(pos_error[:, i], bins=50, alpha=0.7,
                 color='red', density=True)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(pos_error[:, i]), color='r', linestyle='-',
                    label=f'Mean: {np.mean(pos_error[:, i]):.3f}m')
        plt.axvline(x=np.std(pos_error[:, i]), color='orange', linestyle='-',
                    label=f'Std: {np.std(pos_error[:, i]):.3f}m')
        plt.axvline(x=-np.std(pos_error[:, i]), color='orange', linestyle='-')
        plt.xlabel(f'Error {axes_labels[i]} (m)')
        plt.ylabel('Density')
        plt.title(f'Error Distribution {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # PLOT 2: Velocity Estimation Results
    # =============================================================================
    fig2 = plt.figure(figsize=(20, 12))
    fig2.suptitle('Enhanced EKF: Velocity Estimation Results',
                  fontsize=16, fontweight='bold')

    for i in range(3):
        # Velocity comparison
        plt.subplot(3, 3, i+1)
        plt.plot(time, true_vel[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_vel[:, i], 'r-', linewidth=2, label='Enhanced EKF')

        # Plot uncertainty bounds
        vel_std = results['vel_std']
        plt.fill_between(time,
                         est_vel[:, i] - 2*vel_std[:, i],
                         est_vel[:, i] + 2*vel_std[:, i],
                         alpha=0.3, color='red', label='2σ Uncertainty')

        # GPS velocity measurements
        if len(gps_times) > 0:
            plt.scatter(gps_times, gps_vel[:, i], c='blue', s=20, alpha=0.7,
                        marker='s', label='GPS Velocity')

        plt.xlabel('Time (s)')
        plt.ylabel(f'Velocity {axes_labels[i]} (m/s)')
        plt.title(f'Velocity {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Velocity error
        plt.subplot(3, 3, i+4)
        plt.plot(time, vel_error[:, i], 'r-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.fill_between(time, -2*vel_std[:, i], 2*vel_std[:, i],
                         alpha=0.3, color='gray', label='2σ Bounds')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Error {axes_labels[i]} (m/s)')
        plt.title(f'Velocity Error {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error histogram
        plt.subplot(3, 3, i+7)
        plt.hist(vel_error[:, i], bins=50, alpha=0.7,
                 color='red', density=True)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(vel_error[:, i]), color='r', linestyle='-',
                    label=f'Mean: {np.mean(vel_error[:, i]):.3f}m/s')
        plt.axvline(x=np.std(vel_error[:, i]), color='orange', linestyle='-',
                    label=f'Std: {np.std(vel_error[:, i]):.3f}m/s')
        plt.axvline(x=-np.std(vel_error[:, i]), color='orange', linestyle='-')
        plt.xlabel(f'Error {axes_labels[i]} (m/s)')
        plt.ylabel('Density')
        plt.title(f'Error Distribution {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # PLOT 3: Attitude Estimation Results
    # =============================================================================
    fig3 = plt.figure(figsize=(20, 12))
    fig3.suptitle('Enhanced EKF: Attitude Estimation Results',
                  fontsize=16, fontweight='bold')

    att_labels = ['Roll', 'Pitch', 'Yaw']

    for i in range(3):
        # Attitude comparison (in degrees)
        plt.subplot(3, 3, i+1)
        plt.plot(time, np.rad2deg(
            true_attitude[:, i]), 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, np.rad2deg(est_att[:, i]),
                 'r-', linewidth=2, label='Enhanced EKF')

        # Plot uncertainty bounds
        att_std = results['att_std']
        plt.fill_between(time,
                         np.rad2deg(est_att[:, i] - 2*att_std[:, i]),
                         np.rad2deg(est_att[:, i] + 2*att_std[:, i]),
                         alpha=0.3, color='red', label='2σ Uncertainty')

        plt.xlabel('Time (s)')
        plt.ylabel(f'{att_labels[i]} (deg)')
        plt.title(f'Attitude {att_labels[i]}')
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
        plt.ylabel(f'Error {att_labels[i]} (deg)')
        plt.title(f'Attitude Error {att_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error histogram
        plt.subplot(3, 3, i+7)
        plt.hist(np.rad2deg(att_error[:, i]), bins=50,
                 alpha=0.7, color='red', density=True)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=np.rad2deg(np.mean(att_error[:, i])), color='r', linestyle='-',
                    label=f'Mean: {np.rad2deg(np.mean(att_error[:, i])):.2f}°')
        plt.axvline(x=np.rad2deg(np.std(att_error[:, i])), color='orange', linestyle='-',
                    label=f'Std: {np.rad2deg(np.std(att_error[:, i])):.2f}°')
        plt.axvline(
            x=np.rad2deg(-np.std(att_error[:, i])), color='orange', linestyle='-')
        plt.xlabel(f'Error {att_labels[i]} (deg)')
        plt.ylabel('Density')
        plt.title(f'Error Distribution {att_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # PLOT 4: 3D Trajectory and Performance Summary
    # =============================================================================
    fig4 = plt.figure(figsize=(16, 12))

    # 3D trajectory plot
    ax1 = fig4.add_subplot(221, projection='3d')

    # Plot trajectories
    ax1.plot(true_pos[:, 0], true_pos[:, 1], -true_pos[:, 2],
             'g-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(est_pos[:, 0], est_pos[:, 1], -est_pos[:, 2],
             'r-', linewidth=2, label='Enhanced EKF', alpha=0.8)

    # GPS measurements
    if len(gps_times) > 0:
        ax1.scatter(gps_pos[:, 0], gps_pos[:, 1], -gps_pos[:, 2],
                    c='blue', s=30, alpha=0.6, label='GPS Measurements')

    # Mark start and end points
    ax1.scatter([true_pos[0, 0]], [true_pos[0, 1]], [-true_pos[0, 2]],
                c='green', s=100, marker='o', label='Start')
    ax1.scatter([true_pos[-1, 0]], [true_pos[-1, 1]], [-true_pos[-1, 2]],
                c='red', s=100, marker='s', label='End')

    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Up (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)

    # Top view (X-Y plane)
    ax2 = fig4.add_subplot(222)
    ax2.plot(true_pos[:, 0], true_pos[:, 1], 'g-',
             linewidth=3, label='Ground Truth')
    ax2.plot(est_pos[:, 0], est_pos[:, 1], 'r-',
             linewidth=2, label='Enhanced EKF')

    if len(gps_times) > 0:
        ax2.scatter(gps_pos[:, 0], gps_pos[:, 1],
                    c='blue', s=20, alpha=0.6, label='GPS')

    ax2.scatter([true_pos[0, 0]], [true_pos[0, 1]],
                c='green', s=100, marker='o', label='Start')
    ax2.scatter([true_pos[-1, 0]], [true_pos[-1, 1]],
                c='red', s=100, marker='s', label='End')

    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('East (m)')
    ax2.set_title('Top View (N-E Plane)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # Altitude comparison with barometer
    ax3 = fig4.add_subplot(223)
    ax3.plot(time, -true_pos[:, 2], 'g-', linewidth=2, label='True Altitude')
    ax3.plot(time, -est_pos[:, 2], 'r-', linewidth=2,
             label='Enhanced EKF Altitude')

    if len(baro_times) > 0:
        ax3.scatter(baro_times, baro_alt, c='purple',
                    s=15, alpha=0.7, label='Barometer')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Comparison')
    ax3.legend()
    ax3.grid(True)

    # Error magnitude vs time
    ax4 = fig4.add_subplot(224)
    pos_error_mag = np.linalg.norm(pos_error, axis=1)
    vel_error_mag = np.linalg.norm(vel_error, axis=1)
    att_error_mag = np.linalg.norm(att_error, axis=1)

    ax4.plot(time, pos_error_mag, 'r-', linewidth=2, label='Position Error')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time, vel_error_mag, 'b-',
                  linewidth=2, label='Velocity Error')
    ax4_twin.plot(time, np.rad2deg(att_error_mag), 'g-',
                  linewidth=2, label='Attitude Error')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error Magnitude (m)', color='red')
    ax4_twin.set_ylabel('Velocity (m/s) / Attitude (deg) Error', color='blue')
    ax4.set_title('Error Magnitude vs Time')
    ax4.grid(True)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file_path = "logs/all_sensor_data_20250523_040556.csv"

    print("="*80)
    print("ENHANCED POSITION & VELOCITY EKF FOR HEXACOPTER")
    print("="*80)

    results = run_enhanced_position_velocity_ekf(csv_file_path)

    if results is not None:
        ekf, results_data, data, pos_error, vel_error, att_error = results
        print("\nEnhanced EKF processing completed successfully!")

        # Plot comprehensive results
        print("\nGenerating comprehensive plots...")
        plot_enhanced_ekf_results(
            ekf, results_data, data, pos_error, vel_error, att_error)

        print("\nEnhanced EKF analysis completed!")

    else:
        print("Enhanced EKF processing failed!")
