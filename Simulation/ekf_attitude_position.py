import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class ImprovedPositionVelocityEKF:
    """
    Extended Kalman Filter dengan attitude estimation untuk posisi dan velocity

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
        self.P[6:9, 6:9] *= 0.1    # Attitude uncertainty
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

        # Constants
        self.g_ned = np.array([0, 0, 9.81])  # Gravity in NED frame
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

    def initialize_state(self, gps_pos, gps_vel, initial_acc, initial_gyro):
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
        yaw = 0.0

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

# Update run function untuk menggunakan improved EKF


def run_improved_position_velocity_ekf(csv_file_path):
    """Main function dengan improved EKF"""

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

    ekf.initialize_state(gps_pos, gps_vel, initial_acc, initial_gyro)

    # Process all data
    n_samples = len(data)
    results = {
        'timestamp': [],
        'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [],
        'pos_std': [], 'vel_std': [], 'att_std': []
    }

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

        # Barometer update
        if row['baro_available'] == 1:
            ekf.update_barometer(row['baro_altitude'])

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

    pos_error = results['position'] - true_pos
    vel_error = results['velocity'] - true_vel

    # Statistics
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))

    print("\n" + "="*60)
    print("IMPROVED EKF RESULTS")
    print("="*60)
    print(
        f"Position RMSE [X,Y,Z]: [{pos_rmse[0]:.3f}, {pos_rmse[1]:.3f}, {pos_rmse[2]:.3f}] m")
    print(
        f"Velocity RMSE [X,Y,Z]: [{vel_rmse[0]:.3f}, {vel_rmse[1]:.3f}, {vel_rmse[2]:.3f}] m/s")
    print(f"Final acc bias: {results['acc_bias'][-1]}")
    print(f"Final gyro bias: {np.rad2deg(results['gyro_bias'][-1])} deg/s")

    return ekf, results, data


def plot_ekf_results(ekf, results, data):
    """
    Plot comprehensive results dari EKF estimation vs ground truth

    Args:
        ekf: EKF object
        results: Dictionary hasil estimasi dari run_improved_position_velocity_ekf
        data: DataFrame data sensor asli
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

    # Calculate errors
    pos_error = est_pos - true_pos
    vel_error = est_vel - true_vel
    att_error = est_att - true_attitude

    # Handle angle wrapping for attitude errors
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    print("Creating comprehensive EKF plots...")

    # =============================================================================
    # PLOT 1: Position Estimation and Comparison
    # =============================================================================
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle('Position Estimation Results',
                  fontsize=16, fontweight='bold')

    axes_labels = ['North (X)', 'East (Y)', 'Down (Z)']

    for i in range(3):
        # Position comparison
        plt.subplot(3, 3, i+1)
        plt.plot(time, true_pos[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_pos[:, i], 'r-', linewidth=2, label='EKF Estimate')

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

        # Error statistics histogram
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
    plt.savefig('ekf_position_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # PLOT 2: Velocity Estimation and Comparison
    # =============================================================================
    fig2 = plt.figure(figsize=(20, 12))
    fig2.suptitle('Velocity Estimation Results',
                  fontsize=16, fontweight='bold')

    for i in range(3):
        # Velocity comparison
        plt.subplot(3, 3, i+1)
        plt.plot(time, true_vel[:, i], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, est_vel[:, i], 'r-', linewidth=2, label='EKF Estimate')

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

        # Error statistics histogram
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
    plt.savefig('ekf_velocity_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # PLOT 3: Attitude Estimation and Comparison
    # =============================================================================
    fig3 = plt.figure(figsize=(20, 12))
    fig3.suptitle('Attitude Estimation Results',
                  fontsize=16, fontweight='bold')

    att_labels = ['Roll', 'Pitch', 'Yaw']

    for i in range(3):
        # Attitude comparison (in degrees)
        plt.subplot(3, 3, i+1)
        plt.plot(time, np.rad2deg(
            true_attitude[:, i]), 'g-', linewidth=2, label='Ground Truth')
        plt.plot(time, np.rad2deg(est_att[:, i]),
                 'r-', linewidth=2, label='EKF Estimate')

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

        # Error statistics histogram
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
    plt.savefig('ekf_attitude_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # PLOT 4: Bias Estimation
    # =============================================================================
    fig4 = plt.figure(figsize=(16, 10))
    fig4.suptitle('Sensor Bias Estimation', fontsize=16, fontweight='bold')

    # Accelerometer bias
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.plot(time, results['acc_bias'][:, i], 'b-',
                 linewidth=2, label='Acc Bias Estimate')

        # Uncertainty bounds
        plt.fill_between(time,
                         results['acc_bias'][:, i] -
                         2*np.sqrt(ekf.P[9+i, 9+i]),
                         results['acc_bias'][:, i] +
                         2*np.sqrt(ekf.P[9+i, 9+i]),
                         alpha=0.3, color='blue', label='2σ Uncertainty')

        plt.xlabel('Time (s)')
        plt.ylabel(f'Acc Bias {axes_labels[i]} (m/s²)')
        plt.title(f'Accelerometer Bias {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Gyroscope bias (in deg/s)
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.plot(time, np.rad2deg(
            results['gyro_bias'][:, i]), 'r-', linewidth=2, label='Gyro Bias Estimate')

        # Uncertainty bounds
        plt.fill_between(time,
                         np.rad2deg(results['gyro_bias'][:, i] -
                                    2*np.sqrt(ekf.P[12+i, 12+i])),
                         np.rad2deg(results['gyro_bias'][:, i] +
                                    2*np.sqrt(ekf.P[12+i, 12+i])),
                         alpha=0.3, color='red', label='2σ Uncertainty')

        plt.xlabel('Time (s)')
        plt.ylabel(f'Gyro Bias {axes_labels[i]} (deg/s)')
        plt.title(f'Gyroscope Bias {axes_labels[i]}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ekf_bias_estimation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # PLOT 5: 3D Trajectory Comparison
    # =============================================================================
    fig5 = plt.figure(figsize=(16, 12))

    # 3D trajectory plot
    ax1 = fig5.add_subplot(221, projection='3d')

    # Plot trajectories (convert Z to altitude for better visualization)
    ax1.plot(true_pos[:, 0], true_pos[:, 1], -true_pos[:, 2],
             'g-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(est_pos[:, 0], est_pos[:, 1], -est_pos[:, 2],
             'r-', linewidth=2, label='EKF Estimate', alpha=0.8)

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
    ax2 = fig5.add_subplot(222)
    ax2.plot(true_pos[:, 0], true_pos[:, 1], 'g-',
             linewidth=3, label='Ground Truth')
    ax2.plot(est_pos[:, 0], est_pos[:, 1], 'r-',
             linewidth=2, label='EKF Estimate')

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
    ax3 = fig5.add_subplot(223)
    ax3.plot(time, -true_pos[:, 2], 'g-', linewidth=2, label='True Altitude')
    ax3.plot(time, -est_pos[:, 2], 'r-', linewidth=2, label='EKF Altitude')

    if len(baro_times) > 0:
        ax3.scatter(baro_times, baro_alt, c='purple',
                    s=15, alpha=0.7, label='Barometer')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Comparison')
    ax3.legend()
    ax3.grid(True)

    # Position error magnitude
    ax4 = fig5.add_subplot(224)
    pos_error_mag = np.linalg.norm(pos_error, axis=1)
    vel_error_mag = np.linalg.norm(vel_error, axis=1)

    ax4.plot(time, pos_error_mag, 'r-', linewidth=2, label='Position Error')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time, vel_error_mag, 'b-',
                  linewidth=2, label='Velocity Error')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error Magnitude (m)', color='red')
    ax4_twin.set_ylabel('Velocity Error Magnitude (m/s)', color='blue')
    ax4.set_title('Error Magnitude vs Time')
    ax4.grid(True)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('ekf_3d_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================================
    # PLOT 6: Performance Metrics Summary
    # =============================================================================
    fig6, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig6.suptitle('EKF Performance Summary', fontsize=16, fontweight='bold')

    # RMSE comparison
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    categories = ['X/Roll', 'Y/Pitch', 'Z/Yaw']
    x_pos = np.arange(len(categories))

    # Position RMSE
    bars1 = ax1.bar(x_pos, pos_rmse, alpha=0.7, color=['red', 'green', 'blue'])
    ax1.set_xlabel('Axes')
    ax1.set_ylabel('RMSE (m)')
    ax1.set_title('Position RMSE')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(pos_rmse),
                 f'{height:.3f}m', ha='center', va='bottom')

    # Velocity RMSE
    bars2 = ax2.bar(x_pos, vel_rmse, alpha=0.7, color=['red', 'green', 'blue'])
    ax2.set_xlabel('Axes')
    ax2.set_ylabel('RMSE (m/s)')
    ax2.set_title('Velocity RMSE')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.grid(True, alpha=0.3)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(vel_rmse),
                 f'{height:.3f}m/s', ha='center', va='bottom')

    # Attitude RMSE (in degrees)
    bars3 = ax3.bar(x_pos, np.rad2deg(att_rmse), alpha=0.7,
                    color=['red', 'green', 'blue'])
    ax3.set_xlabel('Axes')
    ax3.set_ylabel('RMSE (deg)')
    ax3.set_title('Attitude RMSE')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.grid(True, alpha=0.3)

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(np.rad2deg(att_rmse)),
                 f'{height:.2f}°', ha='center', va='bottom')

    # GPS availability and update frequency
    gps_updates = np.sum(gps_available)
    baro_updates = np.sum(baro_available) if len(baro_times) > 0 else 0
    total_samples = len(time)

    update_types = ['IMU', 'GPS', 'Barometer']
    update_counts = [total_samples, gps_updates, baro_updates]
    update_rates = [100, gps_updates/(time[-1]-time[0]),
                    baro_updates/(time[-1]-time[0]) if baro_updates > 0 else 0]

    bars4 = ax4.bar(update_types, update_rates, alpha=0.7,
                    color=['orange', 'blue', 'purple'])
    ax4.set_ylabel('Update Rate (Hz)')
    ax4.set_title('Sensor Update Rates')
    ax4.grid(True, alpha=0.3)

    for i, (bar, count) in enumerate(zip(bars4, update_counts)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(update_rates),
                 f'{height:.1f} Hz\n({count} updates)', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('ekf_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("EKF PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total simulation time: {time[-1]-time[0]:.2f} seconds")
    print(f"Total samples processed: {len(time)}")
    print(
        f"GPS updates: {gps_updates} ({100*gps_updates/len(time):.1f}% availability)")
    print(f"Barometer updates: {baro_updates}")
    print()
    print("Position RMSE:")
    for i, label in enumerate(['North (X)', 'East (Y)', 'Down (Z)']):
        print(f"  {label}: {pos_rmse[i]:.4f} m")
    print()
    print("Velocity RMSE:")
    for i, label in enumerate(['North (X)', 'East (Y)', 'Down (Z)']):
        print(f"  {label}: {vel_rmse[i]:.4f} m/s")
    print()
    print("Attitude RMSE:")
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
        print(f"  {label}: {np.rad2deg(att_rmse[i]):.3f} degrees")
    print()
    print("Final Bias Estimates:")
    print(
        f"  Accelerometer: [{results['acc_bias'][-1][0]:.4f}, {results['acc_bias'][-1][1]:.4f}, {results['acc_bias'][-1][2]:.4f}] m/s²")
    print(f"  Gyroscope: [{np.rad2deg(results['gyro_bias'][-1][0]):.4f}, {np.rad2deg(results['gyro_bias'][-1][1]):.4f}, {np.rad2deg(results['gyro_bias'][-1][2]):.4f}] deg/s")
    print("="*80)


# Update main function untuk include plotting
if __name__ == "__main__":
    csv_file_path = "logs/all_sensor_data_20250523_040556.csv"

    print("="*60)
    print("IMPROVED POSITION & VELOCITY EKF FOR HEXACOPTER")
    print("="*60)

    results = run_improved_position_velocity_ekf(csv_file_path)

    if results is not None:
        ekf, results_data, data = results
        print("\nImproved EKF processing completed successfully!")

        # Plot comprehensive results
        print("\nGenerating comprehensive plots...")
        plot_ekf_results(ekf, results_data, data)

        print("\nAll plots have been saved as PNG files:")
        print("- ekf_position_results.png")
        print("- ekf_velocity_results.png")
        print("- ekf_attitude_results.png")
        print("- ekf_bias_estimation.png")
        print("- ekf_3d_trajectory.png")
        print("- ekf_performance_summary.png")

    else:
        print("EKF processing failed!")
