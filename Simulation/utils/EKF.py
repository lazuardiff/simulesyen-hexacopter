import numpy as np
from scipy.spatial.transform import Rotation


class AttitudeEstimator:
    """
    Implementasi Robust dari Extended Kalman Filter untuk estimasi attitude
    dengan algoritma TRIAD untuk inisialisasi dan koreksi
    """

    def __init__(self, dt, gyro_noise_matrix, acc_noise_matrix, mag_noise_matrix, gyro_bias_init=None):
        """
        Inisialisasi estimator attitude

        Parameters:
        - dt: time step dalam detik
        - gyro_noise: parameter noise gyroscope (rad/s)
        - acc_noise: parameter noise accelerometer
        - mag_noise: parameter noise magnetometer
        - gyro_bias_init: nilai awal bias gyro (opsional)
        """
        # Konstanta dan parameter
        self.dt = dt
        if isinstance(gyro_noise_matrix, (list, tuple, np.ndarray)) and len(gyro_noise_matrix) == 3:
            self.gyro_noise = np.diag(gyro_noise_matrix) * 0.5
        else:
            self.gyro_noise = gyro_noise_matrix
        if isinstance(acc_noise_matrix, (list, tuple, np.ndarray)) and len(acc_noise_matrix) == 3:
            self.acc_noise = np.diag(acc_noise_matrix) * 0.25
        else:
            self.acc_noise = acc_noise_matrix
        if isinstance(mag_noise_matrix, (list, tuple, np.ndarray)) and len(mag_noise_matrix) == 3:
            self.mag_noise = np.diag(mag_noise_matrix) * 0.25
        else:
            self.mag_noise = mag_noise_matrix
        self.g = 9.81  # Percepatan gravitasi (m/s^2)
        self.epsilon = 1e-8  # Nilai kecil untuk stabilitas numerik

        # State: [quaternion(4), gyro_bias(3)]
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Inisialisasi quaternion sebagai identitas [w, x, y, z]

        # Inisialisasi bias gyro jika diberikan
        if gyro_bias_init is not None:
            self.x[4:7] = gyro_bias_init

        # Vektor referensi untuk accelerometer dan magnetometer di frame bumi
        # Arah gravitasi di frame bumi (unit vector)
        self.acc_ref = np.array([0, 0, 1.0])
        # Arah utara magnetik (akan diupdate saat inisialisasi)
        self.mag_ref = np.array([1.0, 0, 0])

        # Matriks kovariansi
        # Kovariansi untuk [delta_rot(3), bias_gyro(3)]
        self.P = np.zeros((6, 6))
        # Ketidakpastian awal dalam orientasi
        self.P[:3, :3] = np.eye(3) * 0.005
        # Ketidakpastian awal dalam bias gyro
        self.P[3:, 3:] = np.eye(3) * 0.0005

        # Matriks kovariansi noise proses
        self.Q = np.zeros((6, 6))
        # Gunakan matriks noise gyro yang sebenarnya
        self.Q[:3, :3] = self.gyro_noise * (self.dt**2) * 0.8
        # Matriks bias gyro bisa juga diubah menjadi parameter input
        self.Q[3:, 3:] = np.eye(3) * 1e-9

        # Matriks transformasi untuk quaternion
        self.quat_corr_mat = np.zeros((4, 3))

        # Flag inisialisasi
        self.initialized = False

        # Variabel tracking untuk reset filter
        self.consecutive_large_innovations = 0
        self.max_consecutive_innovations = 5
        self.innovation_threshold = 0.5

        # Vektor gravitasi dan medan magnet dalam body frame
        self.gravity_body = None
        self.mag_body = None

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix (DCM)
        q = [w, x, y, z]
        """
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]]
                               )  # scipy uses [x, y, z, w]
        return r.as_matrix()

    def triad_algorithm(self, v1_body, v2_body, v1_ref, v2_ref):
        """
        TRIAD algorithm to determine attitude based on two vector observations

        Parameters:
        - v1_body, v2_body: Two vectors measured in body frame
        - v1_ref, v2_ref: Corresponding reference vectors in reference frame

        Returns:
        - Rotation matrix from body to reference frame
        """
        # Normalize all vectors
        v1_body = v1_body / np.linalg.norm(v1_body)
        v2_body = v2_body / np.linalg.norm(v2_body)
        v1_ref = v1_ref / np.linalg.norm(v1_ref)
        v2_ref = v2_ref / np.linalg.norm(v2_ref)

        # Construct orthogonal unit vectors in body frame
        t1_body = v1_body
        t2_body = np.cross(v1_body, v2_body)
        t2_body = t2_body / np.linalg.norm(t2_body)
        t3_body = np.cross(t1_body, t2_body)

        # Construct orthogonal unit vectors in reference frame
        t1_ref = v1_ref
        t2_ref = np.cross(v1_ref, v2_ref)
        t2_ref = t2_ref / np.linalg.norm(t2_ref)
        t3_ref = np.cross(t1_ref, t2_ref)

        # Construct rotation matrices
        R_body = np.column_stack((t1_body, t2_body, t3_body))
        R_ref = np.column_stack((t1_ref, t2_ref, t3_ref))

        # Compute rotation matrix from body to reference frame
        R_body_to_ref = R_ref @ R_body.T

        return R_body_to_ref

    def compute_quaternion_correction_matrix(self, q):
        """
        Compute matrix that maps angular velocity to quaternion derivative
        Used for error state propagation
        """
        # q = [w, x, y, z]
        self.quat_corr_mat[0, :] = [-q[1], -q[2], -q[3]]
        self.quat_corr_mat[1, :] = [q[0], -q[3], q[2]]
        self.quat_corr_mat[2, :] = [q[3], q[0], -q[1]]
        self.quat_corr_mat[3, :] = [-q[2], q[1], q[0]]

        return 0.5 * self.quat_corr_mat

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        q = [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def quaternion_conjugate(self, q):
        """
        Compute conjugate of quaternion
        q = [w, x, y, z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def rotate_vector_by_quaternion(self, v, q):
        """
        Rotate vector v by quaternion q
        v = [x, y, z]
        q = [w, x, y, z]
        """
        # Convert vector to pure quaternion
        v_quat = np.array([0, v[0], v[1], v[2]])

        # q * v * q^-1
        q_conj = self.quaternion_conjugate(q)
        v_rotated = self.quaternion_multiply(
            self.quaternion_multiply(q, v_quat), q_conj)

        # Return vector part
        return v_rotated[1:]

    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        q = [w, x, y, z]
        """
        # Convert to scipy format [x, y, z, w]
        q_scipy = [q[1], q[2], q[3], q[0]]

        # Convert to euler
        r = Rotation.from_quat(q_scipy)
        euler = r.as_euler('xyz')

        return euler

    def euler_to_quaternion(self, euler):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion
        Returns quaternion in format [w, x, y, z]
        """
        r = Rotation.from_euler('xyz', euler)
        q_scipy = r.as_quat()  # [x, y, z, w]

        # Convert to w, x, y, z format
        q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        return q

    def normalize_quaternion(self, q):
        """
        Normalize quaternion to unit length
        """
        norm = np.linalg.norm(q)
        if norm < self.epsilon:
            return np.array([1, 0, 0, 0])  # Default to identity if near zero
        return q / norm

    def normalize_vector(self, v):
        """
        Normalize vector to unit length
        """
        norm = np.linalg.norm(v)
        if norm < self.epsilon:
            return v  # Return original if near zero
        return v / norm

    def initialize(self, accel, mag):
        """
        Initialize orientation using TRIAD algorithm

        Parameters:
        - accel: accelerometer measurement [ax, ay, az] (m/s^2)
        - mag: magnetometer measurement [mx, my, mz]
        """
        if np.linalg.norm(accel) < self.epsilon or np.linalg.norm(mag) < self.epsilon:
            return False

        # Store normalized measurements as reference
        # Negative because accel measures -g
        self.gravity_body = -self.normalize_vector(accel)
        self.mag_body = self.normalize_vector(mag)

        # Check that vectors aren't parallel
        cross_prod = np.cross(self.gravity_body, self.mag_body)
        if np.linalg.norm(cross_prod) < 0.1:  # Threshold for near-parallel vectors
            return False

        try:
            # Apply TRIAD algorithm to get rotation from body to reference
            # First, need to define reference vectors in the inertial frame
            # gravity points down in reference frame
            grav_ref = np.array([0, 0, 1.0])

            # Rough magnetic field reference (pointing north, horizontal plane)
            # For better accuracy this should be set based on your location
            mag_ref = np.array([1.0, 0, 0])

            # Get rotation matrix using TRIAD
            R_body_to_ref = self.triad_algorithm(
                self.gravity_body, self.mag_body, grav_ref, mag_ref)

            # Convert to quaternion and set initial state
            r = Rotation.from_matrix(R_body_to_ref)
            q_scipy = r.as_quat()  # [x, y, z, w]

            # Convert to w, x, y, z format
            q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

            # Set quaternion in state vector
            self.x[0:4] = q

            # Reset covariance
            self.P[:3, :3] = np.eye(3) * 0.01  # Orientation uncertainty

            self.initialized = True
            return True

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            return False

    def predict(self, gyro, dt=None, gyro_in_deg=True):
        """
        Predict step of EKF using gyroscope data

        Parameters:
        - gyro: angular rates [p, q, r] in deg/s (default) or rad/s
        - dt: time step in seconds (optional)
        - gyro_in_deg: True if input is in deg/s, False if in rad/s
        """
        if dt is None:
            dt = self.dt

        # Convert gyro from deg/s to rad/s if needed
        if gyro_in_deg:
            gyro_rad = np.deg2rad(gyro)
        else:
            gyro_rad = gyro

        # Extract current quaternion and bias
        q = self.x[0:4]
        bias = self.x[4:7]

        # Apply bias correction to gyro
        gyro_corrected = gyro_rad - bias

        # Quaternion derivative (basic kinematic equation)
        self.compute_quaternion_correction_matrix(q)
        q_dot = self.quat_corr_mat @ gyro_corrected

        # Integrate quaternion
        q_new = q + q_dot * dt

        # Normalize quaternion
        q_new = self.normalize_quaternion(q_new)

        # Update state
        self.x[0:4] = q_new

        # Jacobian for error-state propagation
        F = np.eye(6)

        # Transition for orientation error due to gyro error
        F[0:3, 0:3] = np.eye(3) - self.skew_symmetric(gyro_corrected) * dt

        # Transition for orientation error due to gyro bias error
        F[0:3, 3:6] = -np.eye(3) * dt

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        # Keep P symmetric
        self.P = 0.5 * (self.P + self.P.T)

    def skew_symmetric(self, v):
        """
        Create skew-symmetric matrix from vector
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def update_accelerometer(self, accel):
        """
        Update step of EKF using accelerometer data

        Parameters:
        - accel: accelerometer measurement [ax, ay, az] (m/s^2)
        """
        if not self.initialized:
            return False

        # Check for valid input
        if np.linalg.norm(accel) < self.epsilon:
            return False

        # Normalize measurement
        accel_norm = self.normalize_vector(accel)

        # Current quaternion
        q = self.x[0:4]

        # Expected gravity direction in body frame (should point opposite to accelerometer in static case)
        expected_accel = self.rotate_vector_by_quaternion(
            np.array([0, 0, 1]), self.quaternion_conjugate(q))

        # Innovation (error)
        # Note the negative sign
        v_cross = np.cross(accel_norm, -expected_accel)
        v_dot = np.dot(accel_norm, -expected_accel)

        # Check for large innovations (may indicate divergence)
        innov_norm = np.linalg.norm(v_cross)
        if innov_norm > self.innovation_threshold:
            self.consecutive_large_innovations += 1
            if self.consecutive_large_innovations > self.max_consecutive_innovations:
                print("Large accelerometer innovations detected. Reinitializing filter.")
                self.consecutive_large_innovations = 0
                return False
        else:
            self.consecutive_large_innovations = max(
                0, self.consecutive_large_innovations - 1)

        # Observation matrix (maps from error state to innovation)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3) * v_dot  # Jacobian of cross product

        # Innovation covariance
        # Adjustment factor berdasarkan magnitudo
        adjustment_factor = (
            1 + 5 * np.abs(1 - np.linalg.norm(accel) / self.g))
        # Gunakan matriks noise accelerometer
        R = self.acc_noise * (adjustment_factor**2)
        S = H @ self.P @ H.T + R

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # State update (delta form)
        dx = K @ v_cross

        # Extract orientation and bias corrections
        dq = dx[0:3]
        db = dx[3:6]

        # Correction quaternion (small angle approximation)
        q_corr = np.array([1, dq[0]/2, dq[1]/2, dq[2]/2])
        q_corr = self.normalize_quaternion(q_corr)

        # Apply correction to quaternion through multiplication
        q_new = self.quaternion_multiply(q, q_corr)

        # Apply correction to bias
        bias_new = self.x[4:7] + db

        # Update state
        self.x[0:4] = self.normalize_quaternion(q_new)
        self.x[4:7] = bias_new

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ R @ K.T  # Joseph form

        # Keep P symmetric
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def update_magnetometer(self, mag):
        """
        Update step of EKF using magnetometer data

        Parameters:
        - mag: magnetometer measurement [mx, my, mz]
        """
        if not self.initialized:
            return False

        # Check for valid input
        if np.linalg.norm(mag) < self.epsilon:
            return False

        # Normalize measurement
        mag_norm = self.normalize_vector(mag)

        # Current quaternion
        q = self.x[0:4]

        # Expected magnetic field direction in body frame
        # We only care about the yaw component, so project into XY plane
        mag_ref = np.array([1.0, 0, 0])  # Reference (north) in earth frame
        expected_mag = self.rotate_vector_by_quaternion(
            mag_ref, self.quaternion_conjugate(q))

        # Project vectors to horizontal plane (remove Z component)
        mag_norm_h = np.array([mag_norm[0], mag_norm[1], 0])
        expected_mag_h = np.array([expected_mag[0], expected_mag[1], 0])

        # Re-normalize
        mag_norm_h = self.normalize_vector(mag_norm_h)
        expected_mag_h = self.normalize_vector(expected_mag_h)

        # Innovation (error) - only care about heading (yaw) error
        v_cross = np.cross(mag_norm_h, expected_mag_h)
        v_dot = np.dot(mag_norm_h, expected_mag_h)

        # Extract only Z component of cross product (yaw correction)
        v_cross_z = np.array([0, 0, v_cross[2]])

        # Observation matrix (only affects yaw)
        H = np.zeros((1, 6))
        H[0, 2] = v_dot  # Only yaw component

        # Innovation covariance
        z_axis_noise = self.mag_noise[2, 2]  # Ambil nilai noise sumbu Z
        R = np.array([[z_axis_noise]])
        S = H[0:1, :] @ self.P @ H[0:1, :].T + R

        # Kalman gain
        try:
            K = self.P @ H[0:1, :].T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H[0:1, :].T @ np.linalg.pinv(S)

        # State update (delta form)
        dx = K @ np.array([v_cross_z[2]])

        # Extract orientation correction
        dq = np.zeros(3)
        dq[2] = dx[2]  # Only yaw correction

        # Correction quaternion (small angle approximation)
        q_corr = np.array([1, dq[0]/2, dq[1]/2, dq[2]/2])
        q_corr = self.normalize_quaternion(q_corr)

        # Apply correction to quaternion through multiplication
        q_new = self.quaternion_multiply(q, q_corr)

        # Apply minimal bias corrections from magnetometer (only heading-related bias)
        bias_new = self.x[4:7]
        # Only z-bias if available
        bias_new[2] += dx[5] if dx.shape[0] > 5 else 0

        # Update state
        self.x[0:4] = self.normalize_quaternion(q_new)
        self.x[4:7] = bias_new

        # Update covariance
        I = np.eye(6)
        KH = K @ H[0:1, :]
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T  # Joseph form

        # Keep P symmetric
        self.P = 0.5 * (self.P + self.P.T)

        return True

    def get_attitude(self, in_deg=False):
        """
        Get current attitude as Euler angles

        Parameters:
        - in_deg: True to return in degrees, False for radians

        Returns:
        - [roll, pitch, yaw]
        """
        euler = self.quaternion_to_euler(self.x[0:4])
        if in_deg:
            return np.rad2deg(euler)
        return euler

    def get_quaternion(self):
        """
        Get current attitude as quaternion

        Returns:
        - [w, x, y, z]
        """
        return self.x[0:4]

    def get_gyro_bias(self, in_deg=False):
        """
        Get current gyro bias estimate

        Parameters:
        - in_deg: True to return in deg/s, False for rad/s

        Returns:
        - [bias_x, bias_y, bias_z]
        """
        if in_deg:
            return np.rad2deg(self.x[4:7])
        return self.x[4:7]
