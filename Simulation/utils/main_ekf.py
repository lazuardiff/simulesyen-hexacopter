import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Implementasi EKF yang diperbaiki


class ImprovedAttitudeEstimator:
    def __init__(self, dt, gyro_noise_matrix, acc_noise_matrix, mag_noise_matrix, gyro_bias_init=None):
        # Konstanta dan parameter
        self.dt = dt

        # Konversi noise parameter
        if isinstance(gyro_noise_matrix, (list, tuple, np.ndarray)) and len(gyro_noise_matrix) == 3:
            self.gyro_noise = np.diag(gyro_noise_matrix)
        else:
            self.gyro_noise = gyro_noise_matrix

        if isinstance(acc_noise_matrix, (list, tuple, np.ndarray)) and len(acc_noise_matrix) == 3:
            self.acc_noise = np.diag(acc_noise_matrix)
        else:
            self.acc_noise = acc_noise_matrix

        if isinstance(mag_noise_matrix, (list, tuple, np.ndarray)) and len(mag_noise_matrix) == 3:
            self.mag_noise = np.diag(mag_noise_matrix)
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

        # Vektor referensi di frame bumi
        self.acc_ref = np.array([0, 0, 1.0])  # Arah gravitasi
        self.mag_ref = np.array([1.0, 0, 0])  # Arah utara magnetik

        # MODIFIKASI 1: Matriks kovariansi awal yang lebih agresif
        # Kovariansi untuk [delta_rot(3), bias_gyro(3)]
        self.P = np.zeros((6, 6))
        # Ketidakpastian awal dalam orientasi (lebih tinggi)
        self.P[:3, :3] = np.eye(3) * 0.1
        # Ketidakpastian awal dalam bias gyro (lebih tinggi)
        self.P[3:, 3:] = np.eye(3) * 0.01

        # MODIFIKASI 2: Matriks kovariansi noise proses yang ditingkatkan
        self.Q = np.zeros((6, 6))
        # Noise rotasi (gunakan matriks noise gyro yang sebenarnya)
        self.Q[:3, :3] = self.gyro_noise * (self.dt**2)
        # Proses noise bias gyro (ditingkatkan agar lebih cepat beradaptasi)
        # Nilai lebih besar untuk adaptasi lebih cepat
        self.Q[3:, 3:] = np.eye(3) * 1e-5

        # Matriks transformasi quaternion
        self.quat_corr_mat = np.zeros((4, 3))

        # Flag inisialisasi dan variabel
        self.initialized = False
        self.initialization_count = 0
        # MODIFIKASI 3: Lebih banyak upaya inisialisasi
        self.max_initialization_tries = 10

        # MODIFIKASI 4: Parameter adaptive filtering yang lebih halus
        self.consecutive_large_innovations = 0
        self.max_consecutive_innovations = 8  # Lebih toleran terhadap innovation besar
        # Threshold lebih tinggi untuk mengurangi reinisialisasi
        self.innovation_threshold = 0.8

        # MODIFIKASI 5: Variabel buffer untuk akselerasi
        self.static_threshold = 0.3  # Threshold untuk mendeteksi kondisi statis
        self.accel_buffer = []
        self.accel_buffer_size = 10
        self.is_static = False

        # Vektor referensi dalam body frame
        self.gravity_body = None
        self.mag_body = None

        # MODIFIKASI 6: Time constant filter untuk bias gyro
        self.bias_filter_alpha = 0.005  # Filter constant untuk bias (low-pass)
        self.prev_bias_est = np.zeros(3)
        if gyro_bias_init is not None:
            self.prev_bias_est = gyro_bias_init.copy()

    # Fungsi-fungsi helper tetap sama
    def quaternion_to_rotation_matrix(self, q):
        # scipy uses [x, y, z, w]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        return r.as_matrix()

    def triad_algorithm(self, v1_body, v2_body, v1_ref, v2_ref):
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
        # q = [w, x, y, z]
        self.quat_corr_mat[0, :] = [-q[1], -q[2], -q[3]]
        self.quat_corr_mat[1, :] = [q[0], -q[3], q[2]]
        self.quat_corr_mat[2, :] = [q[3], q[0], -q[1]]
        self.quat_corr_mat[3, :] = [-q[2], q[1], q[0]]

        return 0.5 * self.quat_corr_mat

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def quaternion_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def rotate_vector_by_quaternion(self, v, q):
        # Convert vector to pure quaternion
        v_quat = np.array([0, v[0], v[1], v[2]])

        # q * v * q^-1
        q_conj = self.quaternion_conjugate(q)
        v_rotated = self.quaternion_multiply(
            self.quaternion_multiply(q, v_quat), q_conj)

        # Return vector part
        return v_rotated[1:]

    def quaternion_to_euler(self, q):
        # Convert to scipy format [x, y, z, w]
        q_scipy = [q[1], q[2], q[3], q[0]]

        # Convert to euler
        r = Rotation.from_quat(q_scipy)
        euler = r.as_euler('xyz')

        return euler

    def euler_to_quaternion(self, euler):
        r = Rotation.from_euler('xyz', euler)
        q_scipy = r.as_quat()  # [x, y, z, w]
        # Convert to w, x, y, z format
        q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        return q

    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        if norm < self.epsilon:
            return np.array([1, 0, 0, 0])  # Default to identity if near zero
        return q / norm

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm < self.epsilon:
            return v  # Return original if near zero
        return v / norm

    # MODIFIKASI 7: Detection kondisi statis untuk tuning adaptive
    def check_static_condition(self, accel):
        # Update buffer
        self.accel_buffer.append(accel)
        if len(self.accel_buffer) > self.accel_buffer_size:
            self.accel_buffer.pop(0)

        # Hitung variance
        if len(self.accel_buffer) >= self.accel_buffer_size:
            # Normalisasi semua vektor
            norm_accels = [np.linalg.norm(a) for a in self.accel_buffer]
            variance = np.var(norm_accels)
            # Deteksi kondisi statis
            self.is_static = variance < self.static_threshold
            return self.is_static

        return False

    def initialize(self, accel, mag):
        if np.linalg.norm(accel) < self.epsilon or np.linalg.norm(mag) < self.epsilon:
            return False

        self.initialization_count += 1

        # MODIFIKASI 8: Inisialisasi lebih robust dengan rata-rata beberapa pengukuran
        if len(self.accel_buffer) < self.accel_buffer_size:
            # Kumpulkan beberapa pengukuran
            self.accel_buffer.append(accel)
            if len(self.accel_buffer) < self.accel_buffer_size:
                return False

            # Rata-rata
            accel = np.mean(self.accel_buffer, axis=0)

        # Store normalized measurements as reference
        # Negative because accel measures -g
        self.gravity_body = -self.normalize_vector(accel)
        self.mag_body = self.normalize_vector(mag)

        # Check that vectors aren't parallel
        cross_prod = np.cross(self.gravity_body, self.mag_body)
        if np.linalg.norm(cross_prod) < 0.1:  # Threshold for near-parallel vectors
            if self.initialization_count < self.max_initialization_tries:
                return False

        try:
            # Apply TRIAD algorithm to get rotation from body to reference
            # gravity points down in reference frame
            grav_ref = np.array([0, 0, 1.0])
            # magnetic field points north in reference frame
            mag_ref = np.array([1.0, 0, 0])

            # Get rotation matrix using TRIAD
            R_body_to_ref = self.triad_algorithm(
                self.gravity_body, self.mag_body, grav_ref, mag_ref)

            # Convert to quaternion and set initial state
            r = Rotation.from_matrix(R_body_to_ref)
            q_scipy = r.as_quat()  # [x, y, z, w]
            # [w, x, y, z]
            q = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

            # Set quaternion in state vector
            self.x[0:4] = q

            # Reset covariance to initial values
            self.P[:3, :3] = np.eye(3) * 0.1  # Orientation uncertainty
            self.P[3:, 3:] = np.eye(3) * 0.01  # Bias uncertainty

            self.initialized = True
            return True

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            if self.initialization_count < self.max_initialization_tries:
                return False
            else:
                # Force initialization with default orientation
                self.x[0:4] = np.array([1, 0, 0, 0])  # Identity quaternion
                self.initialized = True
                return True

    def skew_symmetric(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict(self, gyro, dt=None, gyro_in_deg=True):
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

        # MODIFIKASI 9: Menggunakan integrasi quaternion yang lebih akurat (RK4)
        # Method 1: Euler integration (standard)
        # q_new = q + q_dot * dt

        # Method 2: Improved integration for better accuracy
        k1 = q_dot
        q_temp = q + k1 * dt/2
        k2 = self.compute_quaternion_correction_matrix(q_temp) @ gyro_corrected
        q_temp = q + k2 * dt/2
        k3 = self.compute_quaternion_correction_matrix(q_temp) @ gyro_corrected
        q_temp = q + k3 * dt
        k4 = self.compute_quaternion_correction_matrix(q_temp) @ gyro_corrected

        q_new = q + (k1 + 2*k2 + 2*k3 + k4) * dt/6

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

    def update_accelerometer(self, accel):
        if not self.initialized:
            return False

        # Check for valid input
        if np.linalg.norm(accel) < self.epsilon:
            return False

        # MODIFIKASI 10: Deteksi kondisi statis untuk tuning adaptive
        is_static = self.check_static_condition(accel)

        # Normalize measurement
        accel_norm = self.normalize_vector(accel)

        # Current quaternion
        q = self.x[0:4]

        # Expected gravity direction in body frame (points opposite to accelerometer in static case)
        expected_accel = self.rotate_vector_by_quaternion(
            np.array([0, 0, 1]), self.quaternion_conjugate(q))

        # Innovation (error) - Note the negative sign
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

        # MODIFIKASI 11: Improved adaptive noise adjustment
        # Mendeteksi percepatan non-gravitasi dengan lebih presisi
        acc_magnitude = np.linalg.norm(accel) / self.g
        acceleration_factor = np.abs(1 - acc_magnitude)

        # Innovation covariance - adaptive tuning berdasarkan deteksi kondisi statis
        if is_static:
            # Lower noise untuk kondisi statis (lebih percaya accelerometer)
            adjustment_factor = 1.0 + 2.0 * acceleration_factor
        else:
            # Higher noise untuk kondisi dinamis (kurang percaya accelerometer)
            adjustment_factor = 1.0 + 10.0 * acceleration_factor

        # Gunakan matriks noise accelerometer
        R = self.acc_noise * (adjustment_factor**2)
        S = H @ self.P @ H.T + R

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # MODIFIKASI 12: Limit Kalman gain untuk stabilitas
        max_gain = 0.5
        K = np.clip(K, -max_gain, max_gain)

        # State update (delta form)
        dx = K @ v_cross

        # Extract orientation and bias corrections
        dq = dx[0:3]
        db = dx[3:6]

        # MODIFIKASI 13: Bias update filter - untuk mengurangi noise pada estimasi bias
        # Apply low-pass filter to bias updates
        bias_update = db
        if is_static:
            # Kondisi statis - update bias lebih agresif
            self.bias_filter_alpha = 0.05
        else:
            # Kondisi dinamis - update bias lebih konservatif
            self.bias_filter_alpha = 0.01

        filtered_bias_update = self.bias_filter_alpha * bias_update

        # Correction quaternion (small angle approximation)
        q_corr = np.array([1, dq[0]/2, dq[1]/2, dq[2]/2])
        q_corr = self.normalize_quaternion(q_corr)

        # Apply correction to quaternion through multiplication
        q_new = self.quaternion_multiply(q, q_corr)

        # Apply correction to bias
        bias_new = self.x[4:7] + filtered_bias_update

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

        # MODIFIKASI 14: Magnetometer correction for both yaw and bias
        # Observation matrix (untuk koreksi yaw dan juga bias)
        H = np.zeros((1, 6))
        H[0, 2] = v_dot  # Yaw component
        # Magnetometer can also correct z-axis gyro bias
        H[0, 5] = 0.1    # Coeffisien untuk koreksi bias-z dari magnetometer

        # Innovation covariance
        z_axis_noise = self.mag_noise[2, 2]  # Ambil nilai noise sumbu Z
        R = np.array([[z_axis_noise]])
        S = H[0:1, :] @ self.P @ H[0:1, :].T + R

        # Kalman gain
        try:
            K = self.P @ H[0:1, :].T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H[0:1, :].T @ np.linalg.pinv(S)

        # MODIFIKASI 15: Limit magnetometer gain untuk stabilitas
        # Magnetometer biasanya memiliki noise lebih tinggi
        max_mag_gain = 0.3
        K = np.clip(K, -max_mag_gain, max_mag_gain)

        # State update (delta form)
        dx = K @ np.array([v_cross_z[2]])

        # Extract orientation correction (hanya untuk yaw)
        dq = np.zeros(3)
        dq[2] = dx[2]  # Only yaw correction

        # Correction quaternion (small angle approximation)
        q_corr = np.array([1, dq[0]/2, dq[1]/2, dq[2]/2])
        q_corr = self.normalize_quaternion(q_corr)

        # Apply correction to quaternion through multiplication
        q_new = self.quaternion_multiply(q, q_corr)

        # Apply minimal bias corrections from magnetometer (only heading-related bias)
        bias_new = self.x[4:7]
        if dx.shape[0] > 5:
            # MODIFIKASI 16: Filter updates untuk bias gyro dari magnetometer
            # Only apply to z-bias and with filtering
            mag_bias_correction = dx[5] * \
                0.01 if self.is_static else dx[5] * 0.001
            bias_new[2] += mag_bias_correction

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

    # Fungsi getter tetap sama
    def get_attitude(self, in_deg=False):
        euler = self.quaternion_to_euler(self.x[0:4])
        if in_deg:
            return np.rad2deg(euler)
        return euler

    def get_quaternion(self):
        return self.x[0:4]

    def get_gyro_bias(self, in_deg=False):
        if in_deg:
            return np.rad2deg(self.x[4:7])
        return self.x[4:7]


# Simulasi dengan parameter yang dioptimalkan
def improved_simulation():
    # Parameter simulasi
    dt = 0.01  # 100 Hz
    duration = 20  # dalam detik
    time = np.arange(0, duration, dt)

    # MODIFIKASI 17: Parameter noise yang dioptimalkan
    # Gyroscope (rad/s)
    # Noise berbeda untuk tiap sumbu
    gyro_noise_matrix = np.array([0.005, 0.007, 0.003])

    # Accelerometer
    # Sumbu Z biasanya lebih noisy
    acc_noise_matrix = np.array([0.01, 0.01, 0.02])

    # Magnetometer
    # Magnetometer secara umum lebih noisy
    mag_noise_matrix = np.array([0.02, 0.02, 0.03])

    # MODIFIKASI 18: Inisialisasi bias gyro yang lebih baik
    # Bisa dari kalibrasi awal atau nol jika tidak ada informasi sebelumnya
    gyro_bias_init = np.array([0.0, 0.0, 0.0])  # Asumsikan nol awalnya

    # Bias gyro sebenarnya (untuk simulasi)
    gyro_bias_true = np.array([0.01, -0.02, 0.005])  # rad/s

    # Inisialisasi filter dengan parameter teroptimasi
    ekf = ImprovedAttitudeEstimator(
        dt, gyro_noise_matrix, acc_noise_matrix, mag_noise_matrix, gyro_bias_init)

    # Penyimpanan untuk hasil
    attitude_true = np.zeros((len(time), 3))
    attitude_est = np.zeros((len(time), 3))
    gyro_bias_est = np.zeros((len(time), 3))
    quaternion_est = np.zeros((len(time), 4))

    # Gerakan dengan perubahan attitude yang lebih kompleks
    # MODIFIKASI 19: Gerakan simulasi yang lebih realistis
    roll_rate = 0.3  # rad/s
    pitch_rate = 0.2  # rad/s
    yaw_rate = 0.15  # rad/s

    # Flag inisialisasi
    initialized = False

    # Proses setiap time step
    for i, t in enumerate(time):
        # Attitude sebenarnya - gerakan sinusoidal multi-frekuensi
        roll_true = 0.3 * np.sin(roll_rate * t) + \
            0.05 * np.sin(2.5 * roll_rate * t)
        pitch_true = 0.2 * np.cos(pitch_rate * t * 0.7) + \
            0.03 * np.cos(3.0 * pitch_rate * t)
        yaw_true = 0.5 * np.sin(yaw_rate * t * 0.5) + \
            0.08 * np.sin(1.8 * yaw_rate * t)

        # Simpan attitude sebenarnya
        attitude_true[i] = [roll_true, pitch_true, yaw_true]

        # Simulasikan pembacaan gyro (turunan dari sudut + dinamika tambahan)
        gyro_x = roll_rate * np.cos(roll_rate * t) * 0.3 + \
            2.5 * roll_rate * 0.05 * np.cos(2.5 * roll_rate * t)
        gyro_y = -pitch_rate * 0.7 * \
            np.sin(pitch_rate * t * 0.7) - 3.0 * pitch_rate * \
            0.03 * np.sin(3.0 * pitch_rate * t)
        gyro_z = yaw_rate * 0.5 * \
            np.cos(yaw_rate * t * 0.5) + 1.8 * yaw_rate * \
            0.08 * np.cos(1.8 * yaw_rate * t)

        # Tambahkan noise pada gyro (berbeda untuk setiap sumbu)
        gyro_x += np.random.normal(0, gyro_noise_matrix[0])
        gyro_y += np.random.normal(0, gyro_noise_matrix[1])
        gyro_z += np.random.normal(0, gyro_noise_matrix[2])

        # Tambahkan bias gyro
        gyro = np.array([gyro_x, gyro_y, gyro_z]) + gyro_bias_true

        # Konversi ke deg/s untuk simulasi data seperti sensor asli
        gyro_deg = np.rad2deg(gyro)

        # Simulasikan pengukuran accelerometer
        r_true = Rotation.from_euler('xyz', [roll_true, pitch_true, yaw_true])
        R_true = r_true.as_matrix()

        # Gravitasi dalam frame earth (unit vector pointing down)
        gravity_earth = np.array([0, 0, 1.0])

        # Transformasikan ke body frame
        gravity_body = R_true.T @ gravity_earth

        # MODIFIKASI 20: Simulasi accelerometer yang lebih realistis
        # Accelerometer mengukur -g dalam body frame (plus noise)
        # Tambahkan komponen percepatan linear untuk simulasi lebih realistis
        linear_accel = np.array([
            0.05 * np.sin(1.2 * t),
            0.07 * np.cos(0.8 * t),
            0.03 * np.sin(1.5 * t)
        ]) * 9.81  # Percepatan linear dalam m/s²

        accel = -gravity_body * 9.81 + linear_accel + np.array([
            np.random.normal(0, acc_noise_matrix[0] * 9.81),
            np.random.normal(0, acc_noise_matrix[1] * 9.81),
            np.random.normal(0, acc_noise_matrix[2] * 9.81)
        ])

        # Simulasikan pengukuran magnetometer
        # MODIFIKASI 21: Model magnetometer yang lebih realistis
        # Medan magnetik dalam frame earth (north, east, down)
        # Menambahkan inklinasi magnetik untuk lebih realistis
        # Inklinasi magnetik (tergantung lokasi geografis)
        inclination = np.radians(45)
        mag_earth = np.array([
            np.cos(inclination),  # Komponen horizontal (north)
            0,                    # Komponen east
            np.sin(inclination)   # Komponen vertical (down)
        ])

        # Transformasikan ke body frame
        mag_body = R_true.T @ mag_earth

        # Tambahkan noise dan gangguan
        # Hard-iron effect (offset bias)
        hard_iron = np.array([0.05, -0.03, 0.02])
        # Tambahkan gangguan dan noise
        mag = mag_body + hard_iron + np.array([
            np.random.normal(0, mag_noise_matrix[0]),
            np.random.normal(0, mag_noise_matrix[1]),
            np.random.normal(0, mag_noise_matrix[2])
        ])

        # Inisialisasi filter dengan pengukuran pertama
        if not initialized:
            initialized = ekf.initialize(accel, mag)
            if not initialized:
                continue

        # Langkah prediksi dengan gyro
        ekf.predict(gyro_deg, gyro_in_deg=True)

        # Update dengan accelerometer dan magnetometer
        accel_update_ok = ekf.update_accelerometer(accel)

        # Jika update accelerometer gagal, reinisialisasi
        if not accel_update_ok:
            initialized = ekf.initialize(accel, mag)
            if not initialized:
                continue

        # Update dengan magnetometer
        ekf.update_magnetometer(mag)

        # Simpan estimasi
        attitude_est[i] = ekf.get_attitude()
        quaternion_est[i] = ekf.get_quaternion()
        gyro_bias_est[i] = ekf.get_gyro_bias()

    # Cetak hasil akhir
    print("Attitude akhir sebenarnya (rad):", attitude_true[-1])
    print("Attitude akhir sebenarnya (deg):", np.rad2deg(attitude_true[-1]))
    print("Attitude akhir estimasi (rad):", attitude_est[-1])
    print("Attitude akhir estimasi (deg):", np.rad2deg(attitude_est[-1]))
    print("Error attitude (rad):", attitude_true[-1] - attitude_est[-1])
    print("Error attitude (deg):", np.rad2deg(
        attitude_true[-1] - attitude_est[-1]))
    print("Estimasi bias gyro (rad/s):", ekf.get_gyro_bias())
    print("Estimasi bias gyro (deg/s):", ekf.get_gyro_bias(in_deg=True))
    print("Bias gyro sebenarnya (rad/s):", gyro_bias_true)
    print("Bias gyro sebenarnya (deg/s):", np.rad2deg(gyro_bias_true))

    # Analisis statistik error
    attitude_error_deg = np.rad2deg(attitude_true - attitude_est)

    roll_error_std = np.std(attitude_error_deg[:, 0])
    pitch_error_std = np.std(attitude_error_deg[:, 1])
    yaw_error_std = np.std(attitude_error_deg[:, 2])

    roll_error_mean = np.mean(np.abs(attitude_error_deg[:, 0]))
    pitch_error_mean = np.mean(np.abs(attitude_error_deg[:, 1]))
    yaw_error_mean = np.mean(np.abs(attitude_error_deg[:, 2]))

    print("\nAnalisis Error:")
    print("Roll Error: Mean = {:.4f}°, Std = {:.4f}°".format(
        roll_error_mean, roll_error_std))
    print("Pitch Error: Mean = {:.4f}°, Std = {:.4f}°".format(
        pitch_error_mean, pitch_error_std))
    print("Yaw Error: Mean = {:.4f}°, Std = {:.4f}°".format(
        yaw_error_mean, yaw_error_std))

    # Plot hasil
    # Plotting hasil
    plt.figure(figsize=(15, 12))

    # Plot Roll
    plt.subplot(3, 2, 1)
    plt.plot(time, attitude_true[:, 0], 'b-', label='True Roll')
    plt.plot(time, attitude_est[:, 0], 'r--', label='Estimated Roll')
    plt.xlabel('Time (s)')
    plt.ylabel('Roll (degrees)')
    plt.legend()
    plt.grid(True)

    # Plot Pitch
    plt.subplot(3, 2, 3)
    plt.plot(time, attitude_true[:, 1], 'b-', label='True Pitch')
    plt.plot(time, attitude_est[:, 1], 'r--', label='Estimated Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (degrees)')
    plt.legend()
    plt.grid(True)

    # Plot Yaw
    plt.subplot(3, 2, 5)
    plt.plot(time, attitude_true[:, 2], 'b-', label='True Yaw')
    plt.plot(time, attitude_est[:, 2], 'r--', label='Estimated Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (degrees)')
    plt.legend()
    plt.grid(True)

    # Plot Error Roll
    plt.subplot(3, 2, 2)
    plt.plot(time, attitude_error_deg[:, 0], 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Roll Error (degrees)')
    plt.grid(True)

    # Plot Error Pitch
    plt.subplot(3, 2, 4)
    plt.plot(time, attitude_error_deg[:, 1], 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Error (degrees)')
    plt.grid(True)

    # Plot Error Yaw
    plt.subplot(3, 2, 6)
    plt.plot(time, attitude_error_deg[:, 2], 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Error (degrees)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('attitude_estimation_results.png')
    plt.show()

    # Plot gyro bias estimation
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, gyro_bias_est[:, 0], 'r-', label='Estimated X Bias')
    plt.axhline(y=gyro_bias_true[0], color='k',
                linestyle='--', label='True X Bias')
    plt.ylabel('Gyro X Bias (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, gyro_bias_est[:, 1], 'g-', label='Estimated Y Bias')
    plt.axhline(y=gyro_bias_true[1], color='k',
                linestyle='--', label='True Y Bias')
    plt.ylabel('Gyro Y Bias (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, gyro_bias_est[:, 2], 'b-', label='Estimated Z Bias')
    plt.axhline(y=gyro_bias_true[2], color='k',
                linestyle='--', label='True Z Bias')
    plt.xlabel('Time (s)')
    plt.ylabel('Gyro Z Bias (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('gyro_bias_estimation.png')
    plt.show()

    # Cetak hasil akhir
    print("Attitude akhir sebenarnya (rad):", attitude_true[-1])
    print("Attitude akhir sebenarnya (deg):", np.rad2deg(attitude_true[-1]))
    print("Attitude akhir estimasi (rad):", attitude_est[-1])
    print("Attitude akhir estimasi (deg):", np.rad2deg(attitude_est[-1]))
    print("Error attitude (rad):", attitude_true[-1] - attitude_est[-1])
    print("Error attitude (deg):", np.rad2deg(
        attitude_true[-1] - attitude_est[-1]))
    print("Estimasi bias gyro (rad/s):", ekf.get_gyro_bias())
    print("Estimasi bias gyro (deg/s):", ekf.get_gyro_bias(in_deg=True))
    print("Bias gyro sebenarnya (rad/s):", gyro_bias_true)
    print("Bias gyro sebenarnya (deg/s):", np.rad2deg(gyro_bias_true))

    # Analisis statistik error
    roll_error_std = np.std(attitude_error_deg[:, 0])
    pitch_error_std = np.std(attitude_error_deg[:, 1])
    yaw_error_std = np.std(attitude_error_deg[:, 2])

    roll_error_mean = np.mean(np.abs(attitude_error_deg[:, 0]))
    pitch_error_mean = np.mean(np.abs(attitude_error_deg[:, 1]))
    yaw_error_mean = np.mean(np.abs(attitude_error_deg[:, 2]))

    print("\nAnalisis Error:")
    print("Roll Error: Mean = {:.4f}°, Std = {:.4f}°".format(
        roll_error_mean, roll_error_std))
    print("Pitch Error: Mean = {:.4f}°, Std = {:.4f}°".format(
        pitch_error_mean, pitch_error_std))
    print("Yaw Error: Mean = {:.4f}°, Std = {:.4f}°".format(
        yaw_error_mean, yaw_error_std))

    # Visualisasi 3D (opsional)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Membuat visual axes
    def create_axes(ax, quaternion, origin=[0, 0, 0], scale=0.1):
        r = Rotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        axes = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        transformed_axes = r.apply(axes)

        # Plot sumbu X (merah), Y (hijau), dan Z (biru)
        ax.quiver(origin[0], origin[1], origin[2],
                  transformed_axes[0, 0], transformed_axes[0,
                                                           1], transformed_axes[0, 2],
                  color='r', lw=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  transformed_axes[1, 0], transformed_axes[1,
                                                           1], transformed_axes[1, 2],
                  color='g', lw=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  transformed_axes[2, 0], transformed_axes[2,
                                                           1], transformed_axes[2, 2],
                  color='b', lw=2)

    # Ambil beberapa sampel untuk visualisasi (terlalu banyak akan membuat gambar penuh)
    step = len(time) // 20
    for i in range(0, len(time), step):
        # Buat rotation dari euler angles sebenarnya
        r_true = Rotation.from_euler('xyz', attitude_true[i])
        q_true = r_true.as_quat()  # [x, y, z, w]
        q_true = np.array([q_true[3], q_true[0], q_true[1],
                          q_true[2]])  # [w, x, y, z]

        # Ambil quaternion estimasi
        q_est = quaternion_est[i]

        # Plot axes
        create_axes(ax, q_true, origin=[0, 0, 0], scale=0.2)
        create_axes(ax, q_est, origin=[0.5, 0, 0], scale=0.2)

    ax.set_xlim([-0.5, 1.0])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('True vs Estimated Orientation')

    # Tambahkan legend
    ax.text(-0.5, 0, 0.5, 'Merah: X-axis', color='r')
    ax.text(-0.5, 0, 0.4, 'Hijau: Y-axis', color='g')
    ax.text(-0.5, 0, 0.3, 'Biru: Z-axis', color='b')
    ax.text(-0.5, 0, 0.2, 'Kiri: True Orientation', color='k')
    ax.text(-0.5, 0, 0.1, 'Kanan: Estimated Orientation', color='k')

    plt.tight_layout()
    plt.savefig('3d_orientation.png')
    plt.show()


# Run the improved simulation
improved_simulation()
