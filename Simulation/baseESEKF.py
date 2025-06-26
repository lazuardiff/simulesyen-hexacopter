import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


class BaseESEKF:
    """
    Base Error-State EKF dengan estimasi vektor gravitasi.
    Sesuai dengan model lengkap di Bab 5 paper Solà.

    Nominal State vector (x): [p(3), v(3), q(4), a_b(3), ω_b(3), g(3)] -> 19 dimensi
    Error State vector (δx): [δp(3), δv(3), δθ(3), δa_b(3), δω_b(3), δg(3)] -> 18 dimensi
    """

    def __init__(self, dt=0.01):
        self.dt = dt

        # --- KEMBALIKAN UKURAN STATE VECTOR KE SEMULA ---
        self.state_dim = 16  # Kembali ke 16 (tanpa g)
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Inisialisasi quaternion

        self.error_state_dim = 15  # Kembali ke 15 (tanpa δg)

        # --- KEMBALIKAN MATRIKS KOVARIANS & NOISE KE UKURAN SEMULA ---
        self.P = np.eye(self.error_state_dim)  # 15x15
        self.P[0:3, 0:3] *= 25.0
        self.P[3:6, 3:6] *= 4.0
        # Gunakan tuning P yang sudah baik dari hasil sebelumnya
        self.P[6:9, 6:9] = np.diag([0.04**2, 0.04**2, 0.1**2])
        self.P[9:12, 9:12] *= 0.005
        self.P[12:15, 12:15] *= 0.005

        # Matriks Q kembali ke ukuran 15x15
        self.Q = np.zeros((self.error_state_dim, self.error_state_dim))
        # Gunakan nilai tuning Q yang sudah baik
        vel_noise_psd = 0.1
        att_noise_psd = 0.001
        acc_bias_psd = 1e-5
        gyro_bias_psd = 3e-5
        self.Q[3:6, 3:6] = np.eye(3) * (vel_noise_psd**2) * self.dt
        self.Q[6:9, 6:9] = np.eye(3) * (att_noise_psd**2) * self.dt
        self.Q[9:12, 9:12] = np.eye(3) * (acc_bias_psd**2) * self.dt
        self.Q[12:15, 12:15] = np.eye(3) * (gyro_bias_psd**2) * self.dt

        self.R_accel = np.eye(3) * 1.1**2
        self.R_gps_pos = np.eye(3) * 1.0**2
        self.R_gps_vel = np.eye(3) * (0.1**2)
        self.R_baro = np.array([[0.5**2]])
        self.R_mag = np.eye(3) * 0.02**2

        # --- DEFINISIKAN KEMBALI g SEBAGAI KONSTANTA ---
        self.g_ned = np.array([0, 0, 9.81])

        # Parameter tuning lainnya tetap sama
        self.yaw_innovation_gate = 8.0
        self.mag_declination = 0.5
        self.enable_mag_bias_learning = True
        self.mag_bias = np.zeros(3)
        self.use_gps_heading = True
        self.gps_heading_threshold = 1.0
        self.gps_heading_weight_max = 0.6

        self.mag_ref_ned = np.array(
            [np.cos(np.deg2rad(0.5)), np.sin(np.deg2rad(0.5)), 0.0])
        self.initialized = False
        self.prediction_mode = "IMU_ONLY"

    def predict(self, accel_body, gyro_body):
        if not self.initialized:
            return

        # --- PREDIKSI NOMINAL STATE MENGGUNAKAN RK4 ---
        # RK4 membutuhkan 4 evaluasi turunan (slope)

        # k1 dihitung pada state awal
        k1 = self._nominal_state_dynamics(self.x, accel_body, gyro_body)

        # k2 dihitung pada state di tengah interval, menggunakan k1
        k2_state = self.x + 0.5 * self.dt * k1
        k2_state[6:10] = self.normalize_quaternion(
            k2_state[6:10])  # Normalisasi kuaternion
        k2 = self._nominal_state_dynamics(k2_state, accel_body, gyro_body)

        # k3 dihitung pada state di tengah interval, menggunakan k2
        k3_state = self.x + 0.5 * self.dt * k2
        k3_state[6:10] = self.normalize_quaternion(
            k3_state[6:10])  # Normalisasi kuaternion
        k3 = self._nominal_state_dynamics(k3_state, accel_body, gyro_body)

        # k4 dihitung pada state di akhir interval, menggunakan k3
        k4_state = self.x + self.dt * k3
        k4_state[6:10] = self.normalize_quaternion(
            k4_state[6:10])  # Normalisasi kuaternion
        k4 = self._nominal_state_dynamics(k4_state, accel_body, gyro_body)

        # Kombinasikan slope untuk mendapatkan update state
        state_increment = (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Update state nominal
        self.x += state_increment
        self.x[6:10] = self.normalize_quaternion(
            self.x[6:10])  # Final normalization

        # --- PREDIKSI KOVARIANS ERROR STATE (TETAP SAMA) ---
        # Logika untuk menghitung F dan P tidak berubah

        # Kita perlu R_bn dan gyro_corrected dari state *sebelum* update untuk F
        q_old = self.x - state_increment  # Perkirakan state lama untuk F
        q_old[6:10] = self.normalize_quaternion(q_old[6:10])
        R_bn_old = self.quaternion_to_rotation_matrix(q_old[6:10])
        accel_corrected_old = accel_body - q_old[10:13]
        gyro_corrected_old = gyro_body - q_old[13:16]

        # Matriks Jacobian F kembali ke ukuran 15x15
        F = np.eye(self.error_state_dim)

        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = - \
            R_bn_old @ self.skew_symmetric(accel_corrected_old) * self.dt
        F[3:6, 9:12] = -R_bn_old * self.dt

        delta_rot_vec = gyro_corrected_old * self.dt
        R_delta = Rotation.from_rotvec(delta_rot_vec).as_matrix()
        F[6:9, 6:9] = R_delta.T
        F[6:9, 12:15] = -np.eye(3) * self.dt

        # Propagasi Kovariansi
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.error_state_dim) * 1e-12

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

    def quaternion_multiply(self, q1, q2):
        """Perkalian quaternion q1 ⊗ q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _inject_error_and_reset(self, delta_x):
        """
        Menyuntikkan estimasi error state (δx) ke dalam state nominal (x)
        dan me-reset error state. Sesuai Bab 6.2 & 6.3.
        """
        # Injeksi error (x ← x ⊕ δx, Eq. 283)
        self.x[0:3] += delta_x[0:3]   # Posisi
        self.x[3:6] += delta_x[3:6]   # Kecepatan

        # Injeksi error attitude (q ← q ⊗ q{δθ}, Eq. 283c)
        d_theta = delta_x[6:9]
        angle = np.linalg.norm(d_theta)
        if angle > 1e-8:
            axis = d_theta / angle
            dq = np.array([np.cos(angle/2),
                           axis[0] * np.sin(angle/2),
                           axis[1] * np.sin(angle/2),
                           axis[2] * np.sin(angle/2)])
            self.x[6:10] = self.quaternion_multiply(self.x[6:10], dq)
            self.x[6:10] = self.normalize_quaternion(self.x[6:10])

        self.x[10:13] += delta_x[9:12]  # Bias Akselerometer
        self.x[13:16] += delta_x[12:15]  # Bias Giroskop
        self.x[16:19] += delta_x[15:18]  # Vektor Gravitasi

        # Reset ESEKF (δx ← 0) secara implisit.
        # Kovariansi P sudah diupdate, dan mean error state (δx) kembali dianggap nol.
        # Paper menyebutkan Jacobian reset G (Eq. 288), namun seringkali G ≈ I.
        # Kita tidak mengimplementasikan G secara eksplisit, yang ekuivalen dengan G=I.

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

    def _nominal_state_dynamics(self, state, accel_body, gyro_body):
        """
        Menghitung turunan dari nominal state (ẋ = f(x, u)).
        Ini adalah fungsi yang akan diintegrasikan oleh RK4.

        Args:
            state (np.array): Vektor state nominal [p, v, q, ab, ωb] pada suatu waktu.
            accel_body (np.array): Pengukuran akselerometer mentah.
            gyro_body (np.array): Pengukuran giroskop mentah.

        Returns:
            np.array: Turunan state (x_dot).
        """
        # Ekstrak state
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        acc_bias = state[10:13]
        gyro_bias = state[13:16]

        # Koreksi pengukuran IMU
        accel_corrected = accel_body - acc_bias
        gyro_corrected = gyro_body - gyro_bias

        # --- Hitung turunan state ---

        # 1. Turunan Posisi (p_dot = v)
        pos_dot = vel

        # 2. Turunan Kecepatan (v_dot = R(a_m - a_b) + g)
        R_bn = self.quaternion_to_rotation_matrix(q)
        accel_ned = R_bn @ accel_corrected + self.g_ned
        vel_dot = accel_ned

        # 3. Turunan Quaternion (q_dot = 1/2 * q ⊗ ω)
        omega_quat = np.array(
            [0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)

        # 4. Turunan Bias (diasumsikan random walk, turunannya nol di model nominal)
        acc_bias_dot = np.zeros(3)
        gyro_bias_dot = np.zeros(3)

        # Gabungkan semua turunan menjadi satu vektor
        x_dot = np.concatenate(
            [pos_dot, vel_dot, q_dot, acc_bias_dot, gyro_bias_dot])
        return x_dot

    def update_accelerometer_for_attitude(self, accel_body, gyro_body):
        """
        Tahap Koreksi untuk orientasi (roll & pitch) menggunakan akselerometer.
        Konsep ini diadaptasi dari jurnal, diimplementasikan dalam kerangka ES-EKF.
        Update ini hanya aktif saat UAV bergerak dengan akselerasi rendah.
        """
        if not self.initialized:
            return 0

        # --- KONDISI AKTIVASI ---
        # 1. Cek apakah akselerasi mendekati 1g (tidak ada akselerasi linear besar)
        acc_norm = np.linalg.norm(accel_body)
        if not (0.9 * 9.81 < acc_norm < 1.1 * 9.81):
            return 0  # Tidak aktif jika ada akselerasi linear yang signifikan

        # 2. Cek apakah rotasi rendah (untuk memastikan akselerometer tidak terpengaruh gaya sentripetal)
        # Gunakan gyro yang sudah dikoreksi bias
        gyro_corrected = gyro_body - self.x[13:16]
        if np.linalg.norm(gyro_corrected) > np.deg2rad(15.0):  # Batas 15 deg/s
            return 0

        # --- 1. Hitung Inovasi ---
        # Prediksi arah gravitasi di body frame berdasarkan orientasi saat ini
        q_nominal = self.x[6:10]
        R_bn = self.quaternion_to_rotation_matrix(q_nominal)
        gravity_expected_body = R_bn.T @ self.g_ned  # g_ned dari NED ke Body

        # Inovasi adalah selisih antara pengukuran akselerometer dan prediksi gravitasi
        # Kita normalisasi keduanya agar fokus pada arah, bukan magnitudo
        innovation = (accel_body / acc_norm) - \
            (gravity_expected_body / np.linalg.norm(gravity_expected_body))

        # --- 2. Hitung Jacobian Pengukuran (H) ---
        # H = ∂h/∂δx, di mana h(δx) = (R_nominal @ R(δθ))^T @ g_ned
        # ∂h/∂δθ = [ (R_bn.T @ g_ned) x ] = [ gravity_expected_body x ]
        H = np.zeros((3, self.error_state_dim))
        # Turunan terhadap error attitude (δθ)
        H[0:3, 6:9] = self.skew_symmetric(gravity_expected_body)

        # --- 3. Hitung Kalman Gain (K) ---
        S = H @ self.P @ H.T + self.R_accel
        try:
            S_inv = np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            return 0
        K = self.P @ H.T @ S_inv

        # --- 4. Update Error State & Kovariansi ---
        delta_x_correction = K @ innovation

        I = np.eye(self.error_state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_accel @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # --- 5. Injeksi Error ke State Nominal & Reset ESEKF ---
        self._inject_error_and_reset(delta_x_correction)

        return 1

    def update_gps_position(self, gps_pos):
        """
        Tahap Koreksi dengan pengukuran Posisi GPS.
        Sesuai dengan Bab 6 dari paper Solà.
        """
        if not self.initialized or gps_pos is None:
            return

        # --- 1. Hitung Inovasi ---
        # h(x) = self.x[0:3] (prediksi posisi dari state nominal)
        # y = z - h(x) (inovasi)
        innovation = gps_pos - self.x[0:3]

        # --- 2. Hitung Jacobian Pengukuran (H) ---
        # H = ∂h/∂δx, di mana h adalah fungsi pengukuran
        H = np.zeros((3, self.error_state_dim))
        H[0:3, 0:3] = np.eye(3)  # ∂h/∂δp = I

        # --- 3. Hitung Kalman Gain (K) ---
        S = H @ self.P @ H.T + self.R_gps_pos
        K = self.P @ H.T @ np.linalg.inv(S)

        # --- 4. Update Error State & Kovariansi (Eq. 275 & 276) ---
        # Update mean error state (estimasi δx)
        delta_x_correction = K @ innovation

        # Update kovariansi (menggunakan Joseph form untuk stabilitas numerik)
        I = np.eye(self.error_state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_pos @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # --- 5. Injeksi Error ke State Nominal & Reset ESEKF (Bab 6.2 & 6.3) ---
        self._inject_error_and_reset(delta_x_correction)

    def update_gps_velocity(self, gps_vel):
        """
        Tahap Koreksi dengan pengukuran Kecepatan GPS.
        Sesuai dengan Bab 6 dari paper Solà.
        """
        if not self.initialized or gps_vel is None:
            return

        # --- 1. Hitung Inovasi ---
        # z = gps_vel (pengukuran aktual)
        # h(x) = self.x[3:6] (prediksi kecepatan dari state nominal)
        innovation = gps_vel - self.x[3:6]

        # --- 2. Hitung Jacobian Pengukuran (H) ---
        # H = ∂h/∂δx
        H = np.zeros((3, self.error_state_dim))
        H[0:3, 3:6] = np.eye(3)  # ∂h/∂δv = I

        # --- 3. Hitung Kalman Gain (K) ---
        S = H @ self.P @ H.T + self.R_gps_vel
        try:
            # Menggunakan pseudoinverse untuk stabilitas jika S singular
            S_inv = np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv

        # --- 4. Update Error State & Kovariansi ---
        delta_x_correction = K @ innovation

        I = np.eye(self.error_state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + \
            K @ self.R_gps_vel @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # --- 5. Injeksi Error ke State Nominal & Reset ESEKF ---
        self._inject_error_and_reset(delta_x_correction)

        # --- Bantuan Heading dari GPS (Logika Terpisah) ---
        # Ini adalah koreksi tambahan di luar EKF standar, jadi dipanggil setelahnya.
        self.update_gps_heading(gps_vel)

    def update_gps_heading(self, gps_vel):
        """Bantuan heading dari GPS untuk koreksi yaw (bukan bagian dari EKF standar)."""
        if not self.use_gps_heading:
            return

        gps_speed = np.linalg.norm(gps_vel[:2])
        if gps_speed > self.gps_heading_threshold:
            gps_heading = np.arctan2(gps_vel[1], gps_vel[0])

            q = self.x[6:10]
            R = self.quaternion_to_rotation_matrix(q)
            current_yaw = np.arctan2(R[1, 0], R[0, 0])

            heading_innovation = gps_heading - current_yaw
            heading_innovation = np.arctan2(
                np.sin(heading_innovation), np.cos(heading_innovation))

            # Batasi inovasi agar tidak terlalu agresif
            if abs(heading_innovation) < np.deg2rad(45):
                weight = min(gps_speed / 5.0, 1.0) * \
                    self.gps_heading_weight_max
                yaw_correction_angle = heading_innovation * weight

                # Koreksi langsung disuntikkan ke state nominal quaternion
                # Ini adalah "trick" atau "bantuan" di luar kerangka EKF murni
                dq_yaw = np.array(
                    [np.cos(yaw_correction_angle / 2), 0, 0, np.sin(yaw_correction_angle / 2)])
                self.x[6:10] = self.quaternion_multiply(self.x[6:10], dq_yaw)
                self.x[6:10] = self.normalize_quaternion(self.x[6:10])

    def update_barometer(self, baro_alt):
        """
        Tahap Koreksi dengan pengukuran Ketinggian dari Barometer.
        Sesuai dengan Bab 6 dari paper Solà.
        """
        if not self.initialized or baro_alt is None:
            return

        # --- 1. Hitung Inovasi ---
        # z = baro_alt
        # h(x) = -self.x[2] (ketinggian adalah negatif dari posisi z di frame NED)
        innovation = np.array([baro_alt - (-self.x[2])])

        # --- 2. Hitung Jacobian Pengukuran (H) ---
        # h = -p_z, maka ∂h/∂δp_z = -1
        H = np.zeros((1, self.error_state_dim))
        H[0, 2] = -1

        # --- 3. Hitung Kalman Gain (K) ---
        S = H @ self.P @ H.T + self.R_baro
        try:
            S_inv = np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv

        # --- 4. Update Error State & Kovariansi ---
        delta_x_correction = K @ innovation

        I = np.eye(self.error_state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_baro @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # --- 5. Injeksi Error ke State Nominal & Reset ESEKF ---
        self._inject_error_and_reset(delta_x_correction)

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

        R_bn = self.quaternion_to_rotation_matrix(self.x[6:10])
        mag_ref_ned_corrected = np.array(
            [np.cos(self.mag_declination), np.sin(self.mag_declination), 0.0])
        mag_expected_body = R_bn.T @ mag_ref_ned_corrected

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

        # --- 1. Hitung Inovasi ---
        # Menggunakan komponen horizontal untuk fokus pada yaw
        mag_h = np.array([mag_body_corrected[0], mag_body_corrected[1], 0.0])
        exp_h = np.array([mag_expected_body[0], mag_expected_body[1], 0.0])

        if np.linalg.norm(mag_h) < 0.05 or np.linalg.norm(exp_h) < 0.05:
            return

        mag_h_norm = mag_h / np.linalg.norm(mag_h)
        exp_h_norm = exp_h / np.linalg.norm(exp_h)

        # z = mag_h_norm[:2]
        # h(x) = exp_h_norm[:2]
        innovation = mag_h_norm[:2] - exp_h_norm[:2]

        # --- 2. Hitung Jacobian Pengukuran (H) ---
        H = np.zeros((2, self.error_state_dim))
        # Primary yaw sensitivity (∂h/∂δθ_yaw)
        H[0, 8] = exp_h_norm[1]
        H[1, 8] = -exp_h_norm[0]

        # Secondary roll/pitch sensitivity (reduced weighting)
        H[0, 6] = -exp_h_norm[2] * 0.05   # Reduced coupling
        H[1, 7] = -exp_h_norm[2] * 0.05   # Reduced coupling

        # Use adaptive noise for horizontal components
        R_mag_h = R_mag_adaptive[:2, :2]

        S = H @ self.P @ H.T + R_mag_h

        # Innovation Gating (Mahalanobis distance)
        y_normalized = innovation.T @ np.linalg.pinv(S) @ innovation
        if y_normalized > self.yaw_innovation_gate**2:
            return

        try:
            K = self.P @ H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            return

        # --- 4. Update Error State & Kovariansi ---
        # Menggunakan inovasi yang dibatasi (clipping) untuk mencegah update yang terlalu besar
        innovation_limited = np.clip(innovation, -0.3, 0.3)
        delta_x_correction = K @ innovation_limited

        # Batasi koreksi error state untuk stabilitas
        # Terutama untuk roll dan pitch yang tidak diobservasi langsung oleh magnetometer
        delta_x_correction[6] = np.clip(
            delta_x_correction[6], -0.05, 0.05)  # Roll
        delta_x_correction[7] = np.clip(
            delta_x_correction[7], -0.05, 0.05)  # Pitch

        I = np.eye(self.error_state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_mag_h @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(self.error_state_dim) * 1e-9

        # --- 5. Injeksi Error ke State Nominal & Reset ESEKF ---
        self._inject_error_and_reset(delta_x_correction)

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
