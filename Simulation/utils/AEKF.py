import numpy as np
from numpy import sin, cos
import config


class AEKF:
    def __init__(self, dt=0.01, debug=False):
        """
        Initialize the Adaptive Extended Kalman Filter for sensor fusion

        Args:
            dt: Time step (s)
            debug: Enable debug output
        """
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
        # x, y, z: position in NED frame
        # vx, vy, vz: velocity in NED frame
        # qw, qx, qy, qz: quaternion attitude
        # bax, bay, baz: accelerometer bias in body frame
        # bgx, bgy, bgz: gyroscope bias in body frame

        self.debug = debug
        self.dt = dt
        self.state_dim = 16
        self.g = 9.81  # gravity constant (m/s^2)

        # Initial state
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initial quaternion is [1, 0, 0, 0]

        # Initial covariance matrix
        self.P = np.eye(self.state_dim)
        self.P[0:3, 0:3] *= 20.0       # Position uncertainty
        self.P[3:6, 3:6] *= 5.0        # Velocity uncertainty
        self.P[6:10, 6:10] *= 0.1      # Attitude uncertainty
        self.P[10:13, 10:13] *= 0.2    # Acc bias uncertainty
        self.P[13:16, 13:16] *= 0.1    # Gyro bias uncertainty

        # Process noise covariance
        self.Q = np.eye(self.state_dim)
        self.Q[0:3, 0:3] *= 0.05       # Position process noise
        self.Q[3:6, 3:6] *= 0.2        # Velocity process noise
        self.Q[6:10, 6:10] *= 0.05     # Attitude process noise
        self.Q[10:13, 10:13] *= 0.01   # Acc bias process noise
        self.Q[13:16, 13:16] *= 0.005  # Gyro bias process noise

        # Measurement noise covariance
        self.R_imu = np.eye(6)
        self.R_imu[0:3, 0:3] *= 0.8    # Accelerometer measurement noise
        self.R_imu[3:6, 3:6] *= 0.3    # Gyroscope measurement noise

        self.R_gps = np.eye(6)
        self.R_gps[0:3, 0:3] = np.diag([1.0, 1.0, 4.0])  # GPS position noise
        self.R_gps[3:6, 3:6] *= 0.01   # GPS velocity noise

        self.R_mag = np.eye(3) * 0.0025  # Magnetometer noise

        # Adaptive filter parameters
        self.adaptive_window = 20
        self.innovation_sequence = []
        self.predicted_cov_sequence = []

        # Gravity vector in NED frame
        self.g_vec = np.array([0, 0, self.g])

        # Magnetic reference in NED frame (magnetic north)
        self.mag_reference = np.array(
            [1.0, 0.0, 0.0])  # Adjust based on location

        # Timestamp handling
        self.last_timestamp = 0

    def skew_symmetric(self, v):
        """Create skew-symmetric matrix from a 3-element vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict(self, acc_m, gyro_m, timestamp=None):
        """
        Prediction step using IMU measurements

        Args:
            acc_m: Accelerometer measurement [ax, ay, az] in body frame (m/s^2)
            gyro_m: Gyroscope measurement [wx, wy, wz] in body frame (rad/s)
            timestamp: Current time in seconds
        """
        # Handle timestamp
        if timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                self.dt = dt
            self.last_timestamp = timestamp

        # Extract current state
        p = self.x[0:3]    # Position in NED
        v = self.x[3:6]    # Velocity in NED
        q = self.x[6:10]   # Quaternion [qw, qx, qy, qz]
        ba = self.x[10:13]  # Accelerometer bias
        bg = self.x[13:16]  # Gyroscope bias

        # Correct measured values for bias
        acc_corrected = acc_m - ba
        gyro_corrected = gyro_m - bg

        # Get rotation matrix from body to NED
        R = self.quat2Dcm(q)

        # Transform acceleration to NED frame and remove gravity
        acc_ned = R @ acc_corrected - self.g_vec

        # State prediction using RK2 integration
        # Velocity update (midpoint)
        v_mid = v + 0.5 * acc_ned * self.dt

        # Position update using midpoint velocity
        p_new = p + v_mid * self.dt

        # Final velocity update
        v_new = v + acc_ned * self.dt

        # Quaternion update using angular velocity
        omega = gyro_corrected
        omega_norm = np.linalg.norm(omega)

        if omega_norm > 1e-10:
            axis = omega / omega_norm
            angle = omega_norm * self.dt

            qw = np.cos(angle/2)
            qxyz = axis * np.sin(angle/2)
            dq = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])

            q_new = self.quatMultiply(q, dq)
        else:
            q_new = q.copy()

        q_new = q_new / np.linalg.norm(q_new)

        # Bias update (random walk model)
        ba_new = ba
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

        # Attitude Jacobian wrt quaternion
        F[6:10, 6:10] = self._quaternion_derivative_matrix(omega, self.dt)

        # Acceleration Jacobian wrt quaternion
        F[3:6, 6:10] = self._acceleration_quaternion_jacobian(acc_corrected, q)

        # Acceleration affected by accelerometer bias
        F[3:6, 10:13] = -R * self.dt

        # Quaternion affected by gyroscope bias
        dq_dbg = -0.5 * self.dt * \
            self._quaternion_left_product_matrix(q)[1:, :]
        F[6:10, 13:16] = dq_dbg.T

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        # Ensure covariance stays symmetric
        self.P = 0.5 * (self.P + self.P.T)

    def update_imu(self, acc_m, gyro_m, timestamp=None):
        """
        Update step using IMU measurements

        Args:
            acc_m: Accelerometer measurement [ax, ay, az] in body frame (m/s^2)
            gyro_m: Gyroscope measurement [wx, wy, wz] in body frame (rad/s)
            timestamp: Current time in seconds
        """
        # Extract current state
        q = self.x[6:10]
        ba = self.x[10:13]
        bg = self.x[13:16]

        # Measurement model
        R = self.quat2Dcm(q)
        gravity_body = R.T @ self.g_vec  # Gravity vector in body frame

        # Expected accelerometer measurement (gravity + bias)
        h_acc = gravity_body + ba

        # Expected gyroscope measurement (bias)
        h_gyro = bg

        # Combined expected measurement
        h = np.hstack((h_acc, h_gyro))
        z = np.hstack((acc_m, gyro_m))

        # Innovation (measurement residual)
        y = z - h

        # Measurement Jacobian
        H = np.zeros((6, self.state_dim))

        # Sensitivity of accelerometer to attitude (quaternion)
        dR_dq = self._rotation_matrix_quaternion_jacobian(q)

        for i in range(4):
            H[0:3, 6+i] = dR_dq[i].T @ self.g_vec

        # Accelerometer affected by acc bias
        H[0:3, 10:13] = np.eye(3)

        # Gyroscope affected by gyro bias
        H[3:6, 13:16] = np.eye(3)

        # Check if acceleration magnitude matches gravity (indicates static conditions)
        acc_mag = np.linalg.norm(acc_m)
        acc_error = abs(acc_mag - self.g)

        # Dynamically adjust measurement noise
        R_adaptive = self.R_imu.copy()
        if acc_error > 0.5:  # Vehicle likely accelerating
            # Trust accelerometer less during dynamic motion
            R_adaptive[0:3, 0:3] *= (1.0 + 5.0 * acc_error)
            if self.debug:
                print(
                    f"Dynamic motion detected: {acc_error:.2f}, increasing R")

        # Innovation covariance
        S = H @ self.P @ H.T + R_adaptive

        # Store for adaptive filtering
        self.innovation_sequence.append(y)
        self.predicted_cov_sequence.append(S)
        if len(self.innovation_sequence) > self.adaptive_window:
            self.innovation_sequence.pop(0)
            self.predicted_cov_sequence.pop(0)

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Update covariance - Joseph form for numerical stability
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_adaptive @ K.T

        # Adapt process noise if enough data is available
        if len(self.innovation_sequence) >= self.adaptive_window:
            self.adapt_process_noise()

    def update_gps(self, pos_m, vel_m, timestamp=None):
        """
        Update step using GPS measurements
        """
        if pos_m is None or vel_m is None:
            return

        # Inisialisasi variabel R_gps_temp di luar blok kondisional
        R_gps_temp = None

        # Inisialisasi counter GPS jika belum ada
        if not hasattr(self, 'gps_update_count'):
            self.gps_update_count = 0

        # Untuk beberapa pengukuran GPS pertama, berikan kepercayaan lebih
        # dan gunakan threshold penolakan yang lebih longgar
        use_relaxed_threshold = self.gps_update_count < 10

        # Simpan value R_gps asli dan tingkatkan kepercayaan untuk awal
        if self.gps_update_count < 3:
            R_gps_temp = self.R_gps.copy()
            self.R_gps = np.eye(6)
            self.R_gps[0:3, 0:3] = np.diag(
                [0.5, 0.5, 2.0])  # Lebih percaya GPS di awal
            self.R_gps[3:6, 3:6] = np.diag(
                [0.005, 0.005, 0.005])  # Velocity noise

        # Increment counter SEBELUM processing
        self.gps_update_count += 1

        # Measurement model
        h = np.hstack((self.x[0:3], self.x[3:6]))
        z = np.hstack((pos_m, vel_m))
        y = z - h

        # Jacobian dan kovariansi inovasi
        H = np.zeros((6, self.state_dim))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        S = H @ self.P @ H.T + self.R_gps

        # Chi-square threshold yang lebih longgar untuk pengukuran awal
        chi2_threshold = 50.0 if use_relaxed_threshold else 15.0

        try:
            # Hitung jarak Mahalanobis dengan penanganan error
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = y.T @ S_inv @ y

            if mahalanobis_dist > chi2_threshold:
                if self.debug:
                    print(
                        f"GPS update rejected: {mahalanobis_dist:.2f} > {chi2_threshold}")
                if R_gps_temp is not None:
                    self.R_gps = R_gps_temp
                return

            # Jika sampai di sini, update berhasil
            if self.debug and self.gps_update_count <= 5:
                print(f"GPS update accepted: {mahalanobis_dist:.2f}")

        except np.linalg.LinAlgError:
            if self.debug:
                print("GPS update rejected: singular innovation covariance")
            if R_gps_temp is not None:
                self.R_gps = R_gps_temp
            return

        # Kalman gain dan update state
        K = self.P @ H.T @ S_inv

        # Update state
        self.x = self.x + K @ y
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Update kovariansi dengan formulasi Joseph untuk stabilitas numerik
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_gps @ K.T

        # Kembalikan R_gps ke nilai semula - TANPA KONDISI initial_gps_count
        if R_gps_temp is not None:
            self.R_gps = R_gps_temp

    def update_magnetometer(self, mag_m, timestamp=None):
        """
        Update step using magnetometer measurements

        Args:
            mag_m: Magnetometer measurement [mx, my, mz] in body frame
            timestamp: Current time in seconds
        """
        if mag_m is None:
            return

        # Extract quaternion from state
        q = self.x[6:10]

        # Normalize magnetometer reading
        mag_norm = np.linalg.norm(mag_m)
        if mag_norm < 1e-10:
            return  # Skip update if magnetometer reading is too small

        mag_m = mag_m / mag_norm

        # Expected magnetic field measurement in body frame
        R = self.quat2Dcm(q)
        h = R.T @ self.mag_reference  # Expected measurement

        # Normalize expected measurement
        h_norm = np.linalg.norm(h)
        if h_norm < 1e-10:
            return

        h = h / h_norm

        # Innovation
        y = mag_m - h

        # Measurement Jacobian
        H = np.zeros((3, self.state_dim))

        # Sensitivity to attitude (quaternion)
        mag_skew = self.skew_symmetric(self.mag_reference)
        dR_dq = self._rotation_matrix_quaternion_jacobian(q)

        for i in range(4):
            H[0:3, 6+i] = -(R.T @ mag_skew @ dR_dq[i])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_mag

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_mag @ K.T

    def update_altitude(self, baro_alt, timestamp=None):
        """
        Update step using barometer altitude measurement

        Args:
            baro_alt: Barometer altitude measurement in NED frame (m)
            timestamp: Current time in seconds
        """
        if baro_alt is None:
            return

        # Measurement model
        h = self.x[2]  # Expected altitude (z position in NED frame)
        z = baro_alt    # Barometer measurement

        # Innovation
        y = z - h

        # Measurement Jacobian
        H = np.zeros((1, self.state_dim))
        H[0, 2] = 1.0  # Direct observation of z position

        # Measurement noise
        R_baro = 2.0  # Barometer noise variance

        # Innovation covariance
        S = H @ self.P @ H.T + R_baro

        # Kalman gain
        K = self.P @ H.T / S

        # Update state
        self.x = self.x + K * y

        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K * R_baro * K.T

    def set_magnetic_parameters(self, declination, inclination, strength=1.0):
        """
        Set local magnetic field parameters

        Args:
            declination: Magnetic declination in degrees
            inclination: Magnetic inclination in degrees
            strength: Magnetic field strength (normalized)
        """
        mag_declination = np.radians(declination)
        mag_inclination = np.radians(inclination)

        # Update magnetic reference vector for NED frame
        self.mag_reference = np.array([
            cos(mag_inclination) * cos(mag_declination),
            cos(mag_inclination) * sin(mag_declination),
            sin(mag_inclination)
        ]) * strength

    def update_zero_velocity(self, threshold=0.05, uncertainty=0.01):
        """
        Apply zero-velocity update when vehicle is detected to be stationary

        Args:
            threshold: Threshold for considering velocity to be zero
            uncertainty: Uncertainty of zero-velocity constraint
        """
        # Check if velocity is close to zero
        vel = self.x[3:6]

        if np.all(np.abs(vel) < threshold):
            # Measurement model: velocity should be zero
            H = np.zeros((3, self.state_dim))
            H[0:3, 3:6] = np.eye(3)  # Measuring velocity states

            # Innovation: difference between current velocity and zero
            y = -vel  # Zero minus current velocity

            # Measurement noise
            R_zupt = np.eye(3) * uncertainty

            # Innovation covariance
            S = H @ self.P @ H.T + R_zupt

            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)

            # Update state
            self.x = self.x + K @ y

            # Update covariance
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_zupt @ K.T

            if self.debug:
                print(
                    f"Zero-velocity update applied, velocity: {np.linalg.norm(vel):.3f} m/s")

    def adapt_process_noise(self):
        """Adapt process noise based on innovation statistics"""
        if len(self.innovation_sequence) < self.adaptive_window:
            return

        # Compute actual innovation covariance from samples
        y_mean = np.mean(self.innovation_sequence, axis=0)
        innovation_cov = np.zeros((6, 6))

        for y in self.innovation_sequence:
            y_diff = y - y_mean
            innovation_cov += np.outer(y_diff, y_diff)

        innovation_cov /= len(self.innovation_sequence)

        # Expected innovation covariance from filter
        expected_cov = np.mean(self.predicted_cov_sequence, axis=0)

        # Adaptasi terpisah untuk posisi dan kecepatan
        pos_scale = np.trace(
            innovation_cov[0:3, 0:3]) / max(np.trace(expected_cov[0:3, 0:3]), 1e-6)
        vel_scale = np.trace(
            innovation_cov[3:6, 3:6]) / max(np.trace(expected_cov[3:6, 3:6]), 1e-6)

        # Batas yang lebih fleksibel
        pos_scale = np.clip(pos_scale, 0.01, 10.0)  # Batas bawah lebih rendah
        vel_scale = np.clip(vel_scale, 0.01, 10.0)  # Batas bawah lebih rendah

        # Adaptasi terpisah berdasarkan jenis state
        self.Q[0:3, 0:3] *= pos_scale  # Position process noise
        self.Q[3:6, 3:6] *= vel_scale  # Velocity process noise

        if self.debug and (abs(pos_scale - 1.0) > 0.1 or abs(vel_scale - 1.0) > 0.1):
            print(
                f"Proses noise diadaptasi: pos={pos_scale:.2f}, vel={vel_scale:.2f}")

    def _quaternion_derivative_matrix(self, omega, dt):
        """Compute transition matrix for quaternion using angular velocity"""
        omega_norm = np.linalg.norm(omega)

        if omega_norm < 1e-10:
            return np.eye(4)

        axis = omega / omega_norm
        angle = omega_norm * dt

        ct = np.cos(angle/2)
        st = np.sin(angle/2)

        # Quaternion derivative matrix
        Omega = np.array([
            [ct, -st*axis[0], -st*axis[1], -st*axis[2]],
            [st*axis[0], ct, st*axis[2], -st*axis[1]],
            [st*axis[1], -st*axis[2], ct, st*axis[0]],
            [st*axis[2], st*axis[1], -st*axis[0], ct]
        ])

        return Omega

    def _quaternion_left_product_matrix(self, q):
        """Create matrix for left quaternion multiplication: q ⊗ p = Q(q) * p"""
        qw, qx, qy, qz = q

        return np.array([
            [qw, -qx, -qy, -qz],
            [qx, qw, -qz, qy],
            [qy, qz, qw, -qx],
            [qz, -qy, qx, qw]
        ])

    def _acceleration_quaternion_jacobian(self, acc_body, q):
        """Compute Jacobian of inertial acceleration with respect to quaternion"""
        # Quaternion to rotation matrix derivatives
        dR_dq = self._rotation_matrix_quaternion_jacobian(q)

        # Jacobian of acceleration wrt quaternion
        J = np.zeros((3, 4))

        # For each quaternion component
        for i in range(4):
            J[:, i] = dR_dq[i] @ acc_body

        return J * self.dt

    def _rotation_matrix_quaternion_jacobian(self, q):
        """Compute Jacobian of rotation matrix with respect to quaternion components"""
        qw, qx, qy, qz = q

        dR_dqw = 2 * np.array([
            [qw, qz, -qy],
            [-qz, qw, qx],
            [qy, -qx, qw]
        ])

        dR_dqx = 2 * np.array([
            [qx, qy, qz],
            [qy, -qx, -qw],
            [qz, qw, -qx]
        ])

        dR_dqy = 2 * np.array([
            [-qy, qx, qw],
            [qx, qy, qz],
            [-qw, qz, -qy]
        ])

        dR_dqz = 2 * np.array([
            [-qz, -qw, qx],
            [qw, -qz, qy],
            [qx, qy, qz]
        ])

        return [dR_dqw, dR_dqx, dR_dqy, dR_dqz]

    def quat2Dcm(self, q):
        """Convert quaternion to Direction Cosine Matrix (body to NED)"""
        qw, qx, qy, qz = q

        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])

    def quatMultiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quatToEuler(self, q):
        """Convert quaternion to Euler angles [roll, pitch, yaw]"""
        qw, qx, qy, qz = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(sinp) if abs(sinp) < 1 else np.sign(sinp) * np.pi / 2

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def get_state(self):
        """Return position, velocity, quaternion, and euler angles"""
        pos = self.x[0:3]
        vel = self.x[3:6]
        quat = self.x[6:10]
        euler = self.quatToEuler(quat)

        return pos, vel, quat, euler
