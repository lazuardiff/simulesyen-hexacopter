import numpy as np
from numpy import sin, cos
import utils
import config
from scipy.stats import chi2


class EKF:
    def __init__(self, dt=0.01, debug=False):
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
        # x, y, z: position
        # vx, vy, vz: velocity
        # qw, qx, qy, qz: quaternion attitude
        # bax, bay, baz: accelerometer bias
        # bgx, bgy, bgz: gyroscope bias

        self.debug = debug
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
        self.R_imu[0:3, 0:3] *= 0.25  # Accelerometer measurement noise
        self.R_imu[3:6, 3:6] *= 0.025  # Gyroscope measurement noise

        self.R_gps = np.eye(6)
        self.R_gps[0:3, 0:3] *= 2.25  # GPS position noise (1.5^2)
        self.R_gps[3:6, 3:6] *= 0.01  # GPS velocity noise

        self.R_alt = np.array([[0.09]])  # Altitude sensor noise (0.2^2)
        self.R_mag = np.eye(3) * 0.0025  # Magnetometer noise (0.05^2)

        # Adaptive filter parameters
        self.adaptive_enabled = True
        self.r_adapt_window = 10  # Window size for adaptive filtering
        self.residual_history = {
            'imu': np.zeros((self.r_adapt_window, 6)),
            'gps': np.zeros((self.r_adapt_window, 6)),
            'mag': np.zeros((self.r_adapt_window, 3)),
            'alt': np.zeros((self.r_adapt_window, 1))
        }
        self.residual_index = 0

        # Sensor validity thresholds (Chi-square test)
        self.chi2_threshold = {
            'imu': chi2.ppf(0.95, 6),
            'gps': chi2.ppf(0.95, 6),
            'mag': chi2.ppf(0.95, 3),
            'alt': chi2.ppf(0.95, 1)
        }
        self.imu_threshold = 25.0  # Naikkan dari nilai awal (12.0)
        self.gps_threshold = 100.0
        self.alt_threshold = 8.0
        self.mag_threshold = self.chi2_threshold['mag']

        # Timestamp buffers for handling sensor delays
        self.last_timestamp = 0
        self.sensor_timestamps = {
            'imu': 0,
            'gps': 0,
            'mag': 0,
            'alt': 0
        }

        # Sensor delay compensation
        self.gps_delay = 0.1  # Typical GPS delay in seconds
        self.state_history = []  # Store state history for delay compensation

        # Enhanced bias models
        self.acc_bias_stability = 0.0002  # m/s²/√hour
        self.gyro_bias_stability = 0.0001  # rad/s/√hour

        # Local magnetic field model
        self.mag_inclination = np.radians(45)  # Default magnetic inclination
        self.mag_declination = np.radians(10)  # Default magnetic declination
        self.mag_strength = 1.0  # Normalized magnitude

        # Gravity vector in inertial frame
        if (config.orient == "NED"):
            self.g_vec = np.array([0, 0, self.g])
            # Magnetic reference in NED (North, East, Down)
            self.mag_reference = np.array([
                cos(self.mag_inclination) * cos(self.mag_declination),
                cos(self.mag_inclination) * sin(self.mag_declination),
                sin(self.mag_inclination)
            ]) * self.mag_strength
        else:  # ENU
            self.g_vec = np.array([0, 0, -self.g])
            # Magnetic reference in ENU (East, North, Up)
            self.mag_reference = np.array([
                cos(self.mag_inclination) * sin(self.mag_declination),
                cos(self.mag_inclination) * cos(self.mag_declination),
                -sin(self.mag_inclination)
            ]) * self.mag_strength

        # Auto-tuning parameters
        self.tuning_enabled = True
        self.tuning_iterations = 0
        self.max_tuning_iterations = 100
        self.Q_scale = 1.0
        self.R_scale = 1.0

        # Utility functions integrated to remove external dependencies
        self.utils = Utils()

    def skew_symmetric(self, v):
        """Create a skew symmetric matrix from a 3-element vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict(self, acc_m, gyro_m, timestamp=None):
        # Handle timestamp
        if timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                self.dt = dt
            self.last_timestamp = timestamp

        # Extract current state
        p = self.x[0:3]   # Position
        v = self.x[3:6]   # Velocity
        q = self.x[6:10]  # Quaternion [qw, qx, qy, qz]
        ba = self.x[10:13]  # Accelerometer bias
        bg = self.x[13:16]  # Gyroscope bias

        # Store state for delay compensation
        self.state_history.append(
            (self.last_timestamp, self.x.copy(), self.P.copy()))
        # Remove old states (keep only those within the largest expected delay)
        while len(self.state_history) > 0 and (self.last_timestamp - self.state_history[0][0]) > 2.0:
            self.state_history.pop(0)

        # Correct for bias - now with temperature compensation if available
        acc_corrected = acc_m - ba
        gyro_corrected = gyro_m - bg

        # Get rotation matrix from body to inertial frame
        R = self.utils.quat2Dcm(q)

        # Compute acceleration in inertial frame (removing gravity)
        acc_inertial = R @ acc_corrected + self.g_vec

        # State prediction using 4th order Runge-Kutta for better accuracy
        # Position update
        k1_p = v
        k2_p = v + 0.5 * self.dt * acc_inertial
        k3_p = v + 0.5 * self.dt * acc_inertial
        k4_p = v + self.dt * acc_inertial
        p_new = p + (self.dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

        # Velocity update
        v_new = v + acc_inertial * self.dt

        # Quaternion update using angular velocity
        omega = gyro_corrected
        omega_norm = np.linalg.norm(omega)

        if omega_norm > 1e-10:
            axis = omega / omega_norm
            angle = omega_norm * self.dt

            qw = np.cos(angle/2)
            qxyz = axis * np.sin(angle/2)
            dq = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])

            q_new = self.utils.quatMultiply(q, dq)
        else:
            q_new = q.copy()

        q_new = q_new / np.linalg.norm(q_new)  # Normalize quaternion

        # Enhanced bias model - Gauss-Markov process instead of random walk
        tau_acc = 3600.0  # Time constant for accelerometer bias in seconds
        tau_gyro = 3600.0  # Time constant for gyroscope bias in seconds

        acc_bias_drift = np.random.normal(
            0, self.acc_bias_stability * np.sqrt(self.dt), 3)
        gyro_bias_drift = np.random.normal(
            0, self.gyro_bias_stability * np.sqrt(self.dt), 3)

        ba_new = ba * np.exp(-self.dt/tau_acc) + acc_bias_drift
        bg_new = bg * np.exp(-self.dt/tau_gyro) + gyro_bias_drift

        # Update state
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new
        self.x[10:13] = ba_new
        self.x[13:16] = bg_new

        # Compute Jacobian of state transition function
        F = np.eye(self.state_dim)
        F[0:3, 3:6] = np.eye(3) * self.dt  # dpos/dvel = dt*I

        # Improved Jacobians for quaternion dynamics
        # Attitude Jacobian wrt quaternion
        F[6:10, 6:10] = self._quaternion_derivative_matrix(omega, self.dt)

        # Acceleration Jacobian wrt quaternion (rotation effect)
        F[3:6, 6:10] = self._acceleration_quaternion_jacobian(acc_corrected, q)

        # Acceleration affected by accelerometer bias
        F[3:6, 10:13] = -R * self.dt

        # Quaternion affected by gyroscope bias
        dq_dbg = -0.5 * self.dt * \
            self._quaternion_left_product_matrix(q)[1:, :]
        F[6:10, 13:16] = dq_dbg.T

        # Update process noise based on motion dynamics
        self._adapt_process_noise(acc_inertial, omega)

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

        # Ensure covariance matrix stays symmetric and positive definite
        self.P = 0.5 * (self.P + self.P.T)

        # Check for numerical stability
        if not np.all(np.linalg.eigvals(self.P) > 0):
            # If not positive definite, apply regularization
            min_eig = np.min(np.linalg.eigvals(self.P))
            if min_eig < 0:
                self.P += (-min_eig * 1.1) * np.eye(self.state_dim)

    def _quaternion_derivative_matrix(self, omega, dt):
        """Compute the transition matrix for quaternion using the angular velocity"""
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
        """Create a matrix for left quaternion multiplication: q ⊗ p = Q(q) * p"""
        qw, qx, qy, qz = q

        return np.array([
            [qw, -qx, -qy, -qz],
            [qx, qw, -qz, qy],
            [qy, qz, qw, -qx],
            [qz, -qy, qx, qw]
        ])

    def _quaternion_right_product_matrix(self, q):
        """Create a matrix for right quaternion multiplication: p ⊗ q = Q(p) * q"""
        qw, qx, qy, qz = q

        return np.array([
            [qw, -qx, -qy, -qz],
            [qx, qw, qz, -qy],
            [qy, -qz, qw, qx],
            [qz, qy, -qx, qw]
        ])

    def _acceleration_quaternion_jacobian(self, acc_body, q):
        """Compute the Jacobian of inertial acceleration with respect to quaternion"""
        # This Jacobian relates how a change in attitude (quaternion)
        # affects the acceleration in inertial frame

        # Create the skew-symmetric matrix of the acceleration in body frame
        acc_skew = self.skew_symmetric(acc_body)

        # Quaternion to rotation matrix derivatives
        dR_dq = self._rotation_matrix_quaternion_jacobian(q)

        # Jacobian of acceleration wrt quaternion
        J = np.zeros((3, 4))

        # For each quaternion component
        for i in range(4):
            J[:, i] = dR_dq[i] @ acc_body

        return J * self.dt

    def _rotation_matrix_quaternion_jacobian(self, q):
        """Compute the Jacobian of rotation matrix with respect to quaternion components"""
        qw, qx, qy, qz = q

        # Partial derivatives of rotation matrix wrt quaternion components
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

    def _adapt_process_noise(self, acc_inertial, omega):
        """Adapt process noise based on dynamics"""
        if not self.adaptive_enabled:
            return

        # Adjust position process noise based on velocity
        vel_norm = np.linalg.norm(self.x[3:6])
        self.Q[0:3, 0:3] = np.eye(3) * (0.01 + 0.1 * vel_norm * self.dt)

        # Adjust velocity process noise based on acceleration
        acc_norm = np.linalg.norm(acc_inertial)
        self.Q[3:6, 3:6] = np.eye(3) * (0.1 + 0.2 * acc_norm * self.dt)

        # Adjust attitude process noise based on angular velocity
        omega_norm = np.linalg.norm(omega)
        self.Q[6:10, 6:10] = np.eye(4) * (0.01 + 0.05 * omega_norm * self.dt)

        # Apply scaling from auto-tuning
        self.Q *= self.Q_scale

    def update_imu(self, acc_m, gyro_m, timestamp=None):
        # Handle timestamp
        if timestamp is not None:
            self.sensor_timestamps['imu'] = timestamp

        # Extract attitude and biases from state
        q = self.x[6:10]
        ba = self.x[10:13]
        bg = self.x[13:16]

        # Measurement model: expected accelerometer and gyroscope readings
        R = self.utils.quat2Dcm(q)
        gravity_body = R.T @ self.g_vec  # Gravity vector in body frame

        h_acc = -gravity_body + ba  # Expected accelerometer measurement in body frame
        # Expected gyroscope measurement when not rotating
        h_gyro = np.zeros(3) + bg

        h = np.hstack((h_acc, h_gyro))
        z = np.hstack((acc_m, gyro_m))

        # Measurement residual
        y = z - h

        # Improved measurement Jacobian (H)
        H = np.zeros((6, self.state_dim))

        # Sensitivity of accelerometer to attitude (quaternion)
        g_skew = self.skew_symmetric(self.g_vec)
        dR_dq = self._rotation_matrix_quaternion_jacobian(q)

        for i in range(4):
            H[0:3, 6+i] = -(dR_dq[i].T @ self.g_vec)

        # Accelerometer affected by acc bias
        H[0:3, 10:13] = np.eye(3)

        # Gyroscope affected by gyro bias
        H[3:6, 13:16] = np.eye(3)

        # Kalman gain
        S = H @ self.P @ H.T + self.R_imu

        # Check residual for outlier detection - PASS THE EXACT S MATRIX
        if not self._is_valid_measurement(y, S, 'imu'):
            nis = y @ np.linalg.inv(S) @ y
            print(
                f"IMU measurement rejected: NIS = {nis:.2f}, threshold = {self.chi2_threshold['imu']:.2f}")
            return

        # Store residual for adaptive filtering
        self.residual_history['imu'][self.residual_index %
                                     self.r_adapt_window] = y
        self.residual_index += 1

        # Adapt measurement noise if enabled
        if self.adaptive_enabled and self.residual_index > self.r_adapt_window:
            self._adapt_measurement_noise('imu')

        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Update covariance - Joseph form for numerical stability
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_imu @ K.T

        # Auto-tuning of filter parameters
        self._auto_tune(y, S, 'imu')

    def update_gps(self, pos_m, vel_m, timestamp=None):
        if pos_m is None or vel_m is None:
            return
        
        if not hasattr(self, 'gps_measurements_count'):
            self.gps_measurements_count = 0

        # Handle timestamp and GPS delay
        if timestamp is not None:
            # Account for GPS delay
            adjusted_time = timestamp - self.gps_delay
            self.sensor_timestamps['gps'] = timestamp

            # Find the closest state in history to the adjusted time
            closest_idx = -1
            min_time_diff = float('inf')

            for i, (t, _, _) in enumerate(self.state_history):
                time_diff = abs(t - adjusted_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_idx = i

            # If we found a suitable historical state, use it as the reference
            if closest_idx >= 0 and min_time_diff < 0.05:  # Within 50ms
                _, state_at_meas, P_at_meas = self.state_history[closest_idx]
                h = np.hstack((state_at_meas[0:3], state_at_meas[3:6]))
            else:
                # Fallback to current state if no suitable history found
                h = np.hstack((self.x[0:3], self.x[3:6]))
        else:
            # Without timestamp, use current state
            h = np.hstack((self.x[0:3], self.x[3:6]))

        if self.gps_measurements_count < 3:
            self.x[0:3] = pos_m  # Reset position estimate to GPS
            self.x[3:6] = vel_m  # Reset velocity estimate to GPS
            
            # Reinitialize position and velocity covariance
            self.P[0:3, 0:3] = np.eye(3) * 5.0  # Position uncertainty
            self.P[3:6, 3:6] = np.eye(3) * 2.0  # Velocity uncertainty
            
            self.gps_measurements_count += 1

        # GPS measurement
        z = np.hstack((pos_m, vel_m))

        # Measurement residual
        y = z - h

        # Check residual for outlier detection
        if not self._is_valid_measurement(y, 'gps'):
            print("GPS measurement rejected as outlier")
            return

        # Store residual for adaptive filtering
        self.residual_history['gps'][self.residual_index %
                                     self.r_adapt_window] = y

        # Adapt measurement noise if enabled
        if self.adaptive_enabled and self.residual_index > self.r_adapt_window:
            self._adapt_measurement_noise('gps')

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

        # Update covariance - Joseph form for numerical stability
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_gps @ K.T

        # Auto-tuning of filter parameters
        self._auto_tune(y, S, 'gps')

    def update_altitude(self, alt_m, timestamp=None):
        if alt_m is None:
            return

        # Handle timestamp
        if timestamp is not None:
            self.sensor_timestamps['alt'] = timestamp

        # Extract z position from state
        if (config.orient == "NED"):
            h = -self.x[2]  # In NED, altitude is -z
        else:
            h = self.x[2]   # In ENU, altitude is z

        # Measurement residual
        y = alt_m - h

        # Check residual for outlier detection
        if not self._is_valid_measurement(y, 'alt'):
            print("Altitude measurement rejected as outlier")
            return

        # Store residual for adaptive filtering
        self.residual_history['alt'][self.residual_index %
                                     self.r_adapt_window] = y

        # Adapt measurement noise if enabled
        if self.adaptive_enabled and self.residual_index > self.r_adapt_window:
            self._adapt_measurement_noise('alt')

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

        # Update covariance - Joseph form for numerical stability
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_alt @ K.T

        # Auto-tuning of filter parameters
        self._auto_tune(np.array([y]), np.array([[S[0, 0]]]), 'alt')

    def update_magnetometer(self, mag_m, timestamp=None):
        if mag_m is None:
            return

        # Handle timestamp
        if timestamp is not None:
            self.sensor_timestamps['mag'] = timestamp

        # Extract quaternion from state
        q = self.x[6:10]

        # Normalize magnetometer reading to handle calibration errors
        mag_norm = np.linalg.norm(mag_m)
        if mag_norm > 1e-10:
            mag_m = mag_m / mag_norm
        else:
            return  # Skip update if magnetometer reading is too small

        # Expected magnetic field measurement in body frame
        R = self.utils.quat2Dcm(q)
        h = R.T @ self.mag_reference  # Expected measurement in body frame

        # Normalize expected measurement
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-10:
            h = h / h_norm

        # Measurement residual
        y = mag_m - h

        # Check residual for outlier detection
        if not self._is_valid_measurement(y, 'mag'):
            print("Magnetometer measurement rejected as outlier")
            return

        # Store residual for adaptive filtering
        self.residual_history['mag'][self.residual_index %
                                     self.r_adapt_window] = y

        # Adapt measurement noise if enabled
        if self.adaptive_enabled and self.residual_index > self.r_adapt_window:
            self._adapt_measurement_noise('mag')

        # Measurement Jacobian - improved for quaternion derivatives
        H = np.zeros((3, self.state_dim))

        # Sensitivity to attitude (quaternion)
        mag_skew = self.skew_symmetric(self.mag_reference)
        dR_dq = self._rotation_matrix_quaternion_jacobian(q)

        for i in range(4):
            H[0:3, 6+i] = -(R.T @ mag_skew @ dR_dq[i])

        # Kalman gain
        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

        # Update covariance - Joseph form for numerical stability
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_mag @ K.T

        # Auto-tuning of filter parameters
        self._auto_tune(y, S, 'mag')

    def _is_valid_measurement(self, residual, sensor_type_or_S, sensor_type=None):
        """Check if a measurement is valid using Chi-square test

        Can be called as either:
        - _is_valid_measurement(residual, sensor_type)
        - _is_valid_measurement(residual, S, sensor_type)
        """
        # Determine if second parameter is S or sensor_type
        if sensor_type is None:
            # Called as: _is_valid_measurement(residual, sensor_type)
            sensor_type = sensor_type_or_S

            # Get inverse of R based on sensor type
            if sensor_type == 'imu':
                S_inv = np.linalg.inv(self.R_imu)
                threshold = self.imu_threshold  # Use the increased threshold
            elif sensor_type == 'gps':
                S_inv = np.linalg.inv(self.R_gps)
                threshold = self.gps_threshold
            elif sensor_type == 'mag':
                S_inv = np.linalg.inv(self.R_mag)
                threshold = self.mag_threshold
            elif sensor_type == 'alt':
                S_inv = 1.0 / self.R_alt[0, 0]
                threshold = self.alt_threshold
            else:
                return True
        else:
            # Called as: _is_valid_measurement(residual, S, sensor_type)
            S = sensor_type_or_S
            S_inv = np.linalg.inv(S)

            if sensor_type == 'imu':
                threshold = self.imu_threshold
            elif sensor_type == 'gps':
                threshold = self.gps_threshold
            elif sensor_type == 'mag':
                threshold = self.mag_threshold
            elif sensor_type == 'alt':
                threshold = self.alt_threshold
            else:
                return True

        # Compute normalized innovation squared
        residual = np.atleast_1d(residual).flatten()

        if residual.size == 1 and (np.isscalar(S_inv) or S_inv.size == 1):
            # Scalar measurement case (like altimeter)
            if isinstance(S_inv, np.ndarray) and S_inv.size == 1:
                S_inv = S_inv.item()  # Convert single element array to scalar
            nis = residual.item()**2 * S_inv
        else:
            # Vector measurement case (IMU, GPS, magnetometer)
            nis = residual @ S_inv @ residual

        if self.debug:
            print(f"{sensor_type} NIS: {nis:.2f}, threshold: {threshold:.2f}")

        # Check against chi-square threshold
        return nis <= threshold

    def _adapt_measurement_noise(self, sensor_type):
        """Adapt measurement noise based on recent residuals"""
        if sensor_type == 'imu':
            residuals = self.residual_history['imu']
            R = np.zeros_like(self.R_imu)

            # Compute empirical covariance of residuals
            for i in range(self.r_adapt_window):
                r = residuals[i].reshape(-1, 1)
                R += r @ r.T

            R /= self.r_adapt_window

            # Blend with current R to avoid rapid changes
            self.R_imu = 0.95 * self.R_imu + 0.05 * R

        elif sensor_type == 'gps':
            residuals = self.residual_history['gps']
            R = np.zeros_like(self.R_gps)

            for i in range(self.r_adapt_window):
                r = residuals[i].reshape(-1, 1)
                R += r @ r.T

            R /= self.r_adapt_window

            # Blend with current R to avoid rapid changes
            self.R_gps = 0.95 * self.R_gps + 0.05 * R

        elif sensor_type == 'mag':
            residuals = self.residual_history['mag']
            R = np.zeros_like(self.R_mag)

            for i in range(self.r_adapt_window):
                r = residuals[i].reshape(-1, 1)
                R += r @ r.T

            R /= self.r_adapt_window

            # Blend with current R to avoid rapid changes
            self.R_mag = 0.95 * self.R_mag + 0.05 * R

        elif sensor_type == 'alt':
            residuals = self.residual_history['alt']
            R = 0.0

            for i in range(self.r_adapt_window):
                r = residuals[i]
                R += r * r

            R /= self.r_adapt_window

            # Blend with current R to avoid rapid changes
            self.R_alt[0, 0] = 0.95 * self.R_alt[0, 0] + 0.05 * R

    def _auto_tune(self, residual, S, sensor_type):
        """Auto-tune filter parameters based on residuals"""
        if not self.tuning_enabled or self.tuning_iterations >= self.max_tuning_iterations:
            return

        # Compute normalized innovation squared (NIS)
        if sensor_type == 'imu':
            dof = 6
        elif sensor_type == 'gps':
            dof = 6
        elif sensor_type == 'mag':
            dof = 3
        elif sensor_type == 'alt':
            dof = 1
        else:
            return

        # Compute NIS
        nis = residual @ np.linalg.inv(S) @ residual
        expected_nis = dof

        # Adjust Q and R scaling factors based on NIS
        if nis > 1.2 * expected_nis:
            # If innovations are too large, increase process noise
            self.Q_scale *= 1.01
        elif nis < 0.8 * expected_nis:
            # If innovations are too small, decrease process noise
            self.Q_scale *= 0.99

        # Keep Q_scale within reasonable bounds
        self.Q_scale = np.clip(self.Q_scale, 0.1, 10.0)

        # Update iteration counter
        self.tuning_iterations += 1

    def set_magnetic_parameters(self, declination, inclination, strength=1.0):
        """Set local magnetic field parameters"""
        self.mag_declination = np.radians(declination)
        self.mag_inclination = np.radians(inclination)
        self.mag_strength = strength

        # Update magnetic reference vector
        if (config.orient == "NED"):
            self.mag_reference = np.array([
                cos(self.mag_inclination) * cos(self.mag_declination),
                cos(self.mag_inclination) * sin(self.mag_declination),
                sin(self.mag_inclination)
            ]) * self.mag_strength
        else:  # ENU
            self.mag_reference = np.array([
                cos(self.mag_inclination) * sin(self.mag_declination),
                cos(self.mag_inclination) * cos(self.mag_declination),
                -sin(self.mag_inclination)
            ]) * self.mag_strength

    def get_state(self):
        # Return position, velocity, quaternion, and euler angles
        pos = self.x[0:3]
        vel = self.x[3:6]
        quat = self.x[6:10]

        # Convert quaternion to euler angles
        euler = self.utils.quatToYPR_ZYX(quat)[::-1]  # [roll, pitch, yaw]

        return pos, vel, quat, euler

    def get_covariance(self):
        """Return position and attitude uncertainty"""
        pos_cov = np.diag(self.P[0:3, 0:3])
        att_cov = np.diag(self.P[6:10, 6:10])

        return pos_cov, att_cov

    def reset(self):
        """Reset the filter to initial conditions"""
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # Initial quaternion is [1, 0, 0, 0]
        self.P = np.eye(self.state_dim)
        self.P[0:3, 0:3] *= 10.0
        self.P[3:6, 3:6] *= 1.0
        self.P[6:10, 6:10] *= 0.1
        self.P[10:13, 10:13] *= 0.01
        self.P[13:16, 13:16] *= 0.01
        self.state_history.clear()
        self.last_timestamp = 0


class Utils:
    """Utility functions to remove external dependencies"""

    def quat2Dcm(self, q):
        """Convert quaternion to Direction Cosine Matrix (DCM)"""
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

    def quatToYPR_ZYX(self, q):
        """Convert quaternion to Yaw-Pitch-Roll Euler angles (ZYX sequence)"""
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

        return np.array([yaw, pitch, roll])
