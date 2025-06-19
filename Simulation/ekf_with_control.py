"""
EKF with Control Input Implementation
====================================

This file contains the Extended Kalman Filter implementation that uses
control input data (motor thrusts, control torques) for enhanced prediction.

Workflow:
1. IMU Prediction + Control Physics Model
2. GPS Position/Velocity Updates  
3. Barometer Altitude Updates
4. Magnetometer Yaw Updates
5. Control Input Validation and Fusion

Author: EKF Implementation Team
Date: 2025
"""

import numpy as np
import pandas as pd
from ekf_common import BaseEKF
import matplotlib.pyplot as plt


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

        # # === STEP 3: PHYSICS-BASED ACCELERATION PREDICTION ===
        # accel_physics = None
        # control_quality = 0.0

        # if control_data is not None:
        #     # Priority 1: Individual motor thrusts (most accurate)
        #     if 'motor_thrusts' in control_data and control_data['motor_thrusts'] is not None:
        #         motor_thrusts = control_data['motor_thrusts']
        #         total_thrust = np.sum(motor_thrusts)

        #         if len(motor_thrusts) == 6 and self.min_thrust_threshold < total_thrust < self.max_thrust_threshold:
        #             # Check thrust distribution quality
        #             thrust_variance = np.var(motor_thrusts)
        #             thrust_mean = np.mean(motor_thrusts)
        #             thrust_cv = thrust_variance / \
        #                 (thrust_mean + 1e-6)  # Coefficient of variation

        #             # Quality metric based on thrust consistency and magnitude
        #             control_quality = min(
        #                 1.0, total_thrust / 20.0) * max(0.1, 1.0 - thrust_cv)

        #             accel_physics = self.dynamics.predict_acceleration_from_motor_thrusts(
        #                 motor_thrusts, R_bn)
        #             self.prediction_mode = "MOTOR_THRUSTS"

        #     # Priority 2: Control thrust vector
        #     elif 'thrust_sp' in control_data and control_data['thrust_sp'] is not None:
        #         thrust_sp = control_data['thrust_sp']
        #         thrust_magnitude = np.linalg.norm(thrust_sp)

        #         if thrust_magnitude > self.min_thrust_threshold:
        #             # Slightly lower quality
        #             control_quality = min(1.0, thrust_magnitude / 20.0) * 0.8

        #             accel_physics = self.dynamics.predict_acceleration_from_thrust_vector(
        #                 thrust_sp, R_bn)
        #             self.prediction_mode = "THRUST_VECTOR"

        #     # Priority 3: Total thrust (assume vertical)
        #     elif 'total_thrust' in control_data and control_data['total_thrust'] is not None:
        #         total_thrust = control_data['total_thrust']

        #         if self.min_thrust_threshold < total_thrust < self.max_thrust_threshold:
        #             control_quality = min(
        #                 1.0, total_thrust / 20.0) * 0.6  # Lower quality

        #             # Vertical thrust assumption
        #             thrust_body = np.array([0, 0, -total_thrust])
        #             accel_physics = self.dynamics.predict_acceleration_from_thrust_vector(
        #                 thrust_body, R_bn)
        #             self.prediction_mode = "TOTAL_THRUST"

        # === STEP 4: IMU-BASED ACCELERATION (BASELINE) ===
        accel_imu = R_bn @ accel_corrected + self.g_ned

        # === STEP 6: KINEMATIC INTEGRATION ===
        # Position and velocity integration
        vel_mid = vel + 0.5 * accel_imu * self.dt  # Midpoint integration
        pos_new = pos + vel_mid * self.dt
        vel_new = vel + accel_imu * self.dt

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


def run_ekf_with_control_data(csv_file_path, use_magnetometer=True, magnetic_declination=0.0):
    """
    Main function to run EKF with control input data

    Workflow:
    1. Load and validate simulation data
    2. Initialize EKF with first valid sensor measurements
    3. Process all timesteps with prediction and measurement updates
    4. Calculate error statistics vs ground truth
    5. Return results for analysis

    Args:
        csv_file_path: Path to simulation data CSV
        use_magnetometer: Enable magnetometer updates for yaw
        magnetic_declination: Local magnetic declination in degrees

    Returns:
        tuple: (ekf_instance, results_dict, raw_data)
    """

    print("\n" + "="*80)
    print("EKF WITH CONTROL INPUT - ENHANCED PREDICTION")
    print("="*80)

    # === STEP 1: LOAD AND VALIDATE DATA ===
    try:
        data = pd.read_csv(csv_file_path)
        print(f"✅ Simulation data loaded: {len(data)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

    # Check control data availability
    control_columns = {
        'motor_thrusts': ['motor_thrust_1', 'motor_thrust_2', 'motor_thrust_3',
                          'motor_thrust_4', 'motor_thrust_5', 'motor_thrust_6'],
        'thrust_sp': ['thrust_sp_x', 'thrust_sp_y', 'thrust_sp_z'],
        'control_torques': ['control_torque_x', 'control_torque_y', 'control_torque_z']
    }

    data_availability = {}
    for key, columns in control_columns.items():
        data_availability[key] = all(col in data.columns for col in columns)

    print(f"\n📊 Control Data Availability:")
    for key, available in data_availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {key}: {available}")

    # Calculate sampling time
    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"🕐 Sampling time: {dt_mean:.4f} s ({1/dt_mean:.1f} Hz)")

    # === STEP 2: INITIALIZE EKF ===
    ekf = EKFWithControl(dt=dt_mean)
    ekf.mag_declination = np.deg2rad(magnetic_declination)
    print(f"🧭 Magnetic declination: {magnetic_declination:.1f}°")

    # Find initialization point with complete sensor data
    init_idx = find_initialization_point(data)
    if init_idx is None:
        print("❌ No suitable initialization data found!")
        return None

    # Initialize EKF state
    success = initialize_ekf_state(ekf, data.iloc[init_idx], use_magnetometer)
    if not success:
        print("❌ EKF initialization failed!")
        return None

    # === STEP 3: PROCESS ALL DATA ===
    results = process_all_data(ekf, data, data_availability, use_magnetometer)

    if len(results['timestamp']) < 100:
        print(f"❌ Insufficient valid results: {len(results['timestamp'])}")
        return None

    # === STEP 4: CALCULATE ERRORS AND STATISTICS ===
    # MODIFIED: RMSE calculation now starts from 5 seconds
    error_stats = calculate_error_statistics(results, data, start_time=5.0)
    print_performance_summary(error_stats, results)

    print("✅ EKF with control input processing completed!")
    return ekf, results, data


def find_initialization_point(data):
    """Find optimal initialization point with complete sensor data"""
    min_start_time = 0.1  # Skip startup period

    for i in range(len(data)):
        row = data.iloc[i]

        if row['timestamp'] < min_start_time:
            continue

        # Check GPS availability
        if row['gps_available'] != 1:
            continue

        # Validate GPS data
        gps_pos = np.array(
            [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array(
            [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

        if np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel)):
            continue

        # Validate IMU data
        acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        if np.any(np.isnan(acc)) or np.any(np.isnan(gyro)) or np.linalg.norm(acc) < 0.1:
            continue

        print(
            f"🎯 Initialization point found: index {i}, time {row['timestamp']:.3f}s")
        return i

    return None


def initialize_ekf_state(ekf, row, use_magnetometer):
    """Initialize EKF state with sensor data"""
    try:
        gps_pos = np.array(
            [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array(
            [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])
        initial_acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        initial_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        # Magnetometer data if available
        initial_mag = None
        if use_magnetometer and 'mag_available' in row and row['mag_available'] == 1:
            initial_mag = np.array([row['mag_x'], row['mag_y'], row['mag_z']])

        # Use ground truth yaw for testing purposes
        true_yaw = row['true_yaw'] if use_magnetometer else None

        ekf.initialize_state(gps_pos, gps_vel, initial_acc,
                             initial_gyro, initial_mag, true_yaw)

        try:
            q_gt = np.array([
                row['true_quat_w'],
                row['true_quat_x'],
                row['true_quat_y'],
                row['true_quat_z']
            ])
            ekf.x[6:10] = ekf.normalize_quaternion(q_gt)
            # Sedikit turunkan kovarians attitude agar EKF “percaya”
            ekf.P[6:9, 6:9] = np.diag([0.04, 0.04, 1.0])
            print("✅  Ground-truth quaternion dipakai untuk inisialisasi")
        except KeyError:
            print("⚠️  Kolom quaternion ground-truth tidak ditemukan, lewati patch")

        return True

    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        return False


def process_all_data(ekf, data, data_availability, use_magnetometer):
    """Process all simulation data through EKF"""
    n_samples = len(data)
    min_processing_time = 0.05

    # Initialize results storage
    results = {
        'timestamp': [], 'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [], 'pos_std': [], 'vel_std': [], 'att_std': [],
        'prediction_modes': [], 'control_quality': []
    }

    # Statistics counters
    stats = {
        'prediction_count': 0, 'gps_updates': 0, 'baro_updates': 0,
        'mag_updates': 0, 'control_used': 0, 'skipped_samples': 0
    }

    print(f"🔄 Processing {n_samples} samples...")

    for i in range(n_samples):
        row = data.iloc[i]

        # Skip early timestamps
        if row['timestamp'] < min_processing_time:
            stats['skipped_samples'] += 1
            continue

        # Validate and prepare IMU data
        try:
            accel_body, gyro_body = prepare_imu_data(row)
        except ValueError:
            stats['skipped_samples'] += 1
            continue

        # Prepare control data
        control_data = prepare_control_data(row, data_availability)
        if control_data:
            stats['control_used'] += 1

        # === PREDICTION STEP ===
        try:
            ekf.predict_with_control_input(accel_body, gyro_body, control_data)
            stats['prediction_count'] += 1
        except Exception as e:
            print(f"⚠️  Prediction failed at sample {i}: {str(e)}")
            continue

        # === MEASUREMENT UPDATES ===
        stats['gps_updates'] += update_gps_measurements(ekf, row)
        stats['baro_updates'] += update_barometer_measurement(ekf, row)

        if use_magnetometer:
            stats['mag_updates'] += update_magnetometer_measurement(ekf, row)

        # Store results
        try:
            state = ekf.get_state()
            store_results(results, row, state)
        except Exception as e:
            print(f"⚠️  Failed to store results at sample {i}: {str(e)}")
            continue

        # Progress indicator
        if i % 5000 == 0 and i > 0:
            print(f"   📈 Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%)")

    # Print processing statistics
    print(f"\n📊 Processing Statistics:")
    print(f"  ✅ Valid predictions: {stats['prediction_count']}")
    print(f"  📡 GPS updates: {stats['gps_updates']}")
    print(f"  🌡️  Barometer updates: {stats['baro_updates']}")
    print(f"  🧭 Magnetometer updates: {stats['mag_updates']}")
    print(f"  🎮 Control data used: {stats['control_used']}")
    print(f"  ⏭️  Skipped samples: {stats['skipped_samples']}")

    return results


def prepare_imu_data(row):
    """Prepare and validate IMU data"""
    accel_body = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
    gyro_body = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

    # Validation
    acc_norm = np.linalg.norm(accel_body)
    if (np.any(np.isnan(accel_body)) or np.any(np.isnan(gyro_body)) or
            acc_norm < 0.1 or acc_norm > 100):
        raise ValueError("Invalid IMU data")

    # Safety clipping
    accel_body = np.clip(accel_body, -50, 50)
    gyro_body = np.clip(gyro_body, -20, 20)

    return accel_body, gyro_body


def prepare_control_data(row, data_availability):
    """Prepare control input data with validation"""
    control_data = {}

    # Motor thrusts (highest priority)
    if data_availability['motor_thrusts']:
        motor_thrusts = np.array([
            row['motor_thrust_1'], row['motor_thrust_2'], row['motor_thrust_3'],
            row['motor_thrust_4'], row['motor_thrust_5'], row['motor_thrust_6']
        ])
        if (np.sum(motor_thrusts) > 0.1 and not np.any(np.isnan(motor_thrusts)) and
                not np.allclose(motor_thrusts, 0, atol=1e-6)):
            control_data['motor_thrusts'] = motor_thrusts

    # Thrust setpoint (second priority)
    if data_availability['thrust_sp'] and 'motor_thrusts' not in control_data:
        thrust_sp = np.array(
            [row['thrust_sp_x'], row['thrust_sp_y'], row['thrust_sp_z']])
        if (np.linalg.norm(thrust_sp) > 0.1 and not np.any(np.isnan(thrust_sp)) and
                not np.allclose(thrust_sp, 0, atol=1e-6)):
            control_data['thrust_sp'] = thrust_sp

    # Total thrust (fallback)
    if 'total_thrust_sp' in row and len(control_data) == 0:
        total_thrust = row['total_thrust_sp']
        if not np.isnan(total_thrust) and 0.1 < total_thrust < 100:
            control_data['total_thrust'] = total_thrust

    # Control torques (additional info)
    if data_availability['control_torques']:
        control_torques = np.array([
            row['control_torque_x'], row['control_torque_y'], row['control_torque_z']
        ])
        if not np.any(np.isnan(control_torques)) and not np.allclose(control_torques, 0, atol=1e-6):
            control_data['control_torques'] = control_torques

    return control_data if control_data else None


def update_gps_measurements(ekf, row):
    """Update GPS measurements if available"""
    if row['gps_available'] != 1:
        return 0

    gps_pos = np.array(
        [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
    gps_vel = np.array(
        [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

    # Validate GPS data
    if (np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel)) or
            np.allclose(gps_pos, 0, atol=1e-6) or np.linalg.norm(gps_pos) > 10000):
        return 0

    ekf.update_gps_position(gps_pos)
    ekf.update_gps_velocity(gps_vel)
    return 1


def update_barometer_measurement(ekf, row):
    """Update barometer measurement if available"""
    if row['baro_available'] != 1:
        return 0

    baro_alt = row['baro_altitude']
    if np.isnan(baro_alt) or not (-1000 < baro_alt < 10000) or abs(baro_alt) <= 1e-6:
        return 0

    ekf.update_barometer(baro_alt)
    return 1


def update_magnetometer_measurement(ekf, row):
    """Update magnetometer measurement if available"""
    if 'mag_available' not in row or row['mag_available'] != 1:
        return 0

    mag_body = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
    if (np.any(np.isnan(mag_body)) or np.allclose(mag_body, 0, atol=1e-6) or
            not (0.1 < np.linalg.norm(mag_body) < 5.0)):
        return 0

    ekf.update_magnetometer(mag_body)
    return 1


def store_results(results, row, state):
    """Store EKF state results"""
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
    results['control_quality'].append(state.get('control_quality', 0.0))


def calculate_error_statistics(results, data, start_time=0.0):
    """
    Calculate error statistics vs ground truth.
    RMSE calculation starts from the specified start_time.
    """
    print(f"\nCalculating RMSE starting from {start_time} seconds...")

    # Convert results to numpy arrays
    for key in ['position', 'velocity', 'attitude', 'acc_bias', 'gyro_bias']:
        results[key] = np.array(results[key])
    results['timestamp'] = np.array(results['timestamp'])

    # --- MODIFICATION: Filter data to start from start_time ---
    time_mask = results['timestamp'] >= start_time
    filtered_results = {}
    for key, value in results.items():
        if isinstance(value, list):  # handles prediction_modes
            filtered_results[key] = [item for i,
                                     item in enumerate(value) if time_mask[i]]
        elif value.ndim > 0:  # handles numpy arrays
            filtered_results[key] = value[time_mask]

    filtered_data = data[data['timestamp'] >= start_time].copy()
    if len(filtered_results['timestamp']) == 0 or len(filtered_data) == 0:
        print("⚠️ Warning: No data available for RMSE calculation after the start time.")
        return {'pos_rmse': np.zeros(3), 'vel_rmse': np.zeros(3), 'att_rmse': np.zeros(3), 'valid_samples': 0}
    # --- END MODIFICATION ---

    # Find matching ground truth data using filtered data
    valid_indices = []
    result_timestamps = filtered_results['timestamp']

    for i in filtered_data.index:
        row = filtered_data.loc[i]
        true_pos = np.array(
            [row['true_pos_x'], row['true_pos_y'], row['true_pos_z']])
        if not np.allclose(true_pos, 0, atol=1e-6):
            valid_indices.append(i)

    if len(valid_indices) < 50:
        print(
            f"⚠️  Warning: Only {len(valid_indices)} valid ground truth samples after {start_time}s.")

    # Extract ground truth
    true_pos = filtered_data.loc[valid_indices, [
        'true_pos_x', 'true_pos_y', 'true_pos_z']].values
    true_vel = filtered_data.loc[valid_indices, [
        'true_vel_x', 'true_vel_y', 'true_vel_z']].values
    true_att = filtered_data.loc[valid_indices, [
        'true_roll', 'true_pitch', 'true_yaw']].values

    # Align EKF results with ground truth
    gt_timestamps = filtered_data.loc[valid_indices, 'timestamp'].values
    matching_indices = []

    for gt_time in gt_timestamps:
        result_idx = np.argmin(np.abs(result_timestamps - gt_time))
        if abs(result_timestamps[result_idx] - gt_time) < 1e-6:
            matching_indices.append(result_idx)

    # Calculate errors
    pos_error = filtered_results['position'][matching_indices] - \
        true_pos[:len(matching_indices)]
    vel_error = filtered_results['velocity'][matching_indices] - \
        true_vel[:len(matching_indices)]
    att_error = filtered_results['attitude'][matching_indices] - \
        true_att[:len(matching_indices)]

    # Handle angle wrapping
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # Calculate RMSE
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    return {
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse,
        'att_rmse': att_rmse,
        'valid_samples': len(matching_indices)
    }


def print_performance_summary(error_stats, results):
    """Print comprehensive performance summary"""
    pos_rmse = error_stats['pos_rmse']
    vel_rmse = error_stats['vel_rmse']
    att_rmse = error_stats['att_rmse']

    print(
        f"\n🎯 PERFORMANCE SUMMARY (Control Input Enhanced, from t={error_stats.get('start_time', 5.0)}s)")
    print("="*60)

    print(
        f"📍 Position RMSE [X,Y,Z]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(f"   Total Position RMSE: {np.linalg.norm(pos_rmse):.4f} m")

    print(
        f"🏃 Velocity RMSE [X,Y,Z]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(f"   Total Velocity RMSE: {np.linalg.norm(vel_rmse):.4f} m/s")

    print(
        f"🎯 Attitude RMSE [R,P,Y]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] deg")

    # Control input analysis
    if 'control_quality' in results:
        control_quality = np.array(results['control_quality'])
        control_usage = np.sum(control_quality > 0.1) / \
            len(control_quality) * 100 if len(control_quality) > 0 else 0

        # Check if there are any quality values > 0 to avoid mean of empty slice warning
        if np.any(control_quality > 0):
            avg_quality = np.mean(control_quality[control_quality > 0])
        else:
            avg_quality = 0.0

        print(f"\n🎮 Control Input Analysis:")
        print(f"   Average control quality: {avg_quality:.3f}")
        print(f"   Control usage: {control_usage:.1f}% of samples")

    # Prediction mode distribution
    modes = results['prediction_modes']
    if len(modes) > 0:
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        print(f"\n🔄 Prediction Mode Distribution:")
        for mode, count in mode_counts.items():
            percentage = 100 * count / len(modes)
            print(f"   {mode}: {count} samples ({percentage:.1f}%)")

    print("="*60)


def plot_results(results, data, start_time=5.0):
    """
    Plot EKF estimation vs ground truth, starting from a specific time.
    This version does NOT plot raw sensor data.

    Args:
        results (dict): Dictionary containing EKF results.
        data (pd.DataFrame): DataFrame containing raw simulation and ground truth data.
        start_time (float): The time in seconds from which to start plotting.
    """
    print(f"\n📈 Generating plots starting from {start_time} seconds...")

    # --- MODIFICATION: Filter data based on start_time ---
    ekf_time = np.array(results['timestamp'])
    ekf_mask = ekf_time >= start_time

    gt_time = data['timestamp'].values
    gt_mask = gt_time >= start_time

    if not np.any(ekf_mask) or not np.any(gt_mask):
        print("⚠️ No data to plot after the specified start time.")
        return

    # --- END MODIFICATION ---

    # --- POSITION PLOT ---
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig1.suptitle(
        f'Position Estimation vs. Ground Truth (from t={start_time}s)', fontsize=16)

    # Position X (North)
    ax1.plot(gt_time[gt_mask], data['true_pos_x'][gt_mask],
             'k-', label='Ground Truth', linewidth=2)
    ax1.plot(ekf_time[ekf_mask], results['position']
             [ekf_mask, 0], 'b-', label='EKF Estimate', linewidth=1.5)
    ax1.set_ylabel('North (m)')
    ax1.legend()
    ax1.grid(True)

    # Position Y (East)
    ax2.plot(gt_time[gt_mask], data['true_pos_y'][gt_mask],
             'k-', label='Ground Truth', linewidth=2)
    ax2.plot(ekf_time[ekf_mask], results['position']
             [ekf_mask, 1], 'b-', label='EKF Estimate', linewidth=1.5)
    ax2.set_ylabel('East (m)')
    ax2.legend()
    ax2.grid(True)

    # Position Z (Down)
    ax3.plot(gt_time[gt_mask], data['true_pos_z'][gt_mask],
             'k-', label='Ground Truth', linewidth=2)
    ax3.plot(ekf_time[ekf_mask], results['position']
             [ekf_mask, 2], 'b-', label='EKF Estimate', linewidth=1.5)
    ax3.set_ylabel('Down (m)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- VELOCITY PLOT ---
    fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig2.suptitle(
        f'Velocity Estimation vs. Ground Truth (from t={start_time}s)', fontsize=16)

    # Velocity X (North)
    ax4.plot(gt_time[gt_mask], data['true_vel_x'][gt_mask],
             'k-', label='Ground Truth', linewidth=2)
    ax4.plot(ekf_time[ekf_mask], results['velocity']
             [ekf_mask, 0], 'b-', label='EKF Estimate', linewidth=1.5)
    ax4.set_ylabel('V_north (m/s)')
    ax4.legend()
    ax4.grid(True)

    # Velocity Y (East)
    ax5.plot(gt_time[gt_mask], data['true_vel_y'][gt_mask],
             'k-', label='Ground Truth', linewidth=2)
    ax5.plot(ekf_time[ekf_mask], results['velocity']
             [ekf_mask, 1], 'b-', label='EKF Estimate', linewidth=1.5)
    ax5.set_ylabel('V_east (m/s)')
    ax5.legend()
    ax5.grid(True)

    # Velocity Z (Down)
    ax6.plot(gt_time[gt_mask], data['true_vel_z'][gt_mask],
             'k-', label='Ground Truth', linewidth=2)
    ax6.plot(ekf_time[ekf_mask], results['velocity']
             [ekf_mask, 2], 'b-', label='EKF Estimate', linewidth=1.5)
    ax6.set_ylabel('V_down (m/s)')
    ax6.set_xlabel('Time (s)')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- ATTITUDE PLOT ---
    fig3, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig3.suptitle(
        f'Attitude Estimation vs. Ground Truth (from t={start_time}s)', fontsize=16)

    # Attitude (Roll)
    ax7.plot(gt_time[gt_mask], np.rad2deg(data['true_roll']
             [gt_mask]), 'k-', label='Ground Truth', linewidth=2)
    ax7.plot(ekf_time[ekf_mask], np.rad2deg(results['attitude']
             [ekf_mask, 0]), 'b-', label='EKF Estimate', linewidth=1.5)
    ax7.set_ylabel('Roll (deg)')
    ax7.legend()
    ax7.grid(True)

    # Attitude (Pitch)
    ax8.plot(gt_time[gt_mask], np.rad2deg(data['true_pitch']
             [gt_mask]), 'k-', label='Ground Truth', linewidth=2)
    ax8.plot(ekf_time[ekf_mask], np.rad2deg(results['attitude']
             [ekf_mask, 1]), 'b-', label='EKF Estimate', linewidth=1.5)
    ax8.set_ylabel('Pitch (deg)')
    ax8.legend()
    ax8.grid(True)

    # Attitude (Yaw)
    ax9.plot(gt_time[gt_mask], np.rad2deg(data['true_yaw']
             [gt_mask]), 'k-', label='Ground Truth', linewidth=2)
    ax9.plot(ekf_time[ekf_mask], np.rad2deg(results['attitude']
             [ekf_mask, 2]), 'b-', label='EKF Estimate', linewidth=1.5)
    ax9.set_ylabel('Yaw (deg)')
    ax9.set_xlabel('Time (s)')
    ax9.legend()
    ax9.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- SENSOR BIAS PLOT ---
    fig4, (ax10, ax11) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig4.suptitle(
        f'Sensor Bias Estimation (from t={start_time}s)', fontsize=16)

    # Accelerometer Bias
    ax10.plot(ekf_time[ekf_mask], results['acc_bias']
              [ekf_mask, 0], label='ax bias')
    ax10.plot(ekf_time[ekf_mask], results['acc_bias']
              [ekf_mask, 1], label='ay bias')
    ax10.plot(ekf_time[ekf_mask], results['acc_bias']
              [ekf_mask, 2], label='az bias')
    ax10.set_ylabel('Accelerometer Bias (m/s²)')
    ax10.legend()
    ax10.grid(True)

    # Gyroscope Bias
    ax11.plot(ekf_time[ekf_mask], np.rad2deg(
        results['gyro_bias'][ekf_mask, 0]), label='gx bias')
    ax11.plot(ekf_time[ekf_mask], np.rad2deg(
        results['gyro_bias'][ekf_mask, 1]), label='gy bias')
    ax11.plot(ekf_time[ekf_mask], np.rad2deg(
        results['gyro_bias'][ekf_mask, 2]), label='gz bias')
    ax11.set_ylabel('Gyroscope Bias (deg/s)')
    ax11.set_xlabel('Time (s)')
    ax11.legend()
    ax11.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    # Example usage
    # Ganti dengan path file CSV Anda yang sebenarnya
    csv_file_path = "logs/angka_8_20250619_043832.csv"

    try:
        # Jalankan EKF
        results = run_ekf_with_control_data(
            csv_file_path,
            use_magnetometer=True,
            magnetic_declination=0.5  # Perkiraan untuk Surabaya
        )

        if results is not None:
            ekf, results_data, raw_data = results
            print("✅ EKF with control input completed successfully!")

            # Panggil fungsi plotting dengan start_time=5.0
            plot_results(results_data, raw_data, start_time=5.0)
        else:
            print("❌ EKF processing failed!")

    except FileNotFoundError:
        print(f"❌ Error: File not found at '{csv_file_path}'")
        print("Please ensure the CSV file exists and the path is correct.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
