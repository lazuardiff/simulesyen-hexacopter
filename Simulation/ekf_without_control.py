"""
EKF without Control Input Implementation
=======================================

This file contains the Extended Kalman Filter implementation that uses
only IMU data for prediction (traditional EKF approach).

Workflow:
1. IMU-only Prediction (accelerometer + gyroscope)
2. GPS Position/Velocity Updates
3. Barometer Altitude Updates  
4. Magnetometer Yaw Updates
5. Pure sensor-based estimation

Author: EKF Implementation Team
Date: 2025
"""

import numpy as np
import pandas as pd
from ekf_common import BaseEKF


class EKFWithoutControl(BaseEKF):
    """
    Extended Kalman Filter without Control Input
    
    This class implements traditional EKF approach using only
    IMU sensors for prediction, without any control input data.
    This serves as a baseline for comparison with control-enhanced EKF.
    """

    def __init__(self, dt=0.01):
        super().__init__(dt)
        
        # IMU-only specific parameters
        self.imu_trust_factor = 1.0  # Full trust in IMU data
        self.bias_adaptation_rate = 0.01  # Rate of bias adaptation
        
        # Quality metrics
        self.imu_quality = 0.0
        self.prediction_confidence = 0.0

    def predict_imu_only(self, accel_body, gyro_body):
        """
        IMU-only prediction step (traditional EKF approach)
        
        Workflow:
        1. Validate IMU sensor data
        2. Apply bias corrections to sensor readings
        3. Integrate acceleration in inertial frame (with gravity)
        4. Integrate angular velocity for attitude update
        5. Propagate state and covariance using error-state model
        
        Args:
            accel_body: IMU accelerometer reading [ax, ay, az] (m/s²)
            gyro_body: IMU gyroscope reading [gx, gy, gz] (rad/s)
        """
        if not self.initialized:
            return

        # === STEP 1: EXTRACT CURRENT STATE ===
        pos = self.x[0:3]
        vel = self.x[3:6]
        q = self.x[6:10]
        acc_bias = self.x[10:13]
        gyro_bias = self.x[13:16]

        # === STEP 2: BIAS CORRECTION ===
        accel_corrected = accel_body - acc_bias
        gyro_corrected = gyro_body - gyro_bias

        # === STEP 3: IMU QUALITY ASSESSMENT ===
        self.assess_imu_quality(accel_body, gyro_body)

        # === STEP 4: COORDINATE TRANSFORMATION ===
        # Get current rotation matrix (body to NED)
        R_bn = self.quaternion_to_rotation_matrix(q)

        # === STEP 5: ACCELERATION INTEGRATION ===
        # Transform body acceleration to NED frame and add gravity
        accel_ned = R_bn @ accel_corrected + self.g_ned

        # Kinematic integration with midpoint method for better accuracy
        vel_mid = vel + 0.5 * accel_ned * self.dt
        pos_new = pos + vel_mid * self.dt
        vel_new = vel + accel_ned * self.dt

        # === STEP 6: ATTITUDE INTEGRATION ===
        omega_norm = np.linalg.norm(gyro_corrected)
        
        if omega_norm > 1e-8:
            # Use Rodrigues rotation formula for numerical stability
            axis = gyro_corrected / omega_norm
            angle = omega_norm * self.dt

            # Quaternion representing rotation increment
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])

            # Quaternion multiplication: q_new = q ⊗ dq
            q_new = np.array([
                q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
                q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
                q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
                q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0]
            ])
        else:
            # No rotation if angular velocity is negligible
            q_new = q.copy()

        # === STEP 7: UPDATE NOMINAL STATE ===
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:10] = self.normalize_quaternion(q_new)
        # Biases evolve through random walk process noise

        # Set prediction mode
        self.prediction_mode = "IMU_ONLY"

        # === STEP 8: ERROR STATE LINEARIZATION ===
        # Compute Jacobian matrix for error state propagation
        F = np.eye(15)

        # Position error depends on velocity error
        F[0:3, 3:6] = np.eye(3) * self.dt

        # Velocity error depends on attitude error and accelerometer bias
        F[3:6, 6:9] = -R_bn @ self.skew_symmetric(accel_corrected) * self.dt
        F[3:6, 9:12] = -R_bn * self.dt

        # Attitude error depends on gyroscope bias
        F[6:9, 6:9] = np.eye(3) - self.skew_symmetric(gyro_corrected) * self.dt
        F[6:9, 12:15] = -np.eye(3) * self.dt

        # Bias states have identity transitions (random walk)

        # === STEP 9: COVARIANCE PROPAGATION ===
        # P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

        # Ensure numerical stability
        self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry
        self.P += np.eye(15) * 1e-12        # Add small diagonal for stability

    def assess_imu_quality(self, accel_body, gyro_body):
        """
        Assess IMU data quality for adaptive processing
        
        Args:
            accel_body: Raw accelerometer data
            gyro_body: Raw gyroscope data
        """
        # Accelerometer quality based on gravity magnitude
        acc_magnitude = np.linalg.norm(accel_body)
        gravity_error = abs(acc_magnitude - 9.81) / 9.81
        acc_quality = max(0.1, 1.0 - gravity_error * 2.0)

        # Gyroscope quality based on reasonable angular rates
        gyro_magnitude = np.linalg.norm(gyro_body)
        gyro_quality = 1.0 if gyro_magnitude < 10.0 else max(0.1, 10.0 / gyro_magnitude)

        # Combined IMU quality
        self.imu_quality = (acc_quality + gyro_quality) / 2.0
        
        # Prediction confidence based on IMU quality
        self.prediction_confidence = min(1.0, self.imu_quality * 1.2)

    def adaptive_bias_correction(self, innovation_gps=None):
        """
        Adaptive bias correction based on GPS innovations
        
        Args:
            innovation_gps: GPS innovation vector for bias adaptation
        """
        if innovation_gps is not None and np.linalg.norm(innovation_gps) > 0.1:
            # Adapt bias estimates based on persistent GPS innovations
            bias_correction = innovation_gps * self.bias_adaptation_rate
            
            # Limit bias corrections to reasonable values
            bias_correction = np.clip(bias_correction, -0.01, 0.01)
            
            # Apply to accelerometer bias (velocity errors often indicate acc bias)
            if len(bias_correction) >= 3:
                self.x[10:13] += bias_correction[:3]

    def get_state(self):
        """Enhanced state getter with IMU-only specific information"""
        base_state = super().get_state()
        
        # Add IMU-only specific information
        base_state.update({
            'imu_quality': self.imu_quality,
            'prediction_confidence': self.prediction_confidence,
            'bias_adaptation_rate': self.bias_adaptation_rate
        })
        
        return base_state


def run_ekf_without_control_data(csv_file_path, use_magnetometer=True, magnetic_declination=0.0):
    """
    Main function to run EKF without control input data
    
    Workflow:
    1. Load and validate simulation data
    2. Initialize EKF with first valid sensor measurements  
    3. Process all timesteps with IMU-only prediction
    4. Apply sensor measurement updates
    5. Calculate error statistics vs ground truth
    6. Return results for comparison analysis
    
    Args:
        csv_file_path: Path to simulation data CSV
        use_magnetometer: Enable magnetometer updates for yaw
        magnetic_declination: Local magnetic declination in degrees
        
    Returns:
        tuple: (ekf_instance, results_dict, raw_data)
    """
    
    print("\n" + "="*80)
    print("EKF WITHOUT CONTROL INPUT - TRADITIONAL IMU-ONLY APPROACH")
    print("="*80)
    
    # === STEP 1: LOAD AND VALIDATE DATA ===
    try:
        data = pd.read_csv(csv_file_path)
        print(f"✅ Simulation data loaded: {len(data)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

    # Calculate sampling time
    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"🕐 Sampling time: {dt_mean:.4f} s ({1/dt_mean:.1f} Hz)")

    # === STEP 2: INITIALIZE EKF ===
    ekf = EKFWithoutControl(dt=dt_mean)
    ekf.mag_declination = np.deg2rad(magnetic_declination)
    print(f"🧭 Magnetic declination: {magnetic_declination:.1f}°")
    print(f"📡 Sensor configuration: IMU + GPS + Barometer" + (" + Magnetometer" if use_magnetometer else ""))

    # Find initialization point
    init_idx = find_initialization_point_imu_only(data)
    if init_idx is None:
        print("❌ No suitable initialization data found!")
        return None

    # Initialize EKF state
    success = initialize_ekf_state_imu_only(ekf, data.iloc[init_idx], use_magnetometer)
    if not success:
        print("❌ EKF initialization failed!")
        return None

    # === STEP 3: PROCESS ALL DATA ===
    results = process_all_data_imu_only(ekf, data, use_magnetometer)
    
    if len(results['timestamp']) < 100:
        print(f"❌ Insufficient valid results: {len(results['timestamp'])}")
        return None

    # === STEP 4: CALCULATE ERRORS AND STATISTICS ===
    error_stats = calculate_error_statistics_imu_only(results, data)
    print_performance_summary_imu_only(error_stats, results)

    print("✅ EKF without control input processing completed!")
    return ekf, results, data


def find_initialization_point_imu_only(data):
    """Find optimal initialization point for IMU-only EKF"""
    min_start_time = 0.1  # Skip startup period
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        if row['timestamp'] < min_start_time:
            continue
            
        # Check GPS availability
        if row['gps_available'] != 1:
            continue
            
        # Validate GPS data
        gps_pos = np.array([row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array([row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])
        
        if np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel)):
            continue
            
        # Validate IMU data (primary requirement for IMU-only EKF)
        acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
        
        if np.any(np.isnan(acc)) or np.any(np.isnan(gyro)):
            continue
            
        # Check accelerometer shows gravity (not in free fall or extreme acceleration)
        acc_magnitude = np.linalg.norm(acc)
        if not (8.0 < acc_magnitude < 12.0):  # Should be close to 9.81 m/s²
            continue
            
        print(f"🎯 IMU-only initialization point: index {i}, time {row['timestamp']:.3f}s")
        print(f"   Accelerometer magnitude: {acc_magnitude:.3f} m/s²")
        print(f"   Gyroscope magnitude: {np.linalg.norm(gyro):.3f} rad/s")
        return i
    
    return None


def initialize_ekf_state_imu_only(ekf, row, use_magnetometer):
    """Initialize EKF state for IMU-only operation"""
    try:
        gps_pos = np.array([row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array([row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])
        initial_acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        initial_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        print(f"🔧 Initial sensor readings:")
        print(f"   GPS position: [{gps_pos[0]:.3f}, {gps_pos[1]:.3f}, {gps_pos[2]:.3f}] m")
        print(f"   IMU accel: [{initial_acc[0]:.3f}, {initial_acc[1]:.3f}, {initial_acc[2]:.3f}] m/s²")
        print(f"   IMU gyro: [{initial_gyro[0]:.4f}, {initial_gyro[1]:.4f}, {initial_gyro[2]:.4f}] rad/s")

        # Magnetometer data if available
        initial_mag = None
        if use_magnetometer and 'mag_available' in row and row['mag_available'] == 1:
            initial_mag = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
            print(f"   Magnetometer: [{initial_mag[0]:.3f}, {initial_mag[1]:.3f}, {initial_mag[2]:.3f}]")

        # Use ground truth yaw for fair comparison (same as control version)
        true_yaw = row['true_yaw'] if use_magnetometer else None

        ekf.initialize_state(gps_pos, gps_vel, initial_acc, initial_gyro, initial_mag, true_yaw)
        print("✅ IMU-only EKF initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        return False


def process_all_data_imu_only(ekf, data, use_magnetometer):
    """Process all simulation data through IMU-only EKF"""
    n_samples = len(data)
    min_processing_time = 0.05
    
    # Initialize results storage
    results = {
        'timestamp': [], 'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [], 'pos_std': [], 'vel_std': [], 'att_std': [],
        'prediction_modes': [], 'imu_quality': []
    }
    
    # Statistics counters
    stats = {
        'prediction_count': 0, 'gps_updates': 0, 'baro_updates': 0, 
        'mag_updates': 0, 'skipped_samples': 0, 'quality_low': 0
    }
    
    print(f"🔄 Processing {n_samples} samples with IMU-only prediction...")
    
    for i in range(n_samples):
        row = data.iloc[i]
        
        # Skip early timestamps
        if row['timestamp'] < min_processing_time:
            stats['skipped_samples'] += 1
            continue

        # Validate and prepare IMU data
        try:
            accel_body, gyro_body = prepare_imu_data_imu_only(row)
        except ValueError:
            stats['skipped_samples'] += 1
            continue

        # === IMU-ONLY PREDICTION STEP ===
        try:
            ekf.predict_imu_only(accel_body, gyro_body)
            stats['prediction_count'] += 1
            
            # Track low quality predictions
            if ekf.imu_quality < 0.5:
                stats['quality_low'] += 1
                
        except Exception as e:
            print(f"⚠️  Prediction failed at sample {i}: {str(e)}")
            continue

        # === MEASUREMENT UPDATES ===
        # GPS updates with adaptive bias correction
        gps_innovation = update_gps_measurements_imu_only(ekf, row)
        if gps_innovation is not None:
            stats['gps_updates'] += 1
            # Apply adaptive bias correction
            ekf.adaptive_bias_correction(gps_innovation)
        
        stats['baro_updates'] += update_barometer_measurement_imu_only(ekf, row)
        
        if use_magnetometer:
            stats['mag_updates'] += update_magnetometer_measurement_imu_only(ekf, row)

        # Store results
        try:
            state = ekf.get_state()
            store_results_imu_only(results, row, state)
        except Exception as e:
            print(f"⚠️  Failed to store results at sample {i}: {str(e)}")
            continue

        # Progress indicator
        if i % 5000 == 0 and i > 0:
            print(f"   📈 Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%)")

    # Print processing statistics
    print(f"\n📊 IMU-Only Processing Statistics:")
    print(f"  ✅ Valid predictions: {stats['prediction_count']}")
    print(f"  📡 GPS updates: {stats['gps_updates']}")
    print(f"  🌡️  Barometer updates: {stats['baro_updates']}")
    print(f"  🧭 Magnetometer updates: {stats['mag_updates']}")
    print(f"  ⚠️  Low quality predictions: {stats['quality_low']} ({100*stats['quality_low']/stats['prediction_count']:.1f}%)")
    print(f"  ⏭️  Skipped samples: {stats['skipped_samples']}")

    return results


def prepare_imu_data_imu_only(row):
    """Prepare and validate IMU data for IMU-only EKF"""
    accel_body = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
    gyro_body = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
    
    # Stricter validation for IMU-only mode
    acc_norm = np.linalg.norm(accel_body)
    gyro_norm = np.linalg.norm(gyro_body)
    
    if (np.any(np.isnan(accel_body)) or np.any(np.isnan(gyro_body)) or 
        acc_norm < 0.5 or acc_norm > 50 or gyro_norm > 20):
        raise ValueError("Invalid IMU data for IMU-only processing")
    
    # Conservative clipping for stability
    accel_body = np.clip(accel_body, -30, 30)
    gyro_body = np.clip(gyro_body, -15, 15)
    
    return accel_body, gyro_body


def update_gps_measurements_imu_only(ekf, row):
    """Update GPS measurements and return innovation for bias adaptation"""
    if row['gps_available'] != 1:
        return None
        
    gps_pos = np.array([row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
    gps_vel = np.array([row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])
    
    # Validate GPS data
    if (np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel)) or
        np.allclose(gps_pos, 0, atol=1e-6) or np.linalg.norm(gps_pos) > 10000):
        return None
    
    # Calculate innovation before update (for bias adaptation)
    pos_innovation = gps_pos - ekf.x[0:3]
    vel_innovation = gps_vel - ekf.x[3:6]
    innovation = np.concatenate([pos_innovation, vel_innovation])
    
    # Apply GPS updates
    ekf.update_gps_position(gps_pos)
    ekf.update_gps_velocity(gps_vel)
    
    return innovation


def update_barometer_measurement_imu_only(ekf, row):
    """Update barometer measurement for IMU-only EKF"""
    if row['baro_available'] != 1:
        return 0
        
    baro_alt = row['baro_altitude']
    if np.isnan(baro_alt) or not (-1000 < baro_alt < 10000) or abs(baro_alt) <= 1e-6:
        return 0
    
    ekf.update_barometer(baro_alt)
    return 1


def update_magnetometer_measurement_imu_only(ekf, row):
    """Update magnetometer measurement for IMU-only EKF"""
    if 'mag_available' not in row or row['mag_available'] != 1:
        return 0
        
    mag_body = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
    if (np.any(np.isnan(mag_body)) or np.allclose(mag_body, 0, atol=1e-6) or
        not (0.1 < np.linalg.norm(mag_body) < 5.0)):
        return 0
    
    ekf.update_magnetometer(mag_body)
    return 1


def store_results_imu_only(results, row, state):
    """Store EKF state results for IMU-only mode"""
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
    results['imu_quality'].append(state.get('imu_quality', 0.0))


def calculate_error_statistics_imu_only(results, data):
    """Calculate error statistics for IMU-only EKF"""
    # Convert results to numpy arrays
    for key in ['position', 'velocity', 'attitude', 'acc_bias', 'gyro_bias', 'pos_std', 'vel_std', 'att_std']:
        results[key] = np.array(results[key])

    # Find matching ground truth data (same logic as control version)
    valid_indices = []
    result_timestamps = np.array(results['timestamp'])
    
    for i, t in enumerate(data['timestamp']):
        if t in result_timestamps:
            # Validate ground truth data
            row = data.iloc[i]
            true_pos = np.array([row['true_pos_x'], row['true_pos_y'], row['true_pos_z']])
            true_att = np.array([row['true_roll'], row['true_pitch'], row['true_yaw']])
            
            if (not np.allclose(true_pos, 0, atol=1e-6) and 
                not np.allclose(true_att, 0, atol=1e-6)):
                valid_indices.append(i)

    if len(valid_indices) < 50:
        print(f"⚠️  Warning: Only {len(valid_indices)} valid ground truth samples")

    # Extract ground truth
    true_pos = np.column_stack([
        data.iloc[valid_indices]['true_pos_x'],
        data.iloc[valid_indices]['true_pos_y'], 
        data.iloc[valid_indices]['true_pos_z']
    ])
    true_vel = np.column_stack([
        data.iloc[valid_indices]['true_vel_x'],
        data.iloc[valid_indices]['true_vel_y'],
        data.iloc[valid_indices]['true_vel_z']
    ])
    true_att = np.column_stack([
        data.iloc[valid_indices]['true_roll'],
        data.iloc[valid_indices]['true_pitch'],
        data.iloc[valid_indices]['true_yaw']
    ])

    # Align EKF results with ground truth
    gt_timestamps = data.iloc[valid_indices]['timestamp'].values
    matching_indices = []
    
    for gt_time in gt_timestamps:
        result_idx = np.argmin(np.abs(result_timestamps - gt_time))
        if abs(result_timestamps[result_idx] - gt_time) < 1e-6:
            matching_indices.append(result_idx)

    # Calculate errors
    pos_error = results['position'][matching_indices] - true_pos[:len(matching_indices)]
    vel_error = results['velocity'][matching_indices] - true_vel[:len(matching_indices)]
    att_error = results['attitude'][matching_indices] - true_att[:len(matching_indices)]

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


def print_performance_summary_imu_only(error_stats, results):
    """Print comprehensive performance summary for IMU-only EKF"""
    pos_rmse = error_stats['pos_rmse']
    vel_rmse = error_stats['vel_rmse'] 
    att_rmse = error_stats['att_rmse']
    
    print(f"\n🎯 PERFORMANCE SUMMARY (IMU-Only Baseline)")
    print("="*60)
    
    print(f"📍 Position RMSE [X,Y,Z]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(f"   Total Position RMSE: {np.linalg.norm(pos_rmse):.4f} m")
    
    print(f"🏃 Velocity RMSE [X,Y,Z]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(f"   Total Velocity RMSE: {np.linalg.norm(vel_rmse):.4f} m/s")
    
    print(f"🎯 Attitude RMSE [R,P,Y]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] deg")
    
    # IMU quality analysis
    if 'imu_quality' in results:
        imu_quality = np.array(results['imu_quality'])
        avg_quality = np.mean(imu_quality)
        low_quality_pct = np.sum(imu_quality < 0.5) / len(imu_quality) * 100
        
        print(f"\n📡 IMU Quality Analysis:")
        print(f"   Average IMU quality: {avg_quality:.3f}")
        print(f"   Low quality samples: {low_quality_pct:.1f}%")
    
    # All predictions should be IMU-only
    print(f"\n🔄 Prediction Mode: 100% IMU-Only")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    csv_file_path = "logs/complete_flight_data_with_geodetic_20250530_012243.csv"
    
    try:
        results = run_ekf_without_control_data(
            csv_file_path, 
            use_magnetometer=True, 
            magnetic_declination=0.5  # Surabaya
        )
        
        if results is not None:
            ekf, results_data, data = results
            print("✅ EKF without control input completed successfully!")
        else:
            print("❌ EKF processing failed!")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()