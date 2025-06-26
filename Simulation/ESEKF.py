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
from baseESEKF import BaseESEKF
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class ESEKF(BaseESEKF):
    """
    ESEKF class inheriting all core logic from BaseESEKF.

    Karena logika input kontrol telah dihapus untuk kesesuaian dengan paper,
    kelas ini tidak perlu menimpa (override) metode apa pun. Ini berfungsi
    sebagai alias langsung ke kelas dasar.
    """
    pass


def run_esekf(csv_file_path, use_magnetometer=True, magnetic_declination=0.0):
    """
    Fungsi utama untuk menjalankan ESEKF murni berbasis IMU dan sensor lainnya.

    Alur Kerja:
    1. Memuat dan memvalidasi data simulasi.
    2. Menginisialisasi ESEKF dengan pengukuran sensor yang valid.
    3. Memproses semua data dengan tahap prediksi dan koreksi.
    4. Menghitung statistik error terhadap ground truth.
    5. Mengembalikan hasil untuk analisis.

    Args:
        csv_file_path: Path ke file CSV data simulasi.
        use_magnetometer: Aktifkan update magnetometer untuk yaw.
        magnetic_declination: Deklinasi magnetik lokal dalam derajat.

    Returns:
        tuple: (ekf_instance, results_dict, raw_data)
    """

    print("\n" + "="*80)
    print("RUNNING ERROR-STATE EKF (IMU + SENSOR FUSION)")
    print("="*80)

    # === LANGKAH 1: MEMUAT DAN MEMVALIDASI DATA ===
    try:
        data = pd.read_csv(csv_file_path)
        print(f"✅ Data simulasi dimuat: {len(data)} sampel")
    except Exception as e:
        print(f"❌ Error memuat data: {str(e)}")
        return None

    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"🕐 Waktu sampling: {dt_mean:.4f} s ({1/dt_mean:.1f} Hz)")

    # === LANGKAH 2: INISIALISASI ESEKF ===
    ekf = ESEKF(dt=dt_mean)
    ekf.mag_declination = np.deg2rad(magnetic_declination)
    print(f"🧭 Deklinasi magnetik diatur: {magnetic_declination:.1f}°")

    init_idx = find_initialization_point(data)
    if init_idx is None:
        print("❌ Tidak ditemukan titik inisialisasi yang sesuai!")
        return None

    success = initialize_ekf_state(ekf, data.iloc[init_idx], use_magnetometer)
    if not success:
        print("❌ Inisialisasi EKF gagal!")
        return None

    # === LANGKAH 3: MEMPROSES SEMUA DATA ===
    results = process_all_data(ekf, data, use_magnetometer)

    if len(results['timestamp']) < 100:
        print(f"❌ Hasil valid tidak mencukupi: {len(results['timestamp'])}")
        return None

    # === LANGKAH 4: MENGHITUNG STATISTIK ERROR ===
    error_stats = calculate_error_statistics(results, data, start_time=5.0)
    print_performance_summary(error_stats)

    print("✅ Pemrosesan ESEKF selesai!")
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


def process_all_data(ekf, data, use_magnetometer):
    """Memproses semua data simulasi melalui ESEKF"""
    n_samples = len(data)
    min_processing_time = 0.05

    results = {
        'timestamp': [], 'position': [], 'velocity': [], 'attitude': [],
        'acc_bias': [], 'gyro_bias': [], 'pos_std': [], 'vel_std': [], 'att_std': []
    }
    stats = {
        'prediction_count': 0, 'gps_updates': 0, 'baro_updates': 0,
        'mag_updates': 0, 'skipped_samples': 0
    }
    print(f"🔄 Memproses {n_samples} sampel...")

    for i in range(n_samples):
        row = data.iloc[i]

        if row['timestamp'] < min_processing_time:
            stats['skipped_samples'] += 1
            continue

        try:
            accel_body, gyro_body = prepare_imu_data(row)
        except ValueError:
            stats['skipped_samples'] += 1
            continue

        # === TAHAP PREDIKSI ===
        try:
            # Memanggil metode predict dari base class, tanpa input kontrol
            ekf.predict(accel_body, gyro_body)
            stats['prediction_count'] += 1
        except Exception as e:
            print(f"⚠️  Prediksi gagal pada sampel {i}: {str(e)}")
            continue

        # === TAHAP KOREKSI (MEASUREMENT UPDATES) ===
        stats['gps_updates'] += update_gps_measurements(ekf, row)
        stats['baro_updates'] += update_barometer_measurement(ekf, row)
        if use_magnetometer:
            stats['mag_updates'] += update_magnetometer_measurement(ekf, row)

        try:
            state = ekf.get_state()
            store_results(results, row, state)
        except Exception as e:
            print(f"⚠️  Gagal menyimpan hasil pada sampel {i}: {str(e)}")
            continue

        if i % 5000 == 0 and i > 0:
            print(f"   📈 Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%)")

    # Cetak statistik pemrosesan
    print("\n📊 Statistik Pemrosesan:")
    print(f"  ✅ Prediksi valid: {stats['prediction_count']}")
    print(f"  📡 Update GPS: {stats['gps_updates']}")
    print(f"  🌡️  Update Barometer: {stats['baro_updates']}")
    print(f"  🧭 Update Magnetometer: {stats['mag_updates']}")
    print(f"  ⏭️  Sampel dilewati: {stats['skipped_samples']}")

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
    """Menyimpan hasil estimasi EKF"""
    results['timestamp'].append(row['timestamp'])
    results['position'].append(state['position'])
    results['velocity'].append(state['velocity'])
    results['attitude'].append(state['attitude_euler'])
    results['acc_bias'].append(state['acc_bias'])
    results['gyro_bias'].append(state['gyro_bias'])
    results['pos_std'].append(state['position_std'])
    results['vel_std'].append(state['velocity_std'])
    results['att_std'].append(state['attitude_std'])


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


def print_performance_summary(error_stats):
    """Mencetak ringkasan performa EKF"""
    pos_rmse = error_stats['pos_rmse']
    vel_rmse = error_stats['vel_rmse']
    att_rmse = error_stats['att_rmse']

    print(f"\n🎯 RINGKASAN PERFORMA (Berdasarkan IMU + Fusi Sensor)")
    print("="*60)
    print(
        f"📍 RMSE Posisi [X,Y,Z]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(f"   Total RMSE Posisi: {np.linalg.norm(pos_rmse):.4f} m")
    print(
        f"🏃 RMSE Kecepatan [X,Y,Z]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(f"   Total RMSE Kecepatan: {np.linalg.norm(vel_rmse):.4f} m/s")
    print(
        f"🎯 RMSE Attitude [R,P,Y]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] deg")
    print("="*60)


def plot_results(results, data, start_time=5.0):
    """
    Plot EKF estimation vs ground truth, starting from a specific time.
    Style sesuai dengan plot_ekf.py
    """
    print(f"\n🎨 Generating plots starting from {start_time} seconds...")

    # Filter data based on start_time
    ekf_time = np.array(results['timestamp'])
    ekf_mask = ekf_time >= start_time

    gt_time = data['timestamp'].values
    gt_mask = gt_time >= start_time

    if not np.any(ekf_mask) or not np.any(gt_mask):
        print("⚠️ No data to plot after the specified start time.")
        return

    # Adjust time to start from 0
    ekf_time_adj = ekf_time[ekf_mask] - start_time
    gt_time_adj = gt_time[gt_mask] - start_time

    # === PLOT 1: Position Estimation (Style seperti plot_ekf.py) ===
    plt.figure(figsize=(14, 10))
    plt.suptitle('Position Estimation vs Ground Truth',
                 fontsize=14, fontweight='bold')

    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(3, 1, i+1)

        # Plot Ground Truth (orange solid line seperti plot_ekf.py)
        if i == 0:
            plt.plot(gt_time_adj, data['true_pos_x'][gt_mask],
                     'r', linewidth=2, label='Ground Truth')
            plt.plot(ekf_time_adj, results['position'][ekf_mask, 0],
                     'b--', linewidth=2, label='EKF Estimate')
        elif i == 1:
            plt.plot(gt_time_adj, data['true_pos_y'][gt_mask],
                     'r', linewidth=2, label='Ground Truth')
            plt.plot(ekf_time_adj, results['position'][ekf_mask, 1],
                     'b--', linewidth=2, label='EKF Estimate')
        else:
            plt.plot(gt_time_adj, data['true_pos_z'][gt_mask],
                     'r', linewidth=2, label='Ground Truth')
            plt.plot(ekf_time_adj, results['position'][ekf_mask, 2],
                     'b--', linewidth=2, label='EKF Estimate')

        plt.ylabel(f'Position {pos_labels[i]} (m)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        if i == 2:
            plt.xlabel('Mission Time (s)')

    plt.tight_layout()

    # === PLOT 2: Velocity Estimation (Style seperti plot_ekf.py) ===
    plt.figure(figsize=(14, 10))
    plt.suptitle('Velocity Estimation vs Ground Truth',
                 fontsize=14, fontweight='bold')

    for i in range(3):
        plt.subplot(3, 1, i+1)

        # Plot Ground Truth (orange solid line seperti plot_ekf.py)
        if i == 0:
            plt.plot(gt_time_adj, data['true_vel_x'][gt_mask],
                     'r', linewidth=2, label='Ground Truth')
            plt.plot(ekf_time_adj, results['velocity'][ekf_mask, 0],
                     'b--', linewidth=2, label='EKF Estimate')
        elif i == 1:
            plt.plot(gt_time_adj, data['true_vel_y'][gt_mask],
                     'r', linewidth=2, label='Ground Truth')
            plt.plot(ekf_time_adj, results['velocity'][ekf_mask, 1],
                     'b--', linewidth=2, label='EKF Estimate')
        else:
            plt.plot(gt_time_adj, data['true_vel_z'][gt_mask],
                     'r', linewidth=2, label='Ground Truth')
            plt.plot(ekf_time_adj, results['velocity'][ekf_mask, 2],
                     'b--', linewidth=2, label='EKF Estimate')

        plt.ylabel(f'Velocity {pos_labels[i]} (m/s)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        if i == 2:
            plt.xlabel('Mission Time (s)')

    plt.tight_layout()

    # === PLOT 3: Attitude Estimation (Style seperti plot_ekf.py) ===
    plt.figure(figsize=(14, 10))
    plt.suptitle('Attitude Estimation vs Ground Truth',
                 fontsize=14, fontweight='bold')

    att_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        plt.subplot(3, 1, i+1)

        # Convert to degrees
        if i == 0:
            true_att = np.rad2deg(data['true_roll'][gt_mask])
            ekf_att = np.rad2deg(results['attitude'][ekf_mask, 0])
        elif i == 1:
            true_att = np.rad2deg(data['true_pitch'][gt_mask])
            ekf_att = np.rad2deg(results['attitude'][ekf_mask, 1])
        else:
            true_att = np.rad2deg(data['true_yaw'][gt_mask])
            ekf_att = np.rad2deg(results['attitude'][ekf_mask, 2])

        # Plot Ground Truth (orange solid line seperti plot_ekf.py)
        plt.plot(gt_time_adj, true_att, 'r',
                 linewidth=2, label='Ground Truth')

        # Plot EKF Estimate (blue dashed line seperti plot_ekf.py)
        plt.plot(ekf_time_adj, ekf_att, 'b--',
                 linewidth=2, label='EKF Estimate')

        plt.ylabel(f'{att_labels[i]} (deg)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        if i == 2:
            plt.xlabel('Mission Time (s)')

    plt.tight_layout()
    plt.show()


def save_ekf_results(results, data, output_dir="ekf_results"):
    """
    Save EKF estimation results to CSV file with comparison to ground truth
    """
    import os
    from datetime import datetime

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"helix_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    print(f"💾 Saving EKF results to: {filepath}")

    # Convert results to numpy arrays
    ekf_timestamps = np.array(results['timestamp'])
    ekf_positions = np.array(results['position'])
    ekf_velocities = np.array(results['velocity'])
    ekf_attitudes = np.array(results['attitude'])

    # Prepare data for CSV
    csv_data = []

    for i, ekf_time in enumerate(ekf_timestamps):
        # Find closest ground truth data
        gt_idx = np.argmin(np.abs(data['timestamp'] - ekf_time))
        gt_row = data.iloc[gt_idx]

        # Compile data row
        row_data = {
            # Time
            'mission_time': ekf_time,

            # EKF Estimates
            'ekf_pos_x': ekf_positions[i, 0],
            'ekf_pos_y': ekf_positions[i, 1],
            'ekf_pos_z': ekf_positions[i, 2],
            'ekf_vel_x': ekf_velocities[i, 0],
            'ekf_vel_y': ekf_velocities[i, 1],
            'ekf_vel_z': ekf_velocities[i, 2],
            'ekf_roll': ekf_attitudes[i, 0],
            'ekf_pitch': ekf_attitudes[i, 1],
            'ekf_yaw': ekf_attitudes[i, 2],

            # Ground Truth
            'true_pos_x': gt_row['true_pos_x'],
            'true_pos_y': gt_row['true_pos_y'],
            'true_pos_z': gt_row['true_pos_z'],
            'true_vel_x': gt_row['true_vel_x'],
            'true_vel_y': gt_row['true_vel_y'],
            'true_vel_z': gt_row['true_vel_z'],
            'true_roll': gt_row['true_roll'],
            'true_pitch': gt_row['true_pitch'],
            'true_yaw': gt_row['true_yaw'],

            # Errors
            'pos_error_x': ekf_positions[i, 0] - gt_row['true_pos_x'],
            'pos_error_y': ekf_positions[i, 1] - gt_row['true_pos_y'],
            'pos_error_z': ekf_positions[i, 2] - gt_row['true_pos_z'],
            'vel_error_x': ekf_velocities[i, 0] - gt_row['true_vel_x'],
            'vel_error_y': ekf_velocities[i, 1] - gt_row['true_vel_y'],
            'vel_error_z': ekf_velocities[i, 2] - gt_row['true_vel_z'],
            'att_error_roll': ekf_attitudes[i, 0] - gt_row['true_roll'],
            'att_error_pitch': ekf_attitudes[i, 1] - gt_row['true_pitch'],
            'att_error_yaw': ekf_attitudes[i, 2] - gt_row['true_yaw'],

            # Additional EKF data
            'acc_bias_x': results['acc_bias'][i][0],
            'acc_bias_y': results['acc_bias'][i][1],
            'acc_bias_z': results['acc_bias'][i][2],
            'gyro_bias_x': results['gyro_bias'][i][0],
            'gyro_bias_y': results['gyro_bias'][i][1],
            'gyro_bias_z': results['gyro_bias'][i][2],
            'pos_std_x': results['pos_std'][i][0],
            'pos_std_y': results['pos_std'][i][1],
            'pos_std_z': results['pos_std'][i][2],
            'vel_std_x': results['vel_std'][i][0],
            'vel_std_y': results['vel_std'][i][1],
            'vel_std_z': results['vel_std'][i][2],
            'att_std_roll': results['att_std'][i][0],
            'att_std_pitch': results['att_std'][i][1],
            'att_std_yaw': results['att_std'][i][2],
            'prediction_mode': results['prediction_modes'][i],
            'control_quality': results['control_quality'][i]
        }

        csv_data.append(row_data)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(filepath, index=False)

    # Print summary
    print(f"✅ EKF results saved successfully!")
    print(f"   📊 Total samples: {len(csv_data)}")
    print(f"   📁 File location: {filepath}")
    print(f"   📝 Columns saved: {len(df.columns)}")

    return filepath


# Main execution block
if __name__ == "__main__":
    # Ganti dengan path file CSV Anda
    csv_file_path = "logs/helix_real_20250619_040431.csv"

    try:
        # Jalankan ESEKF
        results = run_esekf(
            csv_file_path,
            use_magnetometer=True,
            magnetic_declination=0.85  # Deklinasi untuk Surabaya
        )

        if results is not None:
            ekf, results_data, raw_data = results
            print("✅ Pemrosesan ESEKF berhasil diselesaikan!")

            # Simpan hasil ke CSV
            # save_ekf_results(results_data, raw_data) # Aktifkan jika perlu

            # Hasilkan plot
            plot_results(results_data, raw_data, start_time=5.0)

            print("\n🎉 Analisis selesai!")
        else:
            print("❌ Pemrosesan ESEKF gagal!")

    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di '{csv_file_path}'")
    except Exception as e:
        print(f"❌ Terjadi error tak terduga: {str(e)}")
        import traceback
        traceback.print_exc()
