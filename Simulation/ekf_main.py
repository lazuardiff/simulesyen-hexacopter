"""
EKF Analysis for Hexacopter Mission Phase
=========================================

Script untuk menganalisis hasil estimasi EKF pada mission phase hexacopter.
- Hover phase (5 detik awal): Untuk stabilisasi EKF
- Mission phase (setelah 5 detik): Untuk analisis performance

Author: EKF Analysis Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime
import os
from scipy.spatial.transform import Rotation

# Import EKF classes
from ekf_common import EKFWithControl
from ekf_utils import plot_ekf_vs_groundtruth


def load_and_validate_data(csv_file_path):
    """
    Load dan validasi data simulasi hexacopter
    """
    print("📂 Loading simulation data...")

    try:
        data = pd.read_csv(csv_file_path)
        print(f"✅ Data loaded successfully: {len(data)} samples")

        # Basic data info
        print(
            f"   Time range: {data['timestamp'].min():.3f}s to {data['timestamp'].max():.3f}s")
        print(
            f"   Sampling frequency: ~{1/np.mean(np.diff(data['timestamp'])):.1f} Hz")

        # Check flight phases
        hover_samples = np.sum(data['flight_phase'] == 0)
        mission_samples = np.sum(data['flight_phase'] == 1)
        print(
            f"   Hover phase: {hover_samples} samples ({hover_samples*0.001:.3f}s)")
        print(
            f"   Mission phase: {mission_samples} samples ({mission_samples*0.001:.3f}s)")

        # Data quality checks
        print("\n🔍 Data Quality Checks:")

        # Check for zero data at timestamp 0.001
        first_row = data.iloc[0]
        if first_row['timestamp'] <= 0.001:
            print(f"   ⚠️  First timestamp: {first_row['timestamp']:.3f}s")

            # Check if critical data is zero
            critical_zero_checks = {
                'GPS Position': np.allclose([first_row['gps_pos_ned_x'], first_row['gps_pos_ned_y'], first_row['gps_pos_ned_z']], 0),
                'IMU Acceleration': np.allclose([first_row['acc_x'], first_row['acc_y'], first_row['acc_z']], 0),
                'True Position': np.allclose([first_row['true_pos_x'], first_row['true_pos_y'], first_row['true_pos_z']], 0)
            }

            for check_name, is_zero in critical_zero_checks.items():
                status = "❌ Zero data!" if is_zero else "✅ Valid data"
                print(f"   {status} {check_name}")

        # Check data availability
        availability_checks = {
            'GPS': np.mean(data['gps_available'] == 1) * 100,
            'Barometer': np.mean(data['baro_available'] == 1) * 100,
            'Magnetometer': np.mean(data['mag_available'] == 1) * 100
        }

        print("\n📊 Sensor Availability:")
        for sensor, availability in availability_checks.items():
            print(f"   {sensor}: {availability:.1f}%")

        return data

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def run_ekf_analysis(data, magnetic_declination=0.5):
    """
    Jalankan EKF dengan control input pada data simulasi
    """
    print("\n🚁 Running EKF with Control Input...")

    # Calculate sampling time
    dt_mean = np.mean(np.diff(data['timestamp']))
    print(f"   Sampling time: {dt_mean:.4f}s")

    # Initialize EKF
    ekf = EKFWithControl(dt=dt_mean)
    ekf.mag_declination = np.deg2rad(magnetic_declination)

    # Find good initialization point (skip problematic first samples)
    init_idx = find_good_init_point(data)
    if init_idx is None:
        print("❌ No suitable initialization point found!")
        return None, None

    print(
        f"   🎯 Initialization at index {init_idx}, time {data.iloc[init_idx]['timestamp']:.3f}s")

    # Initialize EKF state
    success = initialize_ekf_from_data(ekf, data.iloc[init_idx])
    if not success:
        return None, None

    # Process all data
    results = process_simulation_data(ekf, data)

    print(
        f"✅ EKF processing completed: {len(results['timestamp'])} valid estimates")

    return ekf, results


def find_good_init_point(data):
    """
    Cari titik inisialisasi yang baik (hindari data zero di awal)
    """
    min_time = 0.01  # Skip first 10ms to avoid zero data

    for i in range(len(data)):
        row = data.iloc[i]

        if row['timestamp'] < min_time:
            continue

        # Check GPS availability and validity
        if row['gps_available'] != 1:
            continue

        gps_pos = np.array(
            [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array(
            [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

        if np.any(np.isnan(gps_pos)) or np.any(np.isnan(gps_vel)):
            continue

        # Check for zero data (problematic)
        if np.allclose(gps_pos, 0, atol=1e-6) or np.allclose(gps_vel, 0, atol=1e-6):
            continue

        # Check IMU data
        acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        if np.any(np.isnan(acc)) or np.any(np.isnan(gyro)) or np.linalg.norm(acc) < 0.1:
            continue

        # Check true data for validation
        true_pos = np.array(
            [row['true_pos_x'], row['true_pos_y'], row['true_pos_z']])
        if np.allclose(true_pos, 0, atol=1e-6):
            continue

        return i

    return None


def initialize_ekf_from_data(ekf, row):
    """
    Inisialisasi EKF dari data row tertentu
    """
    try:
        # GPS data
        gps_pos = np.array(
            [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array(
            [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

        # IMU data
        initial_acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        initial_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

        # Magnetometer data
        initial_mag = None
        if row['mag_available'] == 1:
            initial_mag = np.array([row['mag_x'], row['mag_y'], row['mag_z']])

        # Use ground truth quaternion for better initialization
        true_quat = np.array([row['true_quat_w'], row['true_quat_x'],
                             row['true_quat_y'], row['true_quat_z']])

        # Initialize with ground truth yaw for best results
        true_yaw = row['true_yaw']

        ekf.initialize_state(gps_pos, gps_vel, initial_acc,
                             initial_gyro, initial_mag, true_yaw)

        # Override with true quaternion for better stability
        ekf.x[6:10] = ekf.normalize_quaternion(true_quat)

        # Reduce attitude uncertainty since we have good initial estimate
        # Smaller initial uncertainty
        ekf.P[6:9, 6:9] = np.diag([0.01, 0.01, 0.5])

        print("   ✅ EKF initialized with ground truth attitude")
        return True

    except Exception as e:
        print(f"   ❌ Initialization failed: {str(e)}")
        return False


def process_simulation_data(ekf, data):
    """
    Proses semua data simulasi melalui EKF
    """
    results = {
        'timestamp': [], 'position': [], 'velocity': [], 'attitude': [],
        'quaternion': [], 'acc_bias': [], 'gyro_bias': [],
        'pos_std': [], 'vel_std': [], 'att_std': [],
        'prediction_mode': [], 'control_quality': [], 'flight_phase': []
    }

    print(f"   🔄 Processing {len(data)} samples...")

    processed_count = 0
    skipped_count = 0

    for i, row in data.iterrows():
        # Skip early problematic data
        if row['timestamp'] < 0.01:
            skipped_count += 1
            continue

        try:
            # Prepare IMU data
            accel_body = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
            gyro_body = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])

            # Validate IMU data
            if (np.any(np.isnan(accel_body)) or np.any(np.isnan(gyro_body)) or
                    np.linalg.norm(accel_body) < 0.1):
                skipped_count += 1
                continue

            # Prepare control data
            control_data = extract_control_data(row)

            # EKF Prediction step
            ekf.predict_with_control_input(accel_body, gyro_body, control_data)

            # Measurement updates
            update_measurements(ekf, row)

            # Store results
            state = ekf.get_state()
            results['timestamp'].append(row['timestamp'])
            results['position'].append(state['position'].copy())
            results['velocity'].append(state['velocity'].copy())
            results['attitude'].append(state['attitude_euler'].copy())
            results['quaternion'].append(state['quaternion'].copy())
            results['acc_bias'].append(state['acc_bias'].copy())
            results['gyro_bias'].append(state['gyro_bias'].copy())
            results['pos_std'].append(state['position_std'].copy())
            results['vel_std'].append(state['velocity_std'].copy())
            results['att_std'].append(state['attitude_std'].copy())
            results['prediction_mode'].append(state['prediction_mode'])
            results['control_quality'].append(
                state.get('control_quality', 0.0))
            results['flight_phase'].append(row['flight_phase'])

            processed_count += 1

        except Exception as e:
            skipped_count += 1
            if skipped_count % 1000 == 0:  # Only print occasional warnings
                print(
                    f"   ⚠️  Processing error at {row['timestamp']:.3f}s: {str(e)}")
            continue

        # Progress update
        if processed_count % 5000 == 0 and processed_count > 0:
            print(
                f"      📈 Progress: {processed_count} processed, {skipped_count} skipped")

    print(
        f"   ✅ Processing complete: {processed_count} processed, {skipped_count} skipped")

    # Convert lists to numpy arrays
    for key in ['position', 'velocity', 'attitude', 'quaternion', 'acc_bias', 'gyro_bias',
                'pos_std', 'vel_std', 'att_std']:
        results[key] = np.array(results[key])

    return results


def extract_control_data(row):
    """
    Extract control data dari row data
    """
    control_data = {}

    # Motor thrusts
    motor_thrusts = np.array([
        row['motor_thrust_1'], row['motor_thrust_2'], row['motor_thrust_3'],
        row['motor_thrust_4'], row['motor_thrust_5'], row['motor_thrust_6']
    ])

    if (not np.any(np.isnan(motor_thrusts)) and np.sum(motor_thrusts) > 0.1 and
            not np.allclose(motor_thrusts, 0, atol=1e-6)):
        control_data['motor_thrusts'] = motor_thrusts

    # Thrust setpoint
    thrust_sp = np.array(
        [row['thrust_sp_x'], row['thrust_sp_y'], row['thrust_sp_z']])
    if (not np.any(np.isnan(thrust_sp)) and np.linalg.norm(thrust_sp) > 0.1 and
            not np.allclose(thrust_sp, 0, atol=1e-6)):
        control_data['thrust_sp'] = thrust_sp

    # Control torques
    control_torques = np.array(
        [row['control_torque_x'], row['control_torque_y'], row['control_torque_z']])
    if (not np.any(np.isnan(control_torques)) and
            not np.allclose(control_torques, 0, atol=1e-6)):
        control_data['control_torques'] = control_torques

    return control_data if control_data else None


def update_measurements(ekf, row):
    """
    Update EKF dengan measurement yang tersedia
    """
    # GPS updates
    if row['gps_available'] == 1:
        gps_pos = np.array(
            [row['gps_pos_ned_x'], row['gps_pos_ned_y'], row['gps_pos_ned_z']])
        gps_vel = np.array(
            [row['gps_vel_ned_x'], row['gps_vel_ned_y'], row['gps_vel_ned_z']])

        if (not np.any(np.isnan(gps_pos)) and not np.any(np.isnan(gps_vel)) and
                not np.allclose(gps_pos, 0, atol=1e-6)):
            ekf.update_gps_position(gps_pos)
            ekf.update_gps_velocity(gps_vel)

    # Barometer update
    if row['baro_available'] == 1:
        baro_alt = row['baro_altitude']
        if not np.isnan(baro_alt) and abs(baro_alt) > 1e-6:
            ekf.update_barometer(baro_alt)

    # Magnetometer update
    if row['mag_available'] == 1:
        mag_body = np.array([row['mag_x'], row['mag_y'], row['mag_z']])
        if (not np.any(np.isnan(mag_body)) and np.linalg.norm(mag_body) > 0.1 and
                not np.allclose(mag_body, 0, atol=1e-6)):
            ekf.update_magnetometer(mag_body)


def analyze_mission_phase_performance(results, data):
    """
    Analisis performance EKF hanya pada mission phase
    """
    print("\n📊 Mission Phase Performance Analysis...")

    # Filter untuk mission phase saja
    mission_indices = np.array(results['flight_phase']) == 1

    if np.sum(mission_indices) == 0:
        print("❌ No mission phase data found!")
        return None

    print(
        f"   Mission phase samples: {np.sum(mission_indices)} / {len(mission_indices)}")

    # Extract mission phase results
    mission_results = {}
    for key in results.keys():
        if key in ['position', 'velocity', 'attitude', 'quaternion']:
            mission_results[key] = results[key][mission_indices]
        elif key == 'timestamp':
            mission_results[key] = np.array(results[key])[mission_indices]

    # Get corresponding ground truth for mission phase
    mission_timestamps = mission_results['timestamp']
    mission_ground_truth = get_aligned_ground_truth(data, mission_timestamps)

    if mission_ground_truth is None:
        print("❌ Failed to align ground truth data!")
        return None

    # Calculate errors
    pos_error = mission_results['position'] - mission_ground_truth['position']
    vel_error = mission_results['velocity'] - mission_ground_truth['velocity']
    att_error = mission_results['attitude'] - mission_ground_truth['attitude']

    # Handle angle wrapping for attitude
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # Calculate RMSE
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    # Calculate maximum errors
    pos_max_error = np.max(np.linalg.norm(pos_error, axis=1))
    vel_max_error = np.max(np.linalg.norm(vel_error, axis=1))
    att_max_error = np.max(np.linalg.norm(att_error, axis=1))

    # Print results
    print(f"\n🎯 MISSION PHASE PERFORMANCE RESULTS:")
    print("="*60)
    print(
        f"📍 Position RMSE [N,E,D]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(f"   Total Position RMSE: {np.linalg.norm(pos_rmse):.4f} m")
    print(f"   Maximum Position Error: {pos_max_error:.4f} m")
    print()
    print(
        f"🏃 Velocity RMSE [N,E,D]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(f"   Total Velocity RMSE: {np.linalg.norm(vel_rmse):.4f} m/s")
    print(f"   Maximum Velocity Error: {vel_max_error:.4f} m/s")
    print()
    print(
        f"🎯 Attitude RMSE [R,P,Y]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] deg")
    print(
        f"   Total Attitude RMSE: {np.rad2deg(np.linalg.norm(att_rmse)):.3f} deg")
    print(f"   Maximum Attitude Error: {np.rad2deg(att_max_error):.3f} deg")
    print("="*60)

    return {
        'mission_results': mission_results,
        'mission_ground_truth': mission_ground_truth,
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse,
        'att_rmse': att_rmse,
        'pos_error': pos_error,
        'vel_error': vel_error,
        'att_error': att_error,
        'mission_indices': mission_indices
    }


def get_aligned_ground_truth(data, timestamps):
    """
    Get ground truth data aligned dengan timestamps EKF
    """
    try:
        true_pos = np.zeros((len(timestamps), 3))
        true_vel = np.zeros((len(timestamps), 3))
        true_att = np.zeros((len(timestamps), 3))
        true_quat = np.zeros((len(timestamps), 4))

        for i, t in enumerate(timestamps):
            # Find closest timestamp in ground truth
            idx = np.argmin(np.abs(data['timestamp'] - t))
            row = data.iloc[idx]

            true_pos[i] = [row['true_pos_x'],
                           row['true_pos_y'], row['true_pos_z']]
            true_vel[i] = [row['true_vel_x'],
                           row['true_vel_y'], row['true_vel_z']]
            true_att[i] = [row['true_roll'],
                           row['true_pitch'], row['true_yaw']]
            true_quat[i] = [row['true_quat_w'], row['true_quat_x'],
                            row['true_quat_y'], row['true_quat_z']]

        return {
            'position': true_pos,
            'velocity': true_vel,
            'attitude': true_att,
            'quaternion': true_quat
        }

    except Exception as e:
        print(f"❌ Error aligning ground truth: {str(e)}")
        return None


def plot_mission_phase_results(analysis_results, save_plots=True, output_dir="mission_analysis_plots"):
    """
    Plot hasil analisis mission phase
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    mission_results = analysis_results['mission_results']
    mission_gt = analysis_results['mission_ground_truth']
    time = mission_results['timestamp']

    # Adjust time to start from mission start (subtract hover duration)
    mission_start_time = time[0]
    time_adj = time - mission_start_time

    print(f"\n🎨 Generating mission phase plots...")

    # Set plotting style
    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    colors = {'ekf': '#2E86AB', 'truth': '#F18F01', 'error': '#A23B72'}

    # === PLOT 1: Position Estimation ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Mission Phase: Position Estimation vs Ground Truth',
                 fontsize=16, fontweight='bold')

    pos_labels = ['North (X)', 'East (Y)', 'Down (Z)']
    for i in range(3):
        ax = axes[i]
        ax.plot(time_adj, mission_gt['position'][:, i], color=colors['truth'],
                linewidth=2.5, label='Ground Truth', alpha=0.9)
        ax.plot(time_adj, mission_results['position'][:, i], color=colors['ekf'],
                linewidth=2, label='EKF Estimate', linestyle='--', alpha=0.8)

        ax.set_ylabel(f'Position {pos_labels[i]} (m)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add RMSE info
        rmse = np.sqrt(
            np.mean((mission_results['position'][:, i] - mission_gt['position'][:, i])**2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}m', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Mission Time (s)', fontweight='bold')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/mission_position_estimation.png",
                    dpi=300, bbox_inches='tight')

    # === PLOT 2: Velocity Estimation ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Mission Phase: Velocity Estimation vs Ground Truth',
                 fontsize=16, fontweight='bold')

    for i in range(3):
        ax = axes[i]
        ax.plot(time_adj, mission_gt['velocity'][:, i], color=colors['truth'],
                linewidth=2.5, label='Ground Truth', alpha=0.9)
        ax.plot(time_adj, mission_results['velocity'][:, i], color=colors['ekf'],
                linewidth=2, label='EKF Estimate', linestyle='--', alpha=0.8)

        ax.set_ylabel(f'Velocity {pos_labels[i]} (m/s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add RMSE info
        rmse = np.sqrt(
            np.mean((mission_results['velocity'][:, i] - mission_gt['velocity'][:, i])**2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}m/s', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Mission Time (s)', fontweight='bold')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/mission_velocity_estimation.png",
                    dpi=300, bbox_inches='tight')

    # === PLOT 3: Attitude Estimation ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Mission Phase: Attitude Estimation vs Ground Truth',
                 fontsize=16, fontweight='bold')

    att_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax = axes[i]
        ax.plot(time_adj, np.rad2deg(mission_gt['attitude'][:, i]), color=colors['truth'],
                linewidth=2.5, label='Ground Truth', alpha=0.9)
        ax.plot(time_adj, np.rad2deg(mission_results['attitude'][:, i]), color=colors['ekf'],
                linewidth=2, label='EKF Estimate', linestyle='--', alpha=0.8)

        ax.set_ylabel(f'{att_labels[i]} (degrees)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add RMSE info
        att_error = mission_results['attitude'][:,
                                                i] - mission_gt['attitude'][:, i]
        att_error = np.arctan2(
            np.sin(att_error), np.cos(att_error))  # Wrap angles
        rmse = np.sqrt(np.mean(att_error**2))
        ax.text(0.02, 0.98, f'RMSE: {np.rad2deg(rmse):.3f}°', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Mission Time (s)', fontweight='bold')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/mission_attitude_estimation.png",
                    dpi=300, bbox_inches='tight')

    # === PLOT 4: 3D Trajectory ===
    fig = plt.figure(figsize=(15, 12))

    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(mission_gt['position'][:, 0], mission_gt['position'][:, 1], -mission_gt['position'][:, 2],
             color=colors['truth'], linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(mission_results['position'][:, 0], mission_results['position'][:, 1], -mission_results['position'][:, 2],
             color=colors['ekf'], linewidth=2, label='EKF Estimate', linestyle='--', alpha=0.7)

    # Add start/end markers
    ax1.scatter(mission_gt['position'][0, 0], mission_gt['position'][0, 1], -mission_gt['position'][0, 2],
                color='green', marker='o', s=100, label='Start')
    ax1.scatter(mission_gt['position'][-1, 0], mission_gt['position'][-1, 1], -mission_gt['position'][-1, 2],
                color='red', marker='x', s=100, label='End')

    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Up (m)')
    ax1.set_title('3D Mission Trajectory')
    ax1.legend()

    # Top view (X-Y plane)
    ax2 = fig.add_subplot(222)
    ax2.plot(mission_gt['position'][:, 0], mission_gt['position'][:, 1],
             color=colors['truth'], linewidth=3, label='Ground Truth', alpha=0.8)
    ax2.plot(mission_results['position'][:, 0], mission_results['position'][:, 1],
             color=colors['ekf'], linewidth=2, label='EKF Estimate', linestyle='--', alpha=0.7)
    ax2.scatter(mission_gt['position'][0, 0], mission_gt['position'][0, 1],
                color='green', marker='o', s=100, label='Start')
    ax2.scatter(mission_gt['position'][-1, 0], mission_gt['position'][-1, 1],
                color='red', marker='x', s=100, label='End')
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('East (m)')
    ax2.set_title('Mission Trajectory (Top View)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')

    # Position error magnitude over time
    ax3 = fig.add_subplot(223)
    pos_error_norm = np.linalg.norm(analysis_results['pos_error'], axis=1)
    ax3.plot(time_adj, pos_error_norm,
             color=colors['error'], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Mission Time (s)')
    ax3.set_ylabel('Position Error Magnitude (m)')
    ax3.set_title('Position Error vs Time')
    ax3.grid(True, alpha=0.3)

    # Error statistics summary
    ax4 = fig.add_subplot(224)
    error_labels = ['Pos X', 'Pos Y', 'Pos Z', 'Vel X',
                    'Vel Y', 'Vel Z', 'Roll', 'Pitch', 'Yaw']
    rmse_values = np.concatenate([
        analysis_results['pos_rmse'],
        analysis_results['vel_rmse'],
        np.rad2deg(analysis_results['att_rmse'])
    ])

    bars = ax4.bar(range(len(error_labels)), rmse_values, alpha=0.7)
    ax4.set_xticks(range(len(error_labels)))
    ax4.set_xticklabels(error_labels, rotation=45)
    ax4.set_ylabel('RMSE')
    ax4.set_title('Mission Phase RMSE Summary')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, rmse_values)):
        height = bar.get_height()
        if i < 6:  # Position and velocity
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        else:  # Attitude
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{val:.2f}°', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/mission_trajectory_analysis.png",
                    dpi=300, bbox_inches='tight')

    plt.show()
    print(f"✅ Mission phase plots saved to {output_dir}/")


def save_ekf_results(results, analysis_results, output_dir="ekf_results"):
    """
    Simpan hasil estimasi EKF ke file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === SAVE 1: Complete EKF Results ===
    results_df = pd.DataFrame({
        'timestamp': results['timestamp'],
        'flight_phase': results['flight_phase'],

        # Position estimates
        'ekf_pos_x': results['position'][:, 0],
        'ekf_pos_y': results['position'][:, 1],
        'ekf_pos_z': results['position'][:, 2],

        # Velocity estimates
        'ekf_vel_x': results['velocity'][:, 0],
        'ekf_vel_y': results['velocity'][:, 1],
        'ekf_vel_z': results['velocity'][:, 2],

        # Attitude estimates (Euler angles)
        'ekf_roll': results['attitude'][:, 0],
        'ekf_pitch': results['attitude'][:, 1],
        'ekf_yaw': results['attitude'][:, 2],

        # Quaternion estimates
        'ekf_quat_w': results['quaternion'][:, 0],
        'ekf_quat_x': results['quaternion'][:, 1],
        'ekf_quat_y': results['quaternion'][:, 2],
        'ekf_quat_z': results['quaternion'][:, 3],

        # Bias estimates
        'ekf_acc_bias_x': results['acc_bias'][:, 0],
        'ekf_acc_bias_y': results['acc_bias'][:, 1],
        'ekf_acc_bias_z': results['acc_bias'][:, 2],
        'ekf_gyro_bias_x': results['gyro_bias'][:, 0],
        'ekf_gyro_bias_y': results['gyro_bias'][:, 1],
        'ekf_gyro_bias_z': results['gyro_bias'][:, 2],

        # Uncertainty estimates
        'ekf_pos_std_x': results['pos_std'][:, 0],
        'ekf_pos_std_y': results['pos_std'][:, 1],
        'ekf_pos_std_z': results['pos_std'][:, 2],
        'ekf_vel_std_x': results['vel_std'][:, 0],
        'ekf_vel_std_y': results['vel_std'][:, 1],
        'ekf_vel_std_z': results['vel_std'][:, 2],
        'ekf_att_std_roll': results['att_std'][:, 0],
        'ekf_att_std_pitch': results['att_std'][:, 1],
        'ekf_att_std_yaw': results['att_std'][:, 2],

        # Control information
        'prediction_mode': results['prediction_mode'],
        'control_quality': results['control_quality']
    })

    results_file = os.path.join(
        output_dir, f"ekf_complete_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✅ Complete EKF results saved: {results_file}")

    # === SAVE 2: Mission Phase Only Results ===
    if analysis_results is not None:
        mission_results = analysis_results['mission_results']
        mission_gt = analysis_results['mission_ground_truth']

        mission_df = pd.DataFrame({
            'mission_time': mission_results['timestamp'] - mission_results['timestamp'][0],
            'timestamp': mission_results['timestamp'],

            # EKF estimates
            'ekf_pos_x': mission_results['position'][:, 0],
            'ekf_pos_y': mission_results['position'][:, 1],
            'ekf_pos_z': mission_results['position'][:, 2],
            'ekf_vel_x': mission_results['velocity'][:, 0],
            'ekf_vel_y': mission_results['velocity'][:, 1],
            'ekf_vel_z': mission_results['velocity'][:, 2],
            'ekf_roll': mission_results['attitude'][:, 0],
            'ekf_pitch': mission_results['attitude'][:, 1],
            'ekf_yaw': mission_results['attitude'][:, 2],

            # Ground truth
            'true_pos_x': mission_gt['position'][:, 0],
            'true_pos_y': mission_gt['position'][:, 1],
            'true_pos_z': mission_gt['position'][:, 2],
            'true_vel_x': mission_gt['velocity'][:, 0],
            'true_vel_y': mission_gt['velocity'][:, 1],
            'true_vel_z': mission_gt['velocity'][:, 2],
            'true_roll': mission_gt['attitude'][:, 0],
            'true_pitch': mission_gt['attitude'][:, 1],
            'true_yaw': mission_gt['attitude'][:, 2],

            # Errors
            'pos_error_x': analysis_results['pos_error'][:, 0],
            'pos_error_y': analysis_results['pos_error'][:, 1],
            'pos_error_z': analysis_results['pos_error'][:, 2],
            'vel_error_x': analysis_results['vel_error'][:, 0],
            'vel_error_y': analysis_results['vel_error'][:, 1],
            'vel_error_z': analysis_results['vel_error'][:, 2],
            'att_error_roll': analysis_results['att_error'][:, 0],
            'att_error_pitch': analysis_results['att_error'][:, 1],
            'att_error_yaw': analysis_results['att_error'][:, 2],
        })

        mission_file = os.path.join(
            output_dir, f"ekf_mission_phase_results_{timestamp}.csv")
        mission_df.to_csv(mission_file, index=False)
        print(f"✅ Mission phase results saved: {mission_file}")

        # === SAVE 3: Performance Summary ===
        summary_file = os.path.join(
            output_dir, f"ekf_performance_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("EKF HEXACOPTER MISSION PHASE PERFORMANCE SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Analysis timestamp: {timestamp}\n")
            f.write(
                f"Mission phase samples: {len(mission_results['timestamp'])}\n")
            f.write(
                f"Mission duration: {mission_results['timestamp'][-1] - mission_results['timestamp'][0]:.3f} s\n\n")

            f.write("POSITION ESTIMATION PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(
                f"RMSE [N,E,D]: [{analysis_results['pos_rmse'][0]:.4f}, {analysis_results['pos_rmse'][1]:.4f}, {analysis_results['pos_rmse'][2]:.4f}] m\n")
            f.write(
                f"Total Position RMSE: {np.linalg.norm(analysis_results['pos_rmse']):.4f} m\n")
            f.write(
                f"Max Position Error: {np.max(np.linalg.norm(analysis_results['pos_error'], axis=1)):.4f} m\n\n")

            f.write("VELOCITY ESTIMATION PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(
                f"RMSE [N,E,D]: [{analysis_results['vel_rmse'][0]:.4f}, {analysis_results['vel_rmse'][1]:.4f}, {analysis_results['vel_rmse'][2]:.4f}] m/s\n")
            f.write(
                f"Total Velocity RMSE: {np.linalg.norm(analysis_results['vel_rmse']):.4f} m/s\n")
            f.write(
                f"Max Velocity Error: {np.max(np.linalg.norm(analysis_results['vel_error'], axis=1)):.4f} m/s\n\n")

            f.write("ATTITUDE ESTIMATION PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(
                f"RMSE [R,P,Y]: [{np.rad2deg(analysis_results['att_rmse'][0]):.3f}, {np.rad2deg(analysis_results['att_rmse'][1]):.3f}, {np.rad2deg(analysis_results['att_rmse'][2]):.3f}] deg\n")
            f.write(
                f"Total Attitude RMSE: {np.rad2deg(np.linalg.norm(analysis_results['att_rmse'])):.3f} deg\n")
            f.write(
                f"Max Attitude Error: {np.rad2deg(np.max(np.linalg.norm(analysis_results['att_error'], axis=1))):.3f} deg\n\n")

            # Control usage statistics
            mission_indices = analysis_results['mission_indices']
            mission_modes = np.array(results['prediction_mode'])[
                mission_indices]

            f.write("CONTROL INPUT USAGE (MISSION PHASE):\n")
            f.write("-"*40 + "\n")
            mode_counts = {}
            for mode in mission_modes:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

            for mode, count in mode_counts.items():
                percentage = 100 * count / len(mission_modes)
                f.write(f"{mode}: {count} samples ({percentage:.1f}%)\n")

            control_usage = sum(
                1 for mode in mission_modes if mode != "IMU_ONLY") / len(mission_modes) * 100
            f.write(
                f"\nTotal control input utilization: {control_usage:.1f}%\n")

        print(f"✅ Performance summary saved: {summary_file}")

    print(f"\n📁 All results saved to directory: {output_dir}/")


def main():
    """
    Main function untuk analisis EKF mission phase
    """
    print("🚁 HEXACOPTER EKF MISSION PHASE ANALYSIS")
    print("="*80)

    # Specify your CSV file path here
    csv_file_path = input(
        "Enter CSV file path (or press Enter for default): ").strip()
    if not csv_file_path:
        csv_file_path = "logs/hexacopter_ekf_data_with_hovering_fixed_20250609_002136.csv"

    # Load and validate data
    data = load_and_validate_data(csv_file_path)
    if data is None:
        return

    # Run EKF analysis
    ekf, results = run_ekf_analysis(
        data, magnetic_declination=0.5)  # Surabaya declination
    if results is None:
        return

    # Analyze mission phase performance
    analysis_results = analyze_mission_phase_performance(results, data)
    if analysis_results is None:
        return

    # Plot results
    plot_mission_phase_results(analysis_results, save_plots=True)

    # Save results
    save_ekf_results(results, analysis_results)

    print("\n✅ Analysis completed successfully!")
    print("   📊 Mission phase performance calculated")
    print("   📈 Plots generated and saved")
    print("   💾 Results saved to files")


if __name__ == "__main__":
    main()
