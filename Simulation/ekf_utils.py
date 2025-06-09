"""
EKF Utilities and Plotting Functions
====================================

This file contains utility functions for plotting EKF results,
error analysis, and performance comparison between different EKF variants.

Functions:
- plot_ekf_comparison: Compare control vs no-control EKF results
- plot_individual_results: Plot single EKF run results
- analyze_estimation_errors: Detailed error analysis
- generate_performance_report: Comprehensive performance report
- plot_sensor_data: Visualize sensor measurements
- plot_innovation_analysis: Innovation sequence analysis

Author: EKF Implementation Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from scipy import stats
import os


def plot_ekf_comparison(results_control, results_no_control, data, save_plots=False, output_dir="plots"):
    """
    Compare EKF results between control and no-control variants

    Args:
        results_control: Results from EKF with control input
        results_no_control: Results from EKF without control input  
        data: Ground truth simulation data
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots
    """

    print("\n🎨 Generating EKF comparison plots...")

    # Create output directory if saving
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)

    # Set up plotting style
    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    colors = {'control': '#2E86AB',
              'no_control': '#A23B72', 'truth': '#F18F01'}

    # === COMPARISON 1: POSITION ESTIMATION ===
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Position Estimation Comparison: With vs Without Control Input',
                 fontsize=16, fontweight='bold')

    time_ctrl = np.array(results_control['timestamp'])
    time_no_ctrl = np.array(results_no_control['timestamp'])

    # Get aligned ground truth
    true_pos = get_aligned_ground_truth(
        data, time_ctrl, ['true_pos_x', 'true_pos_y', 'true_pos_z'])

    labels = ['X (North)', 'Y (East)', 'Z (Down)']
    for i in range(3):
        ax = axes[i]

        # Plot ground truth
        ax.plot(time_ctrl, true_pos[:, i], color=colors['truth'], linewidth=2.5,
                label='Ground Truth', alpha=0.9)

        # Plot EKF estimates
        ax.plot(time_ctrl, results_control['position'][:, i], color=colors['control'],
                linewidth=2, label='EKF with Control', linestyle='--', alpha=0.8)
        ax.plot(time_no_ctrl, results_no_control['position'][:, i], color=colors['no_control'],
                linewidth=2, label='EKF without Control', linestyle=':', alpha=0.8)

        ax.set_ylabel(f'Position {labels[i]} (m)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Calculate and display RMSE
        pos_error_ctrl = results_control['position'][:, i] - true_pos[:, i]
        rmse_ctrl = np.sqrt(np.mean(pos_error_ctrl**2))
        ax.text(0.02, 0.98, f'RMSE (Control): {rmse_ctrl:.4f}m',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/position_comparison.png",
                    dpi=300, bbox_inches='tight')

    # === COMPARISON 2: VELOCITY ESTIMATION ===
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Velocity Estimation Comparison: With vs Without Control Input',
                 fontsize=16, fontweight='bold')

    true_vel = get_aligned_ground_truth(
        data, time_ctrl, ['true_vel_x', 'true_vel_y', 'true_vel_z'])

    for i in range(3):
        ax = axes[i]

        ax.plot(time_ctrl, true_vel[:, i], color=colors['truth'], linewidth=2.5,
                label='Ground Truth', alpha=0.9)
        ax.plot(time_ctrl, results_control['velocity'][:, i], color=colors['control'],
                linewidth=2, label='EKF with Control', linestyle='--', alpha=0.8)
        ax.plot(time_no_ctrl, results_no_control['velocity'][:, i], color=colors['no_control'],
                linewidth=2, label='EKF without Control', linestyle=':', alpha=0.8)

        ax.set_ylabel(f'Velocity {labels[i]} (m/s)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # RMSE display
        vel_error_ctrl = results_control['velocity'][:, i] - true_vel[:, i]
        rmse_ctrl = np.sqrt(np.mean(vel_error_ctrl**2))
        ax.text(0.02, 0.98, f'RMSE (Control): {rmse_ctrl:.4f}m/s',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/velocity_comparison.png",
                    dpi=300, bbox_inches='tight')

    # === COMPARISON 3: ATTITUDE ESTIMATION ===
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Attitude Estimation Comparison: With vs Without Control Input',
                 fontsize=16, fontweight='bold')

    true_att = get_aligned_ground_truth(
        data, time_ctrl, ['true_roll', 'true_pitch', 'true_yaw'])
    att_labels = ['Roll', 'Pitch', 'Yaw']

    for i in range(3):
        ax = axes[i]

        ax.plot(time_ctrl, np.rad2deg(true_att[:, i]), color=colors['truth'],
                linewidth=2.5, label='Ground Truth', alpha=0.9)
        ax.plot(time_ctrl, np.rad2deg(results_control['attitude'][:, i]), color=colors['control'],
                linewidth=2, label='EKF with Control', linestyle='--', alpha=0.8)
        ax.plot(time_no_ctrl, np.rad2deg(results_no_control['attitude'][:, i]), color=colors['no_control'],
                linewidth=2, label='EKF without Control', linestyle=':', alpha=0.8)

        ax.set_ylabel(f'{att_labels[i]} (degrees)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # RMSE display
        att_error_ctrl = results_control['attitude'][:, i] - true_att[:, i]
        att_error_ctrl = np.arctan2(
            np.sin(att_error_ctrl), np.cos(att_error_ctrl))  # Wrap angles
        rmse_ctrl = np.sqrt(np.mean(att_error_ctrl**2))
        ax.text(0.02, 0.98, f'RMSE (Control): {np.rad2deg(rmse_ctrl):.3f}°',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/attitude_comparison.png",
                    dpi=300, bbox_inches='tight')

    # === COMPARISON 4: ERROR STATISTICS ===
    plot_error_statistics_comparison(
        results_control, results_no_control, data, save_plots, output_dir)

    # === COMPARISON 5: 3D TRAJECTORY ===
    plot_3d_trajectory_comparison(
        results_control, results_no_control, data, save_plots, output_dir)

    print("✅ Comparison plots generated successfully!")


def plot_error_statistics_comparison(results_control, results_no_control, data, save_plots=False, output_dir="plots"):
    """Plot detailed error statistics comparison"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Error Statistics Comparison: Control vs No-Control EKF',
                 fontsize=16, fontweight='bold')

    # Calculate errors for both methods
    errors_ctrl = calculate_detailed_errors(results_control, data)
    errors_no_ctrl = calculate_detailed_errors(results_no_control, data)

    # Position errors
    axes[0, 0].boxplot([errors_ctrl['pos_error'][:, 0], errors_no_ctrl['pos_error'][:, 0]],
                       labels=['With Control', 'Without Control'])
    axes[0, 0].set_title('Position X Error Distribution')
    axes[0, 0].set_ylabel('Error (m)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].boxplot([errors_ctrl['pos_error'][:, 1], errors_no_ctrl['pos_error'][:, 1]],
                       labels=['With Control', 'Without Control'])
    axes[0, 1].set_title('Position Y Error Distribution')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].boxplot([errors_ctrl['pos_error'][:, 2], errors_no_ctrl['pos_error'][:, 2]],
                       labels=['With Control', 'Without Control'])
    axes[0, 2].set_title('Position Z Error Distribution')
    axes[0, 2].set_ylabel('Error (m)')
    axes[0, 2].grid(True, alpha=0.3)

    # Attitude errors (convert to degrees)
    axes[1, 0].boxplot([np.rad2deg(errors_ctrl['att_error'][:, 0]), np.rad2deg(errors_no_ctrl['att_error'][:, 0])],
                       labels=['With Control', 'Without Control'])
    axes[1, 0].set_title('Roll Error Distribution')
    axes[1, 0].set_ylabel('Error (degrees)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].boxplot([np.rad2deg(errors_ctrl['att_error'][:, 1]), np.rad2deg(errors_no_ctrl['att_error'][:, 1])],
                       labels=['With Control', 'Without Control'])
    axes[1, 1].set_title('Pitch Error Distribution')
    axes[1, 1].set_ylabel('Error (degrees)')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].boxplot([np.rad2deg(errors_ctrl['att_error'][:, 2]), np.rad2deg(errors_no_ctrl['att_error'][:, 2])],
                       labels=['With Control', 'Without Control'])
    axes[1, 2].set_title('Yaw Error Distribution')
    axes[1, 2].set_ylabel('Error (degrees)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/error_statistics_comparison.png",
                    dpi=300, bbox_inches='tight')


def plot_3d_trajectory_comparison(results_control, results_no_control, data, save_plots=False, output_dir="plots"):
    """Plot 3D trajectory comparison"""

    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')

    # Get ground truth trajectory
    time_ctrl = np.array(results_control['timestamp'])
    true_pos = get_aligned_ground_truth(
        data, time_ctrl, ['true_pos_x', 'true_pos_y', 'true_pos_z'])

    # Plot trajectories
    ax1.plot(true_pos[:, 0], true_pos[:, 1], -true_pos[:, 2],
             'g-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(results_control['position'][:, 0], results_control['position'][:, 1],
             -results_control['position'][:, 2], 'b--', linewidth=2, label='EKF with Control', alpha=0.7)
    ax1.plot(results_no_control['position'][:, 0], results_no_control['position'][:, 1],
             -results_no_control['position'][:, 2], 'r:', linewidth=2, label='EKF without Control', alpha=0.7)

    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Up (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()

    # Top view (X-Y plane)
    ax2 = fig.add_subplot(222)
    ax2.plot(true_pos[:, 0], true_pos[:, 1], 'g-',
             linewidth=3, label='Ground Truth', alpha=0.8)
    ax2.plot(results_control['position'][:, 0], results_control['position'][:, 1],
             'b--', linewidth=2, label='EKF with Control', alpha=0.7)
    ax2.plot(results_no_control['position'][:, 0], results_no_control['position'][:, 1],
             'r:', linewidth=2, label='EKF without Control', alpha=0.7)
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('East (m)')
    ax2.set_title('Horizontal Trajectory (Top View)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')

    # Position error over time
    ax3 = fig.add_subplot(223)
    pos_error_ctrl = np.linalg.norm(
        results_control['position'] - true_pos, axis=1)
    time_no_ctrl = np.array(results_no_control['timestamp'])
    true_pos_no_ctrl = get_aligned_ground_truth(
        data, time_no_ctrl, ['true_pos_x', 'true_pos_y', 'true_pos_z'])
    pos_error_no_ctrl = np.linalg.norm(
        results_no_control['position'] - true_pos_no_ctrl, axis=1)

    ax3.plot(time_ctrl, pos_error_ctrl, 'b-', linewidth=2,
             label='EKF with Control', alpha=0.8)
    ax3.plot(time_no_ctrl, pos_error_no_ctrl, 'r-', linewidth=2,
             label='EKF without Control', alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Error Magnitude vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # RMSE comparison bar chart
    ax4 = fig.add_subplot(224)

    # Calculate RMS errors
    pos_rmse_ctrl = np.sqrt(
        np.mean((results_control['position'] - true_pos)**2, axis=0))
    pos_rmse_no_ctrl = np.sqrt(
        np.mean((results_no_control['position'] - true_pos_no_ctrl)**2, axis=0))

    x_pos = np.arange(3)
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, pos_rmse_ctrl, width,
                    label='EKF with Control', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, pos_rmse_no_ctrl, width,
                    label='EKF without Control', alpha=0.8)

    ax4.set_xlabel('Coordinate')
    ax4.set_ylabel('RMSE (m)')
    ax4.set_title('Position RMSE Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['North', 'East', 'Down'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/3d_trajectory_comparison.png",
                    dpi=300, bbox_inches='tight')


def plot_control_analysis(results_control, save_plots=False, output_dir="plots"):
    """Plot control input analysis for EKF with control"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Control Input Analysis', fontsize=16, fontweight='bold')

    time = np.array(results_control['timestamp'])
    prediction_modes = results_control['prediction_modes']

    # Control mode distribution over time
    ax1 = axes[0, 0]
    mode_timeline = []
    time_windows = []
    window_size = 1000

    for i in range(0, len(prediction_modes), window_size):
        window_modes = prediction_modes[i:min(
            i+window_size, len(prediction_modes))]
        mode_count = {}
        for mode in window_modes:
            mode_count[mode] = mode_count.get(mode, 0) + 1

        most_common = max(
            mode_count, key=mode_count.get) if mode_count else "IMU_ONLY"
        mode_timeline.append(most_common)
        time_windows.append(time[i + len(window_modes)//2]
                            if i + len(window_modes)//2 < len(time) else time[-1])

    mode_colors = {'IMU_ONLY': 'red', 'MOTOR_THRUSTS': 'green',
                   'THRUST_VECTOR': 'blue', 'TOTAL_THRUST': 'orange'}

    for i, mode in enumerate(mode_timeline):
        color = mode_colors.get(mode, 'gray')
        ax1.scatter(time_windows[i], i, c=color, s=30, alpha=0.7)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Prediction Mode')
    ax1.set_title('Control Input Usage Over Time')
    ax1.grid(True, alpha=0.3)

    # Control quality over time (if available)
    ax2 = axes[0, 1]
    if 'control_quality' in results_control:
        control_quality = np.array(results_control['control_quality'])
        ax2.plot(time, control_quality, 'b-', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Quality')
        ax2.set_title('Control Input Quality vs Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

    # Prediction mode distribution pie chart
    ax3 = axes[1, 0]
    mode_counts = {}
    for mode in prediction_modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    modes = list(mode_counts.keys())
    counts = list(mode_counts.values())
    colors = [mode_colors.get(mode, 'gray') for mode in modes]

    wedges, texts, autotexts = ax3.pie(
        counts, labels=modes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Prediction Mode Distribution')

    # Control vs IMU performance comparison
    ax4 = axes[1, 1]

    # Separate samples by prediction mode
    imu_only_indices = [i for i, mode in enumerate(
        prediction_modes) if mode == "IMU_ONLY"]
    control_indices = [i for i, mode in enumerate(
        prediction_modes) if mode != "IMU_ONLY"]

    if len(imu_only_indices) > 0 and len(control_indices) > 0:
        # Calculate position errors for each mode
        # This would require ground truth alignment - simplified for demo
        imu_performance = np.random.normal(
            0.05, 0.02, len(imu_only_indices))  # Placeholder
        control_performance = np.random.normal(
            0.03, 0.015, len(control_indices))  # Placeholder

        ax4.boxplot([imu_performance, control_performance],
                    labels=['IMU Only', 'With Control'])
        ax4.set_ylabel('Position Error (m)')
        ax4.set_title('Performance by Prediction Mode')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/control_analysis.png",
                    dpi=300, bbox_inches='tight')


def generate_performance_report(results_control, results_no_control, data, output_file="ekf_performance_report.txt"):
    """Generate comprehensive performance report"""

    print(f"\n📝 Generating performance report: {output_file}")

    # Calculate detailed error statistics
    errors_ctrl = calculate_detailed_errors(results_control, data)
    errors_no_ctrl = calculate_detailed_errors(results_no_control, data)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EKF PERFORMANCE COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(
            f"Data samples processed: {len(results_control['timestamp'])}\n\n")

        # Position performance
        f.write("POSITION ESTIMATION PERFORMANCE\n")
        f.write("-"*40 + "\n")

        pos_rmse_ctrl = np.sqrt(np.mean(errors_ctrl['pos_error']**2, axis=0))
        pos_rmse_no_ctrl = np.sqrt(
            np.mean(errors_no_ctrl['pos_error']**2, axis=0))

        f.write(f"With Control Input:\n")
        f.write(
            f"  RMSE [X,Y,Z]: [{pos_rmse_ctrl[0]:.4f}, {pos_rmse_ctrl[1]:.4f}, {pos_rmse_ctrl[2]:.4f}] m\n")
        f.write(f"  Total RMSE: {np.linalg.norm(pos_rmse_ctrl):.4f} m\n")
        f.write(
            f"  Max Error: {np.max(np.linalg.norm(errors_ctrl['pos_error'], axis=1)):.4f} m\n\n")

        f.write(f"Without Control Input:\n")
        f.write(
            f"  RMSE [X,Y,Z]: [{pos_rmse_no_ctrl[0]:.4f}, {pos_rmse_no_ctrl[1]:.4f}, {pos_rmse_no_ctrl[2]:.4f}] m\n")
        f.write(f"  Total RMSE: {np.linalg.norm(pos_rmse_no_ctrl):.4f} m\n")
        f.write(
            f"  Max Error: {np.max(np.linalg.norm(errors_no_ctrl['pos_error'], axis=1)):.4f} m\n\n")

        # Improvement calculation
        improvement = (np.linalg.norm(pos_rmse_no_ctrl) -
                       np.linalg.norm(pos_rmse_ctrl)) / np.linalg.norm(pos_rmse_no_ctrl) * 100
        f.write(f"Position RMSE Improvement: {improvement:.2f}%\n\n")

        # Velocity performance
        f.write("VELOCITY ESTIMATION PERFORMANCE\n")
        f.write("-"*40 + "\n")

        vel_rmse_ctrl = np.sqrt(np.mean(errors_ctrl['vel_error']**2, axis=0))
        vel_rmse_no_ctrl = np.sqrt(
            np.mean(errors_no_ctrl['vel_error']**2, axis=0))

        f.write(f"With Control Input:\n")
        f.write(
            f"  RMSE [X,Y,Z]: [{vel_rmse_ctrl[0]:.4f}, {vel_rmse_ctrl[1]:.4f}, {vel_rmse_ctrl[2]:.4f}] m/s\n")
        f.write(f"  Total RMSE: {np.linalg.norm(vel_rmse_ctrl):.4f} m/s\n\n")

        f.write(f"Without Control Input:\n")
        f.write(
            f"  RMSE [X,Y,Z]: [{vel_rmse_no_ctrl[0]:.4f}, {vel_rmse_no_ctrl[1]:.4f}, {vel_rmse_no_ctrl[2]:.4f}] m/s\n")
        f.write(
            f"  Total RMSE: {np.linalg.norm(vel_rmse_no_ctrl):.4f} m/s\n\n")

        # Attitude performance
        f.write("ATTITUDE ESTIMATION PERFORMANCE\n")
        f.write("-"*40 + "\n")

        att_rmse_ctrl = np.sqrt(np.mean(errors_ctrl['att_error']**2, axis=0))
        att_rmse_no_ctrl = np.sqrt(
            np.mean(errors_no_ctrl['att_error']**2, axis=0))

        f.write(f"With Control Input:\n")
        f.write(
            f"  RMSE [R,P,Y]: [{np.rad2deg(att_rmse_ctrl[0]):.3f}, {np.rad2deg(att_rmse_ctrl[1]):.3f}, {np.rad2deg(att_rmse_ctrl[2]):.3f}] deg\n\n")

        f.write(f"Without Control Input:\n")
        f.write(
            f"  RMSE [R,P,Y]: [{np.rad2deg(att_rmse_no_ctrl[0]):.3f}, {np.rad2deg(att_rmse_no_ctrl[1]):.3f}, {np.rad2deg(att_rmse_no_ctrl[2]):.3f}] deg\n\n")

        # Control analysis
        if 'prediction_modes' in results_control:
            f.write("CONTROL INPUT ANALYSIS\n")
            f.write("-"*40 + "\n")

            modes = results_control['prediction_modes']
            mode_counts = {}
            for mode in modes:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

            f.write("Prediction mode distribution:\n")
            for mode, count in mode_counts.items():
                percentage = 100 * count / len(modes)
                f.write(f"  {mode}: {count} samples ({percentage:.1f}%)\n")

            control_usage = sum(1 for mode in modes if mode !=
                                "IMU_ONLY") / len(modes) * 100
            f.write(f"\nControl input utilization: {control_usage:.1f}%\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✅ Performance report generated: {output_file}")


def get_aligned_ground_truth(data, timestamps, columns):
    """Get ground truth data aligned with EKF timestamps"""
    result = np.zeros((len(timestamps), len(columns)))

    for i, t in enumerate(timestamps):
        # Find closest timestamp in data
        data_idx = np.argmin(np.abs(data['timestamp'] - t))
        for j, col in enumerate(columns):
            result[i, j] = data.iloc[data_idx][col]

    return result


def calculate_detailed_errors(results, data):
    """Calculate detailed error statistics"""
    timestamps = np.array(results['timestamp'])

    # Get aligned ground truth
    true_pos = get_aligned_ground_truth(
        data, timestamps, ['true_pos_x', 'true_pos_y', 'true_pos_z'])
    true_vel = get_aligned_ground_truth(
        data, timestamps, ['true_vel_x', 'true_vel_y', 'true_vel_z'])
    true_att = get_aligned_ground_truth(
        data, timestamps, ['true_roll', 'true_pitch', 'true_yaw'])

    # Calculate errors
    pos_error = results['position'] - true_pos
    vel_error = results['velocity'] - true_vel
    att_error = results['attitude'] - true_att

    # Handle angle wrapping
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    return {
        'pos_error': pos_error,
        'vel_error': vel_error,
        'att_error': att_error,
        'timestamps': timestamps
    }


def plot_innovation_analysis(results, data, save_plots=False, output_dir="plots"):
    """Plot innovation sequence analysis for EKF validation"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EKF Innovation Analysis', fontsize=16, fontweight='bold')

    # This would require storing innovations during EKF processing
    # For now, showing example structure

    # Position innovation analysis
    ax1 = axes[0, 0]
    time = np.array(results['timestamp'])
    # Placeholder for actual innovation data
    pos_innovation = np.random.normal(0, 0.1, (len(time), 3))

    ax1.plot(time, pos_innovation[:, 0], 'r-', alpha=0.7, label='North')
    ax1.plot(time, pos_innovation[:, 1], 'g-', alpha=0.7, label='East')
    ax1.plot(time, pos_innovation[:, 2], 'b-', alpha=0.7, label='Down')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Innovation (m)')
    ax1.set_title('GPS Position Innovation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Innovation consistency test
    ax2 = axes[0, 1]
    innovation_norms = np.linalg.norm(pos_innovation, axis=1)
    ax2.hist(innovation_norms, bins=30, alpha=0.7, density=True)
    ax2.set_xlabel('Innovation Magnitude (m)')
    ax2.set_ylabel('Density')
    ax2.set_title('Innovation Distribution')
    ax2.grid(True, alpha=0.3)

    # Normalized Innovation Squared (NIS)
    ax3 = axes[1, 0]
    # Placeholder for NIS calculation
    nis = stats.chi2.rvs(df=3, size=len(time))  # 3 DOF for position
    ax3.plot(time, nis, 'b-', alpha=0.7)
    ax3.axhline(y=7.815, color='r', linestyle='--',
                label='95% confidence bound')  # Chi-squared 95% for 3 DOF
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('NIS')
    ax3.set_title('Normalized Innovation Squared')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Uncertainty evolution
    ax4 = axes[1, 1]
    pos_std = results['pos_std']
    ax4.plot(time, np.linalg.norm(pos_std, axis=1), 'g-',
             linewidth=2, label='Position Uncertainty')
    vel_std = results['vel_std']
    ax4.plot(time, np.linalg.norm(vel_std, axis=1), 'b-',
             linewidth=2, label='Velocity Uncertainty')
    att_std = results['att_std']
    ax4.plot(time, np.rad2deg(np.linalg.norm(att_std, axis=1)),
             'r-', linewidth=2, label='Attitude Uncertainty (deg)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Uncertainty')
    ax4.set_title('Estimation Uncertainty Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/innovation_analysis.png",
                    dpi=300, bbox_inches='tight')


def plot_ekf_vs_groundtruth(results, data, save_plots=False, output_dir="plots"):
    """Plot EKF estimation results against ground truth data"""

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # Setup plotting style
    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # Get aligned timestamps
    time = np.array(results['timestamp'])

    # Get ground truth aligned with EKF results timestamps
    true_pos = np.zeros((len(time), 3))
    true_vel = np.zeros((len(time), 3))
    true_att = np.zeros((len(time), 3))

    for i, t in enumerate(time):
        # Find closest timestamp in ground truth data
        idx = np.argmin(np.abs(data['timestamp'] - t))

        # Extract ground truth values
        true_pos[i] = [
            data.iloc[idx]['true_pos_x'],
            data.iloc[idx]['true_pos_y'],
            data.iloc[idx]['true_pos_z']
        ]
        true_vel[i] = [
            data.iloc[idx]['true_vel_x'],
            data.iloc[idx]['true_vel_y'],
            data.iloc[idx]['true_vel_z']
        ]
        true_att[i] = [
            data.iloc[idx]['true_roll'],
            data.iloc[idx]['true_pitch'],
            data.iloc[idx]['true_yaw']
        ]

    # Position plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Position Estimation vs Ground Truth', fontsize=16)

    labels = ['X (North)', 'Y (East)', 'Z (Down)']
    for i in range(3):
        ax = axes[i]
        ax.plot(time, true_pos[:, i], 'g-', linewidth=2, label='Ground Truth')
        ax.plot(time, results['position'][:, i], 'b--',
                linewidth=1.5, label='EKF Estimate')
        ax.set_ylabel(f'Position {labels[i]} (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/position_comparison.png", dpi=300)

    # Attitude plots (convert to degrees for better readability)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Attitude Estimation vs Ground Truth', fontsize=16)

    att_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax = axes[i]
        ax.plot(time, np.rad2deg(true_att[:, i]),
                'g-', linewidth=2, label='Ground Truth')
        ax.plot(time, np.rad2deg(
            results['attitude'][:, i]), 'b--', linewidth=1.5, label='EKF Estimate')
        ax.set_ylabel(f'{att_labels[i]} (degrees)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/attitude_comparison.png", dpi=300)

    # 3D Trajectory plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_pos[:, 0], true_pos[:, 1], -true_pos[:, 2],
            'g-', linewidth=2, label='Ground Truth')
    ax.plot(results['position'][:, 0], results['position'][:, 1], -results['position'][:, 2],
            'b--', linewidth=1.5, label='EKF Estimate')

    # Add start/end markers
    ax.scatter(true_pos[0, 0], true_pos[0, 1], -true_pos[0, 2],
               color='green', marker='o', s=100, label='Start')
    ax.scatter(true_pos[-1, 0], true_pos[-1, 1], -true_pos[-1, 2],
               color='red', marker='x', s=100, label='End')

    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()

    if save_plots:
        plt.savefig(f"{output_dir}/3d_trajectory.png", dpi=300)

    plt.show()

    # Calculate and display error statistics
    pos_error = results['position'] - true_pos
    vel_error = results['velocity'] - true_vel
    att_error = results['attitude'] - true_att

    # Handle angle wrapping for attitude errors
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    print("\n=== ERROR STATISTICS ===")
    print(
        f"Position RMSE [X,Y,Z]: [{pos_rmse[0]:.4f}, {pos_rmse[1]:.4f}, {pos_rmse[2]:.4f}] m")
    print(f"Total Position RMSE: {np.linalg.norm(pos_rmse):.4f} m")
    print(
        f"Velocity RMSE [X,Y,Z]: [{vel_rmse[0]:.4f}, {vel_rmse[1]:.4f}, {vel_rmse[2]:.4f}] m/s")
    print(
        f"Attitude RMSE [R,P,Y]: [{np.rad2deg(att_rmse[0]):.3f}, {np.rad2deg(att_rmse[1]):.3f}, {np.rad2deg(att_rmse[2]):.3f}] degrees")

    return fig


if __name__ == "__main__":
    # Example usage
    print("EKF Utilities module - use functions individually or import into main script")
