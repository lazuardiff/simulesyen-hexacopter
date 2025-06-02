"""
EKF Comparison Runner Script
===========================

This is the main script to run and compare EKF implementations
with and without control input integration.

Workflow:
1. Load simulation data and validate
2. Run EKF with control input
3. Run EKF without control input  
4. Compare results and generate reports
5. Create comprehensive visualizations
6. Save results and analysis

Usage:
    python run_ekf_comparison.py

Configuration:
    Edit the get_default_config() function to change file paths and settings

Author: EKF Implementation Team
Date: 2025
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our EKF implementations
from ekf_with_control import run_ekf_with_control_data
from ekf_without_control import run_ekf_without_control_data
from ekf_utils import (
    plot_ekf_comparison,
    plot_control_analysis,
    generate_performance_report,
    plot_innovation_analysis
)


def get_default_config():
    """Get default configuration - no command line arguments needed"""

    # =================================================================
    # 🔧 CONFIGURATION - EDIT THESE PARAMETERS AS NEEDED
    # =================================================================

    config = {
        # Data file path - UPDATE THIS PATH TO YOUR DATA FILE
        'data_file': "logs/complete_flight_data_with_geodetic_20250530_183803.csv",

        # Output directory for results and plots
        'output_dir': "ekf_comparison_results",

        # Local magnetic declination in degrees (Surabaya: ~0.5)
        'magnetic_declination': 0.5,

        # Sensor configuration
        'use_magnetometer': True,

        # Plot and report settings
        'generate_plots': True,
        'save_plots': False,
        'show_plots': True,
        'generate_report': False,
        'save_results': False,

        # Debug settings
        'verbose': False
    }

    return config


def main():
    """Main function to run EKF comparison"""

    # Get configuration (no command line arguments needed)
    config = get_default_config()

    # Print header
    print_header()

    # Print configuration
    print_configuration(config)

    # Validate input file
    if not validate_input_file(config['data_file']):
        return 1

    # Create output directory
    setup_output_directory(config['output_dir'])

    # Start timing
    start_time = time.time()

    try:
        # === STEP 1: RUN EKF WITH CONTROL INPUT ===
        print("\n" + "🎮" + " "*3 + "RUNNING EKF WITH CONTROL INPUT")
        print("="*80)

        results_control = run_ekf_with_control_data(
            csv_file_path=config['data_file'],
            use_magnetometer=config['use_magnetometer'],
            magnetic_declination=config['magnetic_declination']
        )

        if results_control is None:
            print("❌ EKF with control input failed!")
            return 1

        ekf_control, results_data_control, data = results_control
        print("✅ EKF with control input completed successfully!")

        # Print uncertainty statistics for control EKF
        print_uncertainty_statistics(
            results_data_control, "EKF with Control Input")

        # === STEP 2: RUN EKF WITHOUT CONTROL INPUT ===
        print("\n" + "📡" + " "*3 + "RUNNING EKF WITHOUT CONTROL INPUT")
        print("="*80)

        results_no_control = run_ekf_without_control_data(
            csv_file_path=config['data_file'],
            use_magnetometer=config['use_magnetometer'],
            magnetic_declination=config['magnetic_declination']
        )

        if results_no_control is None:
            print("❌ EKF without control input failed!")
            return 1

        ekf_no_control, results_data_no_control, _ = results_no_control
        print("✅ EKF without control input completed successfully!")

        # Print uncertainty statistics for no-control EKF
        print_uncertainty_statistics(
            results_data_no_control, "EKF without Control Input")

        # === STEP 3: PERFORMANCE COMPARISON ===
        print("\n" + "📊" + " "*3 + "PERFORMANCE COMPARISON ANALYSIS")
        print("="*80)

        comparison_results = compare_performance(
            results_data_control,
            results_data_no_control,
            data
        )

        # === STEP 4: GENERATE VISUALIZATIONS ===
        if config['generate_plots']:
            print("\n" + "🎨" + " "*3 + "GENERATING VISUALIZATION PLOTS")
            print("="*80)

            # Main comparison plots (without uncertainty bounds)
            plot_ekf_comparison(
                results_data_control,
                results_data_no_control,
                data,
                save_plots=config['save_plots'],
                output_dir=config['output_dir']
            )

            # Control analysis plots
            plot_control_analysis(
                results_data_control,
                save_plots=config['save_plots'],
                output_dir=config['output_dir']
            )

            # Innovation analysis
            plot_innovation_analysis(
                results_data_control,
                data,
                save_plots=config['save_plots'],
                output_dir=config['output_dir']
            )

            if config['show_plots']:
                plt.show()

        # === STEP 5: GENERATE REPORTS ===
        if config['generate_report']:
            print("\n" + "📝" + " "*3 + "GENERATING PERFORMANCE REPORT")
            print("="*80)

            report_file = os.path.join(
                config['output_dir'], f"ekf_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            generate_performance_report(
                results_data_control,
                results_data_no_control,
                data,
                output_file=report_file
            )

        # === STEP 6: SAVE RESULTS ===
        if config['save_results']:
            save_results_to_files(
                results_data_control,
                results_data_no_control,
                comparison_results,
                config['output_dir']
            )

        # Final summary
        end_time = time.time()
        execution_time = end_time - start_time

        print_summary(comparison_results, execution_time)

        return 0

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def print_configuration(config):
    """Print current configuration settings"""
    print(f"\n⚙️  CONFIGURATION SETTINGS")
    print("-"*50)
    print(f"📁 Data file: {config['data_file']}")
    print(f"📂 Output directory: {config['output_dir']}")
    print(f"🧭 Magnetic declination: {config['magnetic_declination']}°")
    print(f"📡 Use magnetometer: {config['use_magnetometer']}")
    print(f"📊 Generate plots: {config['generate_plots']}")
    print(f"💾 Save results: {config['save_results']}")
    print("-"*50)


def print_header():
    """Print script header"""
    print("\n" + "="*80)
    print("🚁 EKF COMPARISON: CONTROL INPUT vs IMU-ONLY ESTIMATION")
    print("="*80)
    print("📅 Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 Objective: Compare EKF performance with and without control input")
    print("📊 Analysis: Position, Velocity, and Attitude estimation accuracy")
    print("="*80)


def print_uncertainty_statistics(results, ekf_name):
    """Print uncertainty statistics to terminal instead of plotting"""
    print(f"\n📊 {ekf_name.upper()} - UNCERTAINTY ANALYSIS")
    print("-"*60)

    # Calculate average uncertainties
    pos_std = np.array(results['pos_std'])
    vel_std = np.array(results['vel_std'])
    att_std = np.array(results['att_std'])

    # Position uncertainty statistics
    pos_mean = np.mean(pos_std, axis=0)
    pos_max = np.max(pos_std, axis=0)
    pos_final = pos_std[-1] if len(pos_std) > 0 else np.zeros(3)

    print(f"📍 Position Uncertainty (1σ):")
    print(
        f"   Average: [{pos_mean[0]:.4f}, {pos_mean[1]:.4f}, {pos_mean[2]:.4f}] m")
    print(
        f"   Maximum: [{pos_max[0]:.4f}, {pos_max[1]:.4f}, {pos_max[2]:.4f}] m")
    print(
        f"   Final:   [{pos_final[0]:.4f}, {pos_final[1]:.4f}, {pos_final[2]:.4f}] m")

    # Velocity uncertainty statistics
    vel_mean = np.mean(vel_std, axis=0)
    vel_max = np.max(vel_std, axis=0)
    vel_final = vel_std[-1] if len(vel_std) > 0 else np.zeros(3)

    print(f"\n🏃 Velocity Uncertainty (1σ):")
    print(
        f"   Average: [{vel_mean[0]:.4f}, {vel_mean[1]:.4f}, {vel_mean[2]:.4f}] m/s")
    print(
        f"   Maximum: [{vel_max[0]:.4f}, {vel_max[1]:.4f}, {vel_max[2]:.4f}] m/s")
    print(
        f"   Final:   [{vel_final[0]:.4f}, {vel_final[1]:.4f}, {vel_final[2]:.4f}] m/s")

    # Attitude uncertainty statistics (convert to degrees)
    att_mean_deg = np.rad2deg(np.mean(att_std, axis=0))
    att_max_deg = np.rad2deg(np.max(att_std, axis=0))
    att_final_deg = np.rad2deg(
        att_std[-1]) if len(att_std) > 0 else np.zeros(3)

    print(f"\n🎯 Attitude Uncertainty (1σ):")
    print(
        f"   Average: [{att_mean_deg[0]:.3f}, {att_mean_deg[1]:.3f}, {att_mean_deg[2]:.3f}] deg")
    print(
        f"   Maximum: [{att_max_deg[0]:.3f}, {att_max_deg[1]:.3f}, {att_max_deg[2]:.3f}] deg")
    print(
        f"   Final:   [{att_final_deg[0]:.3f}, {att_final_deg[1]:.3f}, {att_final_deg[2]:.3f}] deg")

    # Overall uncertainty assessment
    total_pos_uncertainty = np.mean(np.linalg.norm(pos_std, axis=1))
    total_att_uncertainty = np.rad2deg(
        np.mean(np.linalg.norm(att_std, axis=1)))

    print(f"\n🔍 Overall Assessment:")
    print(
        f"   Average total position uncertainty: {total_pos_uncertainty:.4f} m")
    print(
        f"   Average total attitude uncertainty: {total_att_uncertainty:.3f} deg")

    # Uncertainty quality assessment
    if total_pos_uncertainty < 0.1:
        print("   ✅ Position uncertainty: EXCELLENT")
    elif total_pos_uncertainty < 0.5:
        print("   ✅ Position uncertainty: GOOD")
    elif total_pos_uncertainty < 1.0:
        print("   ⚠️  Position uncertainty: ACCEPTABLE")
    else:
        print("   ❌ Position uncertainty: HIGH")

    if total_att_uncertainty < 1.0:
        print("   ✅ Attitude uncertainty: EXCELLENT")
    elif total_att_uncertainty < 3.0:
        print("   ✅ Attitude uncertainty: GOOD")
    elif total_att_uncertainty < 5.0:
        print("   ⚠️  Attitude uncertainty: ACCEPTABLE")
    else:
        print("   ❌ Attitude uncertainty: HIGH")

    print("-"*60)


def validate_input_file(file_path):
    """Validate input data file"""
    print(f"\n🔍 Validating input file: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("💡 Please run the simulation first to generate data file")
        return False

    try:
        # Quick validation of file structure
        data = pd.read_csv(file_path, nrows=5)
        required_columns = [
            'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
            'gps_pos_ned_x', 'gps_pos_ned_y', 'gps_pos_ned_z',
            'true_pos_x', 'true_pos_y', 'true_pos_z'
        ]

        missing_columns = [
            col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"✅ File validated successfully")
        print(f"📁 File size: {file_size:.2f} MB")
        print(f"📈 Columns available: {len(data.columns)}")

        return True

    except Exception as e:
        print(f"❌ File validation error: {str(e)}")
        return False


def setup_output_directory(output_dir):
    """Create and setup output directory"""
    print(f"\n📁 Setting up output directory: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        subdirs = ['plots', 'data', 'reports']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        print(f"✅ Output directory ready: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"❌ Failed to create output directory: {str(e)}")
        raise


def compare_performance(results_control, results_no_control, data):
    """Compare performance between control and no-control EKF"""

    print("🔍 Calculating performance metrics...")

    # Calculate errors for both methods
    errors_control = calculate_errors(results_control, data)
    errors_no_control = calculate_errors(results_no_control, data)

    # Calculate improvement percentages
    improvements = calculate_improvements(errors_control, errors_no_control)

    # Print comparison summary
    print_performance_comparison(
        errors_control, errors_no_control, improvements)

    return {
        'errors_control': errors_control,
        'errors_no_control': errors_no_control,
        'improvements': improvements,
        'control_samples': len(results_control['timestamp']),
        'no_control_samples': len(results_no_control['timestamp'])
    }


def calculate_errors(results, data):
    """Calculate RMSE errors for EKF results"""

    # Align timestamps
    result_times = np.array(results['timestamp'])
    valid_indices = []

    for i, t in enumerate(data['timestamp']):
        if t in result_times:
            # Validate ground truth
            row = data.iloc[i]
            true_pos = np.array(
                [row['true_pos_x'], row['true_pos_y'], row['true_pos_z']])
            if not np.allclose(true_pos, 0, atol=1e-6):
                valid_indices.append(i)

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

    # Align EKF results
    gt_times = data.iloc[valid_indices]['timestamp'].values
    matching_indices = []

    for gt_time in gt_times:
        result_idx = np.argmin(np.abs(result_times - gt_time))
        if abs(result_times[result_idx] - gt_time) < 1e-6:
            matching_indices.append(result_idx)

    # Calculate errors
    n_matches = min(len(matching_indices), len(true_pos))
    pos_error = results['position'][matching_indices[:n_matches]
                                    ] - true_pos[:n_matches]
    vel_error = results['velocity'][matching_indices[:n_matches]
                                    ] - true_vel[:n_matches]
    att_error = results['attitude'][matching_indices[:n_matches]
                                    ] - true_att[:n_matches]

    # Handle angle wrapping
    att_error = np.arctan2(np.sin(att_error), np.cos(att_error))

    # Calculate RMSE
    pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_error**2, axis=0))

    return {
        'position_rmse': pos_rmse,
        'velocity_rmse': vel_rmse,
        'attitude_rmse': att_rmse,
        'position_total_rmse': np.linalg.norm(pos_rmse),
        'velocity_total_rmse': np.linalg.norm(vel_rmse),
        'attitude_total_rmse_deg': np.rad2deg(np.linalg.norm(att_rmse)),
        'valid_samples': n_matches
    }


def calculate_improvements(errors_control, errors_no_control):
    """Calculate improvement percentages"""

    improvements = {}

    # Position improvement
    pos_improvement = (errors_no_control['position_total_rmse'] -
                       errors_control['position_total_rmse']) / errors_no_control['position_total_rmse'] * 100
    improvements['position'] = pos_improvement

    # Velocity improvement
    vel_improvement = (errors_no_control['velocity_total_rmse'] -
                       errors_control['velocity_total_rmse']) / errors_no_control['velocity_total_rmse'] * 100
    improvements['velocity'] = vel_improvement

    # Attitude improvement
    att_improvement = (errors_no_control['attitude_total_rmse_deg'] -
                       errors_control['attitude_total_rmse_deg']) / errors_no_control['attitude_total_rmse_deg'] * 100
    improvements['attitude'] = att_improvement

    return improvements


def print_performance_comparison(errors_control, errors_no_control, improvements):
    """Print detailed performance comparison"""

    print("\n🏆 PERFORMANCE COMPARISON RESULTS")
    print("="*60)

    # Position comparison
    print("📍 POSITION ESTIMATION:")
    print(f"   With Control:    {errors_control['position_total_rmse']:.4f} m")
    print(
        f"   Without Control: {errors_no_control['position_total_rmse']:.4f} m")
    print(f"   💡 Improvement:   {improvements['position']:+.2f}%")

    # Velocity comparison
    print("\n🏃 VELOCITY ESTIMATION:")
    print(
        f"   With Control:    {errors_control['velocity_total_rmse']:.4f} m/s")
    print(
        f"   Without Control: {errors_no_control['velocity_total_rmse']:.4f} m/s")
    print(f"   💡 Improvement:   {improvements['velocity']:+.2f}%")

    # Attitude comparison
    print("\n🎯 ATTITUDE ESTIMATION:")
    print(
        f"   With Control:    {errors_control['attitude_total_rmse_deg']:.3f}°")
    print(
        f"   Without Control: {errors_no_control['attitude_total_rmse_deg']:.3f}°")
    print(f"   💡 Improvement:   {improvements['attitude']:+.2f}%")

    # Overall assessment
    print("\n🔍 OVERALL ASSESSMENT:")
    avg_improvement = np.mean(
        [improvements['position'], improvements['velocity'], improvements['attitude']])

    if avg_improvement > 5:
        print(
            f"   🏆 EXCELLENT: {avg_improvement:.1f}% average improvement with control input")
    elif avg_improvement > 2:
        print(
            f"   ✅ GOOD: {avg_improvement:.1f}% average improvement with control input")
    elif avg_improvement > 0:
        print(
            f"   ⚠️  MARGINAL: {avg_improvement:.1f}% average improvement with control input")
    else:
        print(
            f"   ❌ NO IMPROVEMENT: {avg_improvement:.1f}% with control input")

    print("="*60)


def save_results_to_files(results_control, results_no_control, comparison_results, output_dir):
    """Save numerical results to CSV files"""

    print("\n💾 Saving results to files...")

    # Save EKF results
    data_dir = os.path.join(output_dir, 'data')

    # Control results
    df_control = pd.DataFrame({
        'timestamp': results_control['timestamp'],
        'pos_x': results_control['position'][:, 0],
        'pos_y': results_control['position'][:, 1],
        'pos_z': results_control['position'][:, 2],
        'vel_x': results_control['velocity'][:, 0],
        'vel_y': results_control['velocity'][:, 1],
        'vel_z': results_control['velocity'][:, 2],
        'roll': np.rad2deg(results_control['attitude'][:, 0]),
        'pitch': np.rad2deg(results_control['attitude'][:, 1]),
        'yaw': np.rad2deg(results_control['attitude'][:, 2]),
        'pos_std_x': results_control['pos_std'][:, 0],
        'pos_std_y': results_control['pos_std'][:, 1],
        'pos_std_z': results_control['pos_std'][:, 2],
        'prediction_mode': results_control['prediction_modes']
    })

    control_file = os.path.join(data_dir, 'ekf_with_control_results.csv')
    df_control.to_csv(control_file, index=False)

    # No-control results
    df_no_control = pd.DataFrame({
        'timestamp': results_no_control['timestamp'],
        'pos_x': results_no_control['position'][:, 0],
        'pos_y': results_no_control['position'][:, 1],
        'pos_z': results_no_control['position'][:, 2],
        'vel_x': results_no_control['velocity'][:, 0],
        'vel_y': results_no_control['velocity'][:, 1],
        'vel_z': results_no_control['velocity'][:, 2],
        'roll': np.rad2deg(results_no_control['attitude'][:, 0]),
        'pitch': np.rad2deg(results_no_control['attitude'][:, 1]),
        'yaw': np.rad2deg(results_no_control['attitude'][:, 2]),
        'pos_std_x': results_no_control['pos_std'][:, 0],
        'pos_std_y': results_no_control['pos_std'][:, 1],
        'pos_std_z': results_no_control['pos_std'][:, 2]
    })

    no_control_file = os.path.join(data_dir, 'ekf_without_control_results.csv')
    df_no_control.to_csv(no_control_file, index=False)

    # Comparison summary
    summary_file = os.path.join(data_dir, 'comparison_summary.csv')
    df_summary = pd.DataFrame({
        'metric': ['position_rmse_m', 'velocity_rmse_ms', 'attitude_rmse_deg'],
        'with_control': [
            comparison_results['errors_control']['position_total_rmse'],
            comparison_results['errors_control']['velocity_total_rmse'],
            comparison_results['errors_control']['attitude_total_rmse_deg']
        ],
        'without_control': [
            comparison_results['errors_no_control']['position_total_rmse'],
            comparison_results['errors_no_control']['velocity_total_rmse'],
            comparison_results['errors_no_control']['attitude_total_rmse_deg']
        ],
        'improvement_percent': [
            comparison_results['improvements']['position'],
            comparison_results['improvements']['velocity'],
            comparison_results['improvements']['attitude']
        ]
    })

    df_summary.to_csv(summary_file, index=False)

    print(f"✅ Results saved to:")
    print(f"   📊 Control EKF: {control_file}")
    print(f"   📊 No-control EKF: {no_control_file}")
    print(f"   📊 Summary: {summary_file}")


def print_summary(comparison_results, execution_time):
    """Print final execution summary"""

    print("\n" + "🎉" + " "*3 + "EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)

    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(
        f"📊 Control samples processed: {comparison_results['control_samples']}")
    print(
        f"📊 No-control samples processed: {comparison_results['no_control_samples']}")

    # Key findings
    improvements = comparison_results['improvements']
    best_improvement = max(improvements.values())
    worst_improvement = min(improvements.values())

    print(f"\n🔍 KEY FINDINGS:")
    print(f"   🏆 Best improvement: {best_improvement:+.2f}%")
    print(f"   📉 Worst improvement: {worst_improvement:+.2f}%")

    if best_improvement > 5:
        print(f"   ✅ Control input provides significant benefit!")
    elif best_improvement > 0:
        print(f"   ⚠️  Control input provides modest benefit")
    else:
        print(f"   ❌ Control input shows no clear benefit")

    print("\n📝 Check the generated reports and plots for detailed analysis.")
    print("="*80)


if __name__ == "__main__":
    """Entry point"""
    exit_code = main()
    sys.exit(exit_code)
