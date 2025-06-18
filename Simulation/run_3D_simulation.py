# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!

Enhanced version with comprehensive plotting capabilities:
- 3D trajectory visualization
- Sensor data plots
- Ground truth vs sensor comparison
- GPS geodetic coordinate visualization

MODIFIED VERSION:
- All plots now show the MISSION PHASE ONLY (t >= 5s).
- 3D trajectory plot is now separate from the 2D trajectory views.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import cProfile
import pandas as pd
import os
from datetime import datetime
import seaborn as sns

# Asumsikan file-file ini berada di path yang benar
from trajectory import Trajectory
from ctrl import Control
from quadFiles.hexacopter import Quadcopter
from utils.windModel import Wind
from utils.SensorModels import IMUSensor, GPSSensor, AltitudeSensor, MagnetometerSensor
import utils
import config


def quad_sim(t, Ts, quad, ctrl, wind, traj, imu, gps, alti, mag):
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Mengambil data sensor
    acc_m, gyro_m = imu.measure(quad, t)

    # ===== ENHANCED GPS MEASUREMENT =====
    # GPS now returns NED position, NED velocity, and geodetic coordinates
    pos_ned_m, vel_ned_m, geodetic_m = gps.measure(quad, t)

    alt_m = alti.measure(quad, t)
    mag_m = mag.measure(quad, t)

    # Trajectory for Desired States
    # ---------------------------
    sDes = traj.desiredState(t, Ts, quad)

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj, quad, sDes, Ts)

    return t, acc_m, gyro_m, pos_ned_m, vel_ned_m, geodetic_m, alt_m, mag_m


class TrajectoryWithHovering:
    """Enhanced trajectory class with initial hovering phase"""

    def __init__(self, original_traj, hover_altitude=1.0, hover_duration=3.0):
        self.original_traj = original_traj
        self.hover_altitude = hover_altitude
        self.hover_duration = hover_duration

        # Copy all attributes from original trajectory
        for attr in dir(original_traj):
            if not attr.startswith('_') and attr != 'desiredState':
                setattr(self, attr, getattr(original_traj, attr))

    def desiredState(self, t, Ts, quad):
        """Enhanced desired state with hovering phase"""

        if t <= self.hover_duration:
            # HOVERING PHASE: Stay at hover_altitude for first 3 seconds
            sDes = np.zeros(19)
            sDes[0:3] = [0, 0, -self.hover_altitude if config.orient ==
                         "NED" else self.hover_altitude]
            if config.orient == "NED":
                sDes[11] = quad.params["mB"] * quad.params["g"]
            else:
                sDes[11] = -quad.params["mB"] * quad.params["g"]
            self.sDes = sDes
            return sDes

        else:
            # MISSION PHASE: Use original trajectory but offset time
            mission_time = t - self.hover_duration
            sDes = self.original_traj.desiredState(mission_time, Ts, quad)

            if hasattr(self, '_position_offset_applied') and not self._position_offset_applied:
                self._initial_mission_pos = sDes[0:3].copy()
                self._position_offset_applied = True

            if hasattr(self, '_initial_mission_pos'):
                if config.orient == "NED":
                    sDes[2] = sDes[2] - self._initial_mission_pos[2] - \
                        self.hover_altitude
                else:
                    sDes[2] = sDes[2] - self._initial_mission_pos[2] + \
                        self.hover_altitude

            self.sDes = sDes
            return sDes

# ========================================================================================
# ===== MODIFIED PLOTTING FUNCTIONS (MISSION PHASE ONLY) =====
# ========================================================================================


def plot_3d_trajectory_mission(pos_all, save_plots=True, output_dir="simulation_plots"):
    """
    Plot 3D trajectory dari fase misi SAJA.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')

    # Plot fase misi dengan warna merah
    ax1.plot(pos_all[:, 0], pos_all[:, 1], -pos_all[:, 2],
             'r-', linewidth=2, label='Mission Phase', alpha=0.8)

    # Tandai awal dan akhir misi
    ax1.scatter(pos_all[0, 0], pos_all[0, 1], -pos_all[0, 2],
                color='green', marker='o', s=100, label='Mission Start (t=5s)')
    ax1.scatter(pos_all[-1, 0], pos_all[-1, 1], -pos_all[-1, 2],
                color='red', marker='x', s=100, label='End')

    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Up (m)')
    ax1.set_title('3D Flight Trajectory (Mission Phase)',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Menyamakan aspek rasio sumbu
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    z_lim = ax1.get_zlim()
    x_range = abs(x_lim[1] - x_lim[0])
    x_middle = np.mean(x_lim)
    y_range = abs(y_lim[1] - y_lim[0])
    y_middle = np.mean(y_lim)
    z_range = abs(z_lim[1] - z_lim[0])
    z_middle = np.mean(z_lim)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax1.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax1.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax1.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{output_dir}/3d_trajectory_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission 3D trajectory plot saved to {output_dir}/3d_trajectory_mission.png")

    plt.show()


def plot_trajectory_views_mission(t_all, pos_all, save_plots=True, output_dir="simulation_plots"):
    """
    Plot tampilan trajectory 2D untuk fase misi SAJA.
    (Menggantikan 'animasi' atau subplot dari plot 3D asli)
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('Trajectory Views (Mission Phase)',
                 fontsize=16, fontweight='bold')

    # Tampilan atas (Bidang X-Y)
    ax1 = fig.add_subplot(131)
    ax1.plot(pos_all[:, 0], pos_all[:, 1], 'r-',
             linewidth=2, label='Mission', alpha=0.8)
    ax1.scatter(pos_all[0, 0], pos_all[0, 1], color='green',
                marker='o', s=100, label='Mission Start')
    ax1.scatter(pos_all[-1, 0], pos_all[-1, 1],
                color='red', marker='x', s=100, label='End')
    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_title('Top View (Horizontal Trajectory)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Tampilan samping (Bidang X-Z)
    ax2 = fig.add_subplot(132)
    ax2.plot(pos_all[:, 0], -pos_all[:, 2], 'r-',
             linewidth=2, label='Mission', alpha=0.8)
    ax2.scatter(pos_all[0, 0], -pos_all[0, 2], color='green',
                marker='o', s=100, label='Mission Start')
    ax2.scatter(pos_all[-1, 0], -pos_all[-1, 2],
                color='red', marker='x', s=100, label='End')
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('Up (m)')
    ax2.set_title('Side View (North-Up Plane)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Waktu vs altitude
    ax3 = fig.add_subplot(133)
    ax3.plot(t_all, -pos_all[:, 2], 'k-', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Time Since Mission Start (s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude vs Time', fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plots:
        plt.savefig(f"{output_dir}/trajectory_views_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission trajectory views plot saved to {output_dir}/trajectory_views_mission.png")

    plt.show()


def plot_sensor_data_mission(t_all, acc_all, gyro_all, gps_pos_ned_all, gps_vel_ned_all,
                             gps_available, baro_alt_all, baro_available, mag_all, mag_available,
                             save_plots=True, output_dir="simulation_plots"):
    """
    Plot semua data sensor untuk fase misi SAJA.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # Plot Data IMU
    fig1, axes1 = plt.subplots(2, 1, figsize=(15, 10))
    fig1.suptitle('IMU Sensor Data (Mission Phase)',
                  fontsize=16, fontweight='bold')

    ax = axes1[0]
    ax.plot(t_all, acc_all[:, 0], 'r-', label='Acc X', alpha=0.8)
    ax.plot(t_all, acc_all[:, 1], 'g-', label='Acc Y', alpha=0.8)
    ax.plot(t_all, acc_all[:, 2], 'b-', label='Acc Z', alpha=0.8)
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Accelerometer Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    ax.plot(t_all, np.rad2deg(gyro_all[:, 0]), 'r-', label='Gyro X', alpha=0.8)
    ax.plot(t_all, np.rad2deg(gyro_all[:, 1]), 'g-', label='Gyro Y', alpha=0.8)
    ax.plot(t_all, np.rad2deg(gyro_all[:, 2]), 'b-', label='Gyro Z', alpha=0.8)
    ax.set_xlabel('Time Since Mission Start (s)')
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('Gyroscope Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/imu_data_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission IMU data plot saved to {output_dir}/imu_data_mission.png")
    plt.show()

    # Plot Data GPS
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10))
    fig2.suptitle('GPS Sensor Data (Mission Phase)',
                  fontsize=16, fontweight='bold')

    gps_times = t_all[gps_available]
    gps_pos = gps_pos_ned_all[gps_available]
    gps_vel = gps_vel_ned_all[gps_available]

    ax = axes2[0]
    if len(gps_times) > 0:
        ax.plot(gps_times, gps_pos[:, 0], 'ro-',
                label='GPS North', markersize=3, alpha=0.8)
        ax.plot(gps_times, gps_pos[:, 1], 'go-',
                label='GPS East', markersize=3, alpha=0.8)
        ax.plot(gps_times, gps_pos[:, 2], 'bo-',
                label='GPS Down', markersize=3, alpha=0.8)
    ax.set_ylabel('Position (m)')
    ax.set_title('GPS Position Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    if len(gps_times) > 0:
        ax.plot(gps_times, gps_vel[:, 0], 'ro-',
                label='GPS Vel North', markersize=3, alpha=0.8)
        ax.plot(gps_times, gps_vel[:, 1], 'go-',
                label='GPS Vel East', markersize=3, alpha=0.8)
        ax.plot(gps_times, gps_vel[:, 2], 'bo-',
                label='GPS Vel Down', markersize=3, alpha=0.8)
    ax.set_xlabel('Time Since Mission Start (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('GPS Velocity Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/gps_data_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission GPS data plot saved to {output_dir}/gps_data_mission.png")
    plt.show()

    # Plot Sensor Lain
    fig3, axes3 = plt.subplots(2, 1, figsize=(15, 10))
    fig3.suptitle('Other Sensor Data (Mission Phase)',
                  fontsize=16, fontweight='bold')

    baro_times = t_all[baro_available]
    baro_data = baro_alt_all[baro_available]
    mag_times = t_all[mag_available]
    mag_data = mag_all[mag_available]

    ax = axes3[0]
    if len(baro_times) > 0:
        ax.plot(baro_times, baro_data, 'ko-',
                label='Barometer Altitude', markersize=2, alpha=0.8)
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Barometer Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    if len(mag_times) > 0:
        ax.plot(mag_times, mag_data[:, 0], 'r-', label='Mag X', alpha=0.8)
        ax.plot(mag_times, mag_data[:, 1], 'g-', label='Mag Y', alpha=0.8)
        ax.plot(mag_times, mag_data[:, 2], 'b-', label='Mag Z', alpha=0.8)
    ax.set_xlabel('Time Since Mission Start (s)')
    ax.set_ylabel('Magnetic Field (Gauss)')
    ax.set_title('Magnetometer Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/other_sensors_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission other sensors plot saved to {output_dir}/other_sensors_mission.png")
    plt.show()


def plot_ground_truth_comparison_mission(t_all, pos_all, vel_all, euler_all,
                                         gps_pos_ned_all, gps_vel_ned_all, gps_available,
                                         save_plots=True, output_dir="simulation_plots"):
    """
    Plot perbandingan ground truth vs sensor untuk fase misi SAJA.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    gps_times = t_all[gps_available]
    gps_pos = gps_pos_ned_all[gps_available]
    gps_vel = gps_vel_ned_all[gps_available]

    # Perbandingan Posisi
    fig1, axes1 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig1.suptitle('Ground Truth vs GPS Position (Mission Phase)',
                  fontsize=16, fontweight='bold')

    labels = ['North (X)', 'East (Y)', 'Down (Z)']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        ax = axes1[i]
        ax.plot(t_all, pos_all[:, i], color=colors[i],
                linewidth=2, label='Ground Truth', alpha=0.8)
        if len(gps_times) > 0:
            ax.plot(gps_times, gps_pos[:, i], 'o', color=colors[i],
                    markersize=3, label='GPS Measurement', alpha=0.6)
        ax.set_ylabel(f'Position {labels[i]} (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes1[2].set_xlabel('Time Since Mission Start (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        plt.savefig(f"{output_dir}/position_comparison_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission position comparison plot saved to {output_dir}/position_comparison_mission.png")
    plt.show()

    # Perbandingan Kecepatan
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig2.suptitle('Ground Truth vs GPS Velocity (Mission Phase)',
                  fontsize=16, fontweight='bold')
    for i in range(3):
        ax = axes2[i]
        ax.plot(t_all, vel_all[:, i], color=colors[i],
                linewidth=2, label='Ground Truth', alpha=0.8)
        if len(gps_times) > 0:
            ax.plot(gps_times, gps_vel[:, i], 'o', color=colors[i],
                    markersize=3, label='GPS Measurement', alpha=0.6)
        ax.set_ylabel(f'Velocity {labels[i]} (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes2[2].set_xlabel('Time Since Mission Start (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        plt.savefig(f"{output_dir}/velocity_comparison_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission velocity comparison plot saved to {output_dir}/velocity_comparison_mission.png")
    plt.show()

    # Plot Attitude
    fig3, axes3 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig3.suptitle('Ground Truth Attitude (Mission Phase)',
                  fontsize=16, fontweight='bold')
    att_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax = axes3[i]
        ax.plot(t_all, np.rad2deg(
            euler_all[:, i]), color=colors[i], linewidth=2, alpha=0.8, label=att_labels[i])
        ax.set_ylabel(f'{att_labels[i]} (degrees)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes3[2].set_xlabel('Time Since Mission Start (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        plt.savefig(f"{output_dir}/attitude_ground_truth_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission attitude plot saved to {output_dir}/attitude_ground_truth_mission.png")
    plt.show()


def plot_gps_geodetic_map_mission(gps_lat_all, gps_lon_all, gps_available,
                                  gps_sensor, save_plots=True, output_dir="simulation_plots"):
    """
    Plot koordinat geodetik GPS di peta untuk fase misi SAJA.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    valid_gps = gps_available & (gps_lat_all != 0) & (gps_lon_all != 0)
    if not np.any(valid_gps):
        print("⚠ No valid GPS data for geodetic map")
        return

    gps_lat_valid = gps_lat_all[valid_gps]
    gps_lon_valid = gps_lon_all[valid_gps]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('GPS Geodetic Coordinates (Mission Phase)',
                 fontsize=16, fontweight='bold')

    # Plot trajectory misi
    ax1.plot(gps_lon_valid, gps_lat_valid, 'ro-', markersize=3,
             linewidth=2, label='Mission Phase', alpha=0.8)

    # Tandai titik awal dan akhir
    ax1.scatter(gps_lon_valid[0], gps_lat_valid[0], color='green',
                marker='o', s=100, label='Mission Start', zorder=5)
    ax1.scatter(gps_lon_valid[-1], gps_lat_valid[-1],
                color='red', marker='x', s=100, label='End', zorder=5)

    ref_pos = gps_sensor.get_reference_position()
    ax1.scatter(ref_pos['longitude'], ref_pos['latitude'], color='black',
                marker='s', s=100, label='Home/Reference', zorder=5)

    ax1.set_xlabel('Longitude (degrees)')
    ax1.set_ylabel('Latitude (degrees)')
    ax1.set_title('GPS Track (Geodetic Coordinates)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots:
        plt.savefig(f"{output_dir}/gps_geodetic_map_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission GPS geodetic map saved to {output_dir}/gps_geodetic_map_mission.png")
    plt.show()


def plot_control_signals_mission(t_all, thrust_sp_all, rate_sp_all, w_cmd_all, thr_all,
                                 control_torques_all, save_plots=True, output_dir="simulation_plots"):
    """
    Plot sinyal kontrol dan output motor untuk fase misi SAJA.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    plt.style.use(
        'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # Plot Perintah Kontrol
    fig1, axes1 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig1.suptitle('Control Signals (Mission Phase)',
                  fontsize=16, fontweight='bold')

    ax = axes1[0]
    ax.plot(t_all, thrust_sp_all[:, 0], 'r-', label='Thrust X', alpha=0.8)
    ax.plot(t_all, thrust_sp_all[:, 1], 'g-', label='Thrust Y', alpha=0.8)
    ax.plot(t_all, thrust_sp_all[:, 2], 'b-', label='Thrust Z', alpha=0.8)
    ax.set_ylabel('Thrust Setpoint (N)')
    ax.set_title('Desired Thrust Vector')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    ax.plot(t_all, np.rad2deg(
        rate_sp_all[:, 0]), 'r-', label='Roll Rate', alpha=0.8)
    ax.plot(t_all, np.rad2deg(rate_sp_all[:, 1]),
            'g-', label='Pitch Rate', alpha=0.8)
    ax.plot(t_all, np.rad2deg(
        rate_sp_all[:, 2]), 'b-', label='Yaw Rate', alpha=0.8)
    ax.set_ylabel('Rate Setpoint (deg/s)')
    ax.set_title('Desired Angular Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes1[2]
    ax.plot(t_all, control_torques_all[:, 0],
            'r-', label='Torque X (Roll)', alpha=0.8)
    ax.plot(t_all, control_torques_all[:, 1],
            'g-', label='Torque Y (Pitch)', alpha=0.8)
    ax.plot(t_all, control_torques_all[:, 2],
            'b-', label='Torque Z (Yaw)', alpha=0.8)
    ax.set_xlabel('Time Since Mission Start (s)')
    ax.set_ylabel('Control Torque (N⋅m)')
    ax.set_title('Control Torques')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/control_signals_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission control signals plot saved to {output_dir}/control_signals_mission.png")
    plt.show()

    # Plot Output Motor
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig2.suptitle('Motor Outputs (Mission Phase)',
                  fontsize=16, fontweight='bold')

    ax = axes2[0]
    for i in range(6):
        ax.plot(t_all, w_cmd_all[:, i], label=f'Motor {i+1}', alpha=0.8)
    ax.set_ylabel('Motor Speed Command (rad/s)')
    ax.set_title('Motor Speed Commands')
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for i in range(6):
        ax.plot(t_all, thr_all[:, i], label=f'Motor {i+1}', alpha=0.8)
    ax.set_xlabel('Time Since Mission Start (s)')
    ax.set_ylabel('Motor Thrust (N)')
    ax.set_title('Individual Motor Thrusts')
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/motor_outputs_mission.png",
                    dpi=300, bbox_inches='tight')
        print(
            f"✓ Mission motor outputs plot saved to {output_dir}/motor_outputs_mission.png")
    plt.show()


def save_data_to_csv(filename, t_all, pos_all, vel_all, euler_all, quat_all,
                     w_cmd_all, thr_all, thrust_sp_all, control_torques_all,
                     acc_all, gyro_all, gps_pos_ned_all, gps_vel_ned_all,
                     baro_alt_all, mag_all, gps_available, baro_available, mag_available):
    """
    Menyimpan semua data simulasi ke file CSV dengan nama kolom yang 
    spesifik agar kompatibel dengan skrip EKF.
    """
    print(f"\n💾 Menyimpan data mentah simulasi ke file: {filename}...")

    # Buat dictionary dengan nama kolom yang EKSPLISIT sesuai kebutuhan EKF
    data_dict = {
        # Timestamp
        'timestamp': t_all,

        # Ground Truth State
        'true_pos_x': pos_all[:, 0],
        'true_pos_y': pos_all[:, 1],
        'true_pos_z': pos_all[:, 2],
        'true_vel_x': vel_all[:, 0],
        'true_vel_y': vel_all[:, 1],
        'true_vel_z': vel_all[:, 2],
        'true_roll': euler_all[:, 0],
        'true_pitch': euler_all[:, 1],
        'true_yaw': euler_all[:, 2],
        'true_quat_w': quat_all[:, 0],
        'true_quat_x': quat_all[:, 1],
        'true_quat_y': quat_all[:, 2],
        'true_quat_z': quat_all[:, 3],

        # Sensor Measurements
        'acc_x': acc_all[:, 0],
        'acc_y': acc_all[:, 1],
        'acc_z': acc_all[:, 2],
        'gyro_x': gyro_all[:, 0],
        'gyro_y': gyro_all[:, 1],
        'gyro_z': gyro_all[:, 2],
        'mag_x': mag_all[:, 0],
        'mag_y': mag_all[:, 1],
        'mag_z': mag_all[:, 2],
        'gps_pos_ned_x': gps_pos_ned_all[:, 0],
        'gps_pos_ned_y': gps_pos_ned_all[:, 1],
        'gps_pos_ned_z': gps_pos_ned_all[:, 2],
        'gps_vel_ned_x': gps_vel_ned_all[:, 0],
        'gps_vel_ned_y': gps_vel_ned_all[:, 1],
        'gps_vel_ned_z': gps_vel_ned_all[:, 2],
        'baro_altitude': baro_alt_all,

        # Sensor Availability Flags
        'gps_available': gps_available.astype(int),
        'baro_available': baro_available.astype(int),
        'mag_available': mag_available.astype(int),

        # Control Inputs
        'thrust_sp_x': thrust_sp_all[:, 0],
        'thrust_sp_y': thrust_sp_all[:, 1],
        'thrust_sp_z': thrust_sp_all[:, 2],
        'control_torque_x': control_torques_all[:, 0],
        'control_torque_y': control_torques_all[:, 1],
        'control_torque_z': control_torques_all[:, 2],
        # EKF menggunakan 'total_thrust_sp'
        'total_thrust_sp': np.sum(thr_all, axis=1),
        'motor_thrust_1': thr_all[:, 0],
        'motor_thrust_2': thr_all[:, 1],
        'motor_thrust_3': thr_all[:, 2],
        'motor_thrust_4': thr_all[:, 3],
        'motor_thrust_5': thr_all[:, 4],
        'motor_thrust_6': thr_all[:, 5],
    }

    try:
        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False, float_format='%.8f')
        print(f"✓ Data kompatibel EKF berhasil disimpan ke {filename}")
    except Exception as e:
        print(f"❌ Gagal menyimpan data ke CSV: {e}")


def main():
    start_time = time.time()

    # Pengaturan Simulasi
    Ti = 0
    Ts = 0.01
    hover_duration = 5.0
    hover_altitude = 1.0
    mission_duration = 80
    Tf = hover_duration + mission_duration

    # Pengaturan Trajectory
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.zeros(3)
    ctrlType = ctrlOptions[0]
    trajSelect[0] = 5
    trajSelect[1] = 3
    trajSelect[2] = 1

    print("=== MISSION-PHASE PLOTTING SIMULATION ===")
    print(f"Total simulation time: {Tf} seconds")

    # Inisialisasi
    quad = Quadcopter(Ti)
    if config.orient == "NED":
        quad.pos[2] = -hover_altitude
        quad.state[2] = -hover_altitude
    else:
        quad.pos[2] = hover_altitude
        quad.state[2] = hover_altitude
    original_traj = Trajectory(quad, ctrlType, trajSelect)
    traj = TrajectoryWithHovering(
        original_traj, hover_altitude, hover_duration)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    sDes = traj.desiredState(0, Ts, quad)
    ctrl.controller(traj, quad, sDes, Ts)

    # Inisialisasi Matriks Hasil
    numTimeStep = int(Tf/Ts+1)
    quat_all = np.zeros([numTimeStep, 4])
    t_all = np.zeros(numTimeStep)
    pos_all = np.zeros([numTimeStep, 3])
    vel_all = np.zeros([numTimeStep, 3])
    euler_all = np.zeros([numTimeStep, 3])
    w_cmd_all = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    thr_all = np.zeros([numTimeStep, len(quad.thr)])
    thrust_sp_all = np.zeros([numTimeStep, 3])
    rate_sp_all = np.zeros([numTimeStep, 3])
    control_torques_all = np.zeros([numTimeStep, 3])

    # Inisialisasi Sensor dan data array
    imu = IMUSensor()
    gps = GPSSensor()
    baro = AltitudeSensor("baro")
    mag = MagnetometerSensor()

    acc_all = np.zeros([numTimeStep, 3])
    gyro_all = np.zeros([numTimeStep, 3])
    gps_pos_ned_all = np.zeros([numTimeStep, 3])
    gps_vel_ned_all = np.zeros([numTimeStep, 3])
    gps_lat_all = np.zeros(numTimeStep)
    gps_lon_all = np.zeros(numTimeStep)
    baro_alt_all = np.zeros(numTimeStep)
    mag_all = np.zeros([numTimeStep, 3])
    gps_available = np.zeros(numTimeStep, dtype=bool)
    baro_available = np.zeros(numTimeStep, dtype=bool)
    mag_available = np.zeros(numTimeStep, dtype=bool)

    # Loop Simulasi
    t = Ti
    i = 0
    while round(t, 3) < Tf:
        t, acc_m, gyro_m, pos_ned_m, vel_ned_m, geodetic_m, alt_m, mag_m = quad_sim(
            t, Ts, quad, ctrl, wind, traj, imu, gps, baro, mag)

        # Simpan data
        if i < numTimeStep:
            t_all[i] = t
            quat_all[i, :] = quad.quat
            pos_all[i, :] = quad.pos
            vel_all[i, :] = quad.vel
            euler_all[i, :] = quad.euler
            w_cmd_all[i, :] = ctrl.w_cmd
            thr_all[i, :] = quad.thr
            thrust_sp_all[i, :] = ctrl.thrust_sp
            rate_sp_all[i, :] = getattr(ctrl, 'rate_sp', [0, 0, 0])
            F_total, Mx, My, Mz = quad.motor_speeds_to_forces_moments(
                quad.wMotor)
            control_torques_all[i, :] = [Mx, My, Mz]

            acc_all[i] = acc_m
            gyro_all[i] = gyro_m
            if pos_ned_m is not None:
                gps_pos_ned_all[i] = pos_ned_m
                gps_vel_ned_all[i] = vel_ned_m
                gps_lat_all[i] = geodetic_m['latitude']
                gps_lon_all[i] = geodetic_m['longitude']
                gps_available[i] = True
            if alt_m is not None:
                baro_alt_all[i] = alt_m
                baro_available[i] = True
            if mag_m is not None:
                mag_all[i] = mag_m
                mag_available[i] = True
        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # Pangkas array ke panjang data aktual
    actual_length = i
    quat_all = quat_all[:actual_length]
    t_all = t_all[:actual_length]
    pos_all = pos_all[:actual_length]
    vel_all = vel_all[:actual_length]
    euler_all = euler_all[:actual_length]
    w_cmd_all = w_cmd_all[:actual_length]
    thr_all = thr_all[:actual_length]
    thrust_sp_all = thrust_sp_all[:actual_length]
    rate_sp_all = rate_sp_all[:actual_length]
    control_torques_all = control_torques_all[:actual_length]
    acc_all = acc_all[:actual_length]
    gyro_all = gyro_all[:actual_length]
    gps_pos_ned_all = gps_pos_ned_all[:actual_length]
    gps_vel_ned_all = gps_vel_ned_all[:actual_length]
    gps_lat_all = gps_lat_all[:actual_length]
    gps_lon_all = gps_lon_all[:actual_length]
    gps_available = gps_available[:actual_length]
    baro_alt_all = baro_alt_all[:actual_length]
    baro_available = baro_available[:actual_length]
    mag_all = mag_all[:actual_length]
    mag_available = mag_available[:actual_length]

    output_dir_logs = "logs"
    os.makedirs(output_dir_logs, exist_ok=True)
    log_filename = f"{output_dir_logs}/helix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    save_data_to_csv(
        log_filename,
        t_all, pos_all, vel_all, euler_all, quat_all, w_cmd_all, thr_all,
        thrust_sp_all, control_torques_all, acc_all, gyro_all,
        gps_pos_ned_all, gps_vel_ned_all, baro_alt_all, mag_all,
        gps_available, baro_available, mag_available
    )

    # ========================================================================
    # ===== BARU: Filter data untuk fase misi (t >= 5s) dan panggil plot =====
    # ========================================================================
    print(f"\nFiltering data for mission phase (t >= {hover_duration}s)...")
    mission_start_index_arr = np.where(t_all >= hover_duration)[0]
    if len(mission_start_index_arr) > 0:
        mission_start_index = mission_start_index_arr[0]
    else:
        print("Warning: No mission phase data found. Plots may be empty.")
        mission_start_index = len(t_all)

    # Buat array waktu relatif untuk plot, dimulai dari 0
    t_mission = t_all[mission_start_index:] - hover_duration

    # Potong semua array data untuk hanya menyertakan fase misi
    pos_mission = pos_all[mission_start_index:]
    vel_mission = vel_all[mission_start_index:]
    euler_mission = euler_all[mission_start_index:]
    acc_mission = acc_all[mission_start_index:]
    gyro_mission = gyro_all[mission_start_index:]
    gps_pos_ned_mission = gps_pos_ned_all[mission_start_index:]
    gps_vel_ned_mission = gps_vel_ned_all[mission_start_index:]
    gps_available_mission = gps_available[mission_start_index:]
    baro_alt_mission = baro_alt_all[mission_start_index:]
    baro_available_mission = baro_available[mission_start_index:]
    mag_mission = mag_all[mission_start_index:]
    mag_available_mission = mag_available[mission_start_index:]
    gps_lat_mission = gps_lat_all[mission_start_index:]
    gps_lon_mission = gps_lon_all[mission_start_index:]
    thrust_sp_mission = thrust_sp_all[mission_start_index:]
    rate_sp_mission = rate_sp_all[mission_start_index:]
    w_cmd_mission = w_cmd_all[mission_start_index:]
    thr_mission = thr_all[mission_start_index:]
    control_torques_mission = control_torques_all[mission_start_index:]

    print(f"Plotting {len(t_mission)} data points from the mission phase.")

    # ===== HASILKAN SEMUA PLOT (DIMODIFIKASI UNTUK FASE MISI) =====
    print(
        f"\n🎨 Generating comprehensive plots for the mission phase (t >= {hover_duration}s)...")

    output_dir = "simulation_plots_mission_phase"

    # Plot 1: 3D Trajectory (Plot Terpisah)
    plot_3d_trajectory_mission(
        pos_mission, save_plots=True, output_dir=output_dir)

    # Plot 2: Tampilan Trajectory 2D (Plot Terpisah)
    plot_trajectory_views_mission(
        t_mission, pos_mission, save_plots=True, output_dir=output_dir)

    # Plot 3: Data Sensor
    plot_sensor_data_mission(t_mission, acc_mission, gyro_mission, gps_pos_ned_mission, gps_vel_ned_mission,
                             gps_available_mission, baro_alt_mission, baro_available_mission, mag_mission, mag_available_mission,
                             save_plots=True, output_dir=output_dir)

    # Plot 4: Perbandingan Ground Truth
    plot_ground_truth_comparison_mission(t_mission, pos_mission, vel_mission, euler_mission,
                                         gps_pos_ned_mission, gps_vel_ned_mission, gps_available_mission,
                                         save_plots=True, output_dir=output_dir)

    # Plot 5: Peta Geodetik GPS
    plot_gps_geodetic_map_mission(gps_lat_mission, gps_lon_mission, gps_available_mission,
                                  gps, save_plots=True, output_dir=output_dir)

    # Plot 6: Sinyal Kontrol
    plot_control_signals_mission(t_mission, thrust_sp_mission, rate_sp_mission, w_cmd_mission, thr_mission,
                                 control_torques_mission, save_plots=True, output_dir=output_dir)


if __name__ == "__main__":
    # Pastikan file-file seperti trajectory.py, ctrl.py, dll. dapat diakses
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
    else:
        raise Exception(
            "{} is not a valid orientation. Verify config.py file.".format(config.orient))
