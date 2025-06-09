# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!

Enhanced version with full GPS geodetic integration
Updated to work with fixed hexacopter mixer matrix
Added 3-second hovering phase at 1m altitude for EKF initialization
FIXED: GPS lat/long logging starts from t=0.001s (not 0.002s)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import pandas as pd
import os
from datetime import datetime

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

            # Position setpoints (hover at specified altitude)
            sDes[0] = 0.0  # x
            sDes[1] = 0.0  # y
            if config.orient == "NED":
                sDes[2] = -self.hover_altitude  # z (negative for NED)
            else:
                sDes[2] = self.hover_altitude   # z (positive for ENU)

            # Velocity setpoints (zero for hovering)
            sDes[3] = 0.0  # vx
            sDes[4] = 0.0  # vy
            sDes[5] = 0.0  # vz

            # Acceleration setpoints (zero for hovering)
            sDes[6] = 0.0  # ax
            sDes[7] = 0.0  # ay
            sDes[8] = 0.0  # az

            # Thrust setpoints (hover thrust)
            sDes[9] = 0.0   # thrust_x
            sDes[10] = 0.0  # thrust_y
            if config.orient == "NED":
                sDes[11] = quad.params["mB"] * \
                    quad.params["g"]  # thrust_z (up in NED)
            else:
                sDes[11] = -quad.params["mB"] * \
                    quad.params["g"]  # thrust_z (up in ENU)

            # Attitude setpoints (level hover)
            sDes[12] = 0.0  # roll
            sDes[13] = 0.0  # pitch
            sDes[14] = 0.0  # yaw

            # Rate setpoints (zero for stable hover)
            sDes[15] = 0.0  # p
            sDes[16] = 0.0  # q
            sDes[17] = 0.0  # r

            # Yaw feedforward
            sDes[18] = 0.0

            self.sDes = sDes
            return sDes

        else:
            # MISSION PHASE: Use original trajectory but offset time
            mission_time = t - self.hover_duration
            sDes = self.original_traj.desiredState(mission_time, Ts, quad)

            # Offset the position to start from hover altitude
            if hasattr(self, '_position_offset_applied') and not self._position_offset_applied:
                self._initial_mission_pos = sDes[0:3].copy()
                self._position_offset_applied = True

            if hasattr(self, '_initial_mission_pos'):
                # Apply position offset to start mission from hover position
                sDes[0] += 0.0  # x offset (start from origin)
                sDes[1] += 0.0  # y offset (start from origin)
                if config.orient == "NED":
                    sDes[2] = sDes[2] - self._initial_mission_pos[2] - \
                        self.hover_altitude
                else:
                    sDes[2] = sDes[2] - self._initial_mission_pos[2] + \
                        self.hover_altitude

            self.sDes = sDes
            return sDes


def main():
    start_time = time.time()

    # Simulation Setup with Hovering Phase
    # ---------------------------
    Ti = 0
    Ts = 0.001
    hover_duration = 5.0  # 3 seconds hovering for EKF initialization
    hover_altitude = 1.0  # 1 meter hovering altitude
    mission_duration = 41  # Original mission duration
    Tf = hover_duration + mission_duration  # Total simulation time
    ifsave = 1

    # Choose trajectory settings
    # ---------------------------
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.zeros(3)

    # Select Control Type             (0: xyz_pos,                  1: xy_vel_z_pos,            2: xyz_vel)
    ctrlType = ctrlOptions[0]
    # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,
    #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
    #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
    #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
    #                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
    trajSelect[0] = 5
    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    trajSelect[1] = 3
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 1

    print("=== ENHANCED SIMULATION WITH HOVERING INITIALIZATION ===")
    print(f"Control type: {ctrlType}")
    print(
        f"Hovering phase: {hover_duration} seconds at {hover_altitude}m altitude")
    print(f"Mission phase: {mission_duration} seconds")
    print(f"Total simulation time: {Tf} seconds")
    print(f"Logging starts at: {Ts} seconds (including t={Ts})")

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    quad = Quadcopter(Ti)

    # Set initial position to hover altitude
    if config.orient == "NED":
        quad.pos[2] = -hover_altitude  # Negative for NED
        quad.state[2] = -hover_altitude
    else:
        quad.pos[2] = hover_altitude   # Positive for ENU
        quad.state[2] = hover_altitude

    # Create original trajectory
    original_traj = Trajectory(quad, ctrlType, trajSelect)

    # Create enhanced trajectory with hovering
    traj = TrajectoryWithHovering(
        original_traj, hover_altitude, hover_duration)

    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States (hovering)
    # ---------------------------
    sDes = traj.desiredState(0, Ts, quad)

    # Generate First Commands
    # ---------------------------
    ctrl.controller(traj, quad, sDes, Ts)

    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(Tf/Ts+1)

    t_all = np.zeros(numTimeStep)
    s_all = np.zeros([numTimeStep, len(quad.state)])
    pos_all = np.zeros([numTimeStep, len(quad.pos)])
    vel_all = np.zeros([numTimeStep, len(quad.vel)])
    quat_all = np.zeros([numTimeStep, len(quad.quat)])
    omega_all = np.zeros([numTimeStep, len(quad.omega)])
    euler_all = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all = np.zeros([numTimeStep, len(quad.wMotor)])
    thr_all = np.zeros([numTimeStep, len(quad.thr)])
    tor_all = np.zeros([numTimeStep, len(quad.tor)])

    # ======= Control Data Arrays =======
    thrust_sp_all = np.zeros([numTimeStep, 3])
    rate_sp_all = np.zeros([numTimeStep, 3])
    rate_ctrl_all = np.zeros([numTimeStep, 3])
    qd_all = np.zeros([numTimeStep, 4])
    pos_sp_all = np.zeros([numTimeStep, 3])
    vel_sp_all = np.zeros([numTimeStep, 3])
    acc_sp_all = np.zeros([numTimeStep, 3])

    total_thrust_all = np.zeros(numTimeStep)
    control_torques_all = np.zeros([numTimeStep, 3])

    acc_body_all = np.zeros([numTimeStep, 3])
    omega_dot_all = np.zeros([numTimeStep, 3])

    # ===== ENHANCED: Initialize Sensors with Reference Position =====
    # GPS sensor with Surabaya reference position (already set in GPS class)
    print("Initializing sensors...")
    imu = IMUSensor()
    gps = GPSSensor()
    baro = AltitudeSensor("baro")
    lidar = AltitudeSensor("lidar")
    mag = MagnetometerSensor()

    # Print GPS reference info for verification
    ref_pos = gps.get_reference_position()
    print(
        f"GPS Reference Position: {ref_pos['latitude']:.6f}°, {ref_pos['longitude']:.6f}°, {ref_pos['altitude']:.1f}m")

    # ===== ENHANCED: Initialize sensor data arrays =====
    # IMU data
    acc_all = np.zeros([numTimeStep, 3])
    gyro_all = np.zeros([numTimeStep, 3])

    # GPS NED data (what EKF uses)
    gps_pos_ned_all = np.zeros([numTimeStep, 3])
    gps_vel_ned_all = np.zeros([numTimeStep, 3])

    # GPS Geodetic data (raw GPS output)
    gps_lat_all = np.zeros(numTimeStep)      # degrees
    gps_lon_all = np.zeros(numTimeStep)      # degrees
    gps_alt_all = np.zeros(numTimeStep)      # meters

    # Other sensors
    baro_alt_all = np.zeros(numTimeStep)
    mag_all = np.zeros([numTimeStep, 3])

    # Availability flags
    gps_available = np.zeros(numTimeStep, dtype=bool)
    baro_available = np.zeros(numTimeStep, dtype=bool)
    mag_available = np.zeros(numTimeStep, dtype=bool)

    # Phase tracking for analysis
    flight_phase = np.zeros(numTimeStep, dtype=int)  # 0=hovering, 1=mission

    # ===== FIXED: Start logging from t=Ts and include first measurement =====
    print(f"Starting simulation and logging from t={Ts} seconds...")
    t = Ts  # Start time at first timestep
    i = 0   # Array index starts from 0

    # ===== FIXED: Take initial sensor readings at t=Ts BEFORE entering loop =====
    print(f"Taking initial sensor readings at t={t:.3f}s...")

    # Take sensor measurements at the current state (t=Ts)
    initial_acc, initial_gyro = imu.measure(quad, t)
    initial_pos_ned, initial_vel_ned, initial_geodetic = gps.measure(quad, t)
    initial_alt = baro.measure(quad, t)
    initial_mag = mag.measure(quad, t)

    # Store initial data (at t=Ts=0.001s)
    t_all[i] = t
    s_all[i, :] = quad.state
    pos_all[i, :] = quad.pos
    vel_all[i, :] = quad.vel
    quat_all[i, :] = quad.quat
    omega_all[i, :] = quad.omega
    euler_all[i, :] = quad.euler
    sDes_traj_all[i, :] = traj.sDes
    sDes_calc_all[i, :] = ctrl.sDesCalc
    w_cmd_all[i, :] = ctrl.w_cmd
    wMotor_all[i, :] = quad.wMotor
    thr_all[i, :] = quad.thr
    tor_all[i, :] = quad.tor

    # Store Control Data
    thrust_sp_all[i, :] = ctrl.thrust_sp
    rate_sp_all[i, :] = ctrl.rate_sp if hasattr(ctrl, 'rate_sp') else [0, 0, 0]
    rate_ctrl_all[i, :] = ctrl.rateCtrl if hasattr(
        ctrl, 'rateCtrl') else [0, 0, 0]
    qd_all[i, :] = ctrl.qd if hasattr(ctrl, 'qd') else [1, 0, 0, 0]
    pos_sp_all[i, :] = ctrl.pos_sp
    vel_sp_all[i, :] = ctrl.vel_sp
    acc_sp_all[i, :] = ctrl.acc_sp

    # Calculate control allocation data
    total_thrust_all[i] = np.linalg.norm(ctrl.thrust_sp)

    # Calculate control torques using mixer matrix
    F_total, Mx, My, Mz = quad.motor_speeds_to_forces_moments(quad.wMotor)
    control_torques_all[i, 0] = Mx  # Roll moment
    control_torques_all[i, 1] = My  # Pitch moment
    control_torques_all[i, 2] = Mz  # Yaw moment

    acc_body_all[i, :] = quad.dcm.T @ quad.acc
    omega_dot_all[i, :] = quad.omega_dot

    # ===== FIXED: Store initial sensor data (at t=0.001s) =====
    # IMU Data
    acc_all[i] = initial_acc
    gyro_all[i] = initial_gyro

    # GPS Data (both NED and Geodetic) - CRITICAL: This is now logged at t=0.001s
    if initial_pos_ned is not None and initial_vel_ned is not None and initial_geodetic is not None:
        gps_pos_ned_all[i] = initial_pos_ned
        gps_vel_ned_all[i] = initial_vel_ned
        # degrees - NOW LOGGED!
        gps_lat_all[i] = initial_geodetic['latitude']
        # degrees - NOW LOGGED!
        gps_lon_all[i] = initial_geodetic['longitude']
        gps_alt_all[i] = initial_geodetic['altitude']    # meters - NOW LOGGED!
        gps_available[i] = True
        print(
            f"✓ Initial GPS data logged: lat={initial_geodetic['latitude']:.6f}°, lon={initial_geodetic['longitude']:.6f}°")

    # Other sensor data
    if initial_alt is not None:
        baro_alt_all[i] = initial_alt
        baro_available[i] = True
    if initial_mag is not None:
        mag_all[i] = initial_mag
        mag_available[i] = True

    # Track flight phase for analysis
    if t <= hover_duration:
        flight_phase[i] = 0  # Hovering phase
    else:
        flight_phase[i] = 1  # Mission phase

    print(f"✓ Initial data logged at t={t:.3f}s (index {i})")

    # Move to next array index
    i += 1

    # ===== SIMULATION LOOP =====
    while round(t, 3) < Tf:
        # ===== Run simulation step =====
        t, acc_m, gyro_m, pos_ned_m, vel_ned_m, geodetic_m, alt_m, mag_m = quad_sim(
            t, Ts, quad, ctrl, wind, traj, imu, gps, baro, mag)

        # Store all data
        t_all[i] = t
        s_all[i, :] = quad.state
        pos_all[i, :] = quad.pos
        vel_all[i, :] = quad.vel
        quat_all[i, :] = quad.quat
        omega_all[i, :] = quad.omega
        euler_all[i, :] = quad.euler
        sDes_traj_all[i, :] = traj.sDes
        sDes_calc_all[i, :] = ctrl.sDesCalc
        w_cmd_all[i, :] = ctrl.w_cmd
        wMotor_all[i, :] = quad.wMotor
        thr_all[i, :] = quad.thr
        tor_all[i, :] = quad.tor

        # Store Control Data
        thrust_sp_all[i, :] = ctrl.thrust_sp
        rate_sp_all[i, :] = ctrl.rate_sp if hasattr(
            ctrl, 'rate_sp') else [0, 0, 0]
        rate_ctrl_all[i, :] = ctrl.rateCtrl if hasattr(
            ctrl, 'rateCtrl') else [0, 0, 0]
        qd_all[i, :] = ctrl.qd if hasattr(ctrl, 'qd') else [1, 0, 0, 0]
        pos_sp_all[i, :] = ctrl.pos_sp
        vel_sp_all[i, :] = ctrl.vel_sp
        acc_sp_all[i, :] = ctrl.acc_sp

        # Calculate control allocation data
        total_thrust_all[i] = np.linalg.norm(ctrl.thrust_sp)

        # Calculate control torques using mixer matrix
        F_total, Mx, My, Mz = quad.motor_speeds_to_forces_moments(quad.wMotor)
        control_torques_all[i, 0] = Mx  # Roll moment
        control_torques_all[i, 1] = My  # Pitch moment
        control_torques_all[i, 2] = Mz  # Yaw moment

        acc_body_all[i, :] = quad.dcm.T @ quad.acc
        omega_dot_all[i, :] = quad.omega_dot

        # ===== Store Sensor Data =====
        # IMU Data
        acc_all[i] = acc_m
        gyro_all[i] = gyro_m

        # GPS Data (both NED and Geodetic)
        if pos_ned_m is not None and vel_ned_m is not None and geodetic_m is not None:
            gps_pos_ned_all[i] = pos_ned_m
            gps_vel_ned_all[i] = vel_ned_m
            gps_lat_all[i] = geodetic_m['latitude']    # degrees
            gps_lon_all[i] = geodetic_m['longitude']   # degrees
            gps_alt_all[i] = geodetic_m['altitude']    # meters
            gps_available[i] = True

        # Other sensor data
        if alt_m is not None:
            baro_alt_all[i] = alt_m
            baro_available[i] = True
        if mag_m is not None:
            mag_all[i] = mag_m
            mag_available[i] = True

        # Track flight phase for analysis
        if t <= hover_duration:
            flight_phase[i] = 0  # Hovering phase
        else:
            flight_phase[i] = 1  # Mission phase

        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # Trim arrays to actual data length
    actual_length = i
    t_all = t_all[:actual_length]
    pos_all = pos_all[:actual_length]
    vel_all = vel_all[:actual_length]
    quat_all = quat_all[:actual_length]
    omega_all = omega_all[:actual_length]
    euler_all = euler_all[:actual_length]
    acc_all = acc_all[:actual_length]
    gyro_all = gyro_all[:actual_length]
    gps_pos_ned_all = gps_pos_ned_all[:actual_length]
    gps_vel_ned_all = gps_vel_ned_all[:actual_length]
    gps_lat_all = gps_lat_all[:actual_length]
    gps_lon_all = gps_lon_all[:actual_length]
    gps_alt_all = gps_alt_all[:actual_length]
    gps_available = gps_available[:actual_length]
    baro_alt_all = baro_alt_all[:actual_length]
    baro_available = baro_available[:actual_length]
    mag_all = mag_all[:actual_length]
    mag_available = mag_available[:actual_length]
    control_torques_all = control_torques_all[:actual_length]
    thrust_sp_all = thrust_sp_all[:actual_length]
    rate_sp_all = rate_sp_all[:actual_length]
    rate_ctrl_all = rate_ctrl_all[:actual_length]
    w_cmd_all = w_cmd_all[:actual_length]
    thr_all = thr_all[:actual_length]
    flight_phase = flight_phase[:actual_length]

    print(f"\n=== FLIGHT PHASE ANALYSIS ===")
    hovering_indices = flight_phase == 0
    mission_indices = flight_phase == 1

    print(
        f"Hovering phase: {np.sum(hovering_indices)} data points ({np.sum(hovering_indices)*Ts:.3f} seconds)")
    print(
        f"Mission phase: {np.sum(mission_indices)} data points ({np.sum(mission_indices)*Ts:.3f} seconds)")

    if np.any(hovering_indices):
        hover_pos_std = np.std(pos_all[hovering_indices], axis=0)
        print(
            f"Hovering position stability (std): X={hover_pos_std[0]:.4f}m, Y={hover_pos_std[1]:.4f}m, Z={hover_pos_std[2]:.4f}m")

    # ===== ENHANCED: Verify GPS data logging =====
    print(f"\n=== GPS DATA VERIFICATION ===")
    first_gps_idx = np.where(gps_available)[0]
    if len(first_gps_idx) > 0:
        first_idx = first_gps_idx[0]
        print(
            f"✓ First GPS data logged at t={t_all[first_idx]:.3f}s (index {first_idx})")
        print(f"  Latitude: {gps_lat_all[first_idx]:.6f}°")
        print(f"  Longitude: {gps_lon_all[first_idx]:.6f}°")
        print(f"  Altitude: {gps_alt_all[first_idx]:.1f}m")

        if t_all[first_idx] == Ts:
            print("✓ GPS lat/long successfully logged from t=0.001s!")
        else:
            print(
                f"⚠ GPS logging starts at t={t_all[first_idx]:.3f}s instead of t={Ts:.3f}s")
    else:
        print("✗ No GPS data logged!")

    # ===== ENHANCED: Save Complete Data with Geodetic GPS =====
    complete_data = pd.DataFrame({
        'timestamp': t_all,
        'flight_phase': flight_phase,  # 0=hovering, 1=mission

        # ===== IMU Data =====
        'acc_x': acc_all[:, 0],
        'acc_y': acc_all[:, 1],
        'acc_z': acc_all[:, 2],
        'gyro_x': gyro_all[:, 0],
        'gyro_y': gyro_all[:, 1],
        'gyro_z': gyro_all[:, 2],

        # ===== GPS NED Data (for EKF) =====
        'gps_pos_ned_x': gps_pos_ned_all[:, 0],
        'gps_pos_ned_y': gps_pos_ned_all[:, 1],
        'gps_pos_ned_z': gps_pos_ned_all[:, 2],
        'gps_vel_ned_x': gps_vel_ned_all[:, 0],
        'gps_vel_ned_y': gps_vel_ned_all[:, 1],
        'gps_vel_ned_z': gps_vel_ned_all[:, 2],

        # ===== GPS Geodetic Data (raw GPS output) =====
        'gps_latitude': gps_lat_all,        # degrees - NOW INCLUDES t=0.001s!
        'gps_longitude': gps_lon_all,       # degrees - NOW INCLUDES t=0.001s!
        'gps_altitude': gps_alt_all,        # meters
        'gps_available': gps_available,

        # ===== Barometer Data =====
        'baro_altitude': baro_alt_all,
        'baro_available': baro_available,

        # ===== Magnetometer Data =====
        'mag_x': mag_all[:, 0],
        'mag_y': mag_all[:, 1],
        'mag_z': mag_all[:, 2],
        'mag_available': mag_available,

        # ===== Motor Commands & Control Signals =====
        'motor_cmd_1': w_cmd_all[:, 0],
        'motor_cmd_2': w_cmd_all[:, 1],
        'motor_cmd_3': w_cmd_all[:, 2],
        'motor_cmd_4': w_cmd_all[:, 3],
        'motor_cmd_5': w_cmd_all[:, 4],
        'motor_cmd_6': w_cmd_all[:, 5],

        'motor_thrust_1': thr_all[:, 0],
        'motor_thrust_2': thr_all[:, 1],
        'motor_thrust_3': thr_all[:, 2],
        'motor_thrust_4': thr_all[:, 3],
        'motor_thrust_5': thr_all[:, 4],
        'motor_thrust_6': thr_all[:, 5],

        'thrust_sp_x': thrust_sp_all[:, 0],
        'thrust_sp_y': thrust_sp_all[:, 1],
        'thrust_sp_z': thrust_sp_all[:, 2],

        'rate_sp_x': rate_sp_all[:, 0],
        'rate_sp_y': rate_sp_all[:, 1],
        'rate_sp_z': rate_sp_all[:, 2],

        'rate_ctrl_x': rate_ctrl_all[:, 0],
        'rate_ctrl_y': rate_ctrl_all[:, 1],
        'rate_ctrl_z': rate_ctrl_all[:, 2],

        'control_torque_x': control_torques_all[:, 0],
        'control_torque_y': control_torques_all[:, 1],
        'control_torque_z': control_torques_all[:, 2],

        # ===== Ground Truth Data =====
        'true_pos_x': pos_all[:, 0],
        'true_pos_y': pos_all[:, 1],
        'true_pos_z': pos_all[:, 2],
        'true_vel_x': vel_all[:, 0],
        'true_vel_y': vel_all[:, 1],
        'true_vel_z': vel_all[:, 2],
        'true_roll': euler_all[:, 0],
        'true_pitch': euler_all[:, 1],
        'true_yaw': euler_all[:, 2],
        'true_omega_x': omega_all[:, 0],
        'true_omega_y': omega_all[:, 1],
        'true_omega_z': omega_all[:, 2],
        'true_quat_w': quat_all[:, 0],
        'true_quat_x': quat_all[:, 1],
        'true_quat_y': quat_all[:, 2],
        'true_quat_z': quat_all[:, 3],
    })

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created directory: {log_dir}")

    # Generate timestamp string for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define file paths with timestamp
    complete_data_file = os.path.join(
        log_dir, f"hexacopter_ekf_data_with_hovering_fixed_{timestamp}.csv")

    # Save complete dataframe to CSV
    complete_data.to_csv(complete_data_file, index=False)

    print("Complete flight data with fixed GPS logging saved successfully:")
    print(f"- {complete_data_file}")

    # Enhanced metadata with hovering info
    metadata_file = os.path.join(
        log_dir, f"simulation_metadata_hovering_fixed_{timestamp}.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(
            f"=== HEXACOPTER SIMULATION WITH HOVERING INITIALIZATION (FIXED GPS LOGGING) ===\n")
        f.write(f"Simulation timestamp: {timestamp}\n")
        f.write(f"Simulation total duration: {Tf} seconds\n")
        f.write(f"Timestep: {Ts} seconds\n")
        f.write(f"Logging start time: {Ts} seconds (INCLUDES t={Ts}s data)\n")
        f.write(
            f"GPS lat/long logging: FIXED - now includes t={Ts}s measurement\n")
        f.write(f"Control type: {ctrlType}\n")
        f.write(f"Trajectory type: {int(trajSelect[0])}\n")
        f.write(f"Yaw trajectory type: {int(trajSelect[1])}\n")
        f.write(f"Waypoint timing mode: {int(trajSelect[2])}\n")
        f.write(f"Total simulation points: {actual_length}\n")
        f.write(f"\n=== GPS LOGGING VERIFICATION ===\n")
        if len(first_gps_idx) > 0:
            f.write(
                f"First GPS data logged at: t={t_all[first_gps_idx[0]]:.3f}s\n")
            f.write(
                f"GPS data properly logged from start: {'YES' if t_all[first_gps_idx[0]] == Ts else 'NO'}\n")
        f.write(f"\n=== HOVERING INITIALIZATION PHASE ===\n")
        f.write(f"Hovering duration: {hover_duration} seconds\n")
        f.write(f"Hovering altitude: {hover_altitude} meters\n")
        f.write(f"Hovering data points: {np.sum(hovering_indices)}\n")
        f.write(f"Purpose: EKF initialization with stable GPS/IMU data\n")
        if np.any(hovering_indices):
            f.write(
                f"Hovering stability (position std): X={hover_pos_std[0]:.4f}m, Y={hover_pos_std[1]:.4f}m, Z={hover_pos_std[2]:.4f}m\n")
        f.write(f"\n=== MISSION PHASE ===\n")
        f.write(f"Mission duration: {mission_duration} seconds\n")
        f.write(f"Mission data points: {np.sum(mission_indices)}\n")
        f.write(f"Mission starts at: t={hover_duration}s\n")
        f.write(f"\n=== GPS CONFIGURATION ===\n")
        f.write(f"Reference Latitude: {ref_pos['latitude']:.6f} degrees\n")
        f.write(f"Reference Longitude: {ref_pos['longitude']:.6f} degrees\n")
        f.write(f"Reference Altitude: {ref_pos['altitude']:.1f}m\n")
        f.write(f"GPS Position Noise: {gps.pos_noise_std:.2f}m\n")
        f.write(f"GPS Velocity Noise: {gps.vel_noise_std:.2f}m/s\n")
        f.write(f"GPS Update Rate: {gps.freq} Hz\n")
        f.write(f"\n=== SENSOR FREQUENCIES ===\n")
        f.write(f"IMU: {imu.freq} Hz\n")
        f.write(f"GPS: {gps.freq} Hz\n")
        f.write(f"Barometer: {baro.freq} Hz\n")
        f.write(f"Magnetometer: {mag.freq} Hz\n")
        f.write(f"\n=== VEHICLE PARAMETERS ===\n")
        f.write(f"Mass: {quad.params['mB']} kg\n")
        f.write(f"Inertia: {quad.params['IB']} kg*m^2\n")
        f.write(f"Arm length: {quad.params.get('L', 'N/A')} m\n")
        f.write(f"Thrust coefficient: {quad.params['kTh']} N/(rad/s)^2\n")
        f.write(f"Torque coefficient: {quad.params['kTo']} N*m/(rad/s)^2\n")
        f.write(f"\n=== MIXER MATRIX CONFIGURATION ===\n")
        f.write(f"Motor Configuration: Fixed Hexacopter X\n")
        f.write(f"Motor 1: kanan CW\n")
        f.write(f"Motor 2: kiri CCW\n")
        f.write(f"Motor 3: kiri atas CW\n")
        f.write(f"Motor 4: kanan bawah CCW\n")
        f.write(f"Motor 5: kanan atas CCW\n")
        f.write(f"Motor 6: kiri bawah CW\n")
        f.write(f"Mixer Matrix: Matrix-based calculation (consistent)\n")
        f.write(f"\n=== EKF INITIALIZATION DATA ===\n")
        f.write(f"- Hovering phase provides stable reference for EKF initialization\n")
        f.write(f"- GPS coordinates and IMU bias estimation during hover\n")
        f.write(
            f"- All sensor data logged continuously from t={Ts}s (FIXED)\n")
        f.write(f"- Flight phase column: 0=hovering, 1=mission\n")
        f.write(f"\n=== DATA STATISTICS ===\n")
        f.write(
            f"GPS update rate: {np.sum(gps_available)/len(gps_available)*100:.1f}%\n")
        f.write(
            f"Barometer update rate: {np.sum(baro_available)/len(baro_available)*100:.1f}%\n")
        f.write(
            f"Magnetometer update rate: {np.sum(mag_available)/len(mag_available)*100:.1f}%\n")

        # GPS statistics
        gps_valid_indices = gps_available & (
            gps_lat_all != 0) & (gps_lon_all != 0)
        if np.any(gps_valid_indices):
            f.write(
                f"GPS latitude range: {np.min(gps_lat_all[gps_valid_indices]):.6f} to {np.max(gps_lat_all[gps_valid_indices]):.6f} degrees\n")
            f.write(
                f"GPS longitude range: {np.min(gps_lon_all[gps_valid_indices]):.6f} to {np.max(gps_lon_all[gps_valid_indices]):.6f} degrees\n")
            f.write(
                f"GPS altitude range: {np.min(gps_alt_all[gps_valid_indices]):.1f}m to {np.max(gps_alt_all[gps_valid_indices]):.1f}m\n")
        else:
            f.write(f"GPS data: No valid GPS measurements recorded\n")

    print(f"- {metadata_file}")
    print(f"\nEKF-ready data with FIXED GPS logging saved!")
    print(f"Total data points: {len(complete_data)}")
    print(f"GPS measurements: {np.sum(gps_available)} points")
    print(
        f"Hovering phase: {np.sum(hovering_indices)} points for EKF initialization")

    # Verification: Print mixer matrix info
    print(f"\n=== MIXER MATRIX VERIFICATION ===")
    print(f"Mixer matrix shape: {quad.params['mixerFM'].shape}")
    print(f"Using consistent matrix-based torque calculation")

    # Test mixer consistency
    print(f"Testing mixer consistency...")
    test_motor_speeds = np.array(
        [100, 100, 100, 100, 100, 100])  # Equal speeds
    F_test, Mx_test, My_test, Mz_test = quad.motor_speeds_to_forces_moments(
        test_motor_speeds)
    print(
        f"Equal motor speeds test: F={F_test:.3f}N, Mx={Mx_test:.6f}N⋅m, My={My_test:.6f}N⋅m, Mz={Mz_test:.6f}N⋅m")
    print(f"Expected: F>0, Mx≈0, My≈0, Mz≈0 (balanced)")

    print(f"\n=== SIMULATION COMPLETE ===")
    print(f"Total simulation time: {Tf} seconds")
    print(f"Data points: {len(t_all)}")
    print(f"Files saved to: {log_dir}/")
    print(
        f"✓ GPS lat/long data now logged from t={Ts}s onwards!")


if __name__ == "__main__":
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception(
            "{} is not a valid orientation. Verify config.py file.".format(config.orient))
