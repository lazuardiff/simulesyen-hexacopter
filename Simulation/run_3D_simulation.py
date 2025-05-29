# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
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
    pos_m, vel_m = gps.measure(quad, t)
    alt_m = alti.measure(quad, t)
    mag_m = mag.measure(quad, t)

    # Trajectory for Desired States
    # ---------------------------
    sDes = traj.desiredState(t, Ts, quad)

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj, quad, sDes, Ts)

    return t, acc_m, gyro_m, pos_m, vel_m, alt_m, mag_m


def main():
    start_time = time.time()

    # Simulation Setup
    # ---------------------------
    Ti = 0
    Ts = 0.001
    Tf = 41
    ifsave = 0

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
    trajSelect[1] = 0
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 1
    print("Control type: {}".format(ctrlType))

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    quad = Quadcopter(Ti)
    traj = Trajectory(quad, ctrlType, trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States
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

    # ======= NEW: Initialize Control Data Arrays =======
    # Motor commands and control signals
    thrust_sp_all = np.zeros([numTimeStep, 3])        # Thrust setpoint [x, y, z]
    rate_sp_all = np.zeros([numTimeStep, 3])          # Rate setpoint [p, q, r]
    rate_ctrl_all = np.zeros([numTimeStep, 3])        # Rate control output
    qd_all = np.zeros([numTimeStep, 4])               # Desired quaternion
    pos_sp_all = np.zeros([numTimeStep, 3])           # Position setpoint
    vel_sp_all = np.zeros([numTimeStep, 3])           # Velocity setpoint
    acc_sp_all = np.zeros([numTimeStep, 3])           # Acceleration setpoint
    
    # Control allocation data (for EKF physics model)
    total_thrust_all = np.zeros(numTimeStep)          # Total thrust magnitude
    control_torques_all = np.zeros([numTimeStep, 3]) # Control torques [Mx, My, Mz]
    
    # Vehicle dynamics data
    acc_body_all = np.zeros([numTimeStep, 3])         # Acceleration in body frame
    omega_dot_all = np.zeros([numTimeStep, 3])        # Angular acceleration

    t_all[0] = Ti
    s_all[0, :] = quad.state
    pos_all[0, :] = quad.pos
    vel_all[0, :] = quad.vel
    quat_all[0, :] = quad.quat
    omega_all[0, :] = quad.omega
    euler_all[0, :] = quad.euler
    sDes_traj_all[0, :] = traj.sDes
    sDes_calc_all[0, :] = ctrl.sDesCalc
    w_cmd_all[0, :] = ctrl.w_cmd
    wMotor_all[0, :] = quad.wMotor
    thr_all[0, :] = quad.thr
    tor_all[0, :] = quad.tor

    # ======= NEW: Store Initial Control Data =======
    thrust_sp_all[0, :] = ctrl.thrust_sp
    rate_sp_all[0, :] = ctrl.rate_sp if hasattr(ctrl, 'rate_sp') else [0, 0, 0]
    rate_ctrl_all[0, :] = ctrl.rateCtrl if hasattr(ctrl, 'rateCtrl') else [0, 0, 0]
    qd_all[0, :] = ctrl.qd if hasattr(ctrl, 'qd') else [1, 0, 0, 0]
    pos_sp_all[0, :] = ctrl.pos_sp
    vel_sp_all[0, :] = ctrl.vel_sp
    acc_sp_all[0, :] = ctrl.acc_sp
    
    # Calculate initial control allocation
    total_thrust_all[0] = np.linalg.norm(ctrl.thrust_sp)
    
    # Calculate control torques using mixer matrix (simplified)
    # This is a basic approximation - you might want to use the actual mixer matrix
    motor_thrusts = quad.thr
    if len(motor_thrusts) >= 6:  # Hexacopter
        # Simplified torque calculation (you should use actual mixer matrix)
        L = quad.params.get("L", 0.225)  # Arm length
        control_torques_all[0, 0] = L * (motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[4] - motor_thrusts[5])  # Roll
        control_torques_all[0, 1] = L * (motor_thrusts[0] + motor_thrusts[1] - motor_thrusts[3] - motor_thrusts[4])  # Pitch  
        control_torques_all[0, 2] = quad.params["kTo"] * (motor_thrusts[0] - motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[3] + motor_thrusts[4] - motor_thrusts[5])  # Yaw
    
    acc_body_all[0, :] = quad.dcm.T @ quad.acc  # Transform acceleration to body frame
    omega_dot_all[0, :] = quad.omega_dot

    # Initialize Sensors
    imu = IMUSensor()
    gps = GPSSensor()
    baro = AltitudeSensor("baro")
    lidar = AltitudeSensor("lidar")
    mag = MagnetometerSensor()  # Initialize magnetometer

    # Initialize arrays for sensor data
    acc_all = np.zeros([numTimeStep, 3])
    gyro_all = np.zeros([numTimeStep, 3])
    gps_pos_all = np.zeros([numTimeStep, 3])
    gps_vel_all = np.zeros([numTimeStep, 3])
    baro_alt_all = np.zeros(numTimeStep)
    mag_all = np.zeros([numTimeStep, 3])  # Magnetometer data array

    gps_available = np.zeros(numTimeStep, dtype=bool)
    baro_available = np.zeros(numTimeStep, dtype=bool)
    # Magnetometer availability
    mag_available = np.zeros(numTimeStep, dtype=bool)

    # Store initial state
    t_all[0] = Ti
    pos_all[0] = quad.pos
    vel_all[0] = quad.vel
    quat_all[0] = quad.quat
    euler_all[0] = quad.euler

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    while round(t, 3) < Tf:

        t, acc_m, gyro_m, pos_m, vel_m, alt_m, mag_m = quad_sim(
            t, Ts, quad, ctrl, wind, traj, imu, gps, baro, mag)

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

        # ======= NEW: Store Control Data =======
        thrust_sp_all[i, :] = ctrl.thrust_sp
        rate_sp_all[i, :] = ctrl.rate_sp if hasattr(ctrl, 'rate_sp') else [0, 0, 0]
        rate_ctrl_all[i, :] = ctrl.rateCtrl if hasattr(ctrl, 'rateCtrl') else [0, 0, 0]
        qd_all[i, :] = ctrl.qd if hasattr(ctrl, 'qd') else [1, 0, 0, 0]
        pos_sp_all[i, :] = ctrl.pos_sp
        vel_sp_all[i, :] = ctrl.vel_sp
        acc_sp_all[i, :] = ctrl.acc_sp
        
        # Calculate control allocation data
        total_thrust_all[i] = np.linalg.norm(ctrl.thrust_sp)
        
        # Calculate control torques (simplified - use actual mixer if available)
        motor_thrusts = quad.thr
        if len(motor_thrusts) >= 6:  # Hexacopter
            L = quad.params.get("L", 0.225)  # Arm length
            control_torques_all[i, 0] = L * (motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[4] - motor_thrusts[5])  # Roll
            control_torques_all[i, 1] = L * (motor_thrusts[0] + motor_thrusts[1] - motor_thrusts[3] - motor_thrusts[4])  # Pitch
            control_torques_all[i, 2] = quad.params["kTo"] * (motor_thrusts[0] - motor_thrusts[1] + motor_thrusts[2] - motor_thrusts[3] + motor_thrusts[4] - motor_thrusts[5])  # Yaw
        
        # Store vehicle dynamics data
        acc_body_all[i, :] = quad.dcm.T @ quad.acc  # Transform acceleration to body frame
        omega_dot_all[i, :] = quad.omega_dot

        # Sensor Data
        acc_all[i] = acc_m
        gyro_all[i] = gyro_m
        if pos_m is not None and vel_m is not None:
            gps_pos_all[i] = pos_m
            gps_vel_all[i] = vel_m
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

    # ======= ENHANCED: Save Complete Data to CSV =======
    
    # Create comprehensive sensor and control data
    complete_data = pd.DataFrame({
        'timestamp': t_all,
        
        # ===== IMU Data =====
        'acc_x': acc_all[:, 0],
        'acc_y': acc_all[:, 1],
        'acc_z': acc_all[:, 2],
        'gyro_x': gyro_all[:, 0],
        'gyro_y': gyro_all[:, 1],
        'gyro_z': gyro_all[:, 2],
        
        # ===== GPS Data =====
        'gps_pos_x': gps_pos_all[:, 0],
        'gps_pos_y': gps_pos_all[:, 1],
        'gps_pos_z': gps_pos_all[:, 2],
        'gps_vel_x': gps_vel_all[:, 0],
        'gps_vel_y': gps_vel_all[:, 1],
        'gps_vel_z': gps_vel_all[:, 2],
        'gps_available': gps_available,
        
        # ===== Barometer Data =====
        'baro_altitude': baro_alt_all,
        'baro_available': baro_available,
        
        # ===== Magnetometer Data =====
        'mag_x': mag_all[:, 0],
        'mag_y': mag_all[:, 1],
        'mag_z': mag_all[:, 2],
        'mag_available': mag_available,
        
        # ===== NEW: Motor Commands & Control Signals =====
        # Individual motor speed commands (rad/s)
        'motor_cmd_1': w_cmd_all[:, 0],
        'motor_cmd_2': w_cmd_all[:, 1],
        'motor_cmd_3': w_cmd_all[:, 2],
        'motor_cmd_4': w_cmd_all[:, 3],
        'motor_cmd_5': w_cmd_all[:, 4],
        'motor_cmd_6': w_cmd_all[:, 5],
        
        # Individual motor thrusts (N)
        'motor_thrust_1': thr_all[:, 0],
        'motor_thrust_2': thr_all[:, 1],
        'motor_thrust_3': thr_all[:, 2],
        'motor_thrust_4': thr_all[:, 3],
        'motor_thrust_5': thr_all[:, 4],
        'motor_thrust_6': thr_all[:, 5],
        
        # Control setpoints
        'thrust_sp_x': thrust_sp_all[:, 0],
        'thrust_sp_y': thrust_sp_all[:, 1],
        'thrust_sp_z': thrust_sp_all[:, 2],
        'total_thrust_sp': total_thrust_all,
        
        'rate_sp_x': rate_sp_all[:, 0],        # Roll rate setpoint (rad/s)
        'rate_sp_y': rate_sp_all[:, 1],        # Pitch rate setpoint (rad/s)  
        'rate_sp_z': rate_sp_all[:, 2],        # Yaw rate setpoint (rad/s)
        
        'rate_ctrl_x': rate_ctrl_all[:, 0],    # Roll rate control output
        'rate_ctrl_y': rate_ctrl_all[:, 1],    # Pitch rate control output
        'rate_ctrl_z': rate_ctrl_all[:, 2],    # Yaw rate control output
        
        # Control allocation outputs
        'control_torque_x': control_torques_all[:, 0],  # Roll torque (N⋅m)
        'control_torque_y': control_torques_all[:, 1],  # Pitch torque (N⋅m)
        'control_torque_z': control_torques_all[:, 2],  # Yaw torque (N⋅m)
        
        # Desired quaternion
        'qd_w': qd_all[:, 0],
        'qd_x': qd_all[:, 1],
        'qd_y': qd_all[:, 2],
        'qd_z': qd_all[:, 3],
        
        # High-level setpoints
        'pos_sp_x': pos_sp_all[:, 0],
        'pos_sp_y': pos_sp_all[:, 1],
        'pos_sp_z': pos_sp_all[:, 2],
        'vel_sp_x': vel_sp_all[:, 0],
        'vel_sp_y': vel_sp_all[:, 1],
        'vel_sp_z': vel_sp_all[:, 2],
        'acc_sp_x': acc_sp_all[:, 0],
        'acc_sp_y': acc_sp_all[:, 1],
        'acc_sp_z': acc_sp_all[:, 2],
        
        # Vehicle dynamics (for validation)
        'acc_body_x': acc_body_all[:, 0],
        'acc_body_y': acc_body_all[:, 1],
        'acc_body_z': acc_body_all[:, 2],
        'omega_dot_x': omega_dot_all[:, 0],
        'omega_dot_y': omega_dot_all[:, 1],
        'omega_dot_z': omega_dot_all[:, 2],
        
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
    complete_data_file = os.path.join(log_dir, f"complete_flight_data_{timestamp}.csv")

    # Save complete dataframe to CSV
    complete_data.to_csv(complete_data_file, index=False)

    print("Complete flight data saved successfully:")
    print(f"- {complete_data_file}")

    # Add enhanced simulation metadata
    metadata_file = os.path.join(log_dir, f"simulation_metadata_{timestamp}.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"=== SIMULATION METADATA ===\n")
        f.write(f"Simulation timestamp: {timestamp}\n")
        f.write(f"Simulation duration: {Tf} seconds\n")
        f.write(f"Timestep: {Ts} seconds\n")
        f.write(f"Control type: {ctrlType}\n")
        f.write(f"Trajectory type: {int(trajSelect[0])}\n")
        f.write(f"Yaw trajectory type: {int(trajSelect[1])}\n")
        f.write(f"Waypoint timing mode: {int(trajSelect[2])}\n")
        f.write(f"Total simulation points: {numTimeStep}\n")
        f.write(f"\n=== SENSOR FREQUENCIES ===\n")
        f.write(f"IMU: {imu.freq} Hz\n")
        f.write(f"GPS: {gps.freq} Hz\n")
        f.write(f"Barometer: {baro.freq} Hz\n")
        f.write(f"Magnetometer: {mag.freq} Hz\n")
        f.write(f"\n=== VEHICLE PARAMETERS ===\n")
        f.write(f"Mass: {quad.params['mB']} kg\n")
        f.write(f"Inertia: {quad.params['IB']} kg⋅m²\n")
        f.write(f"Arm length: {quad.params.get('L', 'N/A')} m\n")
        f.write(f"Thrust coefficient: {quad.params['kTh']} N/(rad/s)²\n")
        f.write(f"Torque coefficient: {quad.params['kTo']} N⋅m/(rad/s)²\n")
        f.write(f"\n=== CONTROL DATA LOGGED ===\n")
        f.write(f"✓ Individual motor speed commands (6 motors)\n")
        f.write(f"✓ Individual motor thrust outputs (6 motors)\n")
        f.write(f"✓ Control thrust setpoints [x, y, z]\n")
        f.write(f"✓ Control rate setpoints [p, q, r]\n")
        f.write(f"✓ Control rate outputs [p, q, r]\n")
        f.write(f"✓ Control torques [Mx, My, Mz]\n")
        f.write(f"✓ Desired quaternion [w, x, y, z]\n")
        f.write(f"✓ Position/velocity/acceleration setpoints\n")
        f.write(f"✓ Vehicle dynamics (acc_body, omega_dot)\n")
        f.write(f"\n=== DATA STATISTICS ===\n")
        f.write(f"GPS update rate: {np.sum(gps_available)/len(gps_available)*100:.1f}%\n")
        f.write(f"Barometer update rate: {np.sum(baro_available)/len(baro_available)*100:.1f}%\n")
        f.write(f"Magnetometer update rate: {np.sum(mag_available)/len(mag_available)*100:.1f}%\n")
        f.write(f"Total thrust range: {np.min(total_thrust_all):.2f} - {np.max(total_thrust_all):.2f} N\n")
        f.write(f"Max control torques: [{np.max(np.abs(control_torques_all[:,0])):.3f}, {np.max(np.abs(control_torques_all[:,1])):.3f}, {np.max(np.abs(control_torques_all[:,2])):.3f}] N⋅m\n")

    print(f"- {metadata_file}")
    print(f"\nData ready for EKF processing with control input!")
    print(f"Total data points: {len(complete_data)}")
    print(f"Control data frequency: {1/Ts:.0f} Hz")

    # View Results
    # ---------------------------
    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all,
                      euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all,
                                  sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
    plot_sensor_data(t_all, pos_all, vel_all, acc_all, gyro_all,
                     gps_pos_all, gps_vel_all, baro_alt_all, mag_all)
    
    # ===== NEW: Plot Control Data =====
    plot_control_data(t_all, thrust_sp_all, rate_sp_all, rate_ctrl_all, 
                     control_torques_all, w_cmd_all, thr_all)
    
    plt.show()


def plot_control_data(t_all, thrust_sp_all, rate_sp_all, rate_ctrl_all, 
                     control_torques_all, w_cmd_all, thr_all):
    """Plot control data untuk validasi"""
    
    # Plot Control Thrust and Torques
    plt.figure(figsize=(15, 12))
    plt.suptitle("Control Commands and Outputs", fontsize=16)
    
    # Thrust setpoints
    plt.subplot(3, 2, 1)
    plt.plot(t_all, thrust_sp_all[:, 0], 'r-', label='Thrust X')
    plt.plot(t_all, thrust_sp_all[:, 1], 'g-', label='Thrust Y')
    plt.plot(t_all, thrust_sp_all[:, 2], 'b-', label='Thrust Z')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Thrust Setpoint (N)')
    plt.title('Control Thrust Setpoints')
    
    # Control torques
    plt.subplot(3, 2, 2)
    plt.plot(t_all, control_torques_all[:, 0], 'r-', label='Torque X (Roll)')
    plt.plot(t_all, control_torques_all[:, 1], 'g-', label='Torque Y (Pitch)')
    plt.plot(t_all, control_torques_all[:, 2], 'b-', label='Torque Z (Yaw)')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Control Torque (N⋅m)')
    plt.title('Control Torques')
    
    # Rate setpoints
    plt.subplot(3, 2, 3)
    plt.plot(t_all, rate_sp_all[:, 0] * 180/np.pi, 'r-', label='Roll Rate SP')
    plt.plot(t_all, rate_sp_all[:, 1] * 180/np.pi, 'g-', label='Pitch Rate SP')
    plt.plot(t_all, rate_sp_all[:, 2] * 180/np.pi, 'b-', label='Yaw Rate SP')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Rate Setpoint (deg/s)')
    plt.title('Angular Rate Setpoints')
    
    # Rate control outputs
    plt.subplot(3, 2, 4)
    plt.plot(t_all, rate_ctrl_all[:, 0], 'r-', label='Roll Ctrl')
    plt.plot(t_all, rate_ctrl_all[:, 1], 'g-', label='Pitch Ctrl')
    plt.plot(t_all, rate_ctrl_all[:, 2], 'b-', label='Yaw Ctrl')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Rate Control Output')
    plt.title('Rate Control Outputs')
    
    # Motor commands
    plt.subplot(3, 2, 5)
    for i in range(6):
        plt.plot(t_all, w_cmd_all[:, i], label=f'Motor {i+1}')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Motor Speed Command (rad/s)')
    plt.xlabel('Time (s)')
    plt.title('Motor Speed Commands')
    
    # Motor thrusts
    plt.subplot(3, 2, 6)
    for i in range(6):
        plt.plot(t_all, thr_all[:, i], label=f'Motor {i+1}')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Motor Thrust (N)')
    plt.xlabel('Time (s)')
    plt.title('Motor Thrust Outputs')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_sensor_data(t_all, pos_all, vel_all, acc_all, gyro_all, gps_pos_all, gps_vel_all, baro_alt_all, mag_all):
    # Plot data IMU
    plt.figure(figsize=(15, 10))
    plt.suptitle("IMU Sensor Data", fontsize=16)

    # Plot accelerometer
    plt.subplot(2, 1, 1)
    plt.plot(t_all, acc_all[:, 0], 'r-', label='Acc X')
    plt.plot(t_all, acc_all[:, 1], 'g-', label='Acc Y')
    plt.plot(t_all, acc_all[:, 2], 'b-', label='Acc Z')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Accelerometer Measurements')

    # Plot gyroscope
    plt.subplot(2, 1, 2)
    # rad/s to deg/s (rad/s * 180/pi)
    gyro_deg_x = gyro_all[:, 0] * 180 / np.pi
    gyro_deg_y = gyro_all[:, 1] * 180 / np.pi
    gyro_deg_z = gyro_all[:, 2] * 180 / np.pi

    plt.plot(t_all, gyro_deg_x, 'r-', label='Gyro X')
    plt.plot(t_all, gyro_deg_y, 'g-', label='Gyro Y')
    plt.plot(t_all, gyro_deg_z, 'b-', label='Gyro Z')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.title('Gyroscope Measurements')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Plot GPS and barometer data
    plt.figure(figsize=(15, 12))
    plt.suptitle("GPS and Barometer Data", fontsize=16)

    # Plot GPS position compared to true position
    plt.subplot(3, 1, 1)
    plt.plot(t_all, pos_all[:, 0], 'r-', label='True X')
    plt.plot(t_all, pos_all[:, 1], 'g-', label='True Y')
    plt.plot(t_all, pos_all[:, 2], 'b-', label='True Z')

    # Find non-zero GPS readings (when GPS updates occurred)
    non_zero_indices = np.where(np.any(gps_pos_all != 0, axis=1))[0]
    plt.plot(t_all[non_zero_indices], gps_pos_all[non_zero_indices,
             0], 'ro', markersize=4, label='GPS X')
    plt.plot(t_all[non_zero_indices], gps_pos_all[non_zero_indices,
             1], 'go', markersize=4, label='GPS Y')
    plt.plot(t_all[non_zero_indices], gps_pos_all[non_zero_indices,
             2], 'bo', markersize=4, label='GPS Z')

    plt.grid(True)
    plt.legend()
    plt.ylabel('Position (m)')
    plt.title('GPS Position Measurements vs True Position')

    # Plot GPS velocity compared to true velocity
    plt.subplot(3, 1, 2)
    plt.plot(t_all, vel_all[:, 0], 'r-', label='True Vx')
    plt.plot(t_all, vel_all[:, 1], 'g-', label='True Vy')
    plt.plot(t_all, vel_all[:, 2], 'b-', label='True Vz')

    # Plot GPS velocity measurements
    plt.plot(t_all[non_zero_indices], gps_vel_all[non_zero_indices,
             0], 'ro', markersize=4, label='GPS Vx')
    plt.plot(t_all[non_zero_indices], gps_vel_all[non_zero_indices,
             1], 'go', markersize=4, label='GPS Vy')
    plt.plot(t_all[non_zero_indices], gps_vel_all[non_zero_indices,
             2], 'bo', markersize=4, label='GPS Vz')

    plt.grid(True)
    plt.legend()
    plt.ylabel('Velocity (m/s)')
    plt.title('GPS Velocity Measurements vs True Velocity')

    # Plot barometer altitude compared to true altitude
    plt.subplot(3, 1, 3)

    # Konversi altitude berdasarkan orientasi sistem koordinat
    if config.orient == "NED":
        true_altitude = -pos_all[:, 2]  # In NED, altitude is -z
    else:
        true_altitude = pos_all[:, 2]   # In ENU, altitude is z

    plt.plot(t_all, true_altitude, 'b-', label='True Altitude')

    # Find non-zero barometer readings
    non_zero_baro = np.where(baro_alt_all != 0)[0]
    plt.plot(t_all[non_zero_baro], baro_alt_all[non_zero_baro],
             'ro', markersize=4, label='Baro Altitude')

    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Barometer Measurements vs True Altitude')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Plot magnetometer data
    plt.figure(figsize=(15, 10))
    plt.suptitle("Magnetometer Data", fontsize=16)

    # Plot magnetometer measurements
    plt.subplot(2, 1, 1)
    # Find non-zero magnetometer readings
    non_zero_mag = np.where(np.any(mag_all != 0, axis=1))[0]

    plt.plot(t_all[non_zero_mag], mag_all[non_zero_mag, 0],
             'r.', markersize=2, label='Mag X')
    plt.plot(t_all[non_zero_mag], mag_all[non_zero_mag, 1],
             'g.', markersize=2, label='Mag Y')
    plt.plot(t_all[non_zero_mag], mag_all[non_zero_mag, 2],
             'b.', markersize=2, label='Mag Z')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Magnetic Field (Gauss)')
    plt.title('Magnetometer Measurements')

    # Plot magnitude of magnetic field
    plt.subplot(2, 1, 2)
    mag_magnitude = np.sqrt(mag_all[non_zero_mag, 0]**2 +
                            mag_all[non_zero_mag, 1]**2 +
                            mag_all[non_zero_mag, 2]**2)
    plt.plot(t_all[non_zero_mag], mag_magnitude, 'k-', label='Magnitude')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field Magnitude (Gauss)')
    plt.title('Magnetometer Magnitude')

    plt.tight_layout(rect=[0, 0, 1, 0.95])


if __name__ == "__main__":
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception(
            "{} is not a valid orientation. Verify config.py file.".format(config.orient))