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
from utils.SensorModels import IMUSensor, GPSSensor, AltitudeSensor
from utils.EKF import EKF
import utils
import config


def quad_sim(t, Ts, quad, ctrl, wind, traj, imu, gps, alti):

    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Mengambil data sensor
    acc_m, gyro_m = imu.measure(quad, t)
    pos_m, vel_m = gps.measure(quad, t)
    alt_m = alti.measure(quad, t)

    # Trajectory for Desired States
    # ---------------------------
    sDes = traj.desiredState(t, Ts, quad)

    # Generate Commands (for next iteration)
    # ---------------------------
    ctrl.controller(traj, quad, sDes, Ts)

    return t, acc_m, gyro_m, pos_m, vel_m, alt_m


def main():
    start_time = time.time()

    # Simulation Setup
    # ---------------------------
    Ti = 0
    Ts = 0.001
    Tf = 20
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

    # Initialize Sensors
    imu = IMUSensor()
    gps = GPSSensor()
    baro = AltitudeSensor("baro")
    lidar = AltitudeSensor("lidar")

    # Initialize arrays for sensor data
    acc_all = np.zeros([numTimeStep, 3])
    gyro_all = np.zeros([numTimeStep, 3])
    gps_pos_all = np.zeros([numTimeStep, 3])
    gps_vel_all = np.zeros([numTimeStep, 3])
    baro_alt_all = np.zeros(numTimeStep)

    gps_available = np.zeros(numTimeStep, dtype=bool)
    baro_available = np.zeros(numTimeStep, dtype=bool)

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

        t, acc_m, gyro_m, pos_m, vel_m, alt_m = quad_sim(
            t, Ts, quad, ctrl, wind, traj, imu, gps, baro)

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

        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # Save data to CSV
    # ground truth data
    ground_truth_data = pd.DataFrame({
        'timestamp': t_all,
        'pos_x': pos_all[:, 0],
        'pos_y': pos_all[:, 1],
        'pos_z': pos_all[:, 2],
        'vel_x': vel_all[:, 0],
        'vel_y': vel_all[:, 1],
        'vel_z': vel_all[:, 2],
        'quat_w': quat_all[:, 0],
        'quat_x': quat_all[:, 1],
        'quat_y': quat_all[:, 2],
        'quat_z': quat_all[:, 3],
        'omega_x': omega_all[:, 0],
        'omega_y': omega_all[:, 1],
        'omega_z': omega_all[:, 2],
        'roll': euler_all[:, 0],
        'pitch': euler_all[:, 1],
        'yaw': euler_all[:, 2]
    })

    # sensor data
    imu_df = pd.DataFrame({
        'timestamp': t_all,
        'acc_x': acc_all[:, 0],
        'acc_y': acc_all[:, 1],
        'acc_z': acc_all[:, 2],
        'gyro_x': gyro_all[:, 0],
        'gyro_y': gyro_all[:, 1],
        'gyro_z': gyro_all[:, 2]
    })

    # gps data
    gps_df = pd.DataFrame({
        'timestamp': t_all,
        'pos_x': gps_pos_all[:, 0],
        'pos_y': gps_pos_all[:, 1],
        'pos_z': gps_pos_all[:, 2],
        'vel_x': gps_vel_all[:, 0],
        'vel_y': gps_vel_all[:, 1],
        'vel_z': gps_vel_all[:, 2],
        'available': gps_available
    })

    # baro data
    baro_df = pd.DataFrame({
        'timestamp': t_all,
        'altitude': baro_alt_all,
        'available': baro_available
    })

    # Save combined sensor data
    sensor_df = pd.DataFrame({
        'timestamp': t_all,
        # IMU
        'acc_x': acc_all[:, 0],
        'acc_y': acc_all[:, 1],
        'acc_z': acc_all[:, 2],
        'gyro_x': gyro_all[:, 0],
        'gyro_y': gyro_all[:, 1],
        'gyro_z': gyro_all[:, 2],
        # GPS
        'gps_pos_x': gps_pos_all[:, 0],
        'gps_pos_y': gps_pos_all[:, 1],
        'gps_pos_z': gps_pos_all[:, 2],
        'gps_vel_x': gps_vel_all[:, 0],
        'gps_vel_y': gps_vel_all[:, 1],
        'gps_vel_z': gps_vel_all[:, 2],
        'gps_available': gps_available,
        # Barometer
        'baro_altitude': baro_alt_all,
        'baro_available': baro_available,
        # Ground truth
        'true_pos_x': pos_all[:, 0],
        'true_pos_y': pos_all[:, 1],
        'true_pos_z': pos_all[:, 2],
        'true_vel_x': vel_all[:, 0],
        'true_vel_y': vel_all[:, 1],
        'true_vel_z': vel_all[:, 2],
        'true_roll': euler_all[:, 0],
        'true_pitch': euler_all[:, 1],
        'true_yaw': euler_all[:, 2],
    })

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created directory: {log_dir}")

    # Generate timestamp string for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define file paths with timestamp
    ground_truth_file = os.path.join(
        log_dir, f"ground_truth_data_{timestamp}.csv")
    imu_file = os.path.join(log_dir, f"imu_data_{timestamp}.csv")
    gps_file = os.path.join(log_dir, f"gps_data_{timestamp}.csv")
    baro_file = os.path.join(log_dir, f"baro_data_{timestamp}.csv")
    all_sensor_file = os.path.join(log_dir, f"all_sensor_data_{timestamp}.csv")

    # Save dataframes to CSV with timestamped filenames in logs directory
    ground_truth_data.to_csv(ground_truth_file, index=False)
    imu_df.to_csv(imu_file, index=False)
    gps_df.to_csv(gps_file, index=False)
    baro_df.to_csv(baro_file, index=False)
    sensor_df.to_csv(all_sensor_file, index=False)

    print("Data saved successfully to CSV files:")
    print(f"- {ground_truth_file}")
    print(f"- {imu_file}")
    print(f"- {gps_file}")
    print(f"- {baro_file}")
    print(f"- {all_sensor_file}")

    # Add simulation parameters to a metadata file
    metadata_file = os.path.join(
        log_dir, f"simulation_metadata_{timestamp}.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Simulation timestamp: {timestamp}\n")
        f.write(f"Simulation duration: {Tf} seconds\n")
        f.write(f"Timestep: {Ts} seconds\n")
        f.write(f"Control type: {ctrlType}\n")
        f.write(f"Trajectory type: {int(trajSelect[0])}\n")
        f.write(f"Yaw trajectory type: {int(trajSelect[1])}\n")
        f.write(f"Waypoint timing mode: {int(trajSelect[2])}\n")
        f.write(f"Total simulation points: {numTimeStep}\n")

    print(f"- {metadata_file}")

    # View Results
    # ---------------------------

    # utils.fullprint(sDes_traj_all[:,3:6])
    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all,
                      euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all,
                                  sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
    plot_sensor_data(t_all, pos_all, vel_all, acc_all, gyro_all,
                     gps_pos_all, gps_vel_all, baro_alt_all)
    plt.show()


def plot_ekf_results(t_all, pos_all, vel_all, euler_all, pos_est_all, vel_est_all, euler_est_all):
    """Plot hasil estimasi EKF dibandingkan dengan ground truth"""

    # Plot Position
    plt.figure(figsize=(15, 12))
    plt.suptitle("EKF Position Estimation", fontsize=16)

    # X Position
    plt.subplot(3, 1, 1)
    plt.plot(t_all, pos_all[:, 0], 'b-', label='True X')
    plt.plot(t_all, pos_est_all[:, 0], 'r--', label='EKF X')
    plt.grid(True)
    plt.legend()
    plt.ylabel('X Position (m)')

    # Y Position
    plt.subplot(3, 1, 2)
    plt.plot(t_all, pos_all[:, 1], 'b-', label='True Y')
    plt.plot(t_all, pos_est_all[:, 1], 'r--', label='EKF Y')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Y Position (m)')

    # Z Position
    plt.subplot(3, 1, 3)
    plt.plot(t_all, pos_all[:, 2], 'b-', label='True Z')
    plt.plot(t_all, pos_est_all[:, 2], 'r--', label='EKF Z')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Z Position (m)')
    plt.xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Plot Velocity
    plt.figure(figsize=(15, 12))
    plt.suptitle("EKF Velocity Estimation", fontsize=16)

    # X Velocity
    plt.subplot(3, 1, 1)
    plt.plot(t_all, vel_all[:, 0], 'b-', label='True Vx')
    plt.plot(t_all, vel_est_all[:, 0], 'r--', label='EKF Vx')
    plt.grid(True)
    plt.legend()
    plt.ylabel('X Velocity (m/s)')

    # Y Velocity
    plt.subplot(3, 1, 2)
    plt.plot(t_all, vel_all[:, 1], 'b-', label='True Vy')
    plt.plot(t_all, vel_est_all[:, 1], 'r--', label='EKF Vy')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Y Velocity (m/s)')

    # Z Velocity
    plt.subplot(3, 1, 3)
    plt.plot(t_all, vel_all[:, 2], 'b-', label='True Vz')
    plt.plot(t_all, vel_est_all[:, 2], 'r--', label='EKF Vz')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Z Velocity (m/s)')
    plt.xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Plot Attitude (Euler angles)
    plt.figure(figsize=(15, 12))
    plt.suptitle("EKF Attitude Estimation", fontsize=16)

    # Roll
    plt.subplot(3, 1, 1)
    plt.plot(t_all, euler_all[:, 0] * 180/np.pi, 'b-', label='True Roll')
    plt.plot(t_all, euler_est_all[:, 0] * 180/np.pi, 'r--', label='EKF Roll')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Roll (deg)')

    # Pitch
    plt.subplot(3, 1, 2)
    plt.plot(t_all, euler_all[:, 1] * 180/np.pi, 'b-', label='True Pitch')
    plt.plot(t_all, euler_est_all[:, 1] * 180/np.pi, 'r--', label='EKF Pitch')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Pitch (deg)')

    # Yaw
    plt.subplot(3, 1, 3)
    plt.plot(t_all, euler_all[:, 2] * 180/np.pi, 'b-', label='True Yaw')
    plt.plot(t_all, euler_est_all[:, 2] * 180/np.pi, 'r--', label='EKF Yaw')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Yaw (deg)')
    plt.xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_sensor_data(t_all, pos_all, vel_all, acc_all, gyro_all, gps_pos_all, gps_vel_all, baro_alt_all):
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


if __name__ == "__main__":
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception(
            "{} is not a valid orientation. Verify config.py file.".format(config.orient))
