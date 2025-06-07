# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!

FIXED VERSION: Updated for consistent hexacopter configuration
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import os

import utils
import config

numFrames = 8


def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave):
    """Fixed animation function with consistent motor configuration"""
    
    try:
        x = pos_all[:, 0]
        y = pos_all[:, 1]
        z = pos_all[:, 2]

        xDes = sDes_tr_all[:, 0]
        yDes = sDes_tr_all[:, 1]
        zDes = sDes_tr_all[:, 2]

        x_wp = waypoints[:, 0]
        y_wp = waypoints[:, 1]
        z_wp = waypoints[:, 2]

        if (config.orient == "NED"):
            z = -z
            zDes = -zDes
            z_wp = -z_wp

        fig = plt.figure(figsize=(12, 9))
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        
        # Motor lines with correct configuration
        line1, = ax.plot([], [], [], lw=3, color='red')    # Motor 1: kanan CW
        line2, = ax.plot([], [], [], lw=3, color='blue')   # Motor 2: kiri CCW
        line3, = ax.plot([], [], [], lw=3, color='green')  # Motor 3: kiri atas CW
        line4, = ax.plot([], [], [], lw=3, color='black')  # Motor 4: kanan bawah CCW
        line5, = ax.plot([], [], [], lw=3, color='purple') # Motor 5: kanan atas CCW
        line6, = ax.plot([], [], [], lw=3, color='orange') # Motor 6: kiri bawah CW
        
        # Trajectory line
        line_traj, = ax.plot([], [], [], '--', lw=2, color='cyan')

        # Setting the axes properties
        extraEachSide = 0.5
        maxRange = 0.5*np.array([x.max()-x.min(), y.max() -
                                y.min(), z.max()-z.min()]).max() + extraEachSide
        mid_x = 0.5*(x.max()+x.min())
        mid_y = 0.5*(y.max()+y.min())
        mid_z = 0.5*(z.max()+z.min())

        ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
        ax.set_xlabel('X (m)')
        if (config.orient == "NED"):
            ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
        elif (config.orient == "ENU"):
            ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
        ax.set_ylabel('Y (m)')
        ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
        ax.set_zlabel('Altitude (m)')

        titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

        # Trajectory type text
        trajType = get_trajectory_type(xyzType)
        yawTrajType = get_yaw_trajectory_type(yawType)

        if (xyzType != 0):  # Not hover
            ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker='o', s=50)
            if (xyzType > 2):  # Smooth trajectories
                ax.plot(xDes, yDes, zDes, ':', lw=2, color='green', alpha=0.7)

        titleType1 = ax.text2D(0.95, 0.95, trajType,
                               transform=ax.transAxes, horizontalalignment='right', fontsize=10)
        titleType2 = ax.text2D(0.95, 0.91, 'Yaw: ' + yawTrajType,
                               transform=ax.transAxes, horizontalalignment='right', fontsize=10)

        def updateLines(i):
            try:
                time = t_all[i*numFrames]
                pos = pos_all[i*numFrames]
                x_curr = pos[0]
                y_curr = pos[1]
                z_curr = pos[2]

                x_from0 = pos_all[0:i*numFrames, 0]
                y_from0 = pos_all[0:i*numFrames, 1]
                z_from0 = pos_all[0:i*numFrames, 2]

                quat = quat_all[i*numFrames]

                if (config.orient == "NED"):
                    z_curr = -z_curr
                    z_from0 = -z_from0
                    quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])

                R = utils.quat2Dcm(quat)
                
                # Get arm length
                if "L" in params:
                    L = params["L"]
                else:
                    L = params.get("dxm", 0.225)
                
                dzm = params.get("dzm", 0.05)

                # FIXED: Motor positions for hexacopter X configuration
                # Motor 1: kanan CW, Motor 2: kiri CCW, Motor 3: kiri atas CW
                # Motor 4: kanan bawah CCW, Motor 5: kanan atas CCW, Motor 6: kiri bawah CW
                
                motor_positions = np.array([
                    # Motor positions in body frame [x, y, z]
                    [L, 0, dzm],                                    # Motor 1: kanan
                    [-L, 0, dzm],                                   # Motor 2: kiri  
                    [L * np.cos(2*np.pi/3), L * np.sin(2*np.pi/3), dzm],  # Motor 3: kiri atas (120°)
                    [L * np.cos(-2*np.pi/3), L * np.sin(-2*np.pi/3), dzm], # Motor 4: kanan bawah (240°)
                    [L * np.cos(np.pi/3), L * np.sin(np.pi/3), dzm],       # Motor 5: kanan atas (60°)
                    [L * np.cos(-np.pi/3), L * np.sin(-np.pi/3), dzm]      # Motor 6: kiri bawah (300°)
                ])

                # Transform to world frame
                center = np.array([x_curr, y_curr, z_curr])
                motor_world = np.zeros((6, 3))
                
                for j in range(6):
                    motor_world[j] = center + R @ motor_positions[j]

                # Update motor lines (from center to motor position)
                lines = [line1, line2, line3, line4, line5, line6]
                for idx, line in enumerate(lines):
                    line.set_data([center[0], motor_world[idx, 0]],
                                  [center[1], motor_world[idx, 1]])
                    line.set_3d_properties([center[2], motor_world[idx, 2]])

                # Update trajectory
                line_traj.set_data(x_from0, y_from0)
                line_traj.set_3d_properties(z_from0)
                titleTime.set_text(u"Time = {:.2f} s".format(time))

                return line1, line2, line3, line4, line5, line6, line_traj
                
            except Exception as e:
                print(f"Animation frame error: {e}")
                return line1, line2, line3, line4, line5, line6, line_traj

        def ini_plot():
            lines = [line1, line2, line3, line4, line5, line6, line_traj]
            for line in lines:
                line.set_data(np.empty([1]), np.empty([1]))
                line.set_3d_properties(np.empty([1]))
            return lines

        # Creating the Animation object
        print("Creating animation...")
        line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, 
                                          frames=len(t_all[0:-2:numFrames]), 
                                          interval=(Ts*1000*numFrames), blit=False)

        if (ifsave):
            # Create directory if it doesn't exist
            gif_dir = 'Gifs/Raw'
            os.makedirs(gif_dir, exist_ok=True)
            
            try:
                filename = f'{gif_dir}/animation_{xyzType:.0f}_{yawType:.0f}.gif'
                line_ani.save(filename, dpi=80, writer='pillow', fps=25)
                print(f"✓ Animation saved: {filename}")
            except Exception as e:
                print(f"✗ Could not save animation: {e}")

        print("✓ Animation created successfully")
        return line_ani
        
    except Exception as e:
        print(f"✗ Animation creation failed: {e}")
        return None


def get_trajectory_type(xyzType):
    """Get trajectory type string"""
    trajectory_types = {
        0: 'Hover',
        1: 'Simple Waypoints', 
        2: 'Simple Waypoint Interpolation',
        3: 'Minimum Velocity Trajectory',
        4: 'Minimum Acceleration Trajectory',
        5: 'Minimum Jerk Trajectory',
        6: 'Minimum Snap Trajectory',
        7: 'Minimum Acceleration Trajectory - Stop',
        8: 'Minimum Jerk Trajectory - Stop',
        9: 'Minimum Snap Trajectory - Stop',
        10: 'Minimum Jerk Trajectory - Fast Stop',
        11: 'Minimum Snap Trajectory - Fast Stop',
        12: 'Simple Waypoints'
    }
    return trajectory_types.get(xyzType, f'Unknown ({xyzType})')


def get_yaw_trajectory_type(yawType):
    """Get yaw trajectory type string"""
    yaw_types = {
        0: 'None',
        1: 'Waypoints', 
        2: 'Interpolation',
        3: 'Follow',
        4: 'Zero'
    }
    return yaw_types.get(yawType, f'Unknown ({yawType})')