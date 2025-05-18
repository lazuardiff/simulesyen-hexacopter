# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

import utils
import config

numFrames = 8


def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave):

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

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    line1, = ax.plot([], [], [], lw=2, color='red')    # Motor 1 (90°)
    line2, = ax.plot([], [], [], lw=2, color='blue')   # Motor 2 (270°)
    line3, = ax.plot([], [], [], lw=2, color='green')  # Motor 3 (330°)
    line4, = ax.plot([], [], [], lw=2, color='black')  # Motor 4 (150°)
    line5, = ax.plot([], [], [], lw=2, color='purple')  # Motor 5 (30°)
    line6, = ax.plot([], [], [], lw=2, color='orange')  # Motor 6 (210°)
    # Trajectory line (sebelumnya line3)
    line_traj, = ax.plot([], [], [], '--', lw=1, color='blue')

    # Setting the axes properties
    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max() -
                            y.min(), z.max()-z.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    mid_z = 0.5*(z.max()+z.min())

    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    if (config.orient == "NED"):
        ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
    elif (config.orient == "ENU"):
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    if (xyzType == 0):
        trajType = 'Hover'
    else:
        ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker='o', s=25)
        if (xyzType == 1 or xyzType == 12):
            trajType = 'Simple Waypoints'
        else:
            ax.plot(xDes, yDes, zDes, ':', lw=1.3, color='green')
            if (xyzType == 2):
                trajType = 'Simple Waypoint Interpolation'
            elif (xyzType == 3):
                trajType = 'Minimum Velocity Trajectory'
            elif (xyzType == 4):
                trajType = 'Minimum Acceleration Trajectory'
            elif (xyzType == 5):
                trajType = 'Minimum Jerk Trajectory'
            elif (xyzType == 6):
                trajType = 'Minimum Snap Trajectory'
            elif (xyzType == 7):
                trajType = 'Minimum Acceleration Trajectory - Stop'
            elif (xyzType == 8):
                trajType = 'Minimum Jerk Trajectory - Stop'
            elif (xyzType == 9):
                trajType = 'Minimum Snap Trajectory - Stop'
            elif (xyzType == 10):
                trajType = 'Minimum Jerk Trajectory - Fast Stop'
            elif (xyzType == 1):
                trajType = 'Minimum Snap Trajectory - Fast Stop'

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'

    titleType1 = ax.text2D(0.95, 0.95, trajType,
                           transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: ' + yawTrajType,
                           transform=ax.transAxes, horizontalalignment='right')

    def updateLines(i):

        time = t_all[i*numFrames]
        pos = pos_all[i*numFrames]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_from0 = pos_all[0:i*numFrames, 0]
        y_from0 = pos_all[0:i*numFrames, 1]
        z_from0 = pos_all[0:i*numFrames, 2]

        dxm = params["dxm"]
        dym = params["dym"]
        dzm = params["dzm"]

        quat = quat_all[i*numFrames]

        if (config.orient == "NED"):
            z = -z
            z_from0 = -z_from0
            quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])

        R = utils.quat2Dcm(quat)
        angles = np.array([90, 270, 330, 150, 30, 210]) * np.pi/180
        if "L" in params:
            L = params["L"]
        else:
            L = params["dxm"]

        motorPoints = np.zeros((18, 3))

        # Pusat drone ditempatkan di indeks genap
        motorPoints[0] = [0, 0, 0]  # pusat untuk line1
        motorPoints[3] = [0, 0, 0]  # pusat untuk line2
        motorPoints[6] = [0, 0, 0]  # pusat untuk line3 (baru)
        motorPoints[9] = [0, 0, 0]  # pusat untuk line4 (baru)
        motorPoints[12] = [0, 0, 0]  # pusat untuk line5 (baru)
        motorPoints[15] = [0, 0, 0]  # pusat untuk line6 (baru)

        # Posisi motor berdasarkan sudut
        # Motor 1 (90°)
        motorPoints[1] = [L * np.cos(angles[0]), L * np.sin(angles[0]), dzm]
        # Motor 2 (270°)
        motorPoints[4] = [L * np.cos(angles[1]), L * np.sin(angles[1]), dzm]
        # Motor 3 (330°)
        motorPoints[7] = [L * np.cos(angles[2]), L * np.sin(angles[2]), dzm]
        # Motor 4 (150°)
        motorPoints[10] = [L * np.cos(angles[3]), L * np.sin(angles[3]), dzm]
        motorPoints[13] = [
            L * np.cos(angles[4]), L * np.sin(angles[4]), dzm]  # Motor 5 (30°)
        # Motor 6 (210°)
        motorPoints[16] = [L * np.cos(angles[5]), L * np.sin(angles[5]), dzm]

        for j in range(18):
            temp = np.dot(R, motorPoints[j])
            motorPoints[j] = [temp[0] + x, temp[1] + y, temp[2] + z]

        line1.set_data([motorPoints[0][0], motorPoints[1][0]],
                       [motorPoints[0][1], motorPoints[1][1]])
        line1.set_3d_properties([motorPoints[0][2], motorPoints[1][2]])

        line2.set_data([motorPoints[3][0], motorPoints[4][0]],
                       [motorPoints[3][1], motorPoints[4][1]])
        line2.set_3d_properties([motorPoints[3][2], motorPoints[4][2]])

        line3.set_data([motorPoints[6][0], motorPoints[7][0]],
                       [motorPoints[6][1], motorPoints[7][1]])
        line3.set_3d_properties([motorPoints[6][2], motorPoints[7][2]])

        line4.set_data([motorPoints[9][0], motorPoints[10][0]],
                       [motorPoints[9][1], motorPoints[10][1]])
        line4.set_3d_properties([motorPoints[9][2], motorPoints[10][2]])

        line5.set_data([motorPoints[12][0], motorPoints[13][0]], [
                       motorPoints[12][1], motorPoints[13][1]])
        line5.set_3d_properties([motorPoints[12][2], motorPoints[13][2]])

        line6.set_data([motorPoints[15][0], motorPoints[16][0]], [
                       motorPoints[15][1], motorPoints[16][1]])
        line6.set_3d_properties([motorPoints[15][2], motorPoints[16][2]])

        line_traj.set_data(x_from0, y_from0)
        line_traj.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time))

        return line1, line2, line3, line4, line5, line6, line_traj

    def ini_plot():
        line1.set_data(np.empty([1]), np.empty([1]))
        line1.set_3d_properties(np.empty([1]))
        line2.set_data(np.empty([1]), np.empty([1]))
        line2.set_3d_properties(np.empty([1]))
        line3.set_data(np.empty([1]), np.empty([1]))
        line3.set_3d_properties(np.empty([1]))
        line4.set_data(np.empty([1]), np.empty([1]))
        line4.set_3d_properties(np.empty([1]))
        line5.set_data(np.empty([1]), np.empty([1]))
        line5.set_3d_properties(np.empty([1]))
        line6.set_data(np.empty([1]), np.empty([1]))
        line6.set_3d_properties(np.empty([1]))
        line_traj.set_data(np.empty([1]), np.empty([1]))
        line_traj.set_3d_properties(np.empty([1]))

        return line1, line2, line3, line4, line5, line6, line_traj

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(
        t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)

    if (ifsave):
        line_ani.save('Gifs/Raw/animation_{0:.0f}_{1:.0f}.gif'.format(
            xyzType, yawType), dpi=80, writer='imagemagick', fps=25)

    plt.show()
    return line_ani
