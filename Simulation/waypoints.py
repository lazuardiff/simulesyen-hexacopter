# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import config

deg2rad = pi/180.0


def makeWaypoints():
    v_average = 1.2
    t_ini = 2

    # Buat 8 titik melingkar
    radius = 2
    num_points = 8
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    wp_ini = np.array([0, 0, 0])  # Mulai dari ketinggian 1 meter
    wp = np.zeros((num_points, 3))

    for i in range(num_points):
        wp[i] = [radius * np.cos(angles[i]), radius * np.sin(angles[i]), 1]

    # Waktu di setiap waypoint (tidak berhenti)
    t = np.ones(num_points) * 1.0

    # Arahkan yaw ke arah gerakan
    yaw_ini = 0
    yaw = np.zeros(num_points)
    for i in range(num_points):
        yaw[i] = np.degrees(angles[i]) + 90  # Hadapkan ke arah gerakan

    t = np.hstack((t_ini, t)).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad

    return t, wp, yaw, v_average
