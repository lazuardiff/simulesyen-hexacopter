# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
MODIFIED: Waypoints now generate a 5-rotation helix trajectory.
"""

import numpy as np
from numpy import pi
import config

deg2rad = pi/180.0


def makeWaypoints():
    v_average = 1.2
    t_ini = 2

    # =================================================================
    # ===== PENGATURAN LINTASAN HELIKS 5 PUTARAN =====
    # =================================================================

    num_rotations = 5  # Jumlah putaran yang diinginkan
    points_per_rotation = 32  # Titik per rotasi untuk kehalusan lintasan

    radius = 5.0  # Radius 5m untuk diameter 10m
    helix_height = 5.0  # Total penurunan ketinggian per satu putaran (meter)
    start_altitude_ned = -2.0  # Ketinggian awal heliks (dalam sistem NED)

    # Menghasilkan jumlah titik dan rentang sudut untuk 5 putaran
    num_points = points_per_rotation * num_rotations
    angles = np.linspace(0, num_rotations * 2*np.pi, num_points, endpoint=True)

    # Titik awal sebelum memulai heliks
    wp_ini = np.array([radius, 0, start_altitude_ned])

    # Waypoints di sepanjang lintasan heliks
    wp = np.zeros((num_points-1, 3))
    for i in range(1, num_points):
        # Ketinggian berubah secara linear, sekarang mencakup 5 putaran
        # Total penurunan akan menjadi num_rotations * helix_height
        altitude = start_altitude_ned - \
            (helix_height * (angles[i] / (2 * np.pi)))

        # Koordinat x, y, z untuk heliks
        wp[i-1] = [radius * np.cos(angles[i]),
                   radius * np.sin(angles[i]), altitude]

    # Waktu di setiap waypoint
    t = np.ones(num_points-1) * 0.5

    # Yaw (orientasi hidung drone) mengikuti arah gerakan
    yaw_ini = 90  # Awalnya menghadap ke arah Y
    yaw = np.zeros(num_points-1)
    for i in range(1, num_points):
        # Arahkan yaw tegak lurus terhadap garis dari pusat (tangensial)
        yaw[i-1] = np.degrees(angles[i] + pi/2)

    # Menggabungkan semua array
    t = np.hstack((t_ini, t)).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad

    return t, wp, yaw, v_average
