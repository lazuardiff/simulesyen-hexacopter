# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
from numpy.linalg import inv
import utils
import config


def sys_params():
    mB = 1.2       # mass (kg)
    g = 9.81      # gravity (m/s/s)
    dxm = 0.16      # arm length (m)
    dym = 0.16      # arm length (m)
    dzm = 0.05      # motor height (m)
    IB = np.array([[0.0123, 0,      0],
                   [0,      0.0123, 0],
                   [0,      0,      0.0224]])  # Inertial tensor (kg*m^2)
    IRzz = 2.7e-5   # Rotor moment of inertia (kg*m^2)

    params = {}
    params["mB"] = mB
    params["g"] = g
    params["dxm"] = dxm
    params["dym"] = dym
    params["dzm"] = dzm
    params["IB"] = IB
    params["invI"] = inv(IB)
    params["IRzz"] = IRzz
    params["L"] = 0.275
    # Include integral gains in linear velocity control
    params["useIntergral"] = bool(False)
    # params["interpYaw"] = bool(False)       # Interpolate Yaw setpoints in waypoint trajectory

    params["Cd"] = 0.1
    params["kTh"] = 1.076e-5  # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
    params["kTo"] = 1.632e-7  # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
    # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
    params["mixerFM"] = makeMixerFM(params)
    params["mixerFMinv"] = np.linalg.pinv(params["mixerFM"])
    params["minThr"] = 0.1*6    # Minimum total thrust
    params["maxThr"] = 9.18*6   # Maximum total thrust
    params["minWmotor"] = 75       # Minimum motor rotation speed (rad/s)
    params["maxWmotor"] = 925      # Maximum motor rotation speed (rad/s)
    params["tau"] = 0.015    # Value for second order system for Motor dynamics
    params["kp"] = 1.0      # Value for second order system for Motor dynamics
    # Value for second order system for Motor dynamics
    params["damp"] = 1.0

    params["motorc1"] = 8.49     # w (rad/s) = cmd*c1 + c0 (cmd in %)
    params["motorc0"] = 74.7
    params["motordeadband"] = 1
    # params["ifexpo"] = bool(False)
    # if params["ifexpo"]:
    #     params["maxCmd"] = 100      # cmd (%) min and max
    #     params["minCmd"] = 0.01
    # else:
    #     params["maxCmd"] = 100
    #     params["minCmd"] = 1

    return params


def makeMixerFM(params):
    kTh = params["kTh"]
    kTo = params["kTo"]
    L = params["L"]  # arm length (meter), ganti sesuai drone Anda
    mixerFM = np.array([
        # Thrust (semua motor berkontribusi positif)
        [kTh, kTh, kTh, kTh, kTh, kTh],

        # Roll moment (positif = roll kiri, negatif = roll kanan)
        # Motor kanan (-), Motor kiri (+)
        [-L*kTh, L*kTh, 0.5*L*kTh, -0.5*L*kTh, -0.5*L*kTh, 0.5*L*kTh],

        # Pitch moment (positif = pitch up, negatif = pitch down)
        # Motor atas (+), Motor bawah (-)
        [0, 0, 0.866*L*kTh, -0.866*L*kTh, 0.866*L*kTh, -0.866*L*kTh],

        # Yaw torque (CW negatif, CCW positif)
        [-kTo, kTo, -kTo, kTo, kTo, -kTo]
    ])

    return mixerFM


def init_cmd(params):
    mB = params["mB"]
    g = params["g"]
    kTh = params["kTh"]
    kTo = params["kTo"]
    c1 = params["motorc1"]
    c0 = params["motorc0"]

    # w = cmd*c1 + c0   and   m*g/4 = kTh*w^2   and   torque = kTo*w^2
    thr_hover = mB*g/6.0
    w_hover = np.sqrt(thr_hover/kTh)
    tor_hover = kTo*w_hover*w_hover
    cmd_hover = (w_hover-c0)/c1
    return [cmd_hover, w_hover, thr_hover, tor_hover]


def init_state(params):

    x0 = 0.  # m
    y0 = 0.  # m
    z0 = 0.  # m
    phi0 = 0.  # rad
    theta0 = 0.  # rad
    psi0 = 0.  # rad

    quat = utils.YPRToQuat(psi0, theta0, phi0)

    if (config.orient == "ENU"):
        z0 = -z0

    s = np.zeros(25)
    s[0] = x0       # x
    s[1] = y0       # y
    s[2] = z0       # z
    s[3] = quat[0]  # q0
    s[4] = quat[1]  # q1
    s[5] = quat[2]  # q2
    s[6] = quat[3]  # q3
    s[7] = 0.       # xdot
    s[8] = 0.       # ydot
    s[9] = 0.       # zdot
    s[10] = 0.       # p
    s[11] = 0.       # q
    s[12] = 0.       # r

    w_hover = params["w_hover"]  # Hovering motor speed
    wdot_hover = 0.              # Hovering motor acc

    s[13] = w_hover
    s[14] = wdot_hover
    s[15] = w_hover
    s[16] = wdot_hover
    s[17] = w_hover
    s[18] = wdot_hover
    s[19] = w_hover
    s[20] = wdot_hover
    s[21] = w_hover
    s[22] = wdot_hover
    s[23] = w_hover
    s[24] = wdot_hover

    return s
