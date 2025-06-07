# PERUBAHAN DETAIL UNTUK hexacopter.py
# =====================================

import numpy as np
from numpy import sin, cos, tan, pi, sign
from scipy.integrate import ode

# 1. TAMBAHKAN IMPORT untuk mixer yang sudah diperbaiki
from quadFiles.initHexa import sys_params, init_cmd, init_state
import utils
import config

deg2rad = pi/180.0


class Quadcopter:
    def __init__(self, Ti):
        # ... (tidak ada perubahan sampai forces())
        self.params = sys_params()

        # Command for initial stable hover
        ini_hover = init_cmd(self.params)
        self.params["FF"] = ini_hover[0]
        self.params["w_hover"] = ini_hover[1]
        self.params["thr_hover"] = ini_hover[2]
        self.thr = np.ones(6)*ini_hover[2]
        self.tor = np.ones(6)*ini_hover[3]

        # Initial State
        self.state = init_state(self.params)

        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17],
                               self.state[19], self.state[21], self.state[23]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator
        self.integrator = ode(self.state_dot).set_integrator(
            'dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)

    def extended_state(self):
        # ... (tidak ada perubahan)
        self.dcm = utils.quat2Dcm(self.quat)
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1]
        self.psi = YPR[0]
        self.theta = YPR[1]
        self.phi = YPR[2]

    def forces(self):
        # ... (tidak ada perubahan)
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    # 2. TAMBAHKAN FUNGSI BARU untuk konversi motor speeds ke forces/moments
    def motor_speeds_to_forces_moments(self, wMotor):
        """
        Konversi kecepatan motor ke total forces dan moments menggunakan mixer matrix.

        Args:
            wMotor: array kecepatan motor [w1, w2, w3, w4, w5, w6] (rad/s)

        Returns:
            [F_total, Mx, My, Mz]: Total thrust dan moments
        """
        omega_sq = wMotor * wMotor
        forces_moments = self.params["mixerFM"] @ omega_sq

        F_total = forces_moments[0]    # Total thrust (N)
        Mx = forces_moments[1]         # Roll moment (N·m)
        My = forces_moments[2]         # Pitch moment (N·m)
        Mz = forces_moments[3]         # Yaw moment (N·m)

        return F_total, Mx, My, Mz

    # 3. UBAH FUNGSI state_dot - INI PERUBAHAN UTAMA
    def state_dot(self, t, state, cmd, wind):

        # Import Params (sama seperti asli)
        mB = self.params["mB"]
        g = self.params["g"]
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        IB = self.params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]
        Cd = self.params["Cd"]

        kTh = self.params["kTh"]
        kTo = self.params["kTo"]
        tau = self.params["tau"]
        kp = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (config.usePrecession):
            uP = 1
        else:
            uP = 0

        # Import State Vector (sama seperti asli)
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]
        xdot = state[7]
        ydot = state[8]
        zdot = state[9]
        p = state[10]
        q = state[11]
        r = state[12]
        wM1 = state[13]
        wdotM1 = state[14]
        wM2 = state[15]
        wdotM2 = state[16]
        wM3 = state[17]
        wdotM3 = state[18]
        wM4 = state[19]
        wdotM4 = state[20]
        wM5 = state[21]
        wdotM5 = state[22]
        wM6 = state[23]
        wdotM6 = state[24]

        # Motor Dynamics (SAMA seperti asli - tidak diubah)
        uMotor = cmd
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*uMotor[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*uMotor[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*uMotor[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*uMotor[3])/(tau**2)
        wddotM5 = (-2.0*damp*tau*wdotM5 - wM5 + kp*uMotor[4])/(tau**2)
        wddotM6 = (-2.0*damp*tau*wdotM6 - wM6 + kp*uMotor[5])/(tau**2)

        wMotor = np.array([wM1, wM2, wM3, wM4, wM5, wM6])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)

        # *** PERUBAHAN UTAMA: Gunakan mixer matrix alih-alih individual thrust ***
        # LAMA (dihapus):
        # thrust = kTh*wMotor*wMotor
        # torque = kTo*wMotor*wMotor
        # ThrM1 = thrust[0]
        # ThrM2 = thrust[1]
        # ... dst

        # BARU: Gunakan mixer matrix
        F_total, Mx, My, Mz = self.motor_speeds_to_forces_moments(wMotor)

        # Wind Model (sama seperti asli)
        velW = 0
        qW1 = 0
        qW2 = 0

        # *** PERUBAHAN UTAMA: State Derivatives menggunakan matrix result ***
        if (config.orient == "NED"):
            DynamicsDot = np.array([
                [xdot],
                [ydot],
                [zdot],
                [-0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [-0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],

                # PERUBAHAN: Gunakan F_total alih-alih individual thrust
                [(Cd*sign(-xdot)*(-xdot)**2 - 2*(q0*q2 + q1*q3) * F_total)/mB],
                [(Cd*sign(-ydot)*(-ydot)**2 + 2*(q0*q1 - q2*q3) * F_total)/mB],
                [(-Cd*sign(+zdot)*(+zdot)**2 - F_total *
                  (q0**2 - q1**2 - q2**2 + q3**2) + g*mB)/mB],

                # PERUBAHAN: Gunakan Mx, My, Mz alih-alih hardcoded coefficients
                # LAMA (dihapus):
                # [((IByy - IBzz)*q*r - uP*IRzz*(wM1 - wM2 + wM3 - wM4 + wM5 - wM6)*q +
                #   (ThrM1 - ThrM2 - 0.5*ThrM3 + 0.5*ThrM4 + 0.5*ThrM5 - 0.5*ThrM6)*dym)/IBxx],

                # BARU:
                [((IByy - IBzz)*q*r - uP*IRzz *
                  (wM1 - wM2 + wM3 - wM4 + wM5 - wM6)*q + Mx)/IBxx],
                [((IBzz - IBxx)*p*r + uP*IRzz *
                  (wM1 - wM2 + wM3 - wM4 + wM5 - wM6)*p + My)/IByy],
                [((IBxx - IByy)*p*q + Mz)/IBzz]
            ])

        elif (config.orient == "ENU"):
            DynamicsDot = np.array([
                [xdot],
                [ydot],
                [zdot],
                [-0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r],
                [0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r],
                [0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r],
                [-0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r],

                # PERUBAHAN: Gunakan F_total untuk ENU juga
                [(Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)
                  * cos(qW2) - xdot)**2 + 2*(q0*q2 + q1*q3)*F_total)/mB],
                [(Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)
                  * cos(qW2) - ydot)**2 - 2*(q0*q1 - q2*q3)*F_total)/mB],
                [(-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot) **
                  2 + F_total*(q0**2 - q1**2 - q2**2 + q3**2) - g*mB)/mB],

                # PERUBAHAN: Gunakan Mx, My, Mz untuk ENU juga
                [((IByy - IBzz)*q*r + uP*IRzz *
                  (wM1 - wM2 + wM3 - wM4 + wM5 - wM6)*q + Mx)/IBxx],
                [((IBzz - IBxx)*p*r - uP*IRzz *
                  (wM1 - wM2 + wM3 - wM4 + wM5 - wM6)*p + My)/IByy],
                [((IBxx - IByy)*p*q + Mz)/IBzz]
            ])

        # State Derivative Vector (sama seperti asli)
        sdot = np.zeros([25])
        sdot[0] = DynamicsDot[0]
        sdot[1] = DynamicsDot[1]
        sdot[2] = DynamicsDot[2]
        sdot[3] = DynamicsDot[3]
        sdot[4] = DynamicsDot[4]
        sdot[5] = DynamicsDot[5]
        sdot[6] = DynamicsDot[6]
        sdot[7] = DynamicsDot[7]
        sdot[8] = DynamicsDot[8]
        sdot[9] = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4
        sdot[21] = wdotM5
        sdot[22] = wddotM5
        sdot[23] = wdotM6
        sdot[24] = wddotM6

        self.acc = sdot[7:10]
        return sdot

    # update() function tetap sama - tidak ada perubahan
    def update(self, t, Ts, cmd, wind):
        prev_vel = self.vel
        prev_omega = self.omega

        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t+Ts)

        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17],
                                self.state[19], self.state[21], self.state[23]])
        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts

        self.extended_state()
        self.forces()


# =====================================
# RANGKUMAN PERUBAHAN:
# =====================================

"""
PERUBAHAN MINIMAL YANG DIPERLUKAN:

1. TAMBAH import untuk initHexa_fixed (dengan mixer matrix yang benar)

2. TAMBAH fungsi motor_speeds_to_forces_moments():
   - Konversi wMotor ke [F_total, Mx, My, Mz] menggunakan mixer matrix
   - Menggantikan perhitungan individual thrust/torque

3. UBAH di state_dot():
   - Hapus: thrust = kTh*wMotor*wMotor dan ThrMx assignments
   - Tambah: F_total, Mx, My, Mz = self.motor_speeds_to_forces_moments(wMotor)
   - Ganti semua (ThrM1 + ThrM2 + ... + ThrM6) dengan F_total
   - Ganti hardcoded coefficients untuk roll/pitch/yaw dengan Mx, My, Mz

4. UPDATE initHexa.py:
   - Perbaiki makeMixerFM() dengan konfigurasi motor yang benar
   - Sesuaikan angles dan yaw_dir

KEUNTUNGAN:
✅ Konsistensi matematiks antara mixer dan dynamics
✅ Konfigurasi motor yang benar sesuai hexacopter X layout
✅ Matrix-based calculation (lebih reliable)
✅ Mudah di-debug dan diverifikasi
"""
