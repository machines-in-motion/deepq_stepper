## This is a demo file for the spring loaded centroidal motion planner
## Author : Avadesh Meduri
## Date : 23/06/2020

import numpy as np
from py_motion_planner.sl_cent_motion_planner import SLCentMotionPlanner
from matplotlib import pyplot as plt 

dt = 0.05
max_force = [25, 25]
max_ht = [0.32, 0.32]

w = np.array([1e-3, 1e-3, 1e+2, 1e+2, 1, 1, 1e+2, 1e+2])
ter_w = np.array([1e+8, 1e+1, 1e+2, 1e+6, 1, 1])

step_time = 0.1
stance_time = 0.1
horizon = 4*(step_time + stance_time)

cnt_plan = [[[1, 0, -0.065, 0.0, 0, step_time],[0, 0, 0.065, 0, 0, step_time]],
            [[0, 0, 0.0, 0.0, step_time, step_time + stance_time],[0, 0, 0.0, 0, step_time, step_time + stance_time]],
            [[0, 0.0, -0.065, 0, step_time + stance_time, 2*step_time + stance_time],[1, 0.0, 0.065, 0, step_time + stance_time, 2*step_time + stance_time]],
            [[0, 0.0, -0.0, 0, 2*step_time + stance_time, 2*step_time + 2*stance_time],[0, 0.0, 0.0, 0, 2*step_time + stance_time, 2*step_time + 2*stance_time]],
            [[1, 0.0, -0.065, 0, 2*(step_time + stance_time), 3*step_time + 2*stance_time],[0, 0.0, 0.065, 0, 2*(step_time + stance_time), 3*step_time + 2*stance_time]],
            [[0, 0.0, -0.0, 0, 3*step_time + 2*stance_time, 3*step_time + 3*stance_time],[0, 0.0, 0.0, 0, 3*step_time + 2*stance_time, 3*step_time + 3*stance_time]],
            [[0, 0.0, -0.065, 0, 3*(step_time + stance_time), 4*step_time + 3*stance_time],[1, 0.0, 0.065, 0, 3*(step_time + stance_time), 4*step_time + 3*stance_time]],
            [[0, 0.0, -0.0, 0, 4*step_time + 3*stance_time, 4*step_time + 4*stance_time],[0, 0.0, 0.0, 0, 4*step_time + 3*stance_time, 4*step_time + 4*stance_time]]]


k_arr = np.tile([[10, 5], [10,5]],(int(np.round(horizon/dt, 2)), 1, 1))
x0 = np.zeros(12)
x0[2] = 0.28
x0[3] = 0.1
xT = np.zeros(12)
xT[2] = 0.28

mp = SLCentMotionPlanner(dt, 2, 1, [1, 1, 1], max_force, max_ht)
traj_full, cent_force_full, force_full = mp.optimize(x0, cnt_plan, k_arr, xT, w, ter_w, horizon)

for i in range(4):
    if np.power(-1, i) > 0:
        cnt_plan_mpc = [[[1, 0, -0.065, 0.0, 0, step_time],[0, 0, 0.065, 0, 0, step_time]],
                        [[0, 0, -0.065, 0.0, step_time, step_time + stance_time],[0, 0, 0.065, 0, step_time, step_time + stance_time]]]
    elif np.power(-1, i) < 0:
        cnt_plan_mpc = [[[0, 0, -0.065, 0.0, 0, step_time],[1, 0, 0.065, 0, 0, step_time]],
                        [[0, 0, -0.065, 0.0, step_time, step_time + stance_time],[0, 0, 0.065, 0, step_time, step_time + stance_time]]]
    
    k_arr = np.tile([[10, 5], [10,5]],(int(np.round((step_time + stance_time)/dt, 2)), 1, 1))
    traj, cent_force, force = mp.optimize(x0, cnt_plan_mpc, k_arr, xT, w, ter_w, step_time + stance_time)
    x0 = traj[:,-1]
    if i == 0:
        traj_mpc = traj
        force_mpc = force
    else:
        traj_mpc = np.concatenate((traj_mpc[:,:-1], traj), axis = 1)
        force_mpc = np.concatenate((force_mpc[:,:-1], force), axis = 1)

print(np.shape(traj_full), np.shape(traj_mpc))

t = np.linspace(0, horizon, int(np.round(horizon/dt, 2)) + 1)

fig, axs = plt.subplots(4,1)
axs[0].plot(t, traj_full[0], label = 'com_x')
axs[0].plot(t, traj_mpc[0], label = 'com_x_mpc')
axs[0].grid()
axs[0].legend()
axs[1].plot(t, traj_full[1], label = 'com_y')
axs[1].plot(t, traj_mpc[1], label = 'com_y_mpc')
axs[1].grid()
axs[1].legend()
axs[2].plot(t, traj_full[2], label = 'com_z')
axs[2].plot(t, traj_mpc[2], label = 'com_z_mpc')
axs[2].grid()
axs[2].legend()
for n in range(2):
    axs[3].plot(t, force_full[n], label = 'f' + str(n))
    axs[3].plot(t, force_mpc[n], label = 'f_mpc' + str(n))
axs[3].plot(t, np.concatenate((cent_force_full[0], [0])),  '--', label = 'f_cent_full')
axs[3].grid()
axs[3].legend()

fig, axs = plt.subplots(3,1)
axs[0].plot(t, traj_full[3], label = 'com_xd')
axs[0].plot(t, traj_mpc[3], label = 'com_xd_mpc')
axs[0].grid()
axs[0].legend()
axs[1].plot(t, traj_full[4], label = 'com_yd')
axs[1].plot(t, traj_mpc[4], label = 'com_yd_mpc')
axs[1].grid()
axs[1].legend()
axs[2].plot(t, traj_full[5], label = 'com_zd')
axs[2].plot(t, traj_mpc[5], label = 'com_zd_mpc')
axs[2].grid()
axs[2].legend()

fig, axs = plt.subplots(3,1)
axs[0].plot(t, traj_full[6], label = 'com_ang_x')
axs[0].plot(t, traj_mpc[6], label = 'com_ang_x_mpc')
axs[0].grid()
axs[0].legend()
axs[1].plot(t, traj_full[7], label = 'com_ang_y')
axs[1].plot(t, traj_mpc[7], label = 'com_ang_y_mpc')
axs[1].grid()
axs[1].legend()
axs[2].plot(t, traj_full[8], label = 'com_ang_z')
axs[2].plot(t, traj_mpc[8], label = 'com_ang_z_mpc')
axs[2].grid()
axs[2].legend()

plt.show()
