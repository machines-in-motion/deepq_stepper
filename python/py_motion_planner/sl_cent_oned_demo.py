# This is a demo to understand angular momementum control with sl motionplanner
# Author : Avadesh Meduri
# Date : 29/06/2020

import numpy as np
from py_motion_planner.sl_cent_motion_planner import SLCentMotionPlanner
from matplotlib import pyplot as plt 

dt = 0.01
max_force = [25, 25]
max_ht = [0.32, 0.32]

w = np.array([1e-3, 1e-3, 1e+2, 1e+2, 1e+3, 1e+3, 1e-1, 1e-1])
ter_w = np.array([1e+7, 1, 1e+2, 1e+2, 1e+3, 1e+3])

step_time = 0.2
stance_time = 0.1
no_steps = 5
horizon = step_time + stance_time
step_size_x = 0.582 #0.4465  #0.582
step_size_y = 0.

k_lipm = 1.28*9.81/0.28
k = [1.2*k_lipm, 0.5*k_lipm]

k_arr = np.tile([k, k],(int(np.round(horizon/dt, 2)), 1, 1))
x0 = np.zeros(12)
x0[2] = 0.28
x0[3] = 1
xT = np.zeros(12)
xT[2] = 0.28

u0x = 0.0
u0y = 0.0

mp = SLCentMotionPlanner(dt, 2, 1.28, [0.016, 0.031, 0.031], max_force, max_ht)

for i in range(no_steps):
    if np.power(-1, i) > 0:
        cnt_plan_mpc = [[[1, u0x, u0y, 0.0, 0, step_time],[0, 0, 0.0, 0, 0, step_time]],
                        [[0, 0, 0.0, 0.0, step_time, step_time + stance_time],[0, 0, 0.0, 0, step_time, step_time + stance_time]]]
    elif np.power(-1, i) < 0:
        cnt_plan_mpc = [[[0, 0, -0.0, 0.0, 0, step_time],[1, u0x, -u0y, 0, 0, step_time]],
                        [[0, 0, -0.0, 0.0, step_time, step_time + stance_time],[0, 0, 0.0, 0, step_time, step_time + stance_time]]]
    
    traj, cent_force, force = mp.optimize(x0, cnt_plan_mpc, k_arr, xT, w, ter_w, horizon)
    x0 = traj[:,-1]
    if i == 0:
        traj_mpc = traj
        force_mpc = force
    else:
        traj_mpc = np.concatenate((traj_mpc[:,:-1], traj), axis = 1)
        force_mpc = np.concatenate((force_mpc[:,:-1], force), axis = 1)

    u0x += step_size_x
    u0x += step_size_y











print(np.shape(traj_mpc))


t = np.linspace(0, no_steps*horizon, no_steps*int(np.round(horizon/dt, 2)) + 1)

fig, axs = plt.subplots(4,1)
axs[0].plot(t, traj_mpc[0], label = 'com_x_mpc')
axs[0].grid()
axs[0].legend()
axs[1].plot(t, traj_mpc[1], label = 'com_y_mpc')
axs[1].grid()
axs[1].legend()
axs[2].plot(t, traj_mpc[2], label = 'com_z_mpc')
axs[2].grid()
axs[2].legend()
for n in range(2):
    axs[3].plot(t, force_mpc[n], label = 'f_mpc' + str(n))
axs[3].grid()
axs[3].legend()

fig, axs = plt.subplots(3,1)
axs[0].plot(t, traj_mpc[3], label = 'com_xd_mpc')
axs[0].grid()
axs[0].legend()
axs[1].plot(t, traj_mpc[4], label = 'com_yd_mpc')
axs[1].grid()
axs[1].legend()
axs[2].plot(t, traj_mpc[5], label = 'com_zd_mpc')
axs[2].grid()
axs[2].legend()

fig, axs = plt.subplots(3,1)
axs[0].plot(t, (180/np.pi)*traj_mpc[6], label = 'com_ang_x_mpc')
axs[0].grid()
axs[0].legend()
axs[1].plot(t, (180/np.pi)*traj_mpc[7], label = 'com_ang_y_mpc')
axs[1].grid()
axs[1].legend()
axs[2].plot(t, (180/np.pi)*traj_mpc[8], label = 'com_ang_z_mpc')
axs[2].grid()
axs[2].legend()

plt.show()
