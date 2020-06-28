## This is the demo of bolt running on flat ground
## Author: Avadesh Meduri
## Date : 28/06/2020

import numpy as np
from py_bullet_env.bullet_slc_bolt_env import BulletSLCBoltEnv
#####################################################################
kp = [35, 35, 25]
kd = [10, 10, 10]
kp_com = [0, 0, 20]
kd_com = [0, 0, 10]
kp_ang_com = [0, 0, 0]
kd_ang_com = [0, 0, 0]

step_time = 0.2
air_time = 0.2
ht = 0.25

bolt_env = BulletSLCBoltEnv(ht, step_time, air_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)


###################################################################
F = [0, 0, 0]
w = 0.0
no_steps = 5
des_com = [0.0, 0, ht]
des_vel = [0.0, 0.0, 0]
x_ori = [0, 0, 0, 1]
x_angvel = [0, 0, 0]

x, xd, u, n = bolt_env.reset_env()
state = [x[0] - u[0], x[1] - u[1], xd[0], xd[1], n, des_vel[0], des_vel[1]]
bolt_env.update_gains([120, 120, 100], [20, 20, 20], [150, 150, 100], [20, 20, 20], [100, 100, 0], [20, 20, 0])

for i in range(no_steps):
    
    u_x = -0.05
    u_y = np.power(-1, i)*0.065
    u_z = 0

    x, xd, u_new, n = bolt_env.step_env([u_x, u_y, u_z])
    u = u_new
    state = [x[0] - u[0], x[1] - u[1], xd[0], xd[1], n, des_vel[0], des_vel[1]]


