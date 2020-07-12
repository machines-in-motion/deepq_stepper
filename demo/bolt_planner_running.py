## This is the demo of bolt running on flat ground
## Author: Avadesh Meduri
## Date : 28/06/2020

import numpy as np
from py_bullet_env.bullet_cent_bolt_env import BulletCentBoltEnv
from py_deepq_stepper.cent_dq_stepper import CentEnv, DQStepper
#####################################################################
kp = [35, 35, 25]
kd = [10, 10, 10]
kp_com = [0, 0, 20]
kd_com = [0, 0, 10]
kp_ang_com = [0, 0, 0]
kd_ang_com = [0, 0, 0]

step_time = 0.1
air_time = 0.1
ht = 0.25

bolt_env = BulletCentBoltEnv(ht, step_time, air_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com)
###################################################################

env = CentEnv(ht, 0.13, 0.6, [0.5, 3.0, 1.5], [11,8])
dqs = DQStepper(env, lr=1e-4, gamma=0.98, use_tarnet= True, trained_model='../models/dqs_1_old')

###################################################################
F = [0, 0, 0]
w = 0.0
no_steps = 5
des_com = [0.0, 0, ht]
des_vel = [0.0, 0.0, 0]
x_ori = [0, 0, 0, 1]
x_angvel = [0, 0, 0]

x, u, n = bolt_env.reset_env()
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], x[3], x[4], n, des_vel[0], des_vel[1]]
bolt_env.update_gains([20, 20, 20], [2, 2, 2], [20, 20, 20], [20, 20, 20], [100, 100, 0], [20, 20, 0])

for i in range(no_steps):
    
   
    terrain = 0.0
    action = dqs.predict_q(state, terrain)[1]
    u_x = u[0] + env.action_space_x[action[0]]
    u_y = u[1] + n*env.action_space_y[action[1]]
    u_z = u[2] + action[2]

    # u_x = -0.0
    u_y = np.power(-1, i)*0.04
    u_z = 0

    x, u_new, n = bolt_env.step_env([u_x, u_y, u_z])
    u = u_new
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], x[3], x[4], n, des_vel[0], des_vel[1]]
    