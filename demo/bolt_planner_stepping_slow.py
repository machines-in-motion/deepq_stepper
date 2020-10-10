## This demo is for bolt stepping with slower step times on plannar grounds
## Author : Avadesh Meduri
## Date : 8/10/2020

import random
import numpy as np
from py_bullet_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv, Buffer
from py_bullet_env.bullet_bolt_env import BoltBulletEnv

import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

kp = [150, 150, 150]
kd = [15, 15, 15]
kp_com = [0, 0, 150.0]
kd_com = [0, 0, 20.0]
kp_ang_com = [100, 100, 0]
kd_ang_com = [25, 25, 0]

F = [0, 0, 0]

step_time = 0.2
stance_time = 0.0
ht = 0.28
w = [0.5, 3, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.2, w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/dqs_1_vel")

# dqs_x = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
#     no_actions= [11,1], trained_model = "../models/dqs_x11")
# dqs_y = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
#     no_actions= [1,9], trained_model = "../models/dqs_y9")


###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_epi = 2
no_steps = 40

##################################################################


v_init = np.round([0.3*random.uniform(-1.0, 1.0), 0.3*random.uniform(-1.0, 1.0)], 2)
# v_des = [0.5*random.randint(-0, 1), 0.5*random.randint(-1, 1)]
# v_init = [0, 0]
print(v_init)
v_des = [0.0, 0.0]
x, xd, u, n = bolt_env.reset_env([0, 0, ht, v_init[0], v_init[1]])
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]

# bolt_env.start_recording("slow_stepping.mp4")

epi_cost = 0
for i in range(no_steps):
    action = dqs.predict_q(state, terrain)[1]
    print(state, action)
    
    # for x axis
    # state_x = state.copy()
    # state_x[1] = 0.0
    # state_x[4] = 0.0
    # action_x = dqs_x.predict_q(state_x, 0.0)[1] 
    # # for y axis
    # state_y = state.copy()
    # state_y[0] = 0.0
    # state_y[3] = 0.0
    # action_y = dqs_y.predict_q(state_y, 0.0)[1] 
    # action = np.array([int(action_x[0]), int(action_y[1]), 0])

    u_x = env.action_space_x[int(action[0])] + u[0]
    u_y = n*env.action_space_y[int(action[1])] + u[1]
    u_z = action[2] + u[2]
    
    x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
    next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
    state = next_state
    u = u_new
    epi_cost += cost


    if done:
        print("terminated.................")
        break
    # if not done:

# bolt_env.stop_recording()
# bolt_env.plot()