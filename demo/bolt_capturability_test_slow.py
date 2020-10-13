## This file compares the capability of the deepQ stepper to recover when sub optimal steps are taken
## by comparing the stepper learning in lipm env and bullet env
## Date : 28/05/2020
## Author : Avadesh Meduri

import random
import numpy as np
import pybullet as p
from py_bullet_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv, Buffer
from py_bullet_env.bullet_bolt_env import BoltBulletEnv

import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

kp = [30, 30, 30]#[150, 150, 150]
kd = [5, 5, 5] #[15, 15, 15]
kp_com = [0, 0, 30] #[0, 0, 150.0]
kd_com = [0, 0, 10] #[0, 0, 20.0]
kp_ang_com = [40, 40, 0]#[100, 100, 0]
kd_ang_com = [5, 5, 0]#[25, 25, 0]

F = [0, 0, 0]

step_time = 0.2
stance_time = 0.00
ht = 0.35
off = 0.0
w = [0.5, 3.5, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, [0.2, 0.2], w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################

dqs_1 = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/dqs_1")

dqs_2 = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/dqs_1_vel")


dqs_arr = [dqs_1, dqs_2]
dqs_ct = 0
###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_epi = 1
no_steps = 10

step_number = 0
opt = 6
option_number = 0
##################################################################
history = []

for dqs in dqs_arr:

    v_init = [0.0, 0.0]
    v_des = [0, 0]
    x, xd, u, n = bolt_env.reset_env([0, 0, ht, v_init[0], v_init[1]])
    state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]
    print(state)        
    epi_cost = 0
    for i in range(no_steps):

        terrain = dqs.x_in[:,7:].copy()
        counter = 0
        for k in range(len(dqs.x_in)):
            u_x = env.action_space_x[int(dqs.x_in[k,8])] + u[0]
            u_y = n*env.action_space_y[int(dqs.x_in[k,9])] + u[1]
            u_z = 0
            terrain[k,3] = np.around(u_z - u[2], 2)
        q_values, _ = dqs.predict_q(state, terrain[:,3])
        terrain[:,0] = np.reshape(q_values, (len(terrain[:,0],)))
        terrain = np.round(terrain[terrain[:,0].argsort()], 2)
        
        # print(terrain[0:12])
        
        # assert False
        if i == step_number:
            option_number = opt

        ## This intentionally chooses the nth best action
        for action_index in range(len(terrain[:,0])):
            if action_index < option_number:
                del_u_x = env.action_space_x[int(terrain[action_index][1])] + u[0]
                del_u_y = n*env.action_space_y[int(terrain[action_index][2])] + u[1]
                del_u_z = terrain[action_index][3]
                if action_index == 0:
                    p.addUserDebugLine([del_u_x, del_u_y, del_u_z],[del_u_x, del_u_y, del_u_z + 0.5],[1, 1, 0], 3)
                else:
                    p.addUserDebugLine([del_u_x, del_u_y, del_u_z],[del_u_x, del_u_y, del_u_z + 0.5],[0, 5*action_index/20,1], 3)
                terrain = np.delete(terrain, action_index, 0)
            else:
                action = terrain[action_index][1:]
                u_x = env.action_space_x[int(action[0])] + u[0]
                u_y = n*env.action_space_y[int(action[1])] + u[1]
                u_z = action[2] + u[2]
                p.addUserDebugLine([u_x, u_y, u_z],[u_x, u_y, u_z + 0.5],[1, 0,1], 3)

                option_number = 0
                break

        if i == step_number:
            time.sleep(1.5)
            # plot_heatmap(q_values, dqs, env)

                    
        x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
        if i == step_number:
            print("state end step : ",  np.round([x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1]], 2))
            print("action:", action)
        next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
        state = next_state
        u = u_new
        epi_cost += cost
        
        
        
        p.removeAllUserDebugItems()
    
    dqs_ct += 1

