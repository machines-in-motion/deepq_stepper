## This demo shows how to use the terrain generation functions
## Author : Avadesh Meduri
## Date : 10/08/2020

import numpy as np
import rospkg
import pybullet as p
import time

from py_bullet_env.bullet_env_handler import TerrainHandler, TerrainGenerator

from py_bullet_deepq_stepper.dq_stepper import DQStepper, InvertedPendulumEnv, Buffer
from py_bullet_env.bullet_bolt_env import BoltBulletEnv

import time
from matplotlib import pyplot as plt
import random

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

step_time = 0.1
stance_time = 0.02
ht = 0.28
off = 0.04
w = [0.0, 1, 0.0]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/dqs_1_old")

# dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
#     no_actions= no_actions, trained_model='../models/bolt/lipm_walking/dqs_3')

##################################################################

terr_gen = TerrainGenerator('/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/shapes')


terr = TerrainHandler(bolt_env.robot.robotId)

max_length = 1.
terr_gen.create_random_terrain(150, max_length)

###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_steps = 100
v_init = [0.0, 0]
v_des = [-0.5, 0.0]
##################################################################


x, xd, u, n = bolt_env.reset_env([0, 0, ht + off, v_init[0], v_init[1]])
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]
epi_cost = 0
for i in range(no_steps):
    
    terrain = dqs.x_in[:,8:].copy()
    for i in range(len(dqs.x_in)):
        u_x = env.action_space_x[int(dqs.x_in[i,8])] + u[0]
        u_y = n*env.action_space_y[int(dqs.x_in[i,9])] + u[1]
        u_z = terr.return_terrain_height(u_x, u_y)
        terrain[i,2] = np.around(u_z - u[2], 2)
    q_values, _ = dqs.predict_q(state, terrain[:,2])
    
    # while True:
    #     action_index = np.argmin(q_values)
    #     if u[2] + terrain[action_index, 2] > -0.02:
    #         action = terrain[action_index]
    #         break
    #     else:
    #         del_u_x = env.action_space_x[int(terrain[action_index][0])] + u[0]
    #         del_u_y = n*env.action_space_y[int(terrain[action_index][1])] + u[1]
    #         del_u_z = terrain[action_index][2]
    #         p.addUserDebugLine([del_u_x, del_u_y, del_u_z],[del_u_x, del_u_y, del_u_z + 0.5],[0,1,1], 3)
    #         terrain = np.delete(terrain, action_index, 0)
    
    action_index = np.argmin(q_values)
    action = terrain[action_index]

    u_x = env.action_space_x[int(action[0])] + u[0]
    u_y = n*env.action_space_y[int(action[1])] + u[1]
    u_z = action[2] + u[2]
    # print(u_z)
    x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
    next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
    state = next_state
    u = u_new
    epi_cost += cost

    p.removeAllUserDebugItems()
    
    if done:
        # print('terminated ..')
        break
    
print('episode cost is : ' + str(epi_cost))
bolt_env.plot()




