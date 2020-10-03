## This demo shows obstacle avoidance in 3d terrain terrain
## Author : Avadesh Meduri
## Date : 18/08/2020

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
w = [0.5, 3.0, 1.5]

bolt_env = BoltBulletEnv(ht, step_time, stance_time, kp, kd, kp_com, kd_com, kp_ang_com, kd_ang_com, w)
##################################################################
env = InvertedPendulumEnv(ht, 0.13, 0.22, w, no_actions= [11, 9])
no_actions = [len(env.action_space_x), len(env.action_space_y)]
print(no_actions)

###################################################################


# dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
#     no_actions= no_actions, trained_model='../models/bolt/lipm_walking/dqs_3')

dqs = DQStepper(lr=1e-4, gamma=0.98, use_tarnet= True, \
    no_actions= no_actions, trained_model = "../models/bolt/bullet_walking/dqs_2")


##################################################################

terr_gen = TerrainGenerator('/home/ameduri/py_devel/workspace/src/catkin/deepq_stepper/python/py_bullet_env/shapes')

terr = TerrainHandler(bolt_env.robot.robotId)

terr_gen.create_random_terrain(30, 0.4, [0.7, 0.])
terr_gen.create_random_terrain(10, 0.4, [1.5, 0.5])


terr_gen.create_random_terrain(2, 0.1, [1.3, 0.5])

terr_gen.load_terrain("../python/py_bullet_env/terrains/uneven.urdf")

###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_steps = 100
##################################################################

v_init = [0, 0]
v_des = [0.5, 0]
x_init = [0, 0]

x, xd, u, n = bolt_env.reset_env([x_init[0], x_init[1], ht + off, v_init[0], v_init[1]])
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]    
epi_cost = 0

# p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[1.0,0,0])

p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[1.0,0,0])

# bolt_env.start_recording("3d_obstacle_avoidance.mp4")

for i in range(no_steps):
    # print(x[0:2])
    if x[0] > 0.9 and x[0] < 1.4 and x[1] < 0.6:
        # print('change1')
        v_des = [0, 0.5]
    elif x[1] < 0.8 and x[1] > 0.5 and x[0] < 1.45 and x[0] > 0.9:
        # print('change2')
        v_des = [0.5, 0]
    elif x[0] > 1.35 and x[1] > 0.0:
        # print('change3')
        v_des = [-.1, -0.5]
    elif x[0] > 1.4 and x[1] < -0.0:
        # print('change4')
        v_des = [-.0, 0]
    terrain = dqs.x_in[:,8:].copy()
     
    for k in range(len(dqs.x_in)):
        u_x = env.action_space_x[int(dqs.x_in[k,8])] + u[0]
        u_y = n*env.action_space_y[int(dqs.x_in[k,9])] + u[1]
        u_z = terr.return_terrain_height(u_x, u_y)
        terrain[k,2] = np.around(u_z - u[2], 2)
    q_values, _ = dqs.predict_q(state, terrain[:,2])
    
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
    
    # if done:
    #     print('terminated ..')
    #     break
  
bolt_env.plot()
# bolt_env.stop_recording()


    


