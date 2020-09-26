## demo of blot stepping on 3d terrain with dq stepper
## Date : 4/06/2020
## Author : Avadesh Meduri

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

terr_gen.create_box([0, 0, 0.0], [0, 0, 0, 1], '3cm')
terr_gen.create_box([0., 0.15, 0.0], [0, 0, 0, 1], '4cm')
terr_gen.create_box([0, -0.15, 0], [0, 0, 0, 1], '3cm')

terr_gen.create_box([0.25, -0.15, 0.0], [0.1, 0, 0., 1], '4cm')
terr_gen.create_box([0.25,  0.15, 0.0], [0, 0.1, 0, 1], '4cm')

terr_gen.create_box([0.5,  -0.05, 0], [0., 0, 0.1, 1], '3cm')
terr_gen.create_box([0.5,  0.05, 0], [0, 0.1, 0, 1], '4cm')

terr_gen.create_box([0.4,  0.3, -0.0], [0, 0, 0, 1], '4cm')
terr_gen.create_box([0.4,  -0.3, -0.0], [0, 0, 0, 1], '4cm')

terr_gen.create_box([0.2,  -0.4, -0.01], [0, 1, 0., 1], '3cm')
terr_gen.create_box([0.2,  0.4, -0.01], [0, 1, 0., 1], '3cm')

terr_gen.create_box([0.6,  -0.4, 0.02], [0, 0, 0., 1], '3cm')
terr_gen.create_box([0.6,  0.4, 0.04], [0, 0, 0., 1], '3cm')

terr_gen.create_sphere([0.7, 0.2, -0.1], '14cm')
terr_gen.create_sphere([0.7, 0.1, -0.15], '20cm')
terr_gen.create_sphere([0.7, 0.0, -0.1], '14cm')

terr_gen.create_sphere([0.8, 0.1, -0.1], '14cm')
terr_gen.create_sphere([0.8, 0.0, -0.14], '20cm')
terr_gen.create_sphere([0.8, -0.2, -0.08], '14cm')

terr_gen.create_sphere([1.0, 0.2, -0.08], '14cm')
terr_gen.create_sphere([1.0, 0.1, -0.15], '20cm')
terr_gen.create_sphere([1.0, 0.0, -0.08], '14cm')

terr_gen.create_sphere([1.1, -0.2, -0.11], '14cm')
terr_gen.create_sphere([1.0, -0.1, -0.16], '20cm')
terr_gen.create_sphere([1.0, 0.0, -0.12], '14cm')

terr_gen.create_sphere([1.0, -0.4, -0.1], '14cm')
terr_gen.create_sphere([1.1, -0.2, -0.15], '20cm')
terr_gen.create_sphere([1.2, -0.3, -0.1], '14cm')

terr_gen.create_sphere([1.3, 0.4, -0.1], '14cm')
terr_gen.create_sphere([1.3, 0.2, -0.15], '20cm')
terr_gen.create_sphere([1.3, 0.3, -0.1], '14cm')




terr_gen.create_box([1.0,  0.4, 0.01], [0, 0, 0.1, 1], '3cm')
terr_gen.create_box([0.8,  0.4, 0.04], [0, 0.1, 0., 1], '3cm')
terr_gen.create_box([1.2,  0.5, 0.02], [0.1, 0, 0., 1], '3cm')

terr_gen.create_random_terrain(10, 0.2, [1.1, 0.75, 0])
terr_gen.create_random_terrain(10, 0.2, [0.8, 0.75, 0])

terr_gen.create_random_terrain(10, 0.3, [1.0, 1., 0])

terr_gen.create_box([1.4, 0.9, 0.01], [0, 0, 0, 1], '3cm')
terr_gen.create_box([1.4, 0.75, 0.01], [0, 0, 0, 1], '3cm')


# terr_gen.create_box([1.65, 0.75, 0.01], [0, 0, 0, 1], '3cm')
# terr_gen.create_box([1.65, 0.95, 0.01], [0, 0, 0, 1], '3cm')


# terr_gen.load_terrain("../python/py_bullet_env/terrains/stairs.urdf")

###################################################################
terrain = np.zeros(no_actions[0]*no_actions[1])
no_steps = 300
##################################################################

v_init = [0, 0]
v_des = [0.5, 0]
x_init = [0, 0]

x, xd, u, n = bolt_env.reset_env([x_init[0], x_init[1], ht + off, v_init[0], v_init[1]])
state = [x[0] - u[0], x[1] - u[1], x[2] - u[2], xd[0], xd[1], n, v_des[0], v_des[1]]    
epi_cost = 0

p.resetDebugVisualizerCamera( cameraDistance=0.7, cameraYaw= 55, cameraPitch=-25, cameraTargetPosition=[0.3,0,0])

# bolt_env.start_recording("3d_stepping_stones.mp4")

for i in range(no_steps):
    print(x[0:2])
    if x[0] > 0.75 and x[0] < 1.0 and x[1] < 0.75:
        # p.resetDebugVisualizerCamera( cameraDistance=1.0, cameraYaw=130, cameraPitch=-25, cameraTargetPosition=[0.5,0.1,0])

        v_des = [0, 0.5]
    elif x[0] > 0.75 and x[1] > 0.7 and x[1] < 1.0:
        # p.resetDebugVisualizerCamera( cameraDistance=1.0, cameraYaw=130, cameraPitch=-25, cameraTargetPosition=[0.5,0.1,0])

        v_des = [0.5, 0]
    terrain = dqs.x_in[:,7:].copy()
    
    for k in range(len(dqs.x_in)):
        u_x = env.action_space_x[int(dqs.x_in[k,8])] + u[0]
        u_y = n*env.action_space_y[int(dqs.x_in[k,9])] + u[1]
        u_z = terr.return_terrain_height(u_x, u_y)
        terrain[k,3] = np.around(u_z - u[2], 2)
    q_values, _ = dqs.predict_q(state, terrain[:,3])
    terrain[:,0] = np.reshape(q_values, (len(terrain[:,0],)))
    terrain = np.round(terrain[terrain[:,0].argsort()], 2)
     
    ## This intentionally chooses the nth best action
    for action_index in range(len(terrain[:,0])):
        if terrain[action_index, 3] + u[2] < 0.01:
            del_u_x = env.action_space_x[int(terrain[action_index][1])] + u[0]
            del_u_y = n*env.action_space_y[int(terrain[action_index][2])] + u[1]
            del_u_z = terrain[action_index][3]
            p.addUserDebugLine([del_u_x, del_u_y, del_u_z],[del_u_x, del_u_y, del_u_z + 0.5],[0, action_index/10,1], 3)
            terrain = np.delete(terrain, action_index, 0)
        else:
            action = terrain[action_index][1:]
            u_x = env.action_space_x[int(action[0])] + u[0]
            u_y = n*env.action_space_y[int(action[1])] + u[1]
            u_z = action[2] + u[2]
            p.addUserDebugLine([u_x, u_y, u_z],[u_x, u_y, u_z + 0.5],[1, 0,1], 3)
            break

    # print(u_z)
    x, xd, u_new, n, cost, done = bolt_env.step_env([u_x, u_y, u_z], v_des, F)
    next_state = np.round([x[0] - u_new[0], x[1] - u_new[1], x[2] - u_new[2], xd[0], xd[1], n, v_des[0], v_des[1]], 2)
    state = next_state
    u = u_new
    epi_cost += cost
    
    # if i == 0:
    #     # time.sleep(100)
    #     width, height, rgbImg, depthImg, segImg = p.getCameraImage(width = 1000, height = 800)
    #     import matplotlib 
    #     matplotlib.image.imsave('stepping_stones.png', rgbImg)
    #     assert False
    
    # if done:
    #     print('terminated ..')
    #     break
    p.removeAllUserDebugItems()

bolt_env.stop_recording()


